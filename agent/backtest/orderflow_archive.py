"""Historical order flow for crypto backtests, from Bybit's public trade archive.

An order book has no public history, but the taker tape does: Bybit publishes
every trade of every day, with its taker side, at ``public.bybit.com``
(``trading/`` for USDT perpetuals, ``spot/`` for spot). This module turns those
daily files into per-minute taker buy and sell volume, caches the minutes under
the runtime root (a day of BTCUSDT perpetual trades is ~90 MB compressed; its
minutes are a few KB), and attaches order-flow columns to backtest bars:

``buy_volume``, ``sell_volume``, ``delta`` (= buy − sell), ``cvd`` (running
delta from the first bar), ``delta_ratio`` (delta / total), ``trade_count`` and
``orderflow_available``. Volumes are in the base asset.

Point-in-time safety: a bar's columns hold only trades inside that bar, which
are known at its close; the engine executes a signal at the next bar's open.
A day the archive does not have yet (today, or a listing gap) is marked
unavailable — its bars get ``orderflow_available = False`` and NaN volumes,
never zeros, so a strategy cannot read a missing tape as "no aggression".

Enable it in the backtest config::

    {"orderflow": {"source": "bybit_archive", "market": "perp"}}

``market`` defaults to ``perp`` for ``-PERP`` codes and ``spot`` otherwise.
"""

from __future__ import annotations

import gzip
import io
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd

from src.config.paths import get_runtime_root

logger = logging.getLogger(__name__)

ARCHIVE_ROOT = "https://public.bybit.com"
COLUMNS = ("buy_volume", "sell_volume", "delta", "cvd", "delta_ratio", "trade_count", "orderflow_available")
_MAX_DAYS = 400


def archive_symbol(code: str) -> tuple[str, str]:
    """Map a project code to Bybit's archive symbol and default market.

    ``BTC-USDT-PERP`` → (``BTCUSDT``, ``perp``); ``ETH-USDT`` → (``ETHUSDT``, ``spot``).
    """
    normalized = code.strip().upper()
    market = "perp" if normalized.endswith("-PERP") else "spot"
    base_quote = normalized.removesuffix("-PERP").split("-")
    if len(base_quote) != 2 or not all(part.isalnum() for part in base_quote):
        raise ValueError(f"order-flow archive needs a BASE-QUOTE crypto code, got {code!r}")
    return "".join(base_quote), market


def archive_url(symbol: str, market: str, day: date) -> str:
    """URL of one day's trade file in the Bybit public archive."""
    if market == "perp":
        return f"{ARCHIVE_ROOT}/trading/{symbol}/{symbol}{day.isoformat()}.csv.gz"
    if market == "spot":
        return f"{ARCHIVE_ROOT}/spot/{symbol}/{symbol}_{day.isoformat()}.csv.gz"
    raise ValueError(f"market must be 'perp' or 'spot', got {market!r}")


def _download(url: str) -> bytes | None:
    """GET one archive file; ``None`` when the archive does not have it (404)."""
    from backtest.loaders._http import throttled_get

    response = throttled_get(url, host_key="public.bybit.com", min_interval=0.2, timeout=120.0)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    return response.content


def parse_day(raw: bytes, market: str) -> pd.DataFrame:
    """Aggregate one compressed archive day into per-minute taker volumes.

    The perpetual file carries ``timestamp`` in seconds, ``side`` as
    ``Buy``/``Sell`` and ``size``; the spot file carries ``timestamp`` in
    milliseconds, ``side`` as ``buy``/``sell`` and ``volume``.

    Returns:
        Frame indexed by UTC minute with ``buy_volume``, ``sell_volume`` and
        ``trade_count``.
    """
    size_column = "size" if market == "perp" else "volume"
    frame = pd.read_csv(
        io.BytesIO(gzip.decompress(raw)),
        usecols=["timestamp", "side", size_column],
        dtype={"timestamp": "float64", "side": "string", size_column: "float64"},
    )
    unit = "s" if market == "perp" else "ms"
    minute = pd.to_datetime(frame["timestamp"], unit=unit, utc=True).dt.floor("min").dt.tz_localize(None)
    side = frame["side"].str.lower()
    if not side.isin(["buy", "sell"]).all():
        raise ValueError("archive file has trades without a buy/sell taker side")
    size = frame[size_column]
    grouped = pd.DataFrame({
        "minute": minute,
        "buy_volume": size.where(side == "buy", 0.0),
        "sell_volume": size.where(side == "sell", 0.0),
        "trade_count": 1,
    }).groupby("minute").sum()
    return grouped


def _cache_path(symbol: str, market: str, day: date) -> Path:
    return get_runtime_root() / "cache" / "orderflow" / "bybit" / market / symbol / f"{day.isoformat()}.csv"


def load_minutes(
    symbol: str,
    market: str,
    start: date,
    end: date,
    *,
    download: Callable[[str], bytes | None] | None = None,
) -> tuple[pd.DataFrame, list[date]]:
    """Per-minute taker volumes for ``[start, end]`` (inclusive), and missing days.

    Each day is downloaded once and cached; a missing day is never cached, so a
    later run picks it up once Bybit publishes it.
    """
    if (end - start).days + 1 > _MAX_DAYS:
        raise ValueError(f"order-flow archive range is limited to {_MAX_DAYS} days")
    fetch = download or _download
    frames, missing = [], []
    day = start
    while day <= end:
        path = _cache_path(symbol, market, day)
        if path.is_file():
            frames.append(pd.read_csv(path, index_col="minute", parse_dates=["minute"]))
        else:
            raw = fetch(archive_url(symbol, market, day))
            if raw is None:
                missing.append(day)
            else:
                minutes = parse_day(raw, market)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                minutes.to_csv(tmp, index_label="minute")
                tmp.replace(path)
                frames.append(minutes)
        day += timedelta(days=1)
    if not frames:
        return pd.DataFrame(columns=["buy_volume", "sell_volume", "trade_count"]), missing
    return pd.concat(frames).sort_index(), missing


def attach_orderflow(frame: pd.DataFrame, minutes: pd.DataFrame, missing_days: list[date]) -> pd.DataFrame:
    """Add order-flow columns to OHLCV bars indexed by their OPEN time.

    Each minute goes to the bar whose open is the latest one at or before it,
    and only while that bar is still open (before the next bar's open), so the
    last bar never absorbs minutes after the data window.
    """
    out = frame.copy()
    index = pd.DatetimeIndex(out.index)
    naive = index.tz_convert(None) if index.tz is not None else index
    if len(naive) == 0:
        for column in COLUMNS:
            out[column] = pd.Series(dtype="float64")
        return out
    step = (naive[1:] - naive[:-1]).min() if len(naive) > 1 else pd.Timedelta(days=1)
    ends = np.append(naive[1:].values, (naive[-1] + step).to_datetime64())
    buy = np.zeros(len(naive))
    sell = np.zeros(len(naive))
    count = np.zeros(len(naive))
    if not minutes.empty:
        minute_index = pd.DatetimeIndex(minutes.index).values
        slot = np.searchsorted(naive.values, minute_index, side="right") - 1
        valid = (slot >= 0) & (minute_index < ends[np.clip(slot, 0, None)])
        np.add.at(buy, slot[valid], minutes["buy_volume"].to_numpy()[valid])
        np.add.at(sell, slot[valid], minutes["sell_volume"].to_numpy()[valid])
        np.add.at(count, slot[valid], minutes["trade_count"].to_numpy()[valid])
    available = np.ones(len(naive), dtype=bool)
    for day in missing_days:
        day_start = np.datetime64(pd.Timestamp(day))
        day_end = day_start + np.timedelta64(1, "D")
        overlaps = (naive.values < day_end) & (ends > day_start)
        available &= ~overlaps
    total = buy + sell
    delta = buy - sell
    out["buy_volume"] = np.where(available, buy, np.nan)
    out["sell_volume"] = np.where(available, sell, np.nan)
    out["delta"] = np.where(available, delta, np.nan)
    out["cvd"] = np.where(available, np.cumsum(np.where(available, delta, 0.0)), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["delta_ratio"] = np.where(available & (total > 0), delta / total, np.nan)
    out["trade_count"] = np.where(available, count, np.nan)
    out["orderflow_available"] = available
    return out


def enrich_with_orderflow(
    data_map: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    *,
    download: Callable[[str], bytes | None] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Attach archive order flow to every crypto frame named in ``config['orderflow']``."""
    settings = config.get("orderflow")
    if settings in (None, False, {}):
        return data_map
    settings = settings if isinstance(settings, dict) else {}
    source = settings.get("source", "bybit_archive")
    if source != "bybit_archive":
        raise ValueError(f"orderflow source must be 'bybit_archive', got {source!r}")
    enriched: Dict[str, pd.DataFrame] = {}
    for code, frame in data_map.items():
        symbol, default_market = archive_symbol(code)
        market = settings.get("market", default_market)
        if frame is None or frame.empty:
            enriched[code] = frame
            continue
        index = pd.DatetimeIndex(frame.index)
        naive = index.tz_convert(None) if index.tz is not None else index
        step = (naive[1:] - naive[:-1]).min() if len(naive) > 1 else pd.Timedelta(days=1)
        start, end = naive[0].date(), (naive[-1] + step - pd.Timedelta(microseconds=1)).date()
        minutes, missing = load_minutes(symbol, market, start, end, download=download)
        if missing:
            logger.warning("Bybit archive has no %s %s trades for %s", market, symbol,
                           ", ".join(day.isoformat() for day in missing))
        enriched[code] = attach_orderflow(frame, minutes, missing)
    return enriched
