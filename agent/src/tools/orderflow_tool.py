"""Read-only crypto order flow: book structure plus the taker tape (Bybit first).

What an order-flow platform such as ATAS shows for a crypto pair is built from
two public feeds every venue publishes without authentication: the L2 order
book and the trade tape with the taker side of each trade. This tool reads both
through ccxt and turns them into the usual readings — book imbalance and walls,
delta and CVD per bar, a footprint, a volume profile, absorption and CVD
divergence. The arithmetic lives in :mod:`src.orderflow.analytics`.

Coverage is reported, never assumed. A REST call returns only the venue's most
recent trades (Bybit: 1000 for perpetuals but only 60 for spot; Binance: 1000),
which may span seconds on a busy pair. ``collect_seconds`` instead streams the
tape and the book over WebSocket (ccxt.pro) for that long, so the window holds
every trade. Every response says which source it used and what time span the
trades actually cover.

The network lives in :func:`_fetch_book`, :func:`_fetch_trades` and
:func:`_stream`, which tests monkeypatch. Nothing here places an order.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from datetime import datetime, timezone
from typing import Any

from backtest.loaders.ccxt_loader import _CCXT_TIMEOUT_MS, _ccxt_proxy_config
from src.agent.tools import BaseTool
from src.orderflow.analytics import (
    book_bands,
    build_bars,
    detect_absorption,
    detect_cvd_divergence,
    find_walls,
    nice_step,
    normalize_trades,
    volume_profile,
)

_EXCHANGES = ("bybit", "binance")
_MARKETS = ("perp", "spot")
_BAR_MINUTES = (1, 3, 5, 15)
_MAX_COLLECT_SECONDS = 180
_MAX_SNAPSHOTS = 5
_MAX_FOOTPRINT_BARS = 10
_BANDS_PCT = (0.1, 0.5, 1.0)
#: Book rows requested per side, within each venue's documented cap.
_BOOK_LIMIT = {"bybit": 200, "binance": 100}
#: Most recent trades a single REST call returns per venue and market.
_TRADE_LIMIT = {("bybit", "perp"): 1000, ("bybit", "spot"): 60, ("binance", "perp"): 1000, ("binance", "spot"): 1000}

_UNITS = {
    "amount_base / *_volume / delta / cvd": "quantity of the BASE asset (BTC in BTC-USDT), never money",
    "price / open / high / low / close / poc_price / value_area_*": "QUOTE-asset price of one base unit",
    "*_notional_quote": "value in the QUOTE asset (USDT for BTC-USDT)",
    "imbalance": "(bid - ask) / (bid + ask) notional within the band, in [-1, 1]; positive = bid-heavy",
    "side": "taker side: a buy is an aggressive buyer lifting the ask",
}
_NOTICE = (
    "Order-flow readings describe the recent tape and the resting book; they are not a "
    "trading signal by themselves. Book walls can be pulled at any moment."
)


def _error(message: str) -> str:
    return json.dumps({"ok": False, "error": message}, ensure_ascii=False)


def _ccxt_id(exchange: str, market: str) -> str:
    return "binanceusdm" if exchange == "binance" and market == "perp" else exchange


def _ccxt_symbol(base: str, quote: str, market: str) -> str:
    return f"{base}/{quote}:{quote}" if market == "perp" else f"{base}/{quote}"


def _exchange_config(market: str, exchange: str) -> dict[str, Any]:
    config: dict[str, Any] = {"enableRateLimit": True, "timeout": _CCXT_TIMEOUT_MS}
    if exchange == "bybit":
        config["options"] = {"defaultType": "swap" if market == "perp" else "spot"}
    proxies = _ccxt_proxy_config()
    if proxies:
        config["proxies"] = proxies
    return config


def _fetch_book(exchange: str, market: str, symbol: str, limit: int) -> dict[str, Any]:
    """One L2 snapshot over public REST."""
    import ccxt

    client = getattr(ccxt, _ccxt_id(exchange, market))(_exchange_config(market, exchange))
    return client.fetch_order_book(symbol, limit)


def _fetch_trades(exchange: str, market: str, symbol: str, limit: int) -> list[dict[str, Any]]:
    """The venue's most recent public trades over REST."""
    import ccxt

    client = getattr(ccxt, _ccxt_id(exchange, market))(_exchange_config(market, exchange))
    return client.fetch_trades(symbol, limit=limit)


def _stream(exchange: str, market: str, symbol: str, seconds: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect every trade and a book snapshot every ~2 s for ``seconds`` (ccxt.pro)."""

    async def collect() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        import ccxt.pro as ccxtpro

        client = getattr(ccxtpro, _ccxt_id(exchange, market))(_exchange_config(market, exchange))
        deadline = time.monotonic() + seconds
        trades: list[dict[str, Any]] = []
        books: list[dict[str, Any]] = []

        async def tape() -> None:
            while time.monotonic() < deadline:
                try:
                    batch = await asyncio.wait_for(client.watch_trades(symbol), deadline - time.monotonic())
                except asyncio.TimeoutError:
                    return
                trades.extend(batch)

        async def book() -> None:
            while time.monotonic() < deadline:
                try:
                    snapshot = await asyncio.wait_for(
                        client.watch_order_book(symbol, _BOOK_LIMIT[exchange] if exchange == "bybit" else None),
                        deadline - time.monotonic(),
                    )
                except asyncio.TimeoutError:
                    return
                books.append({"bids": [list(row[:2]) for row in snapshot["bids"]],
                              "asks": [list(row[:2]) for row in snapshot["asks"]]})
                await asyncio.sleep(min(2.0, max(0.0, deadline - time.monotonic())))

        try:
            await asyncio.gather(tape(), book())
        finally:
            await client.close()
        return trades, books

    result: dict[str, Any] = {}

    def runner() -> None:
        try:
            result["value"] = asyncio.run(collect())
        except Exception as exc:  # noqa: BLE001 - re-raised in the caller's thread
            result["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(seconds + 30)
    if "error" in result:
        raise result["error"]
    if "value" not in result:
        raise TimeoutError(f"stream did not finish within {seconds + 30} s")
    return result["value"]


def _levels(raw: Any) -> list[tuple[float, float]]:
    out = []
    for row in raw if isinstance(raw, list) else []:
        try:
            price, amount = float(row[0]), float(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(price) and math.isfinite(amount) and price > 0 and amount > 0:
            out.append((price, amount))
    return out


def _clean_book(raw: Any) -> dict[str, list[tuple[float, float]]]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "bids": sorted(_levels(raw.get("bids")), key=lambda level: level[0], reverse=True),
        "asks": sorted(_levels(raw.get("asks")), key=lambda level: level[0]),
    }


def _int(value: Any, default: int) -> int | None:
    if value is None:
        return default
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number.is_integer() else None


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


class OrderFlowTool(BaseTool):
    """Crypto order book and taker-tape analytics: walls, delta, CVD, footprint, profile."""

    name = "orderflow"
    repeatable = True
    is_readonly = True
    description = (
        "READ-ONLY crypto order flow from Bybit (default) or Binance public data, spot or "
        "USDT perpetual: the readings an order-flow platform like ATAS shows. Returns (1) the "
        "order book — best bid/ask, spread, bid/ask notional and imbalance within 0.1/0.5/1% "
        "of mid, and large resting 'walls' with how persistent they were across snapshots; "
        "(2) the taker tape aggregated into bars — buy/sell volume, delta, CVD, POC per bar, "
        "stacked footprint imbalances, and a full footprint for the latest bars; (3) a volume "
        "profile with POC and 70% value area; (4) absorption bars and CVD divergence. "
        "collect_seconds=0 uses one REST call (only the venue's most recent trades — Bybit "
        "spot gives just 60); collect_seconds>0 streams the tape and book for that many "
        "seconds so the window holds every trade. 'coverage' states the real time span. "
        "Descriptive only; places nothing. Example: "
        '{"symbol": "BTC-USDT", "exchange": "bybit", "market": "perp", "collect_seconds": 60}.'
    )
    parameters = {
        "type": "object",
        "properties": {
            "symbol": {"type": "string", "description": "BASE-QUOTE pair, e.g. 'BTC-USDT'. Required."},
            "exchange": {"type": "string", "enum": list(_EXCHANGES), "default": "bybit"},
            "market": {
                "type": "string",
                "enum": list(_MARKETS),
                "default": "perp",
                "description": "'perp' = USDT-margined perpetual (linear), 'spot' = spot pair.",
            },
            "bar_minutes": {"type": "integer", "enum": list(_BAR_MINUTES), "default": 1},
            "collect_seconds": {
                "type": "integer",
                "minimum": 0,
                "maximum": _MAX_COLLECT_SECONDS,
                "default": 0,
                "description": "Stream tape and book over WebSocket for this long (0 = one REST snapshot).",
            },
            "snapshots": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_SNAPSHOTS,
                "default": 3,
                "description": "REST mode: book snapshots ~1 s apart, used to judge wall persistence.",
            },
            "footprint_bars": {
                "type": "integer",
                "minimum": 0,
                "maximum": _MAX_FOOTPRINT_BARS,
                "default": 3,
                "description": "How many of the latest bars carry their full footprint.",
            },
            "wall_multiplier": {
                "type": "number",
                "default": 5,
                "description": "A wall is a level at least this many times the median level size nearby.",
            },
        },
        "required": ["symbol"],
    }

    def execute(self, **kwargs: Any) -> str:
        """Read the book and tape once (or stream them) and return the readings as JSON."""
        symbol = str(kwargs.get("symbol") or "").strip().upper().replace("/", "-")
        parts = symbol.split("-")
        if len(parts) != 2 or not all(part.isalnum() for part in parts):
            return _error(f"symbol must be BASE-QUOTE such as 'BTC-USDT' (got {kwargs.get('symbol')!r})")
        exchange = str(kwargs.get("exchange") or "bybit").lower()
        market = str(kwargs.get("market") or "perp").lower()
        if exchange not in _EXCHANGES:
            return _error(f"exchange must be one of {list(_EXCHANGES)}")
        if market not in _MARKETS:
            return _error(f"market must be one of {list(_MARKETS)}")
        bar_minutes = _int(kwargs.get("bar_minutes"), 1)
        collect = _int(kwargs.get("collect_seconds"), 0)
        snapshots = _int(kwargs.get("snapshots"), 3)
        footprint_bars = _int(kwargs.get("footprint_bars"), 3)
        try:
            multiplier = float(kwargs.get("wall_multiplier") if kwargs.get("wall_multiplier") is not None else 5)
        except (TypeError, ValueError):
            multiplier = float("nan")
        if bar_minutes not in _BAR_MINUTES:
            return _error(f"bar_minutes must be one of {list(_BAR_MINUTES)}")
        if collect is None or not 0 <= collect <= _MAX_COLLECT_SECONDS:
            return _error(f"collect_seconds must be an integer from 0 to {_MAX_COLLECT_SECONDS}")
        if snapshots is None or not 1 <= snapshots <= _MAX_SNAPSHOTS:
            return _error(f"snapshots must be an integer from 1 to {_MAX_SNAPSHOTS}")
        if footprint_bars is None or not 0 <= footprint_bars <= _MAX_FOOTPRINT_BARS:
            return _error(f"footprint_bars must be an integer from 0 to {_MAX_FOOTPRINT_BARS}")
        if not math.isfinite(multiplier) or multiplier < 1:
            return _error("wall_multiplier must be a number of at least 1")

        base, quote = parts
        ccxt_symbol = _ccxt_symbol(base, quote, market)
        try:
            if collect > 0:
                raw_trades, raw_books = _stream(exchange, market, ccxt_symbol, collect)
                source = "websocket"
            else:
                raw_books = []
                for index in range(snapshots):
                    if index:
                        time.sleep(1.0)
                    raw_books.append(_fetch_book(exchange, market, ccxt_symbol, _BOOK_LIMIT[exchange]))
                raw_trades = _fetch_trades(exchange, market, ccxt_symbol, _TRADE_LIMIT[(exchange, market)])
                source = "rest"
        except Exception as exc:  # noqa: BLE001 - surfaced as an error envelope
            return _error(f"{exchange} {market} request for {symbol} failed: {exc}")

        books = [book for book in (_clean_book(raw) for raw in raw_books) if book["bids"] and book["asks"]]
        if not books:
            return _error(f"no two-sided order book returned for {symbol} on {exchange} {market}")
        latest = books[-1]
        best_bid, best_ask = latest["bids"][0][0], latest["asks"][0][0]
        if best_bid >= best_ask:
            return _error(f"crossed book for {symbol} (bid {best_bid} >= ask {best_ask}); refusing to read it")
        mid = (best_bid + best_ask) / 2

        trades = normalize_trades(raw_trades)
        coverage: dict[str, Any] = {"source": source, "trades_used": len(trades), "book_snapshots": len(books)}
        flow: dict[str, Any] | None = None
        profile: dict[str, Any] | None = None
        signals: dict[str, Any] = {"absorption": [], "cvd_divergence": None}
        if trades:
            first, last = trades[0].ts_ms, trades[-1].ts_ms
            coverage.update({"first_trade": _iso(first), "last_trade": _iso(last),
                             "seconds_covered": round((last - first) / 1000, 1)})
            if source == "rest" and len(raw_trades or []) >= _TRADE_LIMIT[(exchange, market)]:
                coverage["note"] = (
                    "REST returned the venue's maximum batch of recent trades, so older trades in "
                    "the window are missing; use collect_seconds for a complete tape."
                )
            prices = [trade.price for trade in trades]
            step = nice_step(max(prices) - min(prices), mid)
            bars = build_bars(trades, bar_minutes * 60_000, step)
            for index, bar in enumerate(bars):
                if index < len(bars) - footprint_bars:
                    bar.pop("footprint")
            buy = sum(t.amount for t in trades if t.side == "buy")
            sell = sum(t.amount for t in trades if t.side == "sell")
            notional = sum(t.amount * t.price for t in trades)
            flow = {
                "bar_minutes": bar_minutes,
                "buy_volume": buy,
                "sell_volume": sell,
                "delta": buy - sell,
                "buy_share": buy / (buy + sell),
                "vwap": notional / (buy + sell),
                "bars": bars,
            }
            profile = volume_profile(trades, step)
            signals = {"absorption": detect_absorption(bars), "cvd_divergence": detect_cvd_divergence(bars)}
        else:
            coverage["note"] = "no trades returned in this window; tape readings are omitted"

        payload = {
            "ok": True,
            "source": f"ccxt/{_ccxt_id(exchange, market)}",
            "market": "crypto",
            "notice": _NOTICE,
            "units": _UNITS,
            "data": {
                "symbol": symbol,
                "exchange": exchange,
                "instrument": market,
                "as_of": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "coverage": coverage,
                "book": {
                    "best_bid": best_bid,
                    "best_ask": best_ask,
                    "mid_price": mid,
                    "spread_bps": (best_ask - best_bid) / mid * 1e4,
                    "bands": book_bands(latest["bids"], latest["asks"], mid, _BANDS_PCT),
                    "walls": find_walls(books, mid, multiplier=multiplier),
                },
                "flow": flow,
                "profile": profile,
                "signals": signals,
            },
        }
        return json.dumps(payload, ensure_ascii=False)
