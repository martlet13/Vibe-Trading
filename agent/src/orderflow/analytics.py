"""Order-flow arithmetic over taker trades and L2 book snapshots.

Definitions used throughout (and pinned by ``tests/test_orderflow_analytics.py``):

* **Delta** of a set of trades: taker-buy base volume minus taker-sell base
  volume. **CVD** is the running sum of bar deltas across the window, so it
  starts from zero at the first bar — it is relative, never an absolute level.
* **Footprint** of a bar: base volume traded at each price bin, split into
  taker buys and taker sells. A bin's price is its LOWER edge.
* **Diagonal imbalance** (the usual footprint convention): buys at bin ``p``
  against sells at bin ``p - step``; a buy imbalance when buys are at least
  ``ratio`` times those sells, a sell imbalance when sells at ``p`` are at
  least ``ratio`` times buys at ``p + step``.
* **Value area**: the bins holding ``value_area`` (70%) of the window's volume,
  grown outward from the POC one side at a time, taking the larger neighbour.
* **Absorption**: a bar whose |delta| is at least ``multiplier`` times the
  window's median |delta| while price did not follow — heavy taker selling
  into a bar that closed at or above its open (sellers absorbed, bullish), or
  heavy taker buying into a bar that closed at or below its open (bearish).
* **CVD divergence** at the last bar: price makes the window's extreme while
  CVD stays below (for a high) or above (for a low) its own earlier extreme.

These are descriptive readings of the tape, not trading signals on their own.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import median
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Trade:
    """One taker trade: epoch milliseconds, price, base amount, taker side."""

    ts_ms: int
    price: float
    amount: float
    side: str  # "buy" or "sell" — the taker side


def _finite_positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def normalize_trades(raw: Iterable[Any]) -> list[Trade]:
    """Turn ccxt trade dicts into :class:`Trade` rows, oldest first, deduplicated.

    Rows without a positive finite price and amount, a timestamp, or a
    ``buy``/``sell`` side are dropped: guessing a side would corrupt delta.

    Args:
        raw: ccxt ``fetch_trades`` / ``watch_trades`` rows.

    Returns:
        Trades sorted by time; a repeated trade ``id`` is kept once.
    """
    seen: set[str] = set()
    out: list[Trade] = []
    for row in raw or ():
        if not isinstance(row, dict):
            continue
        side = str(row.get("side") or "").lower()
        price = _finite_positive(row.get("price"))
        amount = _finite_positive(row.get("amount"))
        ts = row.get("timestamp")
        if side not in ("buy", "sell") or price is None or amount is None or not isinstance(ts, (int, float)):
            continue
        trade_id = row.get("id")
        if trade_id is not None:
            key = str(trade_id)
            if key in seen:
                continue
            seen.add(key)
        out.append(Trade(int(ts), price, amount, side))
    out.sort(key=lambda trade: trade.ts_ms)
    return out


def nice_step(price_range: float, reference_price: float, target_bins: int = 30) -> float:
    """A round price-bin size giving roughly ``target_bins`` bins over the range.

    The step is 1, 2 or 5 times a power of ten and never smaller than one
    millionth of the reference price, so a flat window still gets a usable bin.
    """
    floor = max(abs(reference_price) * 1e-6, 1e-12)
    raw = max(price_range / max(target_bins, 1), floor)
    exponent = math.floor(math.log10(raw))
    base = 10.0 ** exponent
    for multiple in (1.0, 2.0, 5.0, 10.0):
        if raw <= multiple * base:
            return multiple * base
    return 10.0 * base


def _decimals(step: float) -> int:
    return max(0, -math.floor(math.log10(step))) if step < 1 else 0


def bin_price(price: float, step: float) -> float:
    """The lower edge of the price bin holding ``price``."""
    return round(math.floor(price / step + 1e-9) * step, _decimals(step) + 2)


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def _footprint(trades: Sequence[Trade], step: float, ratio: float) -> list[dict[str, Any]]:
    buys: dict[float, float] = defaultdict(float)
    sells: dict[float, float] = defaultdict(float)
    for trade in trades:
        level = bin_price(trade.price, step)
        (buys if trade.side == "buy" else sells)[level] += trade.amount
    levels = sorted(set(buys) | set(sells))
    digits = _decimals(step) + 2
    rows = []
    for level in levels:
        buy, sell = buys.get(level, 0.0), sells.get(level, 0.0)
        below = sells.get(round(level - step, digits), 0.0)
        above = buys.get(round(level + step, digits), 0.0)
        rows.append({
            "price": level,
            "buy_volume": buy,
            "sell_volume": sell,
            "delta": buy - sell,
            "buy_imbalance": buy > 0 and buy >= ratio * below and below > 0,
            "sell_imbalance": sell > 0 and sell >= ratio * above and above > 0,
        })
    return rows


def build_bars(
    trades: Sequence[Trade],
    bar_ms: int,
    step: float,
    *,
    imbalance_ratio: float = 3.0,
) -> list[dict[str, Any]]:
    """Aggregate trades into time bars with delta, CVD and a footprint.

    Args:
        trades: Taker trades, oldest first.
        bar_ms: Bar length in milliseconds.
        step: Footprint price-bin size.
        imbalance_ratio: Diagonal ratio that flags a footprint imbalance.

    Returns:
        One dict per bar that had trades, oldest first. Empty input gives ``[]``.
    """
    buckets: dict[int, list[Trade]] = defaultdict(list)
    for trade in trades:
        buckets[trade.ts_ms // bar_ms * bar_ms].append(trade)
    bars: list[dict[str, Any]] = []
    cvd = 0.0
    for start in sorted(buckets):
        rows = buckets[start]
        buy = sum(t.amount for t in rows if t.side == "buy")
        sell = sum(t.amount for t in rows if t.side == "sell")
        delta = buy - sell
        cvd += delta
        footprint = _footprint(rows, step, imbalance_ratio)
        poc = max(footprint, key=lambda level: level["buy_volume"] + level["sell_volume"])
        prices = [t.price for t in rows]
        bars.append({
            "time": _iso(start),
            "open": rows[0].price,
            "high": max(prices),
            "low": min(prices),
            "close": rows[-1].price,
            "trades": len(rows),
            "buy_volume": buy,
            "sell_volume": sell,
            "delta": delta,
            "cvd": cvd,
            "poc": {"price": poc["price"], "volume": poc["buy_volume"] + poc["sell_volume"]},
            "stacked_buy_imbalances": sum(1 for level in footprint if level["buy_imbalance"]),
            "stacked_sell_imbalances": sum(1 for level in footprint if level["sell_imbalance"]),
            "footprint": footprint,
        })
    return bars


def volume_profile(trades: Sequence[Trade], step: float, value_area: float = 0.7) -> dict[str, Any] | None:
    """Window volume profile: POC and the value area around it.

    Returns:
        ``{"bin_size", "poc_price", "value_area_high", "value_area_low",
        "value_area_share", "levels": [...]}``, or ``None`` with no trades.
        ``value_area_high`` is the TOP edge of the highest value-area bin.
    """
    if not trades:
        return None
    volume: dict[float, float] = defaultdict(float)
    buys: dict[float, float] = defaultdict(float)
    for trade in trades:
        level = bin_price(trade.price, step)
        volume[level] += trade.amount
        if trade.side == "buy":
            buys[level] += trade.amount
    levels = sorted(volume)
    total = sum(volume.values())
    poc_index = max(range(len(levels)), key=lambda i: volume[levels[i]])
    low = high = poc_index
    covered = volume[levels[poc_index]]
    while covered < value_area * total and (low > 0 or high < len(levels) - 1):
        below = volume[levels[low - 1]] if low > 0 else -1.0
        above = volume[levels[high + 1]] if high < len(levels) - 1 else -1.0
        if above >= below:
            high += 1
            covered += above
        else:
            low -= 1
            covered += below
    digits = _decimals(step) + 2
    return {
        "bin_size": step,
        "poc_price": levels[poc_index],
        "value_area_high": round(levels[high] + step, digits),
        "value_area_low": levels[low],
        "value_area_share": covered / total,
        "levels": [
            {
                "price": level,
                "volume": volume[level],
                "buy_volume": buys.get(level, 0.0),
                "sell_volume": volume[level] - buys.get(level, 0.0),
            }
            for level in levels
        ],
    }


def detect_absorption(bars: Sequence[dict[str, Any]], multiplier: float = 2.0) -> list[dict[str, Any]]:
    """Bars where heavy one-sided taker flow failed to move price its way."""
    if len(bars) < 3:
        return []
    typical = median(abs(bar["delta"]) for bar in bars)
    if typical <= 0:
        return []
    found = []
    for bar in bars:
        strength = abs(bar["delta"]) / typical
        if strength < multiplier:
            continue
        if bar["delta"] < 0 and bar["close"] >= bar["open"]:
            kind = "sell_absorption"  # sellers hit the bid, price held: bullish read
        elif bar["delta"] > 0 and bar["close"] <= bar["open"]:
            kind = "buy_absorption"  # buyers lifted the ask, price held: bearish read
        else:
            continue
        found.append({
            "time": bar["time"],
            "kind": kind,
            "delta": bar["delta"],
            "delta_vs_median": strength,
            "open": bar["open"],
            "close": bar["close"],
        })
    return found


def detect_cvd_divergence(bars: Sequence[dict[str, Any]], min_bars: int = 5) -> dict[str, Any] | None:
    """Divergence between price and CVD at the last bar, or ``None``."""
    if len(bars) < min_bars:
        return None
    last, earlier = bars[-1], bars[:-1]
    if last["high"] >= max(bar["high"] for bar in earlier):
        prior = max(earlier, key=lambda bar: bar["cvd"])
        if last["cvd"] < prior["cvd"]:
            return {"kind": "bearish", "price_extreme": last["high"], "cvd": last["cvd"],
                    "prior_cvd_extreme": prior["cvd"], "prior_time": prior["time"]}
    if last["low"] <= min(bar["low"] for bar in earlier):
        prior = min(earlier, key=lambda bar: bar["cvd"])
        if last["cvd"] > prior["cvd"]:
            return {"kind": "bullish", "price_extreme": last["low"], "cvd": last["cvd"],
                    "prior_cvd_extreme": prior["cvd"], "prior_time": prior["time"]}
    return None


def book_bands(
    bids: Sequence[tuple[float, float]],
    asks: Sequence[tuple[float, float]],
    mid: float,
    bands_pct: Sequence[float],
) -> list[dict[str, Any]]:
    """Resting notional on each side within ``band`` percent of mid, and imbalance."""
    rows = []
    for band in bands_pct:
        floor, ceiling = mid * (1 - band / 100), mid * (1 + band / 100)
        bid = sum(price * amount for price, amount in bids if price >= floor)
        ask = sum(price * amount for price, amount in asks if price <= ceiling)
        total = bid + ask
        rows.append({
            "band_pct": band,
            "bid_notional_quote": bid,
            "ask_notional_quote": ask,
            "imbalance": (bid - ask) / total if total > 0 else None,
        })
    return rows


def find_walls(
    snapshots: Sequence[dict[str, list[tuple[float, float]]]],
    mid: float,
    *,
    within_pct: float = 2.0,
    multiplier: float = 5.0,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Unusually large resting levels in the latest snapshot, with persistence.

    A wall is a level within ``within_pct`` of mid whose size is at least
    ``multiplier`` times the median level size on its side within that band.
    ``persistence`` is the share of snapshots that showed the same price with
    at least half the current size — a level that vanishes between snapshots
    is likely spoofed or already traded.
    """
    if not snapshots:
        return []
    latest = snapshots[-1]
    floor, ceiling = mid * (1 - within_pct / 100), mid * (1 + within_pct / 100)
    walls = []
    for side, key in (("bid", "bids"), ("ask", "asks")):
        levels = [(p, a) for p, a in latest.get(key, []) if floor <= p <= ceiling]
        if len(levels) < 3:
            continue
        typical = median(amount for _, amount in levels)
        if typical <= 0:
            continue
        for price, amount in levels:
            if amount < multiplier * typical:
                continue
            seen = sum(
                1 for snap in snapshots
                if any(abs(p - price) <= 1e-12 * max(price, 1) and a >= 0.5 * amount for p, a in snap.get(key, []))
            )
            walls.append({
                "side": side,
                "price": price,
                "amount_base": amount,
                "notional_quote": price * amount,
                "size_vs_median": amount / typical,
                "distance_bps": abs(price - mid) / mid * 1e4,
                "persistence": seen / len(snapshots),
            })
    walls.sort(key=lambda wall: wall["notional_quote"], reverse=True)
    return walls[:limit]
