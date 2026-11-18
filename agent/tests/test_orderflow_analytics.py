"""Order-flow arithmetic: delta, CVD, footprint, profile, absorption, divergence, book.

Every value is synthetic and small enough to check by hand.
"""

from __future__ import annotations

import pytest

from src.orderflow.analytics import (
    Trade,
    bin_price,
    book_bands,
    build_bars,
    detect_absorption,
    detect_cvd_divergence,
    find_walls,
    nice_step,
    normalize_trades,
    volume_profile,
)

MIN = 60_000


def _t(minute: float, price: float, amount: float, side: str) -> Trade:
    return Trade(int(minute * MIN), price, amount, side)


def test_normalize_drops_unknown_side_dedupes_ids_and_sorts():
    raw = [
        {"id": "2", "timestamp": 2000, "price": "101", "amount": "1", "side": "Sell"},
        {"id": "1", "timestamp": 1000, "price": 100, "amount": 2, "side": "buy"},
        {"id": "1", "timestamp": 1000, "price": 100, "amount": 2, "side": "buy"},
        {"id": "3", "timestamp": 3000, "price": 100, "amount": 1, "side": None},
        {"id": "4", "timestamp": 4000, "price": -1, "amount": 1, "side": "buy"},
        {"id": "5", "timestamp": None, "price": 100, "amount": 1, "side": "buy"},
    ]
    trades = normalize_trades(raw)
    assert [(t.ts_ms, t.side, t.amount) for t in trades] == [(1000, "buy", 2.0), (2000, "sell", 1.0)]


@pytest.mark.parametrize(("span", "ref", "step"), [(564.4, 64000, 20.0), (0.0, 64000, 0.1), (3.0, 1.5, 0.1), (0.0004, 0.01, 0.00002)])
def test_nice_step_is_round_and_never_zero(span, ref, step):
    assert nice_step(span, ref) == pytest.approx(step)


def test_bar_delta_cvd_footprint_and_poc():
    trades = [
        _t(0.1, 100.0, 2, "buy"), _t(0.2, 100.4, 1, "sell"), _t(0.9, 101.2, 3.5, "buy"),
        _t(1.1, 101.0, 4, "sell"), _t(1.5, 100.6, 1, "buy"),
    ]
    bars = build_bars(trades, MIN, 1.0)
    assert [bar["delta"] for bar in bars] == [4.5, -3.0]
    assert [bar["cvd"] for bar in bars] == [4.5, 1.5]
    first = bars[0]
    assert (first["open"], first["high"], first["low"], first["close"]) == (100.0, 101.2, 100.0, 101.2)
    assert first["poc"] == {"price": 101.0, "volume": 3.5}
    assert [(lvl["price"], lvl["buy_volume"], lvl["sell_volume"]) for lvl in first["footprint"]] == [
        (100.0, 2.0, 1.0), (101.0, 3.5, 0.0)]


def test_diagonal_imbalance_compares_buys_with_sells_one_bin_lower():
    trades = [_t(0, 100.0, 1, "sell"), _t(0, 101.0, 3, "buy"), _t(0, 101.0, 0.5, "sell"), _t(0, 102.0, 0.1, "buy")]
    bars = build_bars(trades, MIN, 1.0)
    by_price = {lvl["price"]: lvl for lvl in bars[0]["footprint"]}
    assert by_price[101.0]["buy_imbalance"] is True   # 3 buys vs 1 sell at 100
    assert by_price[101.0]["sell_imbalance"] is True  # 0.5 sells vs 0.1 buys at 102
    assert by_price[100.0]["sell_imbalance"] is False  # 1 sell vs 3 buys at 101
    assert bars[0]["stacked_buy_imbalances"] == 1


def test_value_area_grows_from_poc_toward_the_heavier_side():
    trades = [_t(0, p, v, "buy") for p, v in [(100, 1), (101, 5), (102, 10), (103, 2), (104, 1)]]
    profile = volume_profile(trades, 1.0)
    assert profile["poc_price"] == 102.0
    # 10 (102) + 5 (101) = 15 of 19 = 79% >= 70%.
    assert (profile["value_area_low"], profile["value_area_high"]) == (101.0, 103.0)
    assert profile["value_area_share"] == pytest.approx(15 / 19)
    assert volume_profile([], 1.0) is None


def test_bin_price_is_the_lower_edge():
    assert bin_price(64019.9, 20.0) == 64000.0
    assert bin_price(64020.0, 20.0) == 64020.0
    assert bin_price(0.12345, 0.001) == pytest.approx(0.123)


def _bar(minute, open_, close, delta, cvd=0.0, high=None, low=None):
    return {"time": str(minute), "open": open_, "close": close, "delta": delta, "cvd": cvd,
            "high": high if high is not None else max(open_, close), "low": low if low is not None else min(open_, close)}


def test_absorption_flags_heavy_flow_that_price_did_not_follow():
    quiet = [_bar(i, 100, 100.2, (-1) ** i) for i in range(4)]  # median |delta| = 1
    bars = quiet + [_bar(4, 100, 100.5, -5), _bar(5, 100.5, 100, 6), _bar(6, 100, 99, -5), _bar(7, 99, 99.1, -1.5)]
    found = {row["time"]: row["kind"] for row in detect_absorption(bars)}
    # 6 fell with its selling; 7 is below the 2x-median bar.
    assert found == {"4": "sell_absorption", "5": "buy_absorption"}


def test_cvd_divergence_at_a_new_high_and_at_a_new_low():
    up = [_bar(i, 100 + i, 100 + i, 1, cvd=c, high=100 + i) for i, c in enumerate([1, 5, 9, 7, 6])]
    up[-1]["high"] = 110
    assert detect_cvd_divergence(up)["kind"] == "bearish"
    down = [_bar(i, 100 - i, 100 - i, -1, cvd=c, low=100 - i) for i, c in enumerate([-1, -5, -9, -7, -6])]
    down[-1]["low"] = 90
    assert detect_cvd_divergence(down)["kind"] == "bullish"
    confirmed = [_bar(i, 100 + i, 100 + i, 1, cvd=i, high=100 + i) for i in range(5)]
    assert detect_cvd_divergence(confirmed) is None
    assert detect_cvd_divergence(up[:3]) is None


def test_book_bands_and_imbalance():
    bids = [(99.95, 10), (99.5, 10), (98.0, 100)]
    asks = [(100.05, 5), (100.6, 50)]
    rows = book_bands(bids, asks, 100.0, (0.1, 1.0))
    assert rows[0]["bid_notional_quote"] == pytest.approx(999.5)
    assert rows[0]["ask_notional_quote"] == pytest.approx(500.25)
    assert rows[0]["imbalance"] == pytest.approx((999.5 - 500.25) / (999.5 + 500.25))
    assert rows[1]["bid_notional_quote"] == pytest.approx(999.5 + 995)


def test_walls_need_size_and_report_persistence():
    def snap(wall):
        bids = [(99.9, 1), (99.8, 1), (99.7, 1), (99.6, wall)]
        return {"bids": bids, "asks": [(100.1, 1), (100.2, 1), (100.3, 1)]}

    walls = find_walls([snap(0.1), snap(10), snap(10)], 100.0, multiplier=5)
    assert len(walls) == 1
    wall = walls[0]
    assert (wall["side"], wall["price"], wall["amount_base"]) == ("bid", 99.6, 10)
    assert wall["persistence"] == pytest.approx(2 / 3)
    assert find_walls([snap(2)], 100.0, multiplier=5) == []
