"""GTJA #170 scales HIGH*RANK(HIGH-CLOSE) by the 5-day mean of HIGH, not of
CLOSE: (SUM(HIGH,5)/5) in the report."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import rank, safe_div, ts_mean, vwap
from src.factors.zoo.gtja191 import alpha_170


def _panel() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2024-01-01", periods=40)
    cols = ["A", "B", "C", "D"]
    close = pd.DataFrame(50 + rng.normal(0, 1, (40, 4)).cumsum(axis=0), index=idx, columns=cols)
    open_ = close.shift(1).fillna(close)
    # A wide upper range keeps mean(HIGH) well away from mean(CLOSE).
    high = np.maximum(open_, close) + rng.uniform(2.0, 4.0, (40, 4))
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.5, (40, 4))
    volume = pd.DataFrame(rng.uniform(1e5, 5e5, (40, 4)), index=idx, columns=cols)
    amount = volume * (high + low + close) / 3.0
    return {"open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "amount": amount}


def test_alpha170_uses_the_mean_of_high():
    p = _panel()
    c, h, v = p["close"], p["high"], p["volume"]
    vw = vwap(p, "equity_cn")
    a = safe_div(rank(1.0 / c) * v, ts_mean(v, 20))
    b = safe_div(h * rank(h - c), ts_mean(h, 5))
    expected = a * b - rank(vw - vw.shift(5))

    got = alpha_170.compute(p)

    assert got.iloc[25:].notna().all().all()
    pd.testing.assert_frame_equal(got, expected, check_exact=False, rtol=1e-9)
