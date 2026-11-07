"""GTJA #55 is the 20-day sum of the swing-index term that #137 reports per
day, so the two must agree, including on bars where |HIGH - DELAY(LOW,1)| is
the widest range and the third denominator branch applies."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.zoo.gtja191 import alpha_055, alpha_137


def _panel() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2024-01-01", periods=80)
    cols = ["A", "B", "C"]
    close = pd.DataFrame(100 + rng.normal(0, 1, (80, 3)).cumsum(axis=0), index=idx, columns=cols)
    open_ = close.shift(1).fillna(close) + rng.normal(0, 0.3, (80, 3))
    high = np.maximum(open_, close) + rng.uniform(0.1, 1.0, (80, 3))
    low = np.minimum(open_, close) - rng.uniform(0.1, 1.0, (80, 3))
    # A tight bar that closes near its high, followed by a bar whose high sits
    # just above that close: |H - prior L| is then the widest of the three.
    for day in range(10, 80, 7):
        prev = idx[day - 1]
        low.loc[prev] = close.loc[prev] - 3.0
        high.loc[prev] = close.loc[prev] + 0.1
        open_.loc[idx[day]] = close.loc[prev]
        high.loc[idx[day]] = close.loc[prev] + 0.5
        low.loc[idx[day]] = close.loc[prev] - 0.2
        close.loc[idx[day]] = close.loc[prev] + 0.3
    return {"open": open_, "high": high, "low": low, "close": close}


def test_third_branch_is_exercised():
    p = _panel()
    pc, pl = p["close"].shift(1), p["low"].shift(1)
    a, b, e = (p["high"] - pc).abs(), (p["low"] - pc).abs(), (p["high"] - pl).abs()
    assert ((e >= a) & (e >= b)).to_numpy().sum() >= 10


def test_alpha055_is_rolling_sum_of_alpha137():
    p = _panel()
    expected = alpha_137.compute(p).rolling(20, min_periods=20).sum()
    got = alpha_055.compute(p)
    pd.testing.assert_frame_equal(got, expected, check_exact=False, rtol=1e-9)
