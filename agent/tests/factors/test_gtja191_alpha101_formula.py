"""GTJA #101's right leg correlates RANK(HIGH*0.1 + VWAP*0.9) with RANK(VOLUME).

Report: ((RANK(CORR(CLOSE, SUM(MEAN(VOLUME,30),37),15))
          < RANK(CORR(RANK(((HIGH*0.1)+(VWAP*0.9))), RANK(VOLUME), 11))) * -1)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import rank, ts_corr, ts_mean
from src.factors.zoo.gtja191 import alpha_101


def _panel() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(101)
    n = 160
    idx = pd.bdate_range("2024-01-01", periods=n)
    cols = [f"S{i}" for i in range(12)]
    close = pd.DataFrame(20 + rng.normal(0, 0.5, (n, 12)).cumsum(axis=0), index=idx, columns=cols)
    high = close + rng.uniform(0.05, 0.5, (n, 12))
    low = close - rng.uniform(0.05, 0.5, (n, 12))
    vw = low + (high - low) * rng.uniform(0.2, 0.8, (n, 12))
    vol = pd.DataFrame(rng.uniform(1e5, 5e5, (n, 12)), index=idx, columns=cols)
    return {"open": close.shift(1).fillna(close), "high": high, "low": low,
            "close": close, "volume": vol, "amount": vw * vol * 0.1, "vwap": vw}


def test_right_leg_uses_high_vwap_blend_against_raw_volume():
    p = _panel()
    c, h, v, vw = p["close"], p["high"], p["volume"], p["vwap"]
    left = rank(ts_corr(c, ts_mean(v, 30).rolling(37).sum(), 15))
    right = rank(ts_corr(rank(h * 0.1 + vw * 0.9), rank(v), 11))
    expected = (left < right).astype("float64") * -1.0

    got = alpha_101.compute(p)

    tail = slice(90, None)
    pd.testing.assert_frame_equal(got.iloc[tail], expected.iloc[tail])
