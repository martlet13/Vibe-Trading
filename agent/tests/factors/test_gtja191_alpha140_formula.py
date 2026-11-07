"""GTJA #140's second leg is TSRANK(DECAYLINEAR(CORR(...),7),3); the
correlation is decayed directly, with no TSRANK between CORR and DECAYLINEAR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.factors.base import decay_linear, rank, ts_corr, ts_mean, ts_rank
from src.factors.zoo.gtja191 import alpha_140


def _panel() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(3)
    n = 120
    idx = pd.bdate_range("2024-01-01", periods=n)
    cols = ["A", "B", "C", "D", "E"]
    close = pd.DataFrame(30 + rng.normal(0, 0.8, (n, 5)).cumsum(axis=0), index=idx, columns=cols)
    open_ = close.shift(1).fillna(close) + rng.normal(0, 0.2, (n, 5))
    high = np.maximum(open_, close) + rng.uniform(0.05, 0.6, (n, 5))
    low = np.minimum(open_, close) - rng.uniform(0.05, 0.6, (n, 5))
    volume = pd.DataFrame(rng.uniform(1e5, 9e5, (n, 5)), index=idx, columns=cols)
    return {"open": open_, "high": high, "low": low, "close": close, "volume": volume}


def test_alpha140_matches_report_formula():
    p = _panel()
    o, h, l, c, v = p["open"], p["high"], p["low"], p["close"], p["volume"]
    left = rank(decay_linear((rank(o) + rank(l)) - (rank(h) + rank(c)), 8))
    corr = ts_corr(ts_rank(c, 8), ts_rank(ts_mean(v, 60), 20), 8)
    right = ts_rank(decay_linear(corr, 7), 3)
    expected = pd.DataFrame(
        np.minimum(left.to_numpy(dtype=float), right.to_numpy(dtype=float)),
        index=c.index, columns=c.columns,
    )

    got = alpha_140.compute(p)

    assert got.iloc[-20:].notna().all().all()
    pd.testing.assert_frame_equal(got, expected, check_exact=False, rtol=1e-9)
