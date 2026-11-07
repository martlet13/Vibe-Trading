"""A single IC observation has no sample spread to report.

``_ic_stats`` fell back to ``ic_std = 0.0`` when only one cross-section cleared
``min_assets``, so the diagnostics table claimed the factor's IC was perfectly
stable on the strength of one date -- while ``ir`` in the same row already
answered None because the spread was unestimable. ``weight_turnover`` answers
None for the same sample size. This pins the third field to that convention.

The repo's position is already stated in
``agent/tests/test_bench_single_ic_observation.py``: "A single IC cannot
estimate sample spread or enter the factor ranking."
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.quantlib.combination import combine_factors

_DATES = pd.date_range("2024-01-01", periods=6, freq="B")
_COLUMNS = [f"S{i}" for i in range(6)]


def _panels(seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    return {
        "mom": pd.DataFrame(rng.normal(size=(6, 6)), index=_DATES, columns=_COLUMNS),
        "val": pd.DataFrame(rng.normal(size=(6, 6)), index=_DATES, columns=_COLUMNS),
    }


def _forward(rows: list[int], seed: int = 11) -> pd.DataFrame:
    """Forward returns readable on ``rows`` only; every other row is all-NaN."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(np.nan, index=_DATES, columns=_COLUMNS)
    for row in rows:
        frame.iloc[row] = rng.normal(size=len(_COLUMNS))
    return frame


def test_one_observation_reports_no_ic_std() -> None:
    result = combine_factors(
        _panels(), _forward([2]), method="ic", min_periods=2, min_assets=3
    )

    table = result.diagnostics["factor_ic"]
    for name in ("mom", "val"):
        assert table.loc[name, "n_obs"] == 1
        # Not 0.0: there is no spread to measure from a single point.
        assert math.isnan(float(table.loc[name, "ic_std"]))
        # ir already answered None for this sample size; ic_std now agrees.
        assert math.isnan(float(table.loc[name, "ir"]))
    # The convention this follows, in the same diagnostics dict.
    assert result.diagnostics["weight_turnover"] is None


def test_the_composite_summary_reports_no_ic_std_either() -> None:
    """composite_ic is a plain dict, so None survives instead of becoming NaN.

    ``insufficient="equal"`` gives every row a weight, so the composite is
    scored on the one date the forward returns cover.
    """
    result = combine_factors(
        _panels(), _forward([2]), method="ic", min_periods=2, min_assets=3,
        insufficient="equal",
    )

    composite = result.diagnostics["composite_ic"]
    assert composite["n_obs"] == 1
    assert composite["ic_std"] is None
    assert composite["ir"] is None


def test_two_observations_keep_the_measured_sample_std() -> None:
    result = combine_factors(
        _panels(), _forward([2, 4]), method="ic", min_periods=2, min_assets=3
    )

    table = result.diagnostics["factor_ic"]
    for name in ("mom", "val"):
        assert table.loc[name, "n_obs"] == 2
        measured = float(table.loc[name, "ic_std"])
        assert math.isfinite(measured)
        assert measured > 0.0
