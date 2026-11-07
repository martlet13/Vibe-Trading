"""``effective_bars_per_year`` divides by calendar years, not whole days.

The function's own docstring promises the factor is "measured from the series
itself instead of assumed: bars observed divided by calendar years elapsed".
Dividing by ``diff.days`` truncates the fractional day off the span, which
turns the denominator into whole calendar days elapsed and inflates every
annualised number in a cross-market intraday run.

These tests pin the documented contract, not merely the current numbers: the
expected factor is recomputed here straight from the stated formula.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import calc_metrics, effective_bars_per_year


def _measured_bars_per_year(index: pd.DatetimeIndex) -> float:
    """The docstring contract, computed independently of the implementation."""
    span_days = (index[-1] - index[0]).total_seconds() / 86400.0
    return len(index) / (span_days / 365.25)


class TestFactorIsMeasuredFromTheFullSpan:
    """``effective_bars_per_year`` divides by calendar *years*, not whole days."""

    def test_hourly_bars_over_three_days_use_the_fractional_span(self):
        # 72 hourly bars span 2 days 23:00 — truncating to 2 calendar days
        # overstates the factor by ~48%.
        idx = pd.date_range("2026-01-01", periods=72, freq="h")
        assert effective_bars_per_year(idx) == int(_measured_bars_per_year(idx))

    def test_fifteen_minute_bars_over_five_days_use_the_fractional_span(self):
        idx = pd.date_range("2026-01-01", periods=480, freq="15min")
        assert effective_bars_per_year(idx) == int(_measured_bars_per_year(idx))

    def test_intraday_factor_is_within_a_tenth_of_a_percent_of_the_contract(self):
        for freq, periods in (("h", 72), ("15min", 480), ("h", 720), ("30min", 96)):
            idx = pd.date_range("2026-01-01", periods=periods, freq=freq)
            expected = _measured_bars_per_year(idx)
            got = effective_bars_per_year(idx)
            assert abs(got / expected - 1.0) < 1e-3, (
                f"{periods}x{freq}: got {got}, contract says {expected:.1f}"
            )

    def test_midnight_aligned_daily_bars_are_unchanged(self):
        idx = pd.date_range("2024-01-01", periods=253, freq="B")
        assert effective_bars_per_year(idx) == int(_measured_bars_per_year(idx))

    def test_span_shorter_than_a_day_still_counts_as_one_year(self):
        # The sub-day clause is what keeps a near-zero divisor out of the
        # measurement; measuring the fractional span must not remove it.
        idx = pd.DatetimeIndex(["2026-01-01T09:30", "2026-01-01T15:00"])
        assert effective_bars_per_year(idx) == 2

    def test_empty_and_non_datetime_indexes_still_fall_back(self):
        assert effective_bars_per_year(pd.DatetimeIndex([])) == 252
        assert effective_bars_per_year(pd.Index([0, 1, 2, 3])) == 4

    def test_intraday_sharpe_is_annualised_on_the_measured_factor(self):
        idx = pd.date_range("2026-01-01", periods=72, freq="h")
        rng = np.random.default_rng(7)
        equity = pd.Series(
            100_000 * np.cumprod(1 + rng.normal(0.0002, 0.004, len(idx))), index=idx
        )
        measured = int(_measured_bars_per_year(idx))
        truncated = int(len(idx) / ((idx[-1] - idx[0]).days / 365.25))
        assert truncated > measured  # premise: whole days overstate the factor
        derived = calc_metrics(equity, [], 100_000.0, None)
        on_contract = calc_metrics(equity, [], 100_000.0, measured)
        on_truncated = calc_metrics(equity, [], 100_000.0, truncated)
        assert derived["sharpe"] == pytest.approx(on_contract["sharpe"], rel=1e-9)
        assert derived["sharpe"] != pytest.approx(on_truncated["sharpe"], rel=1e-3)


def _cross_market_closes(n: int = 365) -> pd.DataFrame:
    """A crypto symbol on the calendar grid and an equity on weekdays only."""
    idx = pd.date_range("2026-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    btc = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, n)), index=idx)
    aapl = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.015, n)), index=idx)
    aapl[~aapl.index.dayofweek.isin([0, 1, 2, 3, 4])] = np.nan
    return pd.DataFrame({"BTC-USDT": btc, "AAPL": aapl})
