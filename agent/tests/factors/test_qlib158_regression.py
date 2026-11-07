"""Qlib Alpha158 compatibility regressions use independent formula oracles."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

_WINDOWS = (5, 10, 20, 30, 60)


def _close_through(window: int) -> pd.DataFrame:
    steps = np.arange(window + 1, dtype=np.float64)
    values = 30.0 + 0.7 * steps + 0.11 * steps**2 + np.sin(steps * 0.8)
    return pd.DataFrame(values, columns=["A"])


def _fit(window: np.ndarray) -> tuple[float, float]:
    slope, intercept = np.polyfit(np.arange(window.size, dtype=np.float64), window, 1)
    return float(slope), float(intercept)


@pytest.mark.parametrize("window", _WINDOWS)
def test_beta_is_the_ols_slope_over_latest_close(window: int) -> None:
    close = _close_through(window)
    module = importlib.import_module(f"src.factors.zoo.qlib158.beta{window}")

    actual = module.compute({"close": close})["A"]

    for end in (window - 1, window):
        values = close["A"].iloc[end - window + 1 : end + 1].to_numpy()
        slope, _ = _fit(values)
        assert actual.iloc[end] == pytest.approx(slope / values[-1], abs=1e-12)


@pytest.mark.parametrize("window", _WINDOWS)
def test_resi_is_the_latest_point_residual_over_latest_close(window: int) -> None:
    close = _close_through(window)
    module = importlib.import_module(f"src.factors.zoo.qlib158.resi{window}")

    actual = module.compute({"close": close})["A"]

    for end in (window - 1, window):
        values = close["A"].iloc[end - window + 1 : end + 1].to_numpy()
        slope, intercept = _fit(values)
        fitted = intercept + slope * (window - 1)
        assert actual.iloc[end] == pytest.approx(
            (values[-1] - fitted) / values[-1], abs=1e-12
        )


@pytest.mark.parametrize("stem", ("beta", "resi"))
def test_regression_features_require_a_complete_gap_free_window(stem: str) -> None:
    close = pd.DataFrame(
        {"A": [20.0, 21.0, 23.0, 24.0, 27.0, 28.0, 32.0, 33.0, 37.0, 39.0, 42.0, 46.0]}
    )
    close.iloc[3, 0] = np.nan
    module = importlib.import_module(f"src.factors.zoo.qlib158.{stem}5")

    actual = module.compute({"close": close})["A"]

    assert actual.iloc[3:8].isna().all()
    assert actual.iloc[8:].notna().all()


def _volume_through(window: int) -> pd.DataFrame:
    steps = np.arange(window + 2, dtype=np.float64)
    values = 1.0 + ((steps * 17.0 + 3.0) % 31.0)
    return pd.DataFrame(values, columns=["A"])


@pytest.mark.parametrize("window", _WINDOWS)
def test_roc_is_delayed_close_over_latest_close(window: int) -> None:
    close = _close_through(window + 1)
    module = importlib.import_module(f"src.factors.zoo.qlib158.roc{window}")

    actual = module.compute({"close": close})["A"]

    assert actual.iloc[:window].isna().all()
    for end in (window, window + 1):
        expected = close["A"].iloc[end - window] / close["A"].iloc[end]
        assert actual.iloc[end] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("window", _WINDOWS)
def test_cord_uses_log_of_one_plus_volume_ratio(window: int) -> None:
    close = _close_through(window + 1)
    volume = _volume_through(window)
    module = importlib.import_module(f"src.factors.zoo.qlib158.cord{window}")

    actual = module.compute({"close": close, "volume": volume})["A"]

    price_ratio = close["A"] / close["A"].shift(1)
    volume_change = np.log(volume["A"] / volume["A"].shift(1) + 1.0)
    assert actual.iloc[:window].isna().all()
    for end in (window, window + 1):
        window_slice = slice(end - window + 1, end + 1)
        expected = np.corrcoef(
            price_ratio.iloc[window_slice].to_numpy(),
            volume_change.iloc[window_slice].to_numpy(),
        )[0, 1]
        assert actual.iloc[end] == pytest.approx(expected, abs=1e-12)


def test_cord_plus_one_placement_can_flip_correlation_sign() -> None:
    close = pd.DataFrame({"A": [10.0, 11.0, 15.0, 14.0, 20.0, 22.0]})
    volume = pd.DataFrame({"A": [15.0, 29.0, 1.0, 1.0, 12.0, 28.0]})
    module = importlib.import_module("src.factors.zoo.qlib158.cord5")

    actual = module.compute({"close": close, "volume": volume})["A"].iloc[5]

    assert actual == pytest.approx(0.34260936518939894, abs=1e-12)


def test_cord_requires_a_complete_gap_free_volume_window() -> None:
    close = _close_through(11)
    volume = _volume_through(10)
    volume.iloc[3, 0] = np.nan
    module = importlib.import_module("src.factors.zoo.qlib158.cord5")

    actual = module.compute({"close": close, "volume": volume})["A"]

    assert actual.iloc[3:9].isna().all()
    assert actual.iloc[9:].notna().all()
