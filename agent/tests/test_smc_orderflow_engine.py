"""SMC + order-flow skill engine: causality, stops and targets, confirmation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ENGINE_PATH = Path(__file__).resolve().parents[1] / "src" / "skills" / "smc-orderflow" / "example_signal_engine.py"


def _module():
    spec = importlib.util.spec_from_file_location("_smc_orderflow_engine", ENGINE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(seed: int, bars: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.8, bars))
    open_ = np.r_[close[0], close[:-1]]
    spread = rng.uniform(0.1, 0.8, bars)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + spread,
            "low": np.minimum(open_, close) - spread,
            "close": close,
            "volume": 1.0,
            "delta": rng.normal(0, 5, bars),
        },
        index=pd.date_range("2026-01-01", periods=bars, freq="5min"),
    )


def test_truncating_history_never_changes_earlier_signals():
    engine = _module().SignalEngine()
    emitted = 0
    for seed in range(5):
        frame = _frame(seed)
        full = engine.generate({"X": frame})["X"]
        emitted += int((full.diff().fillna(0) != 0).sum())
        for cut in range(60, len(frame), 37):
            partial = engine.generate({"X": frame.iloc[:cut]})["X"]
            pd.testing.assert_series_equal(partial, full.iloc[:cut], check_names=False)
    assert emitted > 0


@pytest.mark.parametrize("seed", range(4))
def test_every_trade_has_stop_and_target_on_the_right_side(seed):
    engine = _module().SignalEngine(rr=2.0)
    engine.generate({"X": _frame(seed)})
    trades = engine.trades["X"]
    assert trades
    for trade in trades:
        entry, stop, target = trade["entry_ref"], trade["stop_loss"], trade["take_profit"]
        if trade["side"] == "long":
            assert stop < entry < target
            assert target - entry == pytest.approx(2.0 * (entry - stop))
        else:
            assert target < entry < stop
            assert entry - target == pytest.approx(2.0 * (stop - entry))
        assert trade.get("exit_reason") in (None, "stop_loss", "take_profit")


def test_position_goes_flat_on_the_bar_that_touches_a_level():
    module = _module()
    frame = _frame(1)
    engine = module.SignalEngine()
    signal = engine.generate({"X": frame})["X"]
    for trade in engine.trades["X"]:
        if trade.get("exit_bar"):
            assert signal.loc[pd.Timestamp(trade["exit_bar"])] == 0
            assert signal.loc[pd.Timestamp(trade["time"])] != 0


def test_without_order_flow_columns_the_engine_refuses():
    frame = _frame(0).drop(columns=["delta"])
    with pytest.raises(ValueError, match="orderflow"):
        _module().SignalEngine().generate({"X": frame})
    assert _module().SignalEngine(confirm="none").generate({"X": frame})["X"].abs().sum() > 0


def test_unavailable_order_flow_never_confirms_an_entry():
    frame = _frame(2)
    frame["delta"] = np.nan
    assert _module().SignalEngine().generate({"X": frame})["X"].abs().sum() == 0


def test_confirmation_filters_entries():
    frame = _frame(3)
    loose = _module().SignalEngine(confirm="none")
    strict = _module().SignalEngine(confirm="absorption")
    loose.generate({"X": frame})
    strict.generate({"X": frame})
    assert len(strict.trades["X"]) < len(loose.trades["X"])


def test_short_side_can_be_disabled():
    engine = _module().SignalEngine(allow_short=False)
    signal = engine.generate({"X": _frame(4)})["X"]
    assert (signal >= 0).all()


@pytest.mark.parametrize("kwargs", [{"confirm": "maybe"}, {"rr": 0}, {"swing_length": 0}])
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        _module().SignalEngine(**kwargs)
