"""A rule's entry bounds must survive into the engine that actually executes.

Seam under test: ``save_profile`` -> ``load_profile`` -> ``render_signal_engine``,
observed two ways — the ``RULES`` literal of the generated ``code/signal_engine.py``
and the signals that generated engine produces when it is executed. Nothing here
reaches into ``codegen._rule_to_context``.

Why the round-trip matters: ``models.ShadowRule`` documents a price-feature
value as a bare scalar (read as ``>=``), an ``(op, value)`` pair or a
``{"min", "max"}`` mapping. ``save_profile`` writes JSON, so that pair is read
back as a list, and ``scanner._compare`` already accepts all of those spellings.
The generated engine runs the shadow's own backtest, so a bound it drops
backtests a looser rule than the profile states.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
import pytest

from src.shadow_account.codegen import render_signal_engine
from src.shadow_account.models import ShadowProfile, ShadowRule
from src.shadow_account.storage import load_profile, save_profile

SHADOW_ID = "shadow_0000b0und"

#: Daily bar timestamps start here; one bar per close handed to ``_signals``.
_START = pd.Timestamp("2024-03-01")

#: Twenty monotonically rising closes: RSI(14) reads 100 from bar 14 onwards.
_RISING_CLOSES = [100.0 + i for i in range(20)]

#: First bar of ``_RISING_CLOSES`` with a warm (non-NaN) RSI(14).
_RISING_FIRST_RSI_BAR = 14


def _profile(entry_condition: dict[str, Any]) -> ShadowProfile:
    rule = ShadowRule(
        rule_id="R1",
        human_text="5-day pullback then bounce",
        entry_condition=entry_condition,
        exit_condition={"holding_days": {"min": 2, "max": 5}},
        holding_days_range=(2, 5),
        support_count=3,
        coverage_rate=0.5,
        sample_trades=("AAA@2025-01-02",),
    )
    return ShadowProfile(
        shadow_id=SHADOW_ID,
        created_at="2025-01-01T00:00:00+00:00",
        journal_hash="0123456789abcdef",
        source_market="us",
        profitable_roundtrips=3,
        total_roundtrips=5,
        date_range=("2024-01-01", "2024-12-31"),
        profile_text="test portrait",
        rules=(rule,),
        preferred_markets=("us",),
        typical_holding_days=(3.0, 6.0),
    )


@pytest.fixture()
def saved_profile(tmp_path, monkeypatch) -> Callable[[dict[str, Any]], ShadowProfile]:
    """Persist a rule with the given ``entry_condition`` and load it back."""

    def _save(entry_condition: dict[str, Any]) -> ShadowProfile:
        monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path / "state"))
        save_profile(_profile(entry_condition))
        return load_profile(SHADOW_ID)

    return _save


def _generated(source: str) -> dict[str, Any]:
    """Execute the generated module and return its namespace."""
    namespace: dict[str, Any] = {}
    exec(
        compile(source, "<generated signal_engine>", "exec"), namespace
    )  # noqa: S102 - code under test
    return namespace


def _signals(source: str, closes: list[float]) -> list[float]:
    """Run the generated engine over one symbol and return its signal series."""
    module = _generated(source)
    index = pd.date_range(_START, periods=len(closes), freq="D")
    frame = pd.DataFrame({"close": closes}, index=index)
    return module["SignalEngine"]().generate({"AAA": frame})["AAA"].tolist()


def _first_entry(signals: list[float]) -> int | None:
    """Index of the first non-zero signal, or ``None`` if the engine never enters."""
    return next((i for i, value in enumerate(signals) if value), None)


@pytest.mark.parametrize(
    ("op", "bound_key"),
    [
        (">=", "prior_5d_return_min"),
        ("gte", "prior_5d_return_min"),
        ("ge", "prior_5d_return_min"),
        ("<=", "prior_5d_return_max"),
        ("lte", "prior_5d_return_max"),
        ("le", "prior_5d_return_max"),
    ],
)
def test_a_json_list_bound_reaches_the_generated_engine(
    saved_profile,
    op,
    bound_key,
) -> None:
    loaded = saved_profile({"market": "us", "prior_5d_return": (op, -0.08)})
    # JSON has no tuple: the documented (op, value) pair comes back as a list.
    assert loaded.rules[0].entry_condition["prior_5d_return"] == [op, -0.08]

    rule = _generated(render_signal_engine(loaded))["RULES"][0]
    assert rule[bound_key] == -0.08


def test_a_one_sided_dict_bound_reaches_the_generated_engine(saved_profile) -> None:
    """The scanner enforces `{"max": x}` alone; the engine must too."""
    loaded = saved_profile({"market": "us", "prior_5d_return": {"max": -0.08}})

    rule = _generated(render_signal_engine(loaded))["RULES"][0]
    assert rule["prior_5d_return_max"] == -0.08


def test_a_both_sided_dict_bound_still_reaches_the_generated_engine(
    saved_profile,
) -> None:
    loaded = saved_profile(
        {"market": "us", "prior_5d_return": {"min": -0.20, "max": -0.08}}
    )

    rule = _generated(render_signal_engine(loaded))["RULES"][0]
    assert (rule["prior_5d_return_min"], rule["prior_5d_return_max"]) == (-0.20, -0.08)


@pytest.mark.parametrize(
    ("entry_condition", "closes", "first_entry"),
    [
        # +10% over 5 bars: above the -8% ceiling, so no bar enters.
        ({"prior_5d_return": {"max": -0.08}}, [100.0] * 5 + [110.0], None),
        # -9% over 5 bars: inside the ceiling, so the last bar enters.
        ({"prior_5d_return": {"max": -0.08}}, [100.0] * 5 + [91.0], 5),
        # RSI(14) reads 100 on rising closes: above a 50 ceiling, never enters.
        ({"entry_rsi14": {"max": 50.0}}, _RISING_CLOSES, None),
        # Same closes under a 50 floor: the first warm RSI bar enters.
        ({"entry_rsi14": {"min": 50.0}}, _RISING_CLOSES, _RISING_FIRST_RSI_BAR),
    ],
)
def test_the_generated_engine_applies_a_one_sided_bound(
    saved_profile,
    entry_condition,
    closes,
    first_entry,
) -> None:
    """The executed engine must filter on a one-sided bound, not just print it.

    Each side of a bound carries its own entry, for RSI as well as for the
    prior return. If the template's RSI guard reverted to requiring both sides
    (``rsi_min is not None and rsi_max is not None``), a ``min``-only RSI rule
    would skip the check and enter on bar 0 rather than ``_RISING_FIRST_RSI_BAR``,
    and a ``max``-only RSI rule would enter on every warm bar rather than none.
    """
    loaded = saved_profile({**entry_condition, "market": "us"})

    signals = _signals(render_signal_engine(loaded), closes)
    assert _first_entry(signals) == first_entry


def test_a_documented_scalar_condition_bounds_the_generated_engine(
    saved_profile,
) -> None:
    """`models.ShadowRule` documents scalars; the scanner reads one as `>=`."""
    loaded = saved_profile({"market": "us", "prior_5d_return": -0.08})
    source = render_signal_engine(loaded)

    assert _signals(source, [100.0] * 5 + [91.0]) == [0.0] * 6  # -9%, below
    assert _signals(source, [100.0] * 5 + [95.0])[-1] == 1.0  # -5%, above


@pytest.mark.parametrize("op", [">", "<"])
def test_a_strict_comparison_is_refused_instead_of_dropped(saved_profile, op) -> None:
    """A strict operator has no exact inclusive render, so it errors instead."""
    loaded = saved_profile({"market": "us", "prior_5d_return": (op, 0.05)})

    with pytest.raises(ValueError, match="prior_5d_return"):
        render_signal_engine(loaded)


def test_a_null_bound_is_no_bound_in_the_generated_engine(saved_profile) -> None:
    """``None`` means "no bound" here too, matching the scanner's ``_to_float``."""
    loaded = saved_profile({"market": "us", "prior_5d_return": {"max": None}})

    rule = _generated(render_signal_engine(loaded))["RULES"][0]
    assert "prior_5d_return_max" not in rule
    assert "prior_5d_return_min" not in rule


def test_a_non_finite_bound_is_refused_instead_of_dropped(saved_profile) -> None:
    """A NaN threshold is not a bound the template can render, so it errors."""
    loaded = saved_profile({"market": "us", "prior_5d_return": {"min": float("nan")}})

    with pytest.raises(ValueError, match="prior_5d_return"):
        render_signal_engine(loaded)


def test_a_condition_tuple_cannot_become_a_market_name(saved_profile) -> None:
    """`market` is a documented key; a tuple there must not stringify silently."""
    loaded = saved_profile({"market": ("==", "us")})

    with pytest.raises(ValueError, match="market"):
        render_signal_engine(loaded)
