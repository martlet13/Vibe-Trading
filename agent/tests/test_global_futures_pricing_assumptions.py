"""A global-futures run must say which numbers it assumed (#1393's twin).

``global_futures.py`` keeps two product tables of different sizes. A product
listed in one and absent from the other does not fail the run -- it silently
takes a generic constant that is indistinguishable from a real table entry in
the output: multiplier 50 (ES's own multiplier) and commission ``2.50``.

A product absent from both tables takes both defaults. Its multiplier sets
the contract count: the engine sizes to a target notional, so the error is
neutralised in notional and margin, while the number of contracts -- and with
it the per-contract commission and lot rounding -- is wrong by the ratio of
the contract's real multiplier to 50. A product present in the multiplier
table but absent from the commission table is half real: the contract count
is right and the fee is assumed. 22 of the 43 listed products have no
commission entry at all.

The direction agreed on #1393 is to make the default observable rather than to
fill the tables, because filling them leaves the *next* added product silently
defaulting. Numbers are unchanged here; the run states which of them were
assumed. ``_PRICE_LIMIT`` is deliberately not reported: a product absent from
it is not "priced" on a fallback, it is simply not subject to a limit modelled
for equity index contracts. Leverage is reported for the opposite reason: it
has no product table to miss, so the fixed default is always an assumption.
"""

from __future__ import annotations

import logging

import pytest

from backtest.engines.global_futures import (
    _COMMISSION_PER_CONTRACT,
    _DEFAULT_COMMISSION,
    _DEFAULT_MULTIPLIER,
    _MULTIPLIER,
    GlobalFuturesEngine,
)


def _engine(*codes: str, **config) -> GlobalFuturesEngine:
    return GlobalFuturesEngine(
        {"codes": list(codes), "initial_capital": 1_000_000, **config}
    )


def _assumptions(engine: GlobalFuturesEngine) -> dict[tuple[str, str], object]:
    reported = engine._engine_diagnostics().get("pricing_assumptions", [])
    return {(row["product"], row["field"]): row["value"] for row in reported}


# Sizing has no product table, so every run that does not configure leverage
# is priced on this assumption.
_LEVERAGE_ASSUMED = {("*", "leverage"): 10.0}


class TestPricingAssumptionsAreReported:
    def test_an_unlisted_product_reports_both_defaults(self) -> None:
        engine = _engine("ZZZF25")
        engine.get_contract_multiplier("ZZZF25")
        engine.calc_commission_for_symbol("ZZZF25", 1, 400.0, True)

        assumed = _assumptions(engine)
        assert assumed[("ZZZ", "contract_multiplier")] == float(_DEFAULT_MULTIPLIER)
        assert assumed[("ZZZ", "commission")] == _DEFAULT_COMMISSION

    def test_a_listed_product_missing_from_the_commission_table_is_reported(self) -> None:
        """KC is the half-real contract: multiplier looked up, commission assumed."""
        assert "KC" in _MULTIPLIER
        assert "KC" not in _COMMISSION_PER_CONTRACT

        engine = _engine("KCH25")
        engine.get_contract_multiplier("KCH25")
        engine.calc_commission_for_symbol("KCH25", 1, 380.0, True)

        assumed = _assumptions(engine)
        assert ("KC", "contract_multiplier") not in assumed
        assert assumed[("KC", "commission")] == _DEFAULT_COMMISSION

    def test_the_value_the_run_used_is_unchanged(self) -> None:
        """Observability only: every number stays exactly what it was."""
        engine = _engine("ZZZF25")
        assert engine.get_contract_multiplier("ZZZF25") == float(_DEFAULT_MULTIPLIER)
        assert engine.get_contract_multiplier("ESZ4") == float(_MULTIPLIER["ES"])
        assert engine.calc_commission_for_symbol("ZZZF25", 2, 400.0, True) == 5.0
        assert engine.calc_commission_for_symbol("ESZ4", 2, 6400.0, True) == 4.5

    def test_a_real_entry_equal_to_the_default_is_not_reported(self) -> None:
        """ES's listed multiplier is 50 -- the default's own value."""
        assert _MULTIPLIER["ES"] == _DEFAULT_MULTIPLIER
        engine = _engine("ESZ4")
        assert engine.get_contract_multiplier("ESZ4") == float(_DEFAULT_MULTIPLIER)
        engine.calc_commission_for_symbol("ESZ4", 1, 6400.0, True)

        assert _assumptions(engine) == _LEVERAGE_ASSUMED

    def test_a_fully_covered_run_declares_only_the_leverage_assumption(self) -> None:
        engine = _engine("ESZ4")
        engine.get_contract_multiplier("ESZ4")
        engine.calc_commission_for_symbol("ESZ4", 1, 6400.0, True)

        assert engine._engine_diagnostics() == {
            "pricing_assumptions": [
                {"product": "*", "field": "leverage", "value": 10.0}
            ]
        }

    def test_configured_leverage_is_not_reported_as_assumed(self) -> None:
        engine = _engine("ESZ4", leverage=4.0)

        assert engine.default_leverage == 4.0
        assert engine._engine_diagnostics() == {}

    def test_a_commission_override_is_not_a_table_default(self) -> None:
        """A configured override never consults the table, so nothing is assumed."""
        engine = _engine("ZZZF25", commission_per_contract=0.99)
        assert engine.calc_commission(2.0, 400.0, 1, True) == 1.98

        assert _assumptions(engine) == _LEVERAGE_ASSUMED

    def test_the_warning_fires_once_per_field_not_once_per_bar(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        engine = _engine("ZZZF25")
        with caplog.at_level(
            logging.WARNING, logger="backtest.engines.global_futures"
        ):
            for _ in range(50):
                engine.get_contract_multiplier("ZZZF25")

        multiplier_warnings = [
            record
            for record in caplog.records
            if "contract_multiplier" in record.getMessage()
        ]
        assert len(multiplier_warnings) == 1
        assert "ZZZ" in multiplier_warnings[0].getMessage()


class TestDiagnosticsSeam:
    def test_base_engine_declares_nothing_by_default(self) -> None:
        from backtest.engines.china_a import ChinaAEngine

        assert ChinaAEngine({"initial_capital": 1_000_000})._engine_diagnostics() == {}

    def test_the_report_reaches_the_run_metrics_not_just_the_hook(
        self, tmp_path
    ) -> None:
        """A hook nobody merges into the result is the failure this guards."""
        import numpy as np
        import pandas as pd

        index = pd.date_range("2026-01-01", periods=60, freq="D")
        prices = np.linspace(400.0, 440.0, 60)
        frame = pd.DataFrame(
            {
                "open": prices,
                "high": prices * 1.01,
                "low": prices * 0.99,
                "close": prices,
                "volume": 1e5,
                "pre_settle": np.r_[prices[0], prices[:-1]],
            },
            index=index,
        )

        class _Loader:
            def fetch(self, *args, **kwargs):
                return {"ZZZF25": frame}

        class _Signals:
            def generate(self, data_map):
                return {"ZZZF25": pd.Series(1.0, index=index)}

        config = {
            "codes": ["ZZZF25"],
            "initial_capital": 1_000_000,
            "start_date": "2026-01-01",
            "end_date": "2026-03-01",
        }
        result = GlobalFuturesEngine(config).run_backtest(
            config, _Loader(), _Signals(), tmp_path
        )

        metrics = result.get("metrics", result)
        reported = {
            (row["product"], row["field"]) for row in metrics["pricing_assumptions"]
        }
        assert ("ZZZ", "contract_multiplier") in reported
        assert ("ZZZ", "commission") in reported
        # Sizing is assumed on every unconfigured run, so it must reach the
        # same artifact the table fallbacks do.
        assert ("*", "leverage") in reported
