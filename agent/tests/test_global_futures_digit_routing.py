"""Digit-bearing contracts must reach the engine and its actual pricing tables."""

from __future__ import annotations

import pytest

from backtest.engines._market_hooks import _detect_market, code_currency
from backtest.engines.global_futures import GlobalFuturesEngine, _extract_product
from backtest.runner import _create_market_engine


@pytest.mark.parametrize(
    "symbol,product,multiplier,commission",
    [
        ("M2K=F", "M2K", 5, 0.62),
        ("M2KZ4", "M2K", 5, 0.62),
        ("M2K2503", "M2K", 5, 0.62),
        ("M2K.CME", "M2K", 5, 0.62),
        ("M2KZ2026.CME", "M2K", 5, 0.62),
        ("6E=F", "6E", 125000, 2.25),
        ("6EZ4", "6E", 125000, 2.25),
        ("6E2503", "6E", 125000, 2.25),
        ("6EZ2026.CME", "6E", 125000, 2.25),
        ("MYM=F", "MYM", 0.5, 0.62),
        ("OJ=F", "OJ", 150, 2.5),
        ("ZR=F", "ZR", 2000, 2.5),
    ],
)
@pytest.mark.parametrize("local", [False, True])
def test_contract_reaches_runner_pricing(symbol, product, multiplier, commission, local):
    symbol = f"local:{symbol.lower()}" if local else symbol
    engine = _create_market_engine("local", {"codes": [symbol]}, [symbol])
    assert isinstance(engine, GlobalFuturesEngine)
    assert _detect_market(symbol) == "futures"
    assert code_currency(symbol) == "USD"
    assert _extract_product(symbol) == product
    assert engine.get_contract_multiplier(symbol) == multiplier
    engine._active_symbol = symbol
    assert engine.calc_commission(2, 100, 1, True) == 2 * commission
    assumptions = engine._engine_diagnostics().get("pricing_assumptions", [])
    expected = (
        []
        if product not in {"OJ", "ZR"}
        else [{"product": product, "field": "commission", "value": 2.5}]
    )
    assert assumptions == [
        {"product": "*", "field": "leverage", "value": 10.0},
        *expected,
    ]


@pytest.mark.parametrize("symbol", ["M6E=F", "M6EZ4", "M6EZ2026.CME"])
def test_unlisted_digit_product_is_reported_without_invented_pricing(symbol):
    engine = _create_market_engine("local", {}, [symbol])
    assert isinstance(engine, GlobalFuturesEngine)
    assert _extract_product(symbol) == "M6E"
    engine.get_contract_multiplier(symbol)
    assert engine._engine_diagnostics()["pricing_assumptions"] == [
        {"product": "*", "field": "leverage", "value": 10.0},
        {"product": "M6E", "field": "contract_multiplier", "value": 50},
    ]


@pytest.mark.parametrize(
    "symbol,market",
    [
        ("600519", "a_share"),
        ("600519.SH", "a_share"),
        ("123=F", "a_share"),
        ("123Z4", "a_share"),
        ("BTC-USDT", "crypto"),
        ("EURUSD=X", "forex"),
        ("BRK.B.US", "us_equity"),
        ("IF2406.CFFEX", "futures"),
        ("RB0", "futures"),
    ],
)
def test_other_market_boundaries_are_preserved(symbol, market):
    assert _detect_market(symbol) == market


@pytest.mark.parametrize(
    "symbol,multiplier,fee",
    [("M2K=F", 5, 0.62), ("6E=F", 125000, 2.25), ("OJ=F", 150, 2.5), ("ZR=F", 2000, 2.5), ("M6E=F", 50, 2.5)],
)
def test_local_csv_reaches_runner_and_persisted_trades(tmp_path, monkeypatch, symbol, multiplier, fee):
    """Execute the public runner with real local data, signal loading and output."""
    import json
    import pandas as pd
    import dotenv
    import backtest.loaders.local_loader as local_loader
    from backtest import runner

    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    prices = tmp_path / "prices.csv"
    pd.DataFrame({"date": dates, "open": 10.0, "high": 10.1, "low": 9.9, "close": 10.0, "volume": 1e9}).to_csv(
        prices, index=False
    )
    bridge = tmp_path / "bridge.json"
    bridge.write_text(json.dumps({"sources": [{"symbol": symbol, "type": "csv", "path": str(prices)}]}))
    monkeypatch.setattr(local_loader, "_CONFIG_PATH", bridge)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setenv("VIBE_TRADING_ALLOWED_RUN_ROOTS", str(tmp_path))
    run = tmp_path / "run"
    (run / "code").mkdir(parents=True)
    (run / "code" / "signal_engine.py").write_text(
        "import pandas as pd\nclass SignalEngine:\n"
        "    def generate(self, data_map):\n"
        "        return {code: pd.Series(1.0, index=frame.index) for code, frame in data_map.items()}\n"
    )
    (run / "config.json").write_text(
        json.dumps(
            {
                "source": "local",
                "codes": [f"local:{symbol}"],
                "initial_cash": 10_000_000,
                "start_date": str(dates[0].date()),
                "end_date": str(dates[-1].date()),
                "slippage": 0,
                "leverage": 1,
            }
        )
    )
    runner.main(run)
    trades = pd.read_csv(run / "artifacts" / "trades.csv")
    assert not trades.empty
    assert set(trades["code"]) == {symbol}
    fills = [json.loads(line) for line in (run / "artifacts" / "fills.jsonl").read_text().splitlines()]
    assert len(fills) == 2
    for fill in fills:
        quantity = abs(fill["signed_quantity"])
        assert fill["fee"] / quantity == pytest.approx(fee)
        assert fill["notional"] / (quantity * fill["execution_price"]) == multiplier
    metrics = json.loads((run / "run_card.json").read_text())["structured_metrics"]
    assumptions = metrics.get("pricing_assumptions", [])
    assert (any(row["field"] == "contract_multiplier" for row in assumptions)) == (symbol == "M6E=F")
    # A listed product must not silently take ES's multiplier of 50.
    assert trades.iloc[0]["qty"] <= 10_000_000 / (10 * multiplier)
