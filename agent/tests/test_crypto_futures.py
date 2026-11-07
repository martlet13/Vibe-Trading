"""USDT-perpetual futures on Binance and Bybit: the three protection rules.

Every futures order is held to: leverage 1-5x, isolated margin only, and a
stop-loss plus take-profit on every new position. The exchange doubles
subclass the real ccxt classes and override only their network methods, so
endpoint resolution and host checks run against ccxt's own URLs. No test
reaches the network.
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest

ccxt = pytest.importorskip("ccxt")

from src.live import registry
from src.live.classification import ToolClass, classify_tool
from src.trading import futures_guard as guard
from src.trading import profiles, service
from src.trading.connectors.binance import sdk as bn
from src.trading.connectors.bybit import sdk as bb

pytestmark = pytest.mark.unit

LONG = {"side": "buy", "stop_loss": 90.0, "take_profit": 120.0}


# --------------------------------------------------------------------------- #
# Shared protection rules
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("leverage, expected", [(None, 1), (1, 1), (5, 5), ("3", 3), (2.0, 2)])
def test_leverage_accepts_whole_multiples_up_to_five(leverage, expected) -> None:
    assert guard.parse_leverage(leverage) == expected


@pytest.mark.parametrize("leverage", [0, 6, 10, 2.5, -1, True, "x", float("inf")])
def test_leverage_outside_one_to_five_is_refused(leverage) -> None:
    with pytest.raises(guard.FuturesOrderError):
        guard.parse_leverage(leverage)


def test_max_leverage_is_five_and_margin_isolated() -> None:
    assert guard.MAX_LEVERAGE == 5
    assert guard.MARGIN_MODE == "isolated"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"side": "buy", "take_profit": 120}, "both stop_loss and take_profit"),
        ({"side": "buy", "stop_loss": 90}, "both stop_loss and take_profit"),
        ({"side": "buy", "stop_loss": 105, "take_profit": 120}, "stop_loss < entry < take_profit"),
        ({"side": "buy", "stop_loss": 90, "take_profit": 95}, "stop_loss < entry < take_profit"),
        ({"side": "sell", "stop_loss": 90, "take_profit": 80}, "take_profit < entry < stop_loss"),
        ({"side": "sell", "stop_loss": 120, "take_profit": 105}, "take_profit < entry < stop_loss"),
        ({"side": "buy", "stop_loss": -1, "take_profit": 120}, "stop_loss must be a positive"),
        ({"side": "hold", "stop_loss": 90, "take_profit": 120}, "side must be"),
        ({**LONG, "leverage": 6}, "between 1 and 5x"),
        ({**LONG, "reduce_only": "yes"}, "reduce_only must be true or false"),
    ],
)
def test_opening_orders_need_correct_protection(kwargs, message) -> None:
    with pytest.raises(guard.FuturesOrderError, match=message):
        guard.validate_order(reference_price=100.0, **kwargs)


def test_unknown_entry_price_cannot_be_protected() -> None:
    with pytest.raises(guard.FuturesOrderError, match="reference price"):
        guard.validate_order(reference_price=None, **LONG)


def test_valid_long_and_short() -> None:
    long = guard.validate_order(reference_price=100.0, leverage=5, **LONG)
    assert (long.leverage, long.stop_loss, long.take_profit, long.reduce_only) == (5, 90.0, 120.0, False)
    short = guard.validate_order(side="sell", reference_price=100.0, stop_loss=110, take_profit=80)
    assert short.leverage == 1


@pytest.mark.parametrize("extra", [{"stop_loss": 90}, {"take_profit": 120}, {"leverage": 2}])
def test_reduce_only_takes_no_protection_or_leverage(extra) -> None:
    with pytest.raises(guard.FuturesOrderError):
        guard.validate_order(side="sell", reference_price=None, reduce_only=True, **extra)


def test_reduce_only_needs_no_price() -> None:
    order = guard.validate_order(side="sell", reference_price=None, reduce_only=True)
    assert order.reduce_only and order.leverage is None


# --------------------------------------------------------------------------- #
# Profiles and classification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("broker", ["binance", "bybit"])
def test_futures_profiles(broker) -> None:
    paper = profiles.profile_by_id(f"{broker}-futures-paper-trade")
    live = profiles.profile_by_id(f"{broker}-futures-live-trade")
    assert paper.environment == "paper" and "orders.place" in paper.capabilities
    assert live.environment == "live" and "orders.place.requires_mandate" in live.capabilities
    assert "orders.place" not in live.capabilities
    assert paper.config["futures"] is True and live.config["futures"] is True
    assert paper.readonly is False and live.readonly is False


@pytest.mark.parametrize("broker", ["binance", "bybit"])
def test_margin_and_leverage_writes_are_write_pinned(broker) -> None:
    curated = registry._BROKER_CURATED_MAPS[broker]
    assert classify_tool("set_margin_mode", None, curated) is ToolClass.WRITE
    assert classify_tool("set_leverage", None, curated) is ToolClass.WRITE
    assert classify_tool("fetch_positions", None, curated) is ToolClass.READ


# --------------------------------------------------------------------------- #
# Recording exchanges (real ccxt classes, network methods overridden)
# --------------------------------------------------------------------------- #


class _Recorder:
    """Mixin: record calls and answer from per-test settings."""

    settings: dict = {}

    def _init_recorder(self) -> None:
        self.calls: list[tuple] = []
        self.margin = self.settings.get("margin", "cross")
        self.lev = None

    def fetch(self, *args, **kwargs):
        raise AssertionError(f"test reached the network: {args[:2]}")

    def fetch_balance(self, params=None):
        self.calls.append(("fetch_balance",))
        return {"info": {}, "free": {}, "used": {}, "total": {}, "USDT": {"free": 900.0, "used": 100.0, "total": 1000.0}}

    def fetch_ticker(self, symbol, params=None):
        self.calls.append(("fetch_ticker", symbol))
        return {"last": self.settings.get("last", 100.0), "bid": 99.0, "ask": 101.0}

    def fetch_margin_mode(self, symbol, params=None):
        self.calls.append(("fetch_margin_mode", symbol))
        return {"marginMode": self.margin}

    def set_margin_mode(self, mode, symbol=None, params=None):
        self.calls.append(("set_margin_mode", mode, symbol, dict(params or {})))
        if not self.settings.get("margin_switch_ignored"):
            self.margin = mode
        return {}

    def set_leverage(self, leverage, symbol=None, params=None):
        self.calls.append(("set_leverage", leverage, symbol))
        self.lev = leverage
        return {}

    def fetch_leverage(self, symbol, params=None):
        self.calls.append(("fetch_leverage", symbol))
        return {"longLeverage": self.settings.get("reported_leverage", self.lev), "marginMode": self.margin}

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        params = dict(params or {})
        self.calls.append(("create_order", symbol, type, side, amount, price, params))
        fail = self.settings.get("fail_on")
        if fail and fail in params:
            raise ccxt.ExchangeError(f"rejected {fail}")
        if self.settings.get("fail_close") and params == {"reduceOnly": True}:
            raise ccxt.ExchangeError("close rejected")
        n = len([c for c in self.calls if c[0] == "create_order"])
        return {"id": f"o{n}", "symbol": symbol, "side": side, "type": type, "status": "closed",
                "amount": amount, "filled": amount, "price": price}

    def cancel_order(self, id, symbol=None, params=None):
        self.calls.append(("cancel_order", id, symbol, dict(params or {})))
        return {"id": id, "symbol": symbol, "status": "canceled"}

    def fetch_positions(self, symbols=None, params=None):
        return [
            {"symbol": "BTC/USDT:USDT", "contracts": 0.01, "contractSize": 1, "side": "long",
             "notional": 650.0, "entryPrice": 64000, "leverage": 3, "marginMode": "isolated"},
            {"symbol": "ETH/USDT:USDT", "contracts": 0.5, "contractSize": 1, "side": "short",
             "notional": -1500.0, "leverage": 2, "marginMode": "isolated"},
            {"symbol": "SOL/USDT:USDT", "contracts": 0, "side": "long", "notional": 0},
        ]


class _Bybit(_Recorder, ccxt.bybit):
    instances: list = []

    def __init__(self, config=None):
        ccxt.bybit.__init__(self, config or {})
        self._init_recorder()
        _Bybit.instances.append(self)


class _BinanceUsdm(_Recorder, ccxt.binanceusdm):
    instances: list = []

    def __init__(self, config=None):
        ccxt.binanceusdm.__init__(self, config or {})
        self._init_recorder()
        _BinanceUsdm.instances.append(self)


@pytest.fixture
def exchanges(monkeypatch):
    _Bybit.instances, _BinanceUsdm.instances = [], []
    _Recorder.settings = {}
    fake = type("FakeCcxt", (), {
        "bybit": _Bybit,
        "binanceusdm": _BinanceUsdm,
        "binance": ccxt.binance,
        "ArgumentsRequired": ccxt.ArgumentsRequired,
        "BadSymbol": ccxt.BadSymbol,
        "NotSupported": ccxt.NotSupported,
    })
    monkeypatch.setattr(bb, "_require_ccxt", lambda: fake)
    monkeypatch.setattr(bn, "_require_ccxt", lambda: fake)
    return _Recorder


def _writes(instances) -> list[tuple]:
    return [c for inst in instances for c in inst.calls if c[0] in ("create_order", "set_margin_mode", "set_leverage")]


def _bybit_cfg(profile="paper"):
    return bb.BybitConfig.from_mapping({"api_key": "k", "api_secret": "s", "profile": profile, "futures": True})


def _binance_cfg(profile="paper", **extra):
    return bn.BinanceConfig.from_mapping(
        {"api_key": "k", "api_secret": "s", "profile": profile, "futures": True, **extra}
    )


# --------------------------------------------------------------------------- #
# Profile flag cannot be smuggled in
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("module, filename", [(bb, "bybit.json"), (bn, "binance.json")])
def test_saved_file_or_override_cannot_turn_a_spot_profile_into_futures(monkeypatch, tmp_path, module, filename) -> None:
    monkeypatch.setattr(module, "get_runtime_root", lambda: tmp_path)
    (tmp_path / filename).write_text('{"api_key": "k", "api_secret": "s", "futures": true}', encoding="utf-8")
    assert module.build_config({"profile": "live"}, {"futures": True}).futures is False
    assert module.build_config({"profile": "live", "futures": True}).futures is True


def test_futures_flag_is_refused_on_read_only_profiles() -> None:
    with pytest.raises(bb.BybitConfigError):
        bb.BybitConfig.from_mapping({"profile": "live-readonly", "futures": True})
    with pytest.raises(bn.BinanceConfigError):
        bn.BinanceConfig.from_mapping({"profile": "live-readonly", "futures": True})


@pytest.mark.parametrize("module, cfg", [(bb, lambda: bb.BybitConfig(profile="paper")), (bn, lambda: bn.BinanceConfig(profile="paper"))])
def test_spot_profiles_refuse_futures_fields(exchanges, module, cfg) -> None:
    result = module.place_order(cfg(), symbol="BTC-USDT", side="buy", quantity=1, stop_loss=90, take_profit=120)
    assert result["status"] == "error" and "futures profiles only" in result["error"]


# --------------------------------------------------------------------------- #
# Bybit perpetuals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile, host", [("paper", "api-testnet.bybit.com"), ("live", "api.bybit.com")])
def test_bybit_futures_client_is_linear_on_the_profile_host(exchanges, profile, host) -> None:
    ex = bb._exchange(_bybit_cfg(profile))
    assert ex.options["defaultType"] == "swap" and ex.options["defaultSubType"] == "linear"
    assert {ex.implode_hostname(u) for u in ex.urls["api"].values()} == {f"https://{host}"}


def test_bybit_opening_order_is_isolated_leveraged_and_protected(exchanges) -> None:
    result = bb.place_order(_bybit_cfg(), symbol="BTC-USDT", side="buy", quantity=0.01,
                            stop_loss=90, take_profit=120, leverage=3)
    assert result["status"] == "ok", result
    assert (result["leverage"], result["margin_mode"], result["stop_loss"], result["take_profit"]) == (3, "isolated", 90.0, 120.0)
    calls = _Bybit.instances[-1].calls
    assert ("set_margin_mode", "isolated", "BTC/USDT:USDT", {"leverage": 3}) in calls
    assert ("set_leverage", 3, "BTC/USDT:USDT") in calls
    create = [c for c in calls if c[0] == "create_order"]
    assert len(create) == 1
    _, symbol, otype, side, amount, price, params = create[0]
    assert (symbol, otype, side, amount, price) == ("BTC/USDT:USDT", "market", "buy", 0.01, None)
    assert params == {"stopLoss": {"triggerPrice": 90.0}, "takeProfit": {"triggerPrice": 120.0}, "tpslMode": "Full"}
    # Margin and leverage are confirmed before the order goes out.
    names = [c[0] for c in calls]
    assert names.index("fetch_leverage") < names.index("create_order")


def test_bybit_limit_entry_attaches_protection_to_the_resting_order(exchanges) -> None:
    result = bb.place_order(_bybit_cfg(), symbol="ETHUSDT", side="sell", quantity=1, order_type="limit",
                            limit_price=3000, stop_loss=3100, take_profit=2800, leverage=2)
    assert result["status"] == "ok", result
    create = [c for c in _Bybit.instances[-1].calls if c[0] == "create_order"][0]
    assert create[2:6] == ("limit", "sell", 1.0, 3000.0)
    assert create[6]["timeInForce"] == "GTC" and create[6]["stopLoss"] == {"triggerPrice": 3100.0}


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"margin_switch_ignored": True}, "not isolated"),
        ({"reported_leverage": 10}, "not 3x"),
    ],
)
def test_bybit_unconfirmed_margin_settings_block_the_order(exchanges, settings, message) -> None:
    exchanges.settings = settings
    result = bb.place_order(_bybit_cfg(), symbol="BTC-USDT", side="buy", quantity=0.01,
                            stop_loss=90, take_profit=120, leverage=3)
    assert result["status"] == "error" and message in result["error"]
    assert not [c for c in _Bybit.instances[-1].calls if c[0] == "create_order"]


def test_bybit_unprotected_entry_never_touches_the_account(exchanges) -> None:
    result = bb.place_order(_bybit_cfg(), symbol="BTC-USDT", side="buy", quantity=0.01, stop_loss=90)
    assert result["status"] == "error" and "take_profit" in result["error"]
    assert _writes(_Bybit.instances) == []


def test_bybit_reduce_only_close_skips_margin_and_protection(exchanges) -> None:
    result = bb.place_order(_bybit_cfg(), symbol="BTC-USDT", side="sell", quantity=0.01, reduce_only=True)
    assert result["status"] == "ok" and result["reduce_only"] is True
    writes = _writes(_Bybit.instances)
    assert writes == [("create_order", "BTC/USDT:USDT", "market", "sell", 0.01, None, {"reduceOnly": True})]


def test_bybit_futures_refuse_notional_and_non_usdt(exchanges) -> None:
    assert "not 'notional'" in bb.place_order(_bybit_cfg(), symbol="BTC-USDT", side="buy", notional=50,
                                               stop_loss=90, take_profit=120)["error"]
    assert "USDT-margined" in bb.place_order(_bybit_cfg(), symbol="BTC-USDC", side="buy", quantity=1,
                                              stop_loss=90, take_profit=120)["error"]


def test_bybit_futures_positions_are_signed_with_notional(exchanges) -> None:
    rows = bb.get_positions(_bybit_cfg("live"))["positions"]
    assert {r["symbol"]: (r["quantity"], r["value_usd"]) for r in rows} == {
        "BTC/USDT:USDT": (0.01, 650.0),
        "ETH/USDT:USDT": (-0.5, 1500.0),
    }


# --------------------------------------------------------------------------- #
# Binance USD-M perpetuals
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("profile, host", [("paper", "testnet.binancefuture.com"), ("live", "fapi.binance.com")])
def test_binance_futures_client_resolves_fapi_to_the_profile_host(exchanges, profile, host) -> None:
    ex = bn._exchange(_binance_cfg(profile))
    fapi = {name: url for name, url in ex.urls["api"].items() if name.startswith("fapi")}
    assert fapi and all(host in url for url in fapi.values())


def test_binance_paper_futures_fail_closed_when_sandbox_does_not_apply(monkeypatch, exchanges) -> None:
    monkeypatch.setattr(_BinanceUsdm, "set_sandbox_mode", lambda self, enabled: None)
    result = bn.place_order(_binance_cfg(), symbol="BTC-USDT", side="buy", quantity=0.01,
                            stop_loss=90, take_profit=120)
    assert result["status"] == "error" and "testnet.binancefuture.com" in result["error"]
    assert _writes(_BinanceUsdm.instances) == []


def test_binance_entry_is_followed_by_reduce_only_stop_and_target(exchanges) -> None:
    result = bn.place_order(_binance_cfg(), symbol="BTCUSDT", side="buy", quantity=0.01,
                            stop_loss=90, take_profit=120, leverage=5)
    assert result["status"] == "ok", result
    assert (result["leverage"], result["margin_mode"]) == (5, "isolated")
    assert result["stop_loss_order_id"] == "o2" and result["take_profit_order_id"] == "o3"
    creates = [c for c in _BinanceUsdm.instances[-1].calls if c[0] == "create_order"]
    assert creates[0][1:7] == ("BTC/USDT:USDT", "market", "buy", 0.01, None, {})
    assert creates[1][3:5] == ("sell", 0.01)
    assert creates[1][6] == {"stopLossPrice": 90.0, "reduceOnly": True, "workingType": "MARK_PRICE"}
    assert creates[2][6] == {"takeProfitPrice": 120.0, "reduceOnly": True, "workingType": "MARK_PRICE"}
    calls = _BinanceUsdm.instances[-1].calls
    assert ("set_margin_mode", "isolated", "BTC/USDT:USDT", {}) in calls
    assert ("set_leverage", 5, "BTC/USDT:USDT") in calls


def test_binance_limit_entries_are_refused(exchanges) -> None:
    result = bn.place_order(_binance_cfg(), symbol="BTC-USDT", side="buy", quantity=0.01, order_type="limit",
                            limit_price=95, stop_loss=90, take_profit=120)
    assert result["status"] == "error" and "market orders only" in result["error"]
    assert _writes(_BinanceUsdm.instances) == []
    close = bn.place_order(_binance_cfg(), symbol="BTC-USDT", side="sell", quantity=0.01, order_type="limit",
                           limit_price=95, reduce_only=True)
    assert close["status"] == "error" and "market orders only" in close["error"]
    assert _writes(_BinanceUsdm.instances) == []


def test_binance_reduce_only_close_is_a_plain_reduce_only_market_order(exchanges) -> None:
    result = bn.place_order(_binance_cfg(), symbol="BTC-USDT", side="sell", quantity=0.01, reduce_only=True)
    assert result["status"] == "ok" and result["reduce_only"] is True
    assert _writes(_BinanceUsdm.instances) == [
        ("create_order", "BTC/USDT:USDT", "market", "sell", 0.01, None, {"reduceOnly": True})
    ]


def test_binance_failed_protection_closes_the_entry(exchanges) -> None:
    exchanges.settings = {"fail_on": "takeProfitPrice"}
    result = bn.place_order(_binance_cfg(), symbol="BTC-USDT", side="buy", quantity=0.01,
                            stop_loss=90, take_profit=120)
    assert result["status"] == "error" and result["unwound"] is True
    calls = _BinanceUsdm.instances[-1].calls
    # The stop that was placed is cancelled from the conditional book, then
    # the entry is flattened with a reduce-only market order.
    assert ("cancel_order", "o2", "BTC/USDT:USDT", {"trigger": True}) in calls
    assert calls[-1] == ("create_order", "BTC/USDT:USDT", "market", "sell", 0.01, None, {"reduceOnly": True})


def test_binance_failed_close_reports_an_unprotected_position(exchanges) -> None:
    exchanges.settings = {"fail_on": "stopLossPrice", "fail_close": True}
    result = bn.place_order(_binance_cfg(), symbol="BTC-USDT", side="buy", quantity=0.01,
                            stop_loss=90, take_profit=120)
    assert result["status"] == "error" and result["unprotected_position"] is True
    assert "UNPROTECTED POSITION" in result["error"]


def test_binance_futures_testnet_keys_are_preferred(exchanges) -> None:
    ex = bn._exchange(_binance_cfg(futures_api_key="fk", futures_api_secret="fs"))
    assert (ex.apiKey, ex.secret) == ("fk", "fs")
    assert bn._public_config(_binance_cfg(futures_api_key="fk12", futures_api_secret="fs"))["futures_api_secret"] == "***redacted***"


def test_binance_cancel_never_addresses_the_conditional_book(exchanges) -> None:
    result = bn.cancel_order(_binance_cfg(), "123", symbol="BTC-USDT")
    assert result["status"] == "ok"
    assert ("cancel_order", "123", "BTC/USDT:USDT", {}) in _BinanceUsdm.instances[-1].calls


# --------------------------------------------------------------------------- #
# Service, agent tool and live gate
# --------------------------------------------------------------------------- #


def _isolate(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bb, "get_runtime_root", lambda: tmp_path)
    monkeypatch.setattr(bn, "get_runtime_root", lambda: tmp_path)
    (tmp_path / "bybit.json").write_text('{"api_key": "k", "api_secret": "s"}', encoding="utf-8")
    (tmp_path / "binance.json").write_text('{"api_key": "k", "api_secret": "s"}', encoding="utf-8")


def test_tool_forwards_protection_to_the_paper_futures_profile(monkeypatch, tmp_path, exchanges) -> None:
    import json

    from src.tools.trading_connector_tool import TradingPlaceOrderTool

    _isolate(monkeypatch, tmp_path)
    tool = TradingPlaceOrderTool()
    for field in ("stop_loss", "take_profit", "leverage", "reduce_only"):
        assert field in tool.parameters["properties"]
    assert tool.parameters["properties"]["leverage"]["maximum"] == 5
    out = json.loads(tool.execute(connection="bybit-futures-paper-trade", symbol="BTC-USDT", side="buy",
                                  quantity=0.01, stop_loss=90, take_profit=120, leverage=2))
    assert out["status"] == "ok", out
    assert out["environment"] == "paper" and out["leverage"] == 2


def test_service_does_not_send_futures_fields_to_other_connectors(monkeypatch) -> None:
    seen = {}

    class Conn:
        @staticmethod
        def build_config(profile_config, overrides):
            return object()

        @staticmethod
        def place_order(config, **kwargs):
            seen.update(kwargs)
            return {"status": "ok"}

    monkeypatch.setattr(service, "_sdk_module", lambda c: Conn)
    service.place_order("AAPL", "alpaca-paper-trade", side="buy", quantity=1)
    assert not {"stop_loss", "take_profit", "leverage", "reduce_only"} & set(seen)


def _mandate():
    from src.live.mandate.model import AssetClass, ConsentMeta, HardCaps, InstrumentType, Mandate, UniverseConstraint

    return Mandate(
        schema_version=1,
        hard_caps=HardCaps(account_funding_usd=1_000.0, max_order_notional_usd=500.0,
                           max_total_exposure_usd=5_000.0, max_leverage=5.0,
                           allowed_instruments=(InstrumentType.CRYPTO,), max_trades_per_day=5),
        universe=UniverseConstraint(asset_classes=(AssetClass.CRYPTO,), min_market_cap_usd=None,
                                    min_avg_daily_volume_usd=None, exclude_symbols=()),
        consent=ConsentMeta(created_at="2026-01-01T00:00:00+00:00", consent_token_sha256="x",
                            broker="bybit", account_ref="", expires_at="2999-01-01T00:00:00+00:00"),
    )


def _patch_gate(monkeypatch, tmp_path, mandate) -> None:
    import src.live.paths as live_paths
    from src.live import sdk_order_gate as gate

    monkeypatch.setattr(live_paths, "get_runtime_root", lambda: tmp_path)
    monkeypatch.setattr(gate, "load_mandate", lambda broker: mandate)
    monkeypatch.setattr(gate, "halt_flag_set", lambda broker: False)
    monkeypatch.setattr(gate, "write_live_action", lambda *a, **k: {"audited": True})
    monkeypatch.setattr(gate, "read_daily_count", lambda broker: 0)
    monkeypatch.setattr(gate, "increment_daily_count", lambda broker, action_id=None: 1)
    monkeypatch.setattr(gate, "daily_order_lock", lambda broker: nullcontext())


@pytest.mark.parametrize("profile_id, instances", [
    ("bybit-futures-live-trade", _Bybit.instances),
    ("binance-futures-live-trade", _BinanceUsdm.instances),
])
def test_live_futures_without_mandate_are_blocked(monkeypatch, tmp_path, exchanges, profile_id, instances) -> None:
    _isolate(monkeypatch, tmp_path)
    _patch_gate(monkeypatch, tmp_path, None)
    out = service.place_order("BTC-USDT", profile_id, side="buy", quantity=0.01, stop_loss=90, take_profit=120)
    assert out["status"] == "blocked"
    assert _writes(_Bybit.instances + _BinanceUsdm.instances) == []


def test_live_bybit_futures_within_mandate_reach_mainnet_protected(monkeypatch, tmp_path, exchanges) -> None:
    _isolate(monkeypatch, tmp_path)
    _patch_gate(monkeypatch, tmp_path, _mandate())
    out = service.place_order("BTC-USDT", "bybit-futures-live-trade", side="buy", quantity=1,
                              stop_loss=90, take_profit=120, leverage=2)
    assert out["status"] == "ok", out
    creates = [(inst, c) for inst in _Bybit.instances for c in inst.calls if c[0] == "create_order"]
    assert len(creates) == 1
    inst, call = creates[0]
    assert call[6]["stopLoss"] == {"triggerPrice": 90.0} and call[6]["takeProfit"] == {"triggerPrice": 120.0}
    assert {inst.implode_hostname(u) for u in inst.urls["api"].values()} == {"https://api.bybit.com"}


def test_live_futures_over_the_mandate_order_cap_are_not_sent(monkeypatch, tmp_path, exchanges) -> None:
    _isolate(monkeypatch, tmp_path)
    _patch_gate(monkeypatch, tmp_path, _mandate())
    # 10 BTC-perp units at 100 = 1,000 USD notional, over the 500 USD order cap.
    out = service.place_order("BTC-USDT", "bybit-futures-live-trade", side="buy", quantity=10,
                              stop_loss=90, take_profit=120)
    assert out["status"] != "ok"
    assert not [c for inst in _Bybit.instances for c in inst.calls if c[0] == "create_order"]


# --------------------------------------------------------------------------- #
# Mandate gate: margined futures are capped by leverage, not by cash funding
# --------------------------------------------------------------------------- #


def _check(notional: float, *, margined: bool, side: str = "buy"):
    from src.live.enforcement import OrderIntent, check_mandate
    from src.live.mandate.model import AssetClass, InstrumentType

    intent = OrderIntent(
        symbol="BTC-USDT", side=side, notional_usd=notional, quantity=None,
        instrument_type=InstrumentType.CRYPTO, asset_class=AssetClass.CRYPTO, margined=margined,
    )
    return check_mandate(_mandate(), intent, {"positions": []}, {}, broker="bybit",
                         remote_tool="place_order", daily_count=0)


def test_margined_buy_above_funding_but_within_mandate_leverage_passes() -> None:
    from dataclasses import replace

    from src.live import enforcement

    mandate = _mandate()
    roomy = replace(mandate, hard_caps=replace(mandate.hard_caps, max_order_notional_usd=5_000.0))
    intent = enforcement.OrderIntent(
        symbol="BTC-USDT", side="buy", notional_usd=2_000.0, quantity=None,
        instrument_type=mandate.hard_caps.allowed_instruments[0],
        asset_class=mandate.universe.asset_classes[0], margined=True,
    )
    # 2,000 USD on 1,000 USD funding is 2x: above the cash ceiling, within 5x.
    assert enforcement.check_mandate(roomy, intent, {"positions": []}, {}, broker="bybit",
                                     remote_tool="place_order", daily_count=0) is None


def test_small_orders_pass_with_or_without_margin() -> None:
    assert _check(400.0, margined=True) is None
    assert _check(400.0, margined=False) is None


def test_cash_buy_above_funding_is_still_blocked() -> None:
    from dataclasses import replace

    from src.live import enforcement

    breach = enforcement.check_mandate(
        replace(_mandate(), hard_caps=replace(_mandate().hard_caps, max_order_notional_usd=5_000.0)),
        enforcement.OrderIntent(
            symbol="BTC-USDT", side="buy", notional_usd=2_000.0, quantity=None,
            instrument_type=_mandate().hard_caps.allowed_instruments[0],
            asset_class=_mandate().universe.asset_classes[0],
        ),
        {"positions": []}, {}, broker="bybit", remote_tool="place_order", daily_count=0,
    )
    assert breach is not None and breach.limit == "account_funding_usd"


def test_margined_buy_beyond_mandate_leverage_is_blocked() -> None:
    from dataclasses import replace

    from src.live import enforcement

    mandate = _mandate()
    roomy = replace(mandate, hard_caps=replace(mandate.hard_caps, max_order_notional_usd=10_000.0,
                                               max_total_exposure_usd=10_000.0))
    intent = enforcement.OrderIntent(
        symbol="BTC-USDT", side="buy", notional_usd=6_000.0, quantity=None,
        instrument_type=mandate.hard_caps.allowed_instruments[0],
        asset_class=mandate.universe.asset_classes[0], margined=True,
    )
    breach = enforcement.check_mandate(roomy, intent, {"positions": []}, {}, broker="bybit",
                                       remote_tool="place_order", daily_count=0)
    # 6,000 USD on 1,000 USD funding is 6x, over the mandate's 5x.
    assert breach is not None and breach.limit == "max_leverage"


def test_margined_flag_survives_quantity_pricing_in_the_gate() -> None:
    from src.live.enforcement import OrderIntent
    from src.live.mandate.model import AssetClass, InstrumentType
    from src.live.sdk_order_gate import _normalize_notional

    class Quote:
        @staticmethod
        def get_quote(symbol, config=None):
            return {"status": "ok", "quote": {"last": 100.0}}

    intent = OrderIntent(symbol="BTC-USDT", side="buy", notional_usd=None, quantity=2.0,
                         instrument_type=InstrumentType.CRYPTO, asset_class=AssetClass.CRYPTO, margined=True)
    priced = _normalize_notional(intent, Quote, object())
    assert priced is not None and priced.notional_usd == 200.0 and priced.margined is True
