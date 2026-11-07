"""Bybit spot connector: profiles, host separation, order shaping and gating.

The exchange double subclasses the real ``ccxt.bybit`` and overrides only its
network methods, so endpoint resolution (sandbox switch, regional hostname) is
ccxt's own and the host guard is exercised against real URLs. No test reaches
the network.
"""

from __future__ import annotations

import pytest

ccxt = pytest.importorskip("ccxt")

from src.live import registry
from src.live.classification import ToolClass, classify_tool
from src.trading import profiles, service
from src.trading.connectors.bybit import sdk as bb

pytestmark = pytest.mark.unit


class _RecordingBybit(ccxt.bybit):
    """Real ccxt Bybit client whose network calls are recorded, never sent."""

    instances: list["_RecordingBybit"] = []

    def __init__(self, config=None):
        super().__init__(config or {})
        self.calls: list[tuple] = []
        _RecordingBybit.instances.append(self)

    def fetch_balance(self, params=None):
        self.calls.append(("fetch_balance",))
        return {
            "info": {},
            "free": {},
            "used": {},
            "total": {},
            "USDT": {"free": 90.0, "used": 10.0, "total": 100.0},
            "BTC": {"free": 0.5, "used": 0.0, "total": 0.5},
            "ETH": {"free": 0.0, "used": 0.0, "total": 0.0},
        }

    def fetch_ticker(self, symbol, params=None):
        self.calls.append(("fetch_ticker", symbol))
        return {"bid": 99.0, "ask": 101.0, "last": 100.0, "high": 110.0, "low": 90.0, "baseVolume": 5.0, "timestamp": 1}

    tickers: dict = {"BTC/USDT": {"last": 100.0}}

    def fetch_tickers(self, symbols=None, params=None):
        self.calls.append(("fetch_tickers", tuple(symbols or ())))
        if isinstance(self.tickers, Exception):
            raise self.tickers
        return {s: t for s, t in self.tickers.items() if not symbols or s in symbols}

    def create_order(self, symbol, type, side, amount, price=None, params=None):
        self.calls.append(("create_order", symbol, type, side, amount, price, dict(params or {})))
        return {"id": "o-1", "symbol": symbol, "side": side, "type": type, "status": "open", "amount": amount, "price": price}

    def cancel_order(self, id, symbol=None, params=None):
        self.calls.append(("cancel_order", id, symbol))
        return {"id": id, "symbol": symbol, "side": "buy", "status": "canceled"}


@pytest.fixture
def fake_ccxt(monkeypatch):
    """Route the connector to the recording client; return the instance list."""
    _RecordingBybit.instances = []
    monkeypatch.setattr(_RecordingBybit, "tickers", {"BTC/USDT": {"last": 100.0}})
    fake_module = type("FakeCcxt", (), {
        "bybit": _RecordingBybit,
        "ArgumentsRequired": ccxt.ArgumentsRequired,
        "BadSymbol": ccxt.BadSymbol,
        "NotSupported": ccxt.NotSupported,
    })
    monkeypatch.setattr(bb, "_require_ccxt", lambda: fake_module)
    return _RecordingBybit.instances


def _cfg(profile: str = "paper", **extra) -> bb.BybitConfig:
    return bb.BybitConfig.from_mapping({"api_key": "key1234", "api_secret": "secret", "profile": profile, **extra})


# --------------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------------- #


def test_profiles_registered_with_expected_permissions() -> None:
    by_id = {p.id: p for p in profiles.list_profiles() if p.connector == "bybit" and not p.config.get("futures")}
    assert set(by_id) == {"bybit-paper-sdk", "bybit-live-sdk-readonly", "bybit-paper-trade", "bybit-live-trade"}
    for pid in ("bybit-paper-sdk", "bybit-live-sdk-readonly"):
        assert by_id[pid].readonly is True
        assert not any(".place" in cap for cap in by_id[pid].capabilities)
    assert "orders.place" in by_id["bybit-paper-trade"].capabilities
    assert by_id["bybit-paper-trade"].environment == "paper"
    live_trade = by_id["bybit-live-trade"]
    assert live_trade.environment == "live" and live_trade.readonly is False
    assert "orders.place.requires_mandate" in live_trade.capabilities
    assert "orders.place" not in live_trade.capabilities


def test_order_ops_write_pinned_through_registry() -> None:
    curated = registry._BROKER_CURATED_MAPS["bybit"]
    assert classify_tool("create_order", None, curated) is ToolClass.WRITE
    assert classify_tool("cancel_order", None, curated) is ToolClass.WRITE
    assert classify_tool("fetch_balance", None, curated) is ToolClass.READ
    assert classify_tool("fetch_tickers", None, curated) is ToolClass.READ


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "payload, message",
    [
        ({"profile": "demo"}, "profile must be"),
        ({"profile": "live", "hostname": "evil.example"}, "hostname must be one of"),
        ({"profile": "paper", "hostname": "bybit.eu"}, "testnet is served only from bybit.com"),
        ({"profile": "live", "timeout": 0}, "timeout must be a positive number"),
    ],
)
def test_invalid_config_is_rejected(payload, message) -> None:
    with pytest.raises(bb.BybitConfigError, match=message):
        bb.BybitConfig.from_mapping(payload)


def test_overrides_cannot_change_readonly_or_timeout() -> None:
    cfg = bb.build_config({"profile": "live-readonly"}, {"readonly": False, "timeout": 999, "hostname": "bybit.eu"})
    assert cfg.readonly is True
    assert cfg.timeout == 15.0
    assert cfg.hostname == "bybit.eu"


def test_saved_config_is_owner_only_and_round_trips(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bb, "get_runtime_root", lambda: tmp_path)
    path = bb.save_config(_cfg("live", hostname="bybit.nl"))
    assert path.stat().st_mode & 0o077 == 0
    assert bb.load_config() == _cfg("live", hostname="bybit.nl")


def test_invalid_saved_config_names_the_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bb, "get_runtime_root", lambda: tmp_path)
    (tmp_path / "bybit.json").write_text('{"profile": "live", "hostname": "evil.example"}', encoding="utf-8")
    with pytest.raises(bb.BybitConfigError, match="bybit.json"):
        bb.load_config()


# --------------------------------------------------------------------------- #
# Host separation against ccxt's real endpoint resolution
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "profile, hostname, expected",
    [
        ("paper", "bybit.com", "api-testnet.bybit.com"),
        ("live-readonly", "bybit.com", "api.bybit.com"),
        ("live", "bybit.eu", "api.bybit.eu"),
    ],
)
def test_client_resolves_every_endpoint_to_profile_host(fake_ccxt, profile, hostname, expected) -> None:
    ex = bb._exchange(_cfg(profile, hostname=hostname))
    resolved = {ex.implode_hostname(url) for url in ex.urls["api"].values()}
    assert resolved == {f"https://{expected}"}
    assert ex.options["defaultType"] == "spot"
    assert ex.options["adjustForTimeDifference"] is True


def test_paper_profile_fails_closed_when_sandbox_does_not_apply(monkeypatch, fake_ccxt) -> None:
    monkeypatch.setattr(_RecordingBybit, "set_sandbox_mode", lambda self, enabled: None)
    result = bb.place_order(_cfg("paper"), symbol="BTC-USDT", side="buy", quantity=0.01)
    assert result["status"] == "error"
    assert "api-testnet.bybit.com" in result["error"]
    assert all(call[0] != "create_order" for inst in fake_ccxt for call in inst.calls)


def test_endpoint_guard_rejects_a_client_without_endpoints() -> None:
    with pytest.raises(bb.BybitConfigError, match="no API endpoints"):
        bb._assert_endpoints(_cfg("paper"), type("Bare", (), {"urls": {}})())


# --------------------------------------------------------------------------- #
# Reads
# --------------------------------------------------------------------------- #


def test_positions_are_nonzero_spot_balances(fake_ccxt) -> None:
    result = bb.get_positions(_cfg("live-readonly"))
    assert result["status"] == "ok" and result["is_testnet"] is False
    assert {row["symbol"]: row["quantity"] for row in result["positions"]} == {"USDT": 100.0, "BTC": 0.5}
    assert all(row["source"] == "spot" for row in result["positions"])
    # Stablecoin cash is not exposure; BTC is valued at its USDT last price.
    assert {row["symbol"]: row["value_usd"] for row in result["positions"]} == {"USDT": 0.0, "BTC": 50.0}
    assert "note" not in result


def test_unpriceable_holding_is_left_unvalued_and_noted(monkeypatch, fake_ccxt) -> None:
    monkeypatch.setattr(_RecordingBybit, "tickers", {})
    result = bb.get_positions(_cfg("live-readonly"))
    btc = next(row for row in result["positions"] if row["symbol"] == "BTC")
    assert "value_usd" not in btc
    assert result["note"] == "No USDT price for: BTC."


def test_ticker_failure_keeps_balances_and_notes_the_gap(monkeypatch, fake_ccxt) -> None:
    monkeypatch.setattr(_RecordingBybit, "tickers", RuntimeError("down"))
    result = bb.get_positions(_cfg("live-readonly"))
    assert result["status"] == "ok" and len(result["positions"]) == 2
    assert "USDT prices unavailable" in result["note"]


def test_quote_uses_public_client_without_credentials(fake_ccxt) -> None:
    result = bb.get_quote("btcusdt", config=_cfg("live-readonly"))
    assert result["symbol"] == "BTC/USDT"
    assert result["quote"]["last"] == 100.0
    assert fake_ccxt[-1].apiKey == "" and fake_ccxt[-1].secret == ""


def test_check_status_reports_missing_credentials_without_network() -> None:
    report = bb.check_status(bb.BybitConfig(profile="paper"))
    assert report["status"] == "error"
    assert "api_key" in report["error"] and "api_secret" in report["error"]


def test_public_config_redacts_secrets() -> None:
    public = bb._public_config(_cfg("live"))
    assert public["api_secret"] == "***redacted***"
    assert public["api_key"] == "key1***"
    assert public["host"] == "api.bybit.com"


@pytest.mark.parametrize(
    "raw, expected",
    [("BTC-USDT", "BTC/USDT"), ("btcusdt", "BTC/USDT"), ("ETH/BTC", "ETH/BTC"), ("SOLUSDC", "SOL/USDC")],
)
def test_normalize_symbol(raw, expected) -> None:
    assert bb.normalize_symbol(raw) == expected


# --------------------------------------------------------------------------- #
# Order shaping
# --------------------------------------------------------------------------- #


def _create_call(instances):
    calls = [call for inst in instances for call in inst.calls if call[0] == "create_order"]
    assert len(calls) == 1
    return calls[0]


def test_quantity_market_order_is_base_sized(fake_ccxt) -> None:
    result = bb.place_order(_cfg("paper"), symbol="BTC-USDT", side="buy", quantity=0.01)
    assert result["status"] == "ok" and result["is_testnet"] is True
    _, symbol, otype, side, amount, price, params = _create_call(fake_ccxt)
    assert (symbol, otype, side, amount, price, params) == ("BTC/USDT", "market", "buy", 0.01, None, {})


@pytest.mark.parametrize("side", ["buy", "sell"])
def test_notional_market_order_is_sent_as_quote_cost(fake_ccxt, side) -> None:
    bb.place_order(_cfg("paper"), symbol="BTC-USDT", side=side, notional=50)
    _, _, otype, sent_side, amount, price, params = _create_call(fake_ccxt)
    assert (otype, sent_side, amount, price) == ("market", side, 50.0, None)
    assert params == {"cost": 50.0}


def test_limit_order_maps_day_to_gtc(fake_ccxt) -> None:
    bb.place_order(_cfg("paper"), symbol="ETHUSDT", side="sell", quantity=1, order_type="limit", limit_price=3000)
    _, symbol, otype, _, amount, price, params = _create_call(fake_ccxt)
    assert (symbol, otype, amount, price, params) == ("ETH/USDT", "limit", 1.0, 3000.0, {"timeInForce": "GTC"})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"side": "hold", "quantity": 1}, "side must be"),
        ({"side": "buy", "quantity": 1, "order_type": "stop"}, "order_type must be"),
        ({"side": "buy"}, "exactly one of"),
        ({"side": "buy", "quantity": 1, "notional": 5}, "exactly one of"),
        ({"side": "buy", "quantity": -1}, "quantity must be a positive"),
        ({"side": "buy", "quantity": float("nan")}, "quantity must be a positive"),
        ({"side": "buy", "notional": float("inf")}, "notional must be a positive"),
        ({"side": "buy", "notional": 10, "order_type": "limit", "limit_price": 1}, "require 'quantity'"),
        ({"side": "buy", "quantity": 1, "order_type": "limit"}, "positive 'limit_price'"),
        ({"side": "buy", "quantity": 1, "order_type": "limit", "limit_price": 1, "time_in_force": "gtd"}, "time_in_force"),
    ],
)
def test_invalid_order_never_reaches_the_exchange(fake_ccxt, kwargs, message) -> None:
    result = bb.place_order(_cfg("paper"), symbol="BTC-USDT", **kwargs)
    assert result["status"] == "error" and message in result["error"]
    assert fake_ccxt == []


def test_unresolvable_symbol_is_rejected(fake_ccxt) -> None:
    result = bb.place_order(_cfg("paper"), symbol="BITCOIN", side="buy", quantity=1)
    assert result["status"] == "error" and "trading pair" in result["error"]
    assert fake_ccxt == []


def test_cancel_requires_symbol_and_forwards_it(fake_ccxt) -> None:
    assert "requires symbol" in bb.cancel_order(_cfg("paper"), "o-1")["error"]
    result = bb.cancel_order(_cfg("paper"), "o-1", symbol="BTC-USDT")
    assert result["status"] == "ok" and result["order_status"] == "canceled"
    assert ("cancel_order", "o-1", "BTC/USDT") in fake_ccxt[-1].calls


# --------------------------------------------------------------------------- #
# Service routing: paper places directly, live needs the mandate
# --------------------------------------------------------------------------- #


def _isolate_saved_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bb, "get_runtime_root", lambda: tmp_path)
    bb.save_config(_cfg("paper"))


def test_service_paper_trade_places_on_testnet(monkeypatch, tmp_path, fake_ccxt) -> None:
    _isolate_saved_config(monkeypatch, tmp_path)
    out = service.place_order("BTC-USDT", "bybit-paper-trade", side="buy", quantity=0.01)
    assert out["status"] == "ok", out
    assert out["environment"] == "paper"
    assert _create_call(fake_ccxt)[1] == "BTC/USDT"


def test_service_live_trade_without_mandate_is_blocked(monkeypatch, tmp_path, fake_ccxt) -> None:
    _isolate_saved_config(monkeypatch, tmp_path)
    monkeypatch.setattr("src.live.sdk_order_gate.load_mandate", lambda broker: None)
    monkeypatch.setattr("src.live.sdk_order_gate.write_live_action", lambda *a, **k: {"audited": True})
    out = service.place_order("BTC-USDT", "bybit-live-trade", side="buy", notional=50.0)
    assert out["status"] == "blocked"
    assert out["environment"] == "live"
    assert all(call[0] != "create_order" for inst in fake_ccxt for call in inst.calls)


def test_service_readonly_profile_refuses_orders(monkeypatch, tmp_path, fake_ccxt) -> None:
    _isolate_saved_config(monkeypatch, tmp_path)
    out = service.place_order("BTC-USDT", "bybit-live-sdk-readonly", side="buy", quantity=1)
    assert out["status"] != "ok"
    assert fake_ccxt == []


def test_service_unconfigured_reports_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(bb, "get_runtime_root", lambda: tmp_path)
    result = service.check_connection("bybit-paper-sdk")
    assert result["status"] == "error"
    assert result["connector"] == "bybit"


# --------------------------------------------------------------------------- #
# Live gate with a committed crypto mandate, over the connector's real reads
# --------------------------------------------------------------------------- #


def _crypto_mandate(*, max_order: float):
    from src.live.mandate.model import AssetClass, ConsentMeta, HardCaps, InstrumentType, Mandate, UniverseConstraint

    return Mandate(
        schema_version=1,
        hard_caps=HardCaps(
            account_funding_usd=1_000.0,
            max_order_notional_usd=max_order,
            max_total_exposure_usd=1_000.0,
            max_leverage=1.0,
            allowed_instruments=(InstrumentType.CRYPTO,),
            max_trades_per_day=5,
        ),
        universe=UniverseConstraint(
            asset_classes=(AssetClass.CRYPTO,),
            min_market_cap_usd=None,
            min_avg_daily_volume_usd=None,
            exclude_symbols=(),
        ),
        consent=ConsentMeta(
            created_at="2026-01-01T00:00:00+00:00",
            consent_token_sha256="deadbeef",
            broker="bybit",
            account_ref="",
            expires_at="2999-01-01T00:00:00+00:00",
        ),
    )


def _patch_live_gate(monkeypatch, tmp_path, mandate) -> None:
    from contextlib import nullcontext

    import src.live.paths as live_paths
    from src.live import sdk_order_gate as gate

    monkeypatch.setattr(live_paths, "get_runtime_root", lambda: tmp_path)
    monkeypatch.setattr(gate, "load_mandate", lambda broker: mandate)
    monkeypatch.setattr(gate, "halt_flag_set", lambda broker: False)
    monkeypatch.setattr(gate, "write_live_action", lambda *a, **k: {"audited": True})
    monkeypatch.setattr(gate, "read_daily_count", lambda broker: 0)
    monkeypatch.setattr(gate, "increment_daily_count", lambda broker, action_id=None: 1)
    monkeypatch.setattr(gate, "daily_order_lock", lambda broker: nullcontext())


def test_live_order_within_mandate_reaches_mainnet(monkeypatch, tmp_path, fake_ccxt) -> None:
    _isolate_saved_config(monkeypatch, tmp_path)
    _patch_live_gate(monkeypatch, tmp_path, _crypto_mandate(max_order=100.0))
    out = service.place_order("BTC-USDT", "bybit-live-trade", side="buy", notional=50.0)
    assert out["status"] == "ok", out
    assert out["environment"] == "live" and out["is_testnet"] is False
    created = [inst for inst in fake_ccxt if any(call[0] == "create_order" for call in inst.calls)]
    assert len(created) == 1
    assert {created[0].implode_hostname(u) for u in created[0].urls["api"].values()} == {"https://api.bybit.com"}


def test_live_order_with_unpriceable_holding_is_not_sent(monkeypatch, tmp_path, fake_ccxt) -> None:
    _isolate_saved_config(monkeypatch, tmp_path)
    _patch_live_gate(monkeypatch, tmp_path, _crypto_mandate(max_order=100.0))
    monkeypatch.setattr(_RecordingBybit, "tickers", {})
    out = service.place_order("BTC-USDT", "bybit-live-trade", side="buy", notional=50.0)
    assert out["status"] != "ok"
    assert all(call[0] != "create_order" for inst in fake_ccxt for call in inst.calls)


def test_live_order_over_mandate_cap_is_not_sent(monkeypatch, tmp_path, fake_ccxt) -> None:
    _isolate_saved_config(monkeypatch, tmp_path)
    _patch_live_gate(monkeypatch, tmp_path, _crypto_mandate(max_order=20.0))
    out = service.place_order("BTC-USDT", "bybit-live-trade", side="buy", notional=50.0)
    assert out["status"] != "ok"
    assert all(call[0] != "create_order" for inst in fake_ccxt for call in inst.calls)


# --------------------------------------------------------------------------- #
# Portfolio: Bybit balances are crypto spot holdings with a derived total
# --------------------------------------------------------------------------- #


def test_portfolio_classifies_bybit_balances_as_crypto_quoted_in_usdt() -> None:
    from src.portfolio.normalization import normalize_position

    btc = normalize_position("bybit", {"symbol": "BTC", "quantity": 0.5, "source": "spot"})
    usdt = normalize_position("bybit", {"symbol": "USDT", "quantity": 100.0, "source": "spot"})
    assert (btc["asset_type"], btc["quote_symbol"]) == ("crypto", "BTC/USDT")
    assert usdt["asset_type"] == "stablecoin"


def test_portfolio_snapshot_with_unpriced_bybit_holding_is_incomplete() -> None:
    from src.portfolio.store import _derived_totals_are_complete

    snapshot = {
        "accounts": [{"broker": "bybit", "unpriced_position_count": 1}],
        "positions": [{"broker": "bybit", "symbol": "XYZ", "priced": False}],
    }
    assert _derived_totals_are_complete(snapshot) is False
    legacy = {"accounts": [{"broker": "bybit"}], "positions": [{"broker": "bybit", "priced": False}]}
    assert _derived_totals_are_complete(legacy) is False
    other = {"accounts": [{"broker": "bybit"}], "positions": [{"broker": "alpaca", "priced": False}]}
    assert _derived_totals_are_complete(other) is True


def test_check_status_explains_a_country_block(monkeypatch, fake_ccxt) -> None:
    def blocked(self, params=None):
        raise ccxt.RateLimitExceeded(
            'bybit GET https://api.bybit.com/v5/market/time 403 Forbidden {"error":"The Amazon CloudFront '
            'distribution is configured to block access from your country."}'
        )

    monkeypatch.setattr(_RecordingBybit, "fetch_balance", blocked)
    report = bb.check_status(_cfg("live-readonly"))
    assert report["status"] == "error"
    assert report["error_code"] == "bybit_restricted_location"
    assert "api.bybit.com" in report["error"] and "RateLimit" not in report["error"]
