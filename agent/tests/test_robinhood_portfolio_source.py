"""Read-only Robinhood portfolio source (#1428), end to end through the real read path.

``PortfolioService`` uses its default readers here, so a refresh goes through
``trading.service`` → the Robinhood mapping → the portfolio contract. Only the
MCP adapter is stubbed, and it answers with the shapes posted on #1428.
"""

from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import connection_routes
from src.portfolio.config import PortfolioSettingsStore
from src.portfolio.service import PortfolioService
from src.portfolio.store import PortfolioStore
from src.trading.connections import ConnectionStore, is_portfolio_connection_profile
from src.trading.profiles import profile_by_id
from tests import robinhood_mcp_helpers as rh

pytestmark = pytest.mark.unit

ACCOUNT = "5QR12345"


class _Broker:
    """Records every MCP call and answers with the configured replies."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.replies: dict[str, dict] = {
            "get_accounts": rh.accounts([rh.account(ACCOUNT, nickname="Main"), rh.account("5QR00009")]),
            "get_portfolio": rh.portfolio(total_value="1500.00", cash="500.00"),
            "get_equity_positions": rh.positions([]),
        }
        self.quotes_by_symbol: dict[str, str | dict] = {}
        self.page_replies: dict[str, dict] = {}

    def reply_for(self, remote_name: str, arguments: dict) -> dict:
        cursor = arguments.get("cursor")
        if cursor:
            return self.page_replies[cursor]
        if remote_name == "get_equity_quotes":
            symbol = arguments["symbols"][0]
            quoted = self.quotes_by_symbol.get(symbol)
            if isinstance(quoted, dict):
                return quoted  # a fully-built envelope the test arranged
            return rh.quotes([rh.quote_result(rh.quote(symbol, last_trade_price=quoted or "0"))])
        return self.replies[remote_name]

    def adapter_class(self):
        broker = self

        class _Adapter:
            def __init__(self, server_name, server_config, **kwargs):  # noqa: ANN001
                assert server_name == "robinhood"
                assert kwargs.get("interactive_oauth") is False

            def call_tool(self, remote_name, arguments):  # noqa: ANN001
                broker.calls.append((remote_name, dict(arguments)))
                return broker.reply_for(remote_name, dict(arguments))

        return _Adapter


@pytest.fixture
def broker(monkeypatch: pytest.MonkeyPatch) -> _Broker:
    fake = _Broker()
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=["get_accounts", "get_portfolio", "get_equity_positions", "get_equity_quotes"],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-token"),
    )
    monkeypatch.setattr(
        "src.config.loader.load_agent_config", lambda: SimpleNamespace(mcp_servers={"robinhood": server})
    )
    monkeypatch.setattr("src.live.registry.has_cached_oauth_token", lambda *_: True)
    monkeypatch.setattr("src.tools.mcp.MCPServerAdapter", fake.adapter_class())
    return fake


def _service(tmp_path, *, account_ref: str = ACCOUNT) -> PortfolioService:
    settings = PortfolioSettingsStore(tmp_path / "portfolio.json")
    settings.connection_store.ensure("robinhood-live", "robinhood-live-mcp-readonly", "Robinhood")
    if account_ref:
        settings.connection_store.select_account("robinhood-live", account_ref)
    settings.save({"display_currency": "USD", "sources": [{"connection_id": "robinhood-live", "label": "Robinhood"}]})
    return PortfolioService(
        PortfolioStore(tmp_path / "portfolio.sqlite3"),
        settings_store=settings,
        fx_fetcher=lambda: (Decimal("7.2"), Decimal("7.8"), "2026-09-17T00:00:00+00:00"),
    )


def test_only_the_readonly_profile_can_back_a_portfolio_connection() -> None:
    assert is_portfolio_connection_profile(profile_by_id("robinhood-live-mcp-readonly")) is True
    assert is_portfolio_connection_profile(profile_by_id("robinhood-live-mcp")) is False


def test_a_source_without_a_selected_account_errors_before_any_call(tmp_path, broker: _Broker) -> None:
    snapshot = _service(tmp_path, account_ref="").refresh()

    account = snapshot["accounts"][0]
    assert account["status"] == "error"
    assert "Select a robinhood account" in account["error"]
    assert snapshot["complete"] is False
    assert broker.calls == []


def test_an_empty_account_is_a_complete_snapshot_read_for_the_selected_account(tmp_path, broker: _Broker) -> None:
    snapshot = _service(tmp_path).refresh()

    account = snapshot["accounts"][0]
    assert snapshot["complete"] is True
    assert account["status"] == "ok"
    assert account["total_usd"] == 1500.0
    assert account["cash_usd"] == 500.0
    assert broker.calls == [
        ("get_portfolio", {"account_number": ACCOUNT}),
        ("get_equity_positions", {"account_number": ACCOUNT}),
    ]


def test_positions_are_priced_through_get_equity_quotes(tmp_path, broker: _Broker) -> None:
    broker.replies["get_portfolio"] = rh.portfolio(total_value="2500.00", cash="500.00")
    broker.replies["get_equity_positions"] = rh.positions(
        [
            rh.position("AAPL", "4", average_buy_price="150.00"),
            rh.position("MSFT", "2", average_buy_price=None),
        ]
    )
    broker.quotes_by_symbol = {"AAPL": "150.25", "MSFT": "400.10"}

    snapshot = _service(tmp_path).refresh()

    rows = {row["symbol"]: row for row in snapshot["positions"]}
    assert rows["AAPL"]["quantity"] == 4.0
    assert rows["AAPL"]["cost_price"] == 150.0
    assert rows["AAPL"]["market_price"] == 150.25
    assert rows["AAPL"]["priced"] is True
    assert rows["AAPL"]["market_value_usd"] == 601.0
    assert not rows["AAPL"].get("price_error")
    assert rows["MSFT"]["cost_price"] is None  # omitted while reconciling, never zero
    assert rows["MSFT"]["market_price"] == 400.10
    assert snapshot["accounts"][0]["status"] == "ok"
    assert snapshot["accounts"][0]["total_usd"] == 2500.0
    # 2500 total - 500 cash - 4*150.25 - 2*400.10 priced
    assert snapshot["accounts"][0]["unpriced_or_other_usd"] == 598.8
    assert not any("No price available" in warning for warning in snapshot["warnings"])
    assert snapshot["complete"] is True
    assert ("get_equity_quotes", {"symbols": ["AAPL"]}) in broker.calls
    assert ("get_equity_quotes", {"symbols": ["MSFT"]}) in broker.calls


def test_a_quote_read_that_fails_to_map_leaves_the_row_unpriced_with_the_reason(
    tmp_path, broker: _Broker
) -> None:
    broker.replies["get_equity_positions"] = rh.positions(
        [
            rh.position("AAPL", "4", average_buy_price="150.00"),
            rh.position("MSFT", "2", average_buy_price=None),
        ]
    )
    missing_state = rh.quote("MSFT")
    del missing_state["state"]
    broker.quotes_by_symbol = {"AAPL": "150.25", "MSFT": rh.quotes([rh.quote_result(missing_state)])}

    snapshot = _service(tmp_path).refresh()

    rows = {row["symbol"]: row for row in snapshot["positions"]}
    assert rows["AAPL"]["priced"] is True
    assert rows["MSFT"]["priced"] is False
    assert rows["MSFT"]["market_price"] is None
    assert "quote is missing state" in rows["MSFT"]["price_error"]
    assert any("robinhood:MSFT" in warning for warning in snapshot["warnings"])


@pytest.mark.parametrize("field", ["options_value", "crypto_value", "futures_value", "fixed_income_value"])
def test_holdings_outside_equities_fail_the_source_instead_of_shortening_it(
    tmp_path, broker: _Broker, field: str
) -> None:
    broker.replies["get_portfolio"] = rh.portfolio(total_value="3000.00", **{field: "125.00"})
    broker.replies["get_equity_positions"] = rh.positions([rh.position("AAPL", "1")])

    snapshot = _service(tmp_path).refresh()

    account = snapshot["accounts"][0]
    assert account["status"] == "error"
    assert field in account["error"]
    assert account["total_usd"] is None
    assert snapshot["positions"] == []
    assert snapshot["complete"] is False


def test_an_unreported_non_equity_value_is_not_read_as_zero(tmp_path, broker: _Broker) -> None:
    envelope = rh.portfolio()
    del envelope["structured_content"]["data"]["event_contracts_value"]
    broker.replies["get_portfolio"] = envelope

    account = _service(tmp_path).refresh()["accounts"][0]

    assert account["status"] == "error"
    assert "did not report event_contracts_value" in account["error"]


@pytest.mark.parametrize(
    ("reply", "message"),
    [
        (rh.positions(None), "no positions list"),
        (rh.positions([None]), "is not an object"),
    ],
)
def test_a_partial_or_null_position_read_is_an_error_not_an_empty_account(
    tmp_path, broker: _Broker, reply: dict, message: str
) -> None:
    broker.replies["get_equity_positions"] = reply

    account = _service(tmp_path).refresh()["accounts"][0]

    assert account["status"] == "error"
    assert message in account["error"]


def test_positions_across_pages_become_one_whole_account(tmp_path, broker: _Broker) -> None:
    broker.replies["get_equity_positions"] = rh.positions(
        [rh.position("AAPL", "1")], next_page="cursor-2"
    )
    broker.page_replies["cursor-2"] = rh.positions([rh.position("MSFT", "2")], next_page="")
    broker.quotes_by_symbol = {"AAPL": "150.25", "MSFT": "400.10"}

    snapshot = _service(tmp_path).refresh()

    account = snapshot["accounts"][0]
    assert account["status"] == "ok"
    assert account["total_usd"] == 1500.0
    assert sorted(row["symbol"] for row in snapshot["positions"]) == ["AAPL", "MSFT"]
    assert all(row["priced"] for row in snapshot["positions"])
    assert snapshot["complete"] is True
    assert broker.calls == [
        ("get_portfolio", {"account_number": ACCOUNT}),
        ("get_equity_positions", {"account_number": ACCOUNT}),
        ("get_equity_positions", {"account_number": ACCOUNT, "cursor": "cursor-2"}),
        ("get_equity_quotes", {"symbols": ["AAPL"]}),
        ("get_equity_quotes", {"symbols": ["MSFT"]}),
    ]


def test_a_failed_second_page_errors_the_source_instead_of_showing_partial_holdings(
    tmp_path, broker: _Broker
) -> None:
    broker.replies["get_equity_positions"] = rh.positions(
        [rh.position("AAPL", "1")], next_page="cursor-2"
    )  # no page_replies entry: the follow-up call fails

    snapshot = _service(tmp_path).refresh()

    account = snapshot["accounts"][0]
    assert account["status"] == "error"
    assert "next-page call failed" in account["error"]
    assert snapshot["positions"] == []
    assert account["total_usd"] is None
    assert snapshot["complete"] is False


def test_a_missing_total_value_is_an_error_not_a_zero_account(tmp_path, broker: _Broker) -> None:
    envelope = rh.portfolio()
    del envelope["structured_content"]["data"]["total_value"]
    broker.replies["get_portfolio"] = envelope

    account = _service(tmp_path).refresh()["accounts"][0]

    assert account["status"] == "error"
    assert "total_value is missing" in account["error"]


def test_the_connection_store_refuses_an_account_on_a_profile_that_takes_none(tmp_path) -> None:
    store = ConnectionStore(tmp_path / "connections.json")
    store.ensure("ibkr", "ibkr-live-local-readonly", "IBKR")

    with pytest.raises(ValueError, match="does not take an account"):
        store.select_account("ibkr", "U1234")


def test_the_connection_store_refuses_a_control_character_in_an_account(tmp_path) -> None:
    store = ConnectionStore(tmp_path / "connections.json")
    store.ensure("robinhood-live", "robinhood-live-mcp-readonly", "Robinhood")

    with pytest.raises(ValueError, match="printable"):
        store.select_account("robinhood-live", "5QR\n1")


def _client(tmp_path, monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, ConnectionStore]:
    store = ConnectionStore(tmp_path / "connections.json")
    store.ensure("robinhood-live", "robinhood-live-mcp-readonly", "Robinhood")
    monkeypatch.setattr(connection_routes, "ConnectionStore", lambda: store)
    monkeypatch.setattr(connection_routes, "require_auth", lambda: None)
    monkeypatch.setattr(connection_routes, "require_settings_write_auth", lambda: None)
    app = FastAPI()
    connection_routes.register_connection_routes(app)
    return TestClient(app), store


def test_accounts_route_lists_the_login_accounts(tmp_path, monkeypatch, broker: _Broker) -> None:
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get("/api/connections/robinhood-live/accounts")

    assert response.status_code == 200
    body = response.json()
    assert [row["account_ref"] for row in body["accounts"]] == [ACCOUNT, "5QR00009"]
    assert body["account_ref"] == ""
    assert broker.calls == [("get_accounts", {})]


@pytest.mark.parametrize(
    ("accounts", "status", "detail"),
    [
        ([rh.account("5QR00009")], 400, "not one of the accounts"),
        ([rh.account(ACCOUNT, deactivated=True)], 400, "deactivated"),
    ],
)
def test_selecting_an_account_the_login_cannot_use_is_refused(
    tmp_path, monkeypatch, broker: _Broker, accounts: list, status: int, detail: str
) -> None:
    broker.replies["get_accounts"] = rh.accounts(accounts)
    client, store = _client(tmp_path, monkeypatch)

    response = client.put("/api/connections/robinhood-live/account", json={"account_ref": ACCOUNT})

    assert response.status_code == status
    assert detail in response.json()["detail"]
    assert store.get("robinhood-live").account_ref == ""


def test_selecting_clearing_and_renaming_keep_the_account_where_it_belongs(
    tmp_path, monkeypatch, broker: _Broker
) -> None:
    client, store = _client(tmp_path, monkeypatch)

    assert client.put("/api/connections/robinhood-live/account", json={"account_ref": ACCOUNT}).status_code == 200
    assert store.get("robinhood-live").account_ref == ACCOUNT

    renamed = client.put(
        "/api/connections/robinhood-live",
        json={"id": "robinhood-live", "profile_id": "robinhood-live-mcp-readonly", "label": "RH main"},
    )
    assert renamed.status_code == 200
    assert store.get("robinhood-live").label == "RH main"
    assert store.get("robinhood-live").account_ref == ACCOUNT

    calls_before_clear = len(broker.calls)
    assert client.put("/api/connections/robinhood-live/account", json={"account_ref": ""}).status_code == 200
    assert store.get("robinhood-live").account_ref == ""
    assert len(broker.calls) == calls_before_clear  # clearing needs no broker read


def test_accounts_route_reports_an_unavailable_broker_as_503(tmp_path, monkeypatch, broker: _Broker) -> None:
    broker.replies["get_accounts"] = {"status": "error", "error": "token expired"}
    client, _ = _client(tmp_path, monkeypatch)

    response = client.get("/api/connections/robinhood-live/accounts")

    assert response.status_code == 503
    assert "token expired" in response.json()["detail"]


@pytest.mark.parametrize("case", ["wrong_symbol", "wrong_close_symbol", "never_traded", "inactive"])
def test_quote_identity_and_listing_status_cannot_price_another_position(tmp_path, broker, case):
    broker.replies["get_equity_positions"] = rh.positions([rh.position("AAPL", "4")])
    quote = rh.quote("MSFT" if case == "wrong_symbol" else "AAPL", last_trade_price="0", bid_price="0", ask_price="0")
    close = rh.settled_close("MSFT" if case == "wrong_close_symbol" else quote["symbol"], price="999")
    if case == "never_traded":
        quote["has_traded"] = False
    if case == "inactive":
        quote["state"] = "delisted"
    broker.quotes_by_symbol["AAPL"] = rh.quotes([rh.quote_result(quote, close_row=close)])
    row = _service(tmp_path).refresh()["positions"][0]
    assert row["priced"] is False
    assert row["market_price"] is None
    assert row.get("price_error")


def test_quote_uses_the_newer_extended_hours_trade_and_retains_its_time(tmp_path, broker):
    broker.replies["get_equity_positions"] = rh.positions([rh.position("AAPL", "4")])
    quote = rh.quote("AAPL", last_trade_price="100")
    quote.update(last_non_reg_trade_price="120", venue_last_non_reg_trade_time="2026-10-12T21:00:00Z")
    broker.quotes_by_symbol["AAPL"] = rh.quotes([rh.quote_result(quote)])
    row = _service(tmp_path).refresh()["positions"][0]
    assert row["market_price"] == 120
    assert "2026-10-12T21:00:00Z" in row["pricing_basis"]
