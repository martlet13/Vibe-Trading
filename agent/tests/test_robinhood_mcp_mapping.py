"""Robinhood reply mapping (#1428 / #1442): the one place that reads the nested shape.

Fixtures come from ``tests/robinhood_mcp_helpers.py``, which reproduces the
shapes posted on #1428. Each rejection case is a way a reply could be read as
something it is not: a first page as the whole account, a null list as an empty
account, an omitted cost as zero.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.trading import service
from src.trading.connectors.robinhood import mcp
from tests import robinhood_mcp_helpers as rh

pytestmark = pytest.mark.unit

ACCOUNT = "5QR12345"


def test_portfolio_summary_reads_the_nested_data_object() -> None:
    summary = mcp.portfolio_summary(rh.portfolio(total_value="1234.50", cash="200.00", buying_power="180.25"))

    assert summary == {
        "portfolio_value": "1234.50",
        "cash": "200.00",
        "buying_power": "180.25",
        "currency": "USD",
        "non_equity_values": {field: "0.00" for field in mcp.NON_EQUITY_VALUE_FIELDS},
    }


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"total_value": None}, "total_value is missing"),
        ({"total_value": "n/a"}, "total_value is not a number"),
        ({"total_value": "NaN"}, "total_value is not finite"),
        ({"currency": ""}, "currency is missing"),
    ],
)
def test_portfolio_summary_refuses_a_missing_or_unreadable_total(override: dict, message: str) -> None:
    envelope = rh.portfolio()
    envelope["structured_content"]["data"].update(override)

    with pytest.raises(mcp.RobinhoodShapeError, match=message):
        mcp.portfolio_summary(envelope)


def test_an_omitted_non_equity_value_is_unknown_not_zero() -> None:
    envelope = rh.portfolio()
    del envelope["structured_content"]["data"]["crypto_value"]

    assert mcp.portfolio_summary(envelope)["non_equity_values"]["crypto_value"] is None


@pytest.mark.parametrize("buying_power", [[], ["5000.00"], "5000.00", 5000, False])
def test_portfolio_summary_rejects_an_unmapped_buying_power_shape(buying_power) -> None:
    envelope = rh.portfolio()
    envelope["structured_content"]["data"]["buying_power"] = buying_power

    with pytest.raises(mcp.RobinhoodShapeError, match="buying_power is not an object"):
        mcp.portfolio_summary(envelope)


@pytest.mark.parametrize("buying_power", [None, {}, {"buying_power": None}])
def test_portfolio_summary_preserves_unknown_buying_power(buying_power) -> None:
    envelope = rh.portfolio(total_value="1234.50")
    envelope["structured_content"]["data"]["buying_power"] = buying_power

    summary = mcp.portfolio_summary(envelope)

    assert summary["buying_power"] is None
    assert summary["portfolio_value"] == "1234.50"


def test_portfolio_summary_allows_omitted_buying_power() -> None:
    envelope = rh.portfolio()
    del envelope["structured_content"]["data"]["buying_power"]

    assert mcp.portfolio_summary(envelope)["buying_power"] is None


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "invalid", [], {}])
def test_portfolio_summary_rejects_invalid_nested_buying_power(value) -> None:
    envelope = rh.portfolio()
    envelope["structured_content"]["data"]["buying_power"] = {"buying_power": value}

    with pytest.raises(mcp.RobinhoodShapeError, match="buying_power.buying_power"):
        mcp.portfolio_summary(envelope)


@pytest.mark.parametrize("value", ["0", 0, "180.25"])
def test_portfolio_summary_keeps_finite_buying_power(value) -> None:
    envelope = rh.portfolio()
    envelope["structured_content"]["data"]["buying_power"] = {"buying_power": value}

    assert mcp.portfolio_summary(envelope)["buying_power"] == str(value)


def test_position_rows_map_symbol_quantity_and_cost() -> None:
    rows = mcp.position_rows(
        rh.positions(
            [
                rh.position("aapl", "10.5", average_buy_price="150.25"),
                rh.position("MSFT", "2", average_buy_price=None),
            ]
        )
    )

    assert rows == [
        {"symbol": "AAPL", "quantity": "10.5", "average_cost": "150.25", "broker_type": "long"},
        {"symbol": "MSFT", "quantity": "2", "average_cost": None, "broker_type": "long"},
    ]


def test_the_position_type_is_carried_through_never_interpreted() -> None:
    rows = mcp.position_rows(rh.positions([rh.position("AAPL", "-3", type="short")]))

    assert rows[0]["broker_type"] == "short"
    assert rows[0]["quantity"] == "-3"  # the sign stays the broker's own fact


@pytest.mark.parametrize("broken", [None, "", "  "])
def test_a_position_row_without_its_required_type_is_an_error(broken: str | None) -> None:
    with pytest.raises(mcp.RobinhoodShapeError, match=r"positions\[0\] has no type"):
        mcp.position_rows(rh.positions([{**rh.position("AAPL", "1"), "type": broken}]))


def test_an_empty_position_list_is_a_complete_empty_account() -> None:
    assert mcp.position_rows(rh.positions([])) == []
    assert mcp.position_rows(rh.positions([], next_page="")) == []


@pytest.mark.parametrize(
    ("envelope", "message"),
    [
        (rh.positions(None), "no positions list"),
        (rh.positions([None]), r"positions\[0\] is not an object"),
        (rh.positions([rh.position("AAPL", "1")], next_page="cursor-2"), "more than one page"),
        (rh.envelope("get_equity_positions", {"positions": [], "total": "3"}), "unmapped data keys: total"),
        (rh.positions([{**rh.position("AAPL", "1"), "symbol": ""}]), "has no symbol"),
        (rh.positions([{**rh.position("AAPL", "1"), "quantity": "lots"}]), "quantity is not a number"),
        ({"status": "ok", "structured_content": {"guide": "no data"}}, "no data object"),
        ({"status": "error", "error": "token expired"}, "token expired"),
    ],
)
def test_position_rows_refuse_what_is_not_a_whole_account(envelope: dict, message: str) -> None:
    with pytest.raises(mcp.RobinhoodShapeError, match=message):
        mcp.position_rows(envelope)


def test_account_choices_keep_only_what_a_picker_shows() -> None:
    choices = mcp.account_choices(
        rh.accounts(
            [
                rh.account("5QR12345", nickname="Main", is_default=True),
                rh.account("5QR99887", agentic_allowed=False),
                {**rh.account("5QR55555"), "permanently_deactivated": True},
            ]
        )
    )

    assert choices == [
        {
            "account_ref": "5QR12345",
            "label": "Main ····2345",
            "is_default": True,
            "agentic_allowed": True,
            "deactivated": False,
        },
        {
            "account_ref": "5QR99887",
            "label": "individual ····9887",
            "is_default": False,
            "agentic_allowed": False,
            "deactivated": False,
        },
        {
            "account_ref": "5QR55555",
            "label": "individual ····5555",
            "is_default": False,
            "agentic_allowed": True,
            "deactivated": True,
        },
    ]


def test_account_choices_refuse_an_account_without_a_number() -> None:
    with pytest.raises(mcp.RobinhoodShapeError, match="has no account_number"):
        mcp.account_choices(rh.accounts([{**rh.account("5QR1"), "account_number": None}]))


def test_normalize_result_adds_the_view_and_keeps_the_raw_reply() -> None:
    envelope = rh.positions([rh.position("AAPL", "1")])

    normalized = mcp.normalize_result("positions", envelope)

    assert normalized["positions"][0]["symbol"] == "AAPL"
    assert normalized["structured_content"] is envelope["structured_content"]
    assert "mapping_error" not in normalized
    assert "positions" not in envelope  # the caller's reply is not mutated


def test_normalize_result_marks_an_unmappable_reply_instead_of_failing_the_read() -> None:
    normalized = mcp.normalize_result("positions", rh.positions([rh.position("AAPL", "1")], next_page="p2"))

    assert normalized["status"] == "ok"
    assert "positions" not in normalized
    assert "more than one page" in normalized["mapping_error"]


def test_normalize_result_leaves_error_envelopes_and_unmapped_operations_alone() -> None:
    error = {"status": "error", "error": "boom"}
    orders = {"status": "ok", "data": {"anything": 1}}

    assert mcp.normalize_result("positions", error) is error
    assert mcp.normalize_result("orders", orders) is orders


def test_account_scoped_operations_send_account_number_and_accounts_sends_nothing() -> None:
    assert mcp.remote_arguments("positions", {"account": "5QR1"}) == {"account_number": "5QR1"}
    assert mcp.remote_arguments("account", {"account_number": "5QR1"}) == {"account_number": "5QR1"}
    assert mcp.remote_arguments("orders", {}) == {}
    assert mcp.remote_arguments("accounts", {"account": "5QR1"}) == {}


def test_a_cursor_goes_only_to_the_tools_whose_schema_has_one() -> None:
    assert mcp.remote_arguments("positions", {"account": "5QR1", "cursor": " c1 "}) == {
        "account_number": "5QR1",
        "cursor": " c1 ",
    }
    assert mcp.remote_arguments("orders", {"cursor": "c1"}) == {"cursor": "c1"}
    assert mcp.remote_arguments("positions", {"account": "5QR1", "cursor": ""}) == {"account_number": "5QR1"}
    assert mcp.remote_arguments("positions", {"account": "5QR1", "cursor": 7}) == {"account_number": "5QR1"}
    assert mcp.remote_arguments("account", {"account": "5QR1", "cursor": "c1"}) == {"account_number": "5QR1"}
    assert mcp.remote_arguments("accounts", {"cursor": "c1"}) == {}


# --------------------------------------------------------------------------- #
# get_equity_quotes                                                            #
# --------------------------------------------------------------------------- #


def test_quote_summary_maps_the_reply_onto_the_shared_quote_reader_names() -> None:
    summary = mcp.quote_summary(
        rh.quotes([rh.quote_result(rh.quote("AAPL", last_trade_price="150.25", bid_price="150", ask_price="151"))])
    )

    assert summary == {
        "symbol": "AAPL",
        "last": "150.25",
        "close": None,
        "bid": "150",
        "ask": "151",
        "has_traded": True,
        "state": "active",
        "price_as_of": "2026-10-12T15:59:00Z",
        "price_error": None,
    }


def test_quote_summary_maps_the_official_settled_close_when_present() -> None:
    summary = mcp.quote_summary(
        rh.quotes([rh.quote_result(rh.quote("AAPL"), close_row=rh.settled_close("AAPL", price="149.50"))])
    )

    assert summary["close"] == "149.50"


def test_quote_summary_maps_a_zero_book_as_no_price_not_zero() -> None:
    summary = mcp.quote_summary(rh.quotes([rh.quote_result(rh.quote("AAPL", bid_price="0", ask_price="0"))]))

    assert summary["last"] == "100.00"
    assert summary["bid"] == "0"  # a closed book's sentinel, not a price
    assert summary["ask"] == "0"


def test_a_quote_that_has_never_traded_maps_no_price_fields() -> None:
    summary = mcp.quote_summary(rh.quotes([rh.quote_result(rh.quote("AAPL", has_traded=False))]))

    assert summary["has_traded"] is False
    assert summary["last"] is None and summary["bid"] is None and summary["ask"] is None
    assert mcp.normalize_result("quote", rh.quotes([rh.quote_result(rh.quote("AAPL", has_traded=False))]))[
        "quote"
    ] == summary


def test_normalize_result_adds_the_quote_view_and_keeps_the_raw_reply() -> None:
    envelope = rh.quotes([rh.quote_result(rh.quote("MSFT", last_trade_price="400.10"))])

    normalized = mcp.normalize_result("quote", envelope)

    assert normalized["quote"]["last"] == "400.10"
    assert normalized["structured_content"] is envelope["structured_content"]
    assert "mapping_error" not in normalized


def _broken_quote(**overrides: Any) -> dict:
    """A quote result with the required quote object replaced or damaged."""
    entry = rh.quote_result(rh.quote("AAPL"))
    entry.update(overrides)
    return entry


@pytest.mark.parametrize(
    ("envelope", "message"),
    [
        (rh.quotes(None), ""),  # a null results list maps to no quote, not an error
        (rh.quotes([]), ""),
        (rh.quotes([{"quote": None}]), ""),  # an explicitly null quote is the same
        (rh.quotes([rh.quote_result(rh.quote("AAPL")), rh.quote_result(rh.quote("MSFT"))]), "2 quote rows"),
        (rh.envelope("get_equity_quotes", {"results": [], "surprise": 1}), "unmapped data keys: surprise"),
        (rh.quotes([{"close": rh.settled_close("AAPL")}]), "results[0] has no quote"),
        (rh.quotes([_broken_quote(quote={**rh.quote("AAPL"), "ghost": "1"})]), "unmapped fields: ghost"),
        (
            rh.quotes([_broken_quote(quote={k: v for k, v in rh.quote("AAPL").items() if k != "state"})]),
            "quote is missing state",
        ),
        (rh.quotes([_broken_quote(quote={**rh.quote("AAPL"), "has_traded": "yes"})]), "has_traded is not a boolean"),
        (rh.quotes([_broken_quote(close={"price": "1"})]), "close is missing"),
        (rh.quotes(["a row"]), "results[0] is not an object"),
    ],
)
def test_quote_replies_that_are_not_the_advertised_shape(envelope: dict, message: str) -> None:
    normalized = mcp.normalize_result("quote", envelope)

    if message:
        assert message in normalized.get("mapping_error", "")
        assert "quote" not in normalized
    else:
        assert normalized.get("quote") is None


# --------------------------------------------------------------------------- #
# pagination                                                                   #
# --------------------------------------------------------------------------- #


def test_records_still_refuse_a_lone_first_page() -> None:
    with pytest.raises(mcp.RobinhoodShapeError, match="more than one page"):
        mcp.records(rh.positions([rh.position("AAPL", "1")], next_page="cursor-2"), "positions", "get_equity_positions")


def test_with_all_pages_follows_the_cursor_and_merges_every_page() -> None:
    fetched: list[str] = []

    def refetch(cursor: str) -> dict:
        fetched.append(cursor)
        if cursor == "cursor-2":
            return rh.positions([rh.position("MSFT", "2")], next_page="cursor-3")
        return rh.positions([rh.position("NVDA", "3")], next_page="")

    merged = mcp.with_all_pages(
        rh.positions([rh.position("AAPL", "1")], next_page="cursor-2"),
        "positions",
        "get_equity_positions",
        refetch,
    )

    assert fetched == ["cursor-2", "cursor-3"]
    assert [row["symbol"] for row in mcp.records(merged, "positions", "get_equity_positions")] == [
        "AAPL",
        "MSFT",
        "NVDA",
    ]
    data = merged["structured_content"]["data"]
    assert data["next"] == ""  # the merged reply is one complete page
    assert merged["data"] is merged["structured_content"]  # both views stay the same object
    assert merged["structured_content"]["guide"] == "Values are strings."  # the first page's guide


def test_with_all_pages_leaves_a_complete_reply_and_error_envelopes_untouched() -> None:
    complete = rh.positions([])
    failed = {"status": "error", "error": "token expired"}
    unmappable = {"status": "ok"}  # no data object: nothing to follow, mappers will name it

    def unused_refetch(cursor: str) -> dict:
        raise AssertionError(f"no page was owed for cursor {cursor!r}")

    assert mcp.with_all_pages(complete, "positions", "get_equity_positions", unused_refetch) is complete
    assert mcp.with_all_pages(failed, "positions", "get_equity_positions", unused_refetch) is failed
    assert mcp.with_all_pages(unmappable, "positions", "get_equity_positions", unused_refetch) is unmappable


def test_with_all_pages_drops_every_page_when_a_follow_up_fails() -> None:
    def refetch(cursor: str) -> dict:
        raise RuntimeError("token expired")

    with pytest.raises(mcp.RobinhoodShapeError, match="next-page call failed: token expired"):
        mcp.with_all_pages(
            rh.positions([rh.position("AAPL", "1")], next_page="cursor-2"),
            "positions",
            "get_equity_positions",
            refetch,
        )


def test_with_all_pages_refuses_a_cursor_that_never_ends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp, "_MAX_PAGES", 3)

    def refetch(cursor: str) -> dict:
        return rh.positions([rh.position(f"S{len(cursor)}", "1")], next_page=f"{cursor}x")

    with pytest.raises(mcp.RobinhoodShapeError, match="did not reach a last page within 3 pages"):
        mcp.with_all_pages(
            rh.positions([rh.position("AAPL", "1")], next_page="c"),
            "positions",
            "get_equity_positions",
            refetch,
        )


def test_the_trading_service_follows_position_pages_before_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict]] = []
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=["get_equity_positions"],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-token"),
    )

    class _Adapter:
        def __init__(self, server_name, server_config, **kwargs):  # noqa: ANN001
            assert kwargs == {"interactive_oauth": False}

        def call_tool(self, remote_name, arguments):  # noqa: ANN001
            calls.append((remote_name, dict(arguments)))
            if arguments.get("cursor") == "cursor-2":
                return rh.positions([rh.position("MSFT", "2")], next_page="")
            return rh.positions([rh.position("AAPL", "1")], next_page="cursor-2")

    monkeypatch.setattr(
        "src.config.loader.load_agent_config", lambda: SimpleNamespace(mcp_servers={"robinhood": server})
    )
    monkeypatch.setattr("src.live.registry.has_cached_oauth_token", lambda *_: True)
    monkeypatch.setattr("src.tools.mcp.MCPServerAdapter", _Adapter)

    result = service.get_positions(
        "robinhood-live-mcp-readonly", account=ACCOUNT, interactive_oauth=False
    )

    assert calls == [
        ("get_equity_positions", {"account_number": ACCOUNT}),
        ("get_equity_positions", {"account_number": ACCOUNT, "cursor": "cursor-2"}),
    ]
    assert [row["symbol"] for row in result["positions"]] == ["AAPL", "MSFT"]
    assert "mapping_error" not in result


def test_trading_service_lists_accounts_through_get_accounts(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict]] = []
    server = SimpleNamespace(
        url="https://agent.robinhood.com/mcp/trading",
        enabled_tools=["get_accounts"],
        auth=SimpleNamespace(cache_dir="/tmp/vibe-token"),
    )

    class _Adapter:
        def __init__(self, server_name, server_config, **kwargs):  # noqa: ANN001
            assert kwargs == {"interactive_oauth": False}

        def call_tool(self, remote_name, arguments):  # noqa: ANN001
            calls.append((remote_name, dict(arguments)))
            return rh.accounts([rh.account("5QR12345", nickname="Main")])

    monkeypatch.setattr(
        "src.config.loader.load_agent_config", lambda: SimpleNamespace(mcp_servers={"robinhood": server})
    )
    monkeypatch.setattr("src.live.registry.has_cached_oauth_token", lambda *_: True)
    monkeypatch.setattr("src.tools.mcp.MCPServerAdapter", _Adapter)

    result = service.get_accounts("robinhood-live-mcp-readonly", interactive_oauth=False)

    assert calls == [("get_accounts", {})]
    assert result["accounts"][0]["account_ref"] == "5QR12345"
    assert result["profile_id"] == "robinhood-live-mcp-readonly"


def test_accounts_are_unsupported_off_remote_mcp() -> None:
    result = service.get_accounts("ibkr-paper-local")

    assert result["status"] == "error"
    assert "accounts.read" in result["error"]


def test_the_trading_profile_stays_first_for_the_broker_on_ramp() -> None:
    assert service.connector_profile_id_for_broker("robinhood") == "robinhood-live-mcp"
    assert service.live_runner_profile_for_broker("robinhood").id == "robinhood-live-mcp"
    assert service.runner_requires_account("robinhood") is True
    assert service.runner_requires_account("ibkr") is False


def test_the_robinhood_live_profile_discloses_options_order_limit() -> None:
    profile = service.live_runner_profile_for_broker("robinhood")

    assert "options order placement is not supported" in profile.notes.lower()


@pytest.mark.parametrize("stamp", [None, "invalid", "2026-10-12T21:00:00"])
def test_extended_trade_without_a_timezone_cannot_be_selected(stamp):
    quote = rh.quote("AAPL")
    quote.update(last_non_reg_trade_price="120", venue_last_non_reg_trade_time=stamp)
    result = mcp.normalize_result("quote", rh.quotes([rh.quote_result(quote)]))
    assert "timestamp with timezone" in result["mapping_error"]
    assert "quote" not in result


def test_trade_selection_compares_instants_instead_of_timestamp_strings():
    quote = rh.quote("AAPL", last_trade_price="110")
    quote.update(
        venue_last_trade_time="2026-10-12T15:00:00-04:00",
        last_non_reg_trade_price="100",
        venue_last_non_reg_trade_time="2026-10-12T18:00:00Z",
    )
    result = mcp.quote_summary(rh.quotes([rh.quote_result(quote)]))
    assert result["last"] == "110"
    assert result["price_as_of"] == "2026-10-12T15:00:00-04:00"
