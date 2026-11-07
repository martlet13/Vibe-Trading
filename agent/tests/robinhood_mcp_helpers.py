"""Robinhood MCP reply builders, taken from the shapes posted on #1428.

Every builder reproduces the adapter envelope ``MCPServerAdapter.call_tool``
returns around a Robinhood reply: the broker payload ``{"data": ..., "guide":
...}`` sits under both ``structured_content`` and ``data``. So the records a
test cares about are at ``structured_content.data.<key>``, one level deeper than
the ``data.positions`` the live path used to read (#1442).

Field names and types come from balu1866's posts: observed ``get_accounts`` and
``get_portfolio`` replies (2026-09-14) and the advertised
``get_equity_positions`` ``outputSchema`` (2026-09-16), plus the server's
advertised ``tools/list`` metadata for ``get_equity_quotes`` and the paginated
tools' argument schemas (transcribed in ``kevin1chun/robinhood-for-agents``,
live capture 2026-10-12). Values are invented.
"""

from __future__ import annotations

from typing import Any

#: Required ``get_equity_positions`` item fields other than symbol / quantity / type.
_POSITION_SHARE_FIELDS = (
    "intraday_quantity",
    "shares_available_for_sells",
    "shares_held_for_sells",
    "shares_held_for_stock_grants",
    "shares_held_for_options_events",
    "shares_held_for_asset_transfer",
    "shares_pending_from_options_events",
)


def envelope(tool: str, data: Any, *, guide: str = "Values are strings.") -> dict[str, Any]:
    """Wrap a broker ``data`` object the way the MCP adapter returns it."""
    structured = {"data": data, "guide": guide}
    return {
        "status": "ok",
        "data": structured,
        "structured_content": structured,
        "server": "robinhood",
        "remote_tool": tool,
        "tool": tool,
    }


def position(
    symbol: str,
    quantity: str,
    *,
    average_buy_price: str | None = "100.00",
    type: str = "long",  # noqa: A002 - the remote field's own name
) -> dict[str, Any]:
    """Return one ``get_equity_positions`` item with every required field."""
    # `type` is the server-documented position direction ('long', 'short',
    # 'boxed', 'empty'); positive fixtures are 'long', matching the schema's
    # "positive quantity" description.
    row: dict[str, Any] = {"symbol": symbol, "quantity": quantity, "type": type}
    row.update({field: "0" for field in _POSITION_SHARE_FIELDS})
    row["shares_available_for_sells"] = quantity
    if average_buy_price is not None:
        row["average_buy_price"] = average_buy_price
    return row


def positions(rows: list[dict[str, Any]] | None, *, next_page: str | None = None) -> dict[str, Any]:
    """Return a ``get_equity_positions`` envelope."""
    data: dict[str, Any] = {"positions": rows}
    if next_page is not None:
        data["next"] = next_page
    return envelope("get_equity_positions", data)


def portfolio(
    *,
    total_value: str = "5000.00",
    cash: str = "5000.00",
    buying_power: str = "5000.00",
    currency: str = "USD",
    **values: str,
) -> dict[str, Any]:
    """Return a ``get_portfolio`` envelope; non-equity values default to zero."""
    data: dict[str, Any] = {
        "total_value": total_value,
        "equity_value": "0.00",
        "options_value": "0.00",
        "futures_value": "0.00",
        "event_contracts_value": "0.00",
        "crypto_value": "0.00",
        "cash": cash,
        "pending_deposits": "0.00",
        "mutual_funds_value": "0.00",
        "fixed_income_value": "0.00",
        "currency": currency,
        "buying_power": {
            "buying_power": buying_power,
            "unleveraged_buying_power": buying_power,
            "display_currency": currency,
        },
        "crypto_buying_power": {"buying_power": "0.00"},
    }
    data.update(values)
    return envelope("get_portfolio", data)


def account(
    number: str,
    *,
    nickname: str | None = None,
    is_default: bool = False,
    agentic_allowed: bool = True,
    deactivated: bool = False,
) -> dict[str, Any]:
    """Return one ``get_accounts`` item."""
    row: dict[str, Any] = {
        "account_number": number,
        "affiliate": "rhf",
        "agentic_allowed": agentic_allowed,
        "brokerage_account_type": "individual",
        "deactivated": deactivated,
        "is_default": is_default,
        "management_type": "self_directed",
        "option_level": "option_level_2",
        "permanently_deactivated": False,
        "rhs_account_number": f"rhs-{number}",
        "state": "active",
        "type": "cash",
        "unsettled_funds": "0.00",
    }
    if nickname is not None:
        row["nickname"] = nickname
    return row


def accounts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return a ``get_accounts`` envelope."""
    return envelope("get_accounts", {"accounts": rows})


def quote(
    symbol: str,
    *,
    last_trade_price: str = "100.00",
    bid_price: str = "99.50",
    ask_price: str = "100.50",
    has_traded: bool = True,
    state: str = "active",
) -> dict[str, Any]:
    """Return one ``get_equity_quotes`` ``quote`` object with every required field."""
    return {
        "symbol": symbol,
        "last_trade_price": last_trade_price,
        "venue_last_trade_time": "2026-10-12T15:59:00Z",
        "last_non_reg_trade_price": None,
        "venue_last_non_reg_trade_time": None,
        "adjusted_previous_close": "99.00",
        "previous_close": "99.00",
        "previous_close_date": "2026-10-11",
        "bid_price": bid_price,
        "venue_bid_time": "2026-10-12T15:59:00Z",
        "ask_price": ask_price,
        "venue_ask_time": "2026-10-12T15:59:00Z",
        "has_traded": has_traded,
        "state": state,
    }


def settled_close(symbol: str, *, price: str | None = "98.50") -> dict[str, Any]:
    """Return one ``get_equity_quotes`` official settled ``close`` object."""
    return {
        "symbol": symbol,
        "date": "2026-10-11",
        "price": price,
        "interpolated": False,
        "source": "sip-close",
    }


def quote_result(quote_row: dict[str, Any], *, close_row: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return one ``get_equity_quotes`` ``results`` entry."""
    entry: dict[str, Any] = {"quote": quote_row}
    if close_row is not None:
        entry["close"] = close_row
    return entry


def quotes(entries: list[dict[str, Any]] | None, *, closes_error: str | None = None) -> dict[str, Any]:
    """Return a ``get_equity_quotes`` envelope."""
    data: dict[str, Any] = {"results": entries}
    if closes_error is not None:
        data["closes_error"] = closes_error
    return envelope("get_equity_quotes", data)
