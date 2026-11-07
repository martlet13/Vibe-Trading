"""Built-in Robinhood connector profiles."""

from __future__ import annotations

from src.trading.types import TradingProfile

ROBINHOOD_PROFILES: tuple[TradingProfile, ...] = (
    TradingProfile(
        id="robinhood-live-mcp",
        connector="robinhood",
        label="Robinhood Live · Agentic MCP",
        environment="live",
        transport="remote_mcp",
        capabilities=(
            "account.read",
            "positions.read",
            "orders.read",
            "quotes.read",
            "orders.place.requires_mandate",
            "runner.manage.requires_mandate",
        ),
        readonly=False,
        config={"server": "robinhood", "account_selection": "required"},
        notes=(
            "Live order placement supports equities only; options order placement is not supported. "
            "Robinhood's options order and required review schemas are not mapped into the mandate gate."
        ),
    ),
    # The portfolio view of the same MCP server and OAuth grant. Positions are
    # priced through get_equity_quotes, mapped from the server's advertised
    # outputSchema (see connectors/robinhood/mcp.py for the evidence trail).
    TradingProfile(
        id="robinhood-live-mcp-readonly",
        connector="robinhood",
        label="Robinhood Live · Agentic MCP Read-Only",
        environment="live",
        transport="remote_mcp",
        capabilities=("account.read", "positions.read", "quotes.read"),
        readonly=True,
        config={"server": "robinhood", "account_selection": "required"},
        notes=(
            "Reads get_portfolio, get_equity_positions and get_equity_quotes for one "
            "selected account. Positions are priced from the quote read; an account "
            "holding options, crypto, futures, event contracts, mutual funds or fixed "
            "income is reported as an error instead of an equity-only view."
        ),
    ),
)
