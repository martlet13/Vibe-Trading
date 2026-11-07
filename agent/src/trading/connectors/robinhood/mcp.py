"""Robinhood remote MCP generic-operation mapping and reply shapes.

Every consumer that reads a Robinhood reply goes through this module: the
trading service's generic reads, the portfolio source, the live runner, the
pre-trade order gate and the mandate ceiling fetch. They used to read the reply
one level too shallow (``data.positions`` where Robinhood nests
``data.data.positions``), each in its own way (#1442).

What the shapes are built from, and nothing else:

* ``get_accounts``: observed field names and types posted on #1428
  (2026-09-14). Not a schema.
* ``get_portfolio``: observed on #1428 (2026-09-14). Not a schema.
* ``get_equity_positions``: the server's advertised ``outputSchema`` posted on
  #1428 (2026-09-16). ``data`` allows exactly ``positions`` and ``next``. An item
  carries ``symbol``, ``quantity`` and an optional ``average_buy_price``, and no
  price or currency.
* ``get_equity_quotes`` and pagination schemas: the third-party transcription
  at ``kevin1chun/robinhood-for-agents@16b3d800e7f29e9d450f45663d054a7b5adcd4b3``
  in ``docs/official-mcp-tools.json``. This is captured schema evidence, not a
  claim that this adapter has been verified against a live authenticated account.

The descriptions this mapping relies on: a position item's ``type`` is the
position direction — ``'long' (positive quantity), 'short' (negative),
'boxed' (both, rare), or 'empty' (zero)`` — carried through, never interpreted.
``next`` is "an opaque cursor for the next page — NOT a URL ... call this tool
again with ``cursor`` set to this value verbatim, leaving the other arguments
unchanged. Empty on the last page." Quotes reply as ``data.results``, one
entry per requested symbol, each required to carry a ``quote`` whose ``bid``
/ ``ask`` are "Zero when the book is closed or unavailable" and whose
``has_traded=False`` means the price fields are meaningless, plus an optional
official settled ``close``.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Callable

_REMOTE_TOOL_NAMES = {
    "accounts": "get_accounts",
    "account": "get_portfolio",
    "positions": "get_equity_positions",
    "orders": "get_equity_orders",
    "quote": "get_equity_quotes",
}

_RUNNER_TOOL_NAMES = {
    "accounts": "get_accounts",
    "account": "get_portfolio",
    "positions": "get_equity_positions",
    "orders": "get_equity_orders",
    "quote": "get_equity_quotes",
    "submit_order": "place_equity_order",
    "cancel_order": "cancel_equity_order",
}

#: ``get_portfolio`` values held outside equities. ``get_equity_positions``
#: covers none of them, so a holdings view is only complete when all are zero.
NON_EQUITY_VALUE_FIELDS = (
    "options_value",
    "crypto_value",
    "futures_value",
    "event_contracts_value",
    "mutual_funds_value",
    "fixed_income_value",
)

#: Every ``get_equity_quotes`` ``quote`` field. The advertised outputSchema
#: marks all fourteen required with ``additionalProperties: false``; the
#: mapping validates the object against exactly these names.
QUOTE_FIELDS = frozenset(
    {
        "symbol",
        "last_trade_price",
        "venue_last_trade_time",
        "last_non_reg_trade_price",
        "venue_last_non_reg_trade_time",
        "adjusted_previous_close",
        "previous_close",
        "previous_close_date",
        "bid_price",
        "venue_bid_time",
        "ask_price",
        "venue_ask_time",
        "has_traded",
        "state",
    }
)

#: The optional official settled ``close`` object of a quote result.
CLOSE_FIELDS = frozenset({"symbol", "date", "price", "interpolated", "source"})

#: Operations whose remote tool takes the selected account as ``account_number``.
ACCOUNT_SCOPED_OPERATIONS = frozenset({"account", "positions", "orders"})

#: Operations whose remote tool takes a pagination ``cursor``. The transcribed
#: input schemas carry it on ``get_equity_positions`` and ``get_equity_orders``;
#: ``get_portfolio`` and ``get_accounts`` take no arguments beyond (for
#: ``get_portfolio``) the account, and a cursor sent there would be refused.
PAGINATED_OPERATIONS = frozenset({"positions", "orders"})


class RobinhoodShapeError(ValueError):
    """A Robinhood reply that does not match the shape this mapping was built from."""


def remote_tool_name(operation: str) -> str | None:
    """Return the Robinhood remote tool name for a generic operation."""
    return _REMOTE_TOOL_NAMES.get(operation)


def runner_tool_name(operation: str) -> str | None:
    """Return the Robinhood remote tool name used by live runner plumbing."""
    return _RUNNER_TOOL_NAMES.get(operation)


def remote_arguments(operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Normalize generic arguments for a Robinhood remote MCP operation."""
    if operation == "quote":
        symbol = arguments.get("symbol")
        symbols = arguments.get("symbols")
        return {"symbols": symbols or ([symbol] if symbol else [])}
    if operation in ACCOUNT_SCOPED_OPERATIONS:
        # Robinhood's get_portfolio / get_equity_positions / get_equity_orders
        # MCP tools require account_number. The generic trading service passes
        # the CLI/agent-supplied account code in under the "account" key; map
        # it to the field name Robinhood's schema actually expects. A cursor
        # goes through only for the tools whose advertised input schema has
        # one, passed verbatim as the schema's description directs.
        wire: dict[str, Any] = {}
        account_number = arguments.get("account_number") or arguments.get("account")
        if account_number:
            wire["account_number"] = account_number
        if operation in PAGINATED_OPERATIONS:
            cursor = arguments.get("cursor")
            if isinstance(cursor, str) and cursor:
                wire["cursor"] = cursor
        return wire
    return {}


def broker_data(envelope: Any, tool: str) -> dict[str, Any]:
    """Return the broker's inner ``data`` object from an adapter envelope.

    Args:
        envelope: ``MCPServerAdapter.call_tool`` result.
        tool: Remote tool name, for the error message.

    Returns:
        The ``data`` object of the broker payload.

    Raises:
        RobinhoodShapeError: If the call failed, or the payload is not an
            object whose ``data`` is an object.
    """
    if not isinstance(envelope, dict):
        raise RobinhoodShapeError(f"{tool} returned {type(envelope).__name__}, not an object")
    if str(envelope.get("status") or "").lower() != "ok":
        raise RobinhoodShapeError(str(envelope.get("error") or f"{tool} failed"))
    payload = envelope.get("structured_content")
    if payload is None:
        payload = envelope.get("data")
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise RobinhoodShapeError(f"{tool} reply has no data object")
    return payload["data"]


#: Upper bound on pages one complete read may follow. The cursor is opaque and
#: the server advertises no page count, so a cursor that has not reached an
#: empty ``next`` within this many pages is treated as a loop, not a big
#: account — and a partial read is never passed off as complete.
_MAX_PAGES = 100


def _page(envelope: Any, key: str, tool: str) -> tuple[list[dict[str, Any]], str]:
    """Validate one reply page and split it into records and the next cursor.

    Raises:
        RobinhoodShapeError: If the list is missing or null, holds a null or
            non-object item, ``data`` carries a key other than ``key`` and
            ``next``, or ``next`` is not the advertised optional string.
    """
    data = broker_data(envelope, tool)
    unexpected = sorted(set(data) - {key, "next"})
    if unexpected:
        raise RobinhoodShapeError(f"{tool} reply has unmapped data keys: {', '.join(unexpected)}")
    next_value = data.get("next")
    if next_value is not None and not isinstance(next_value, str):
        raise RobinhoodShapeError(f"{tool} reply has a non-string next")
    rows = data.get(key)
    if not isinstance(rows, list):
        raise RobinhoodShapeError(f"{tool} reply has no {key} list")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RobinhoodShapeError(f"{tool} {key}[{index}] is not an object")
    return rows, next_value or ""


def records(envelope: Any, key: str, tool: str) -> list[dict[str, Any]]:
    """Return the complete record list a Robinhood list read carries.

    This is the single-page contract: a reply that advertises a further page
    is refused, because the first page is not the whole account. Readers that
    follow cursors go through :func:`with_all_pages` first and hand it the
    merged reply.

    Args:
        envelope: ``MCPServerAdapter.call_tool`` result.
        key: The list field inside ``data`` (``positions``, ``orders``,
            ``accounts``).
        tool: Remote tool name, for error messages.

    Returns:
        Every record on the page, each an object.

    Raises:
        RobinhoodShapeError: If the list is missing or null, holds a null or
            non-object item, ``data`` carries a key other than ``key`` and
            ``next``, or a non-empty ``next`` says the reply is one page of
            several.
    """
    rows, next_cursor = _page(envelope, key, tool)
    if next_cursor:
        raise RobinhoodShapeError(
            f"{tool} returned more than one page and this read does not follow cursors; "
            "the first page is not the whole account"
        )
    return rows


def with_all_pages(envelope: Any, key: str, tool: str, refetch: Callable[[str], Any]) -> Any:
    """Follow the reply's advertised ``next`` cursor and merge every page.

    This implements the contract the server's own schema states: ``next`` is
    an opaque cursor, not a URL, and the next page is fetched by calling the
    tool again with ``cursor`` set to that value verbatim, other arguments
    unchanged, until a reply carries an empty ``next``. ``refetch`` must do
    exactly that and return the follow-up call's envelope.

    Args:
        envelope: The first page's ``MCPServerAdapter.call_tool`` result.
        key: The list field inside ``data`` (``positions``, ``orders``).
        tool: Remote tool name, for error messages.
        refetch: Callable taking the verbatim cursor and returning the next
            page's envelope.

    Returns:
        The first envelope unchanged when the call failed or its ``next`` is
        empty. Otherwise a shallow copy of the first envelope whose
        ``data`` / ``structured_content`` hold every page's records under
        ``key`` with an empty ``next`` — one complete read in the shape the
        single-page mappers already validate.

    Raises:
        RobinhoodShapeError: If a follow-up call fails or breaks the page
            shape, or the cursor does not reach a last page within
            :data:`_MAX_PAGES` pages. A partial multi-page read is never
            returned: the pages fetched so far are dropped with the error.
    """
    if str(envelope.get("status") or "").lower() != "ok":
        return envelope
    try:
        rows, next_cursor = _page(envelope, key, tool)
    except RobinhoodShapeError:
        # No readable page means no cursor to follow: leave the envelope for
        # the mappers, whose mapping_error names what is actually wrong with
        # it instead of a pagination failure that did not happen.
        return envelope
    if not next_cursor:
        return envelope
    payload = envelope.get("structured_content")
    if payload is None:
        payload = envelope.get("data")
    guide = payload.get("guide") if isinstance(payload, dict) else None
    all_rows = list(rows)
    for _ in range(_MAX_PAGES):
        try:
            page = refetch(next_cursor)
        except Exception as exc:
            raise RobinhoodShapeError(f"{tool} next-page call failed: {exc}") from None
        page_rows, next_cursor = _page(page, key, tool)
        all_rows.extend(page_rows)
        if not next_cursor:
            structured = {"data": {key: all_rows, "next": ""}, "guide": guide}
            merged = dict(envelope)
            merged["data"] = structured
            merged["structured_content"] = structured
            return merged
    raise RobinhoodShapeError(
        f"{tool} did not reach a last page within {_MAX_PAGES} pages; "
        "the pages read so far are dropped rather than shown as the whole account"
    )


def _decimal_text(value: Any, field: str, tool: str) -> str:
    """Return ``value`` when it is a finite decimal string, else raise."""
    if isinstance(value, bool) or value is None:
        raise RobinhoodShapeError(f"{tool} {field} is missing")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise RobinhoodShapeError(f"{tool} {field} is not a number") from None
    if not parsed.is_finite():
        raise RobinhoodShapeError(f"{tool} {field} is not finite")
    return str(value)


def portfolio_summary(envelope: Any) -> dict[str, Any]:
    """Map a ``get_portfolio`` reply to the account summary every consumer reads.

    Args:
        envelope: ``MCPServerAdapter.call_tool`` result for ``get_portfolio``.

    Returns:
        ``portfolio_value``, ``cash``, ``buying_power`` and ``currency``, plus
        ``non_equity_values`` mapping each field in
        :data:`NON_EQUITY_VALUE_FIELDS` to its value, or ``None`` when the
        reply omitted it. Values stay decimal strings.

    Raises:
        RobinhoodShapeError: If ``total_value`` or ``currency`` is missing,
            non-null ``buying_power`` is not an object, or a value that is
            present is not a finite number.
    """
    tool = "get_portfolio"
    data = broker_data(envelope, tool)
    currency = data.get("currency")
    if not isinstance(currency, str) or not currency.strip():
        raise RobinhoodShapeError(f"{tool} currency is missing")
    buying_power = data.get("buying_power")
    if buying_power is not None and not isinstance(buying_power, dict):
        raise RobinhoodShapeError(f"{tool} buying_power is not an object")
    raw_buying_power = buying_power.get("buying_power") if isinstance(buying_power, dict) else None
    return {
        "portfolio_value": _decimal_text(data.get("total_value"), "total_value", tool),
        "cash": _decimal_text(data["cash"], "cash", tool) if data.get("cash") is not None else None,
        "buying_power": (
            _decimal_text(raw_buying_power, "buying_power.buying_power", tool)
            if raw_buying_power is not None
            else None
        ),
        "currency": currency.strip().upper(),
        "non_equity_values": {
            field: _decimal_text(data[field], field, tool) if data.get(field) is not None else None
            for field in NON_EQUITY_VALUE_FIELDS
        },
    }


def quote_summary(envelope: Any, *, expected_symbol: str | None = None) -> dict[str, Any] | None:
    """Map a ``get_equity_quotes`` reply to the quote view the portfolio prices with.

    The generic read asks for one symbol, so the reply must carry exactly one
    ``results`` entry. The mapping hands the reply's prices to the shared quote
    reader under its own names: the more recent timestamped regular/extended
    hours trade becomes ``last``; the official settled ``close.price`` becomes
    ``close``; ``bid_price`` / ``ask_price`` become ``bid`` / ``ask``. A closed
    book reports those as ``"0"``, which the shared reader treats as no price.
    ``has_traded=False`` means the price fields are meaningless, so they are
    mapped as absent instead of passed on as prices; the listing ``state`` is
    carried through for consumers that need to qualify the quote.

    Args:
        envelope: ``MCPServerAdapter.call_tool`` result for
            ``get_equity_quotes``.
        expected_symbol: Requested symbol, when called through the trading service.

    Returns:
        ``None`` when the reply advertises no result rows (the symbol resolved
        to nothing, or its ``quote`` is null), else the quote mapping with
        decimal-string values and ``None`` for absent prices.

    Raises:
        RobinhoodShapeError: If the reply does not match the advertised
            schema, or carries more than one result for a one-symbol read.
    """
    tool = "get_equity_quotes"
    data = broker_data(envelope, tool)
    unexpected = sorted(set(data) - {"results", "closes_error"})
    if unexpected:
        raise RobinhoodShapeError(f"{tool} reply has unmapped data keys: {', '.join(unexpected)}")
    results = data.get("results")
    if results is None:
        return None
    if not isinstance(results, list):
        raise RobinhoodShapeError(f"{tool} results is not a list")
    if len(results) > 1:
        raise RobinhoodShapeError(f"{tool} returned {len(results)} quote rows for a one-symbol read")
    if not results:
        return None
    entry = results[0]
    if not isinstance(entry, dict):
        raise RobinhoodShapeError(f"{tool} results[0] is not an object")
    unexpected = sorted(set(entry) - {"quote", "close"})
    if unexpected:
        raise RobinhoodShapeError(f"{tool} results[0] has unmapped keys: {', '.join(unexpected)}")
    if "quote" not in entry:
        # The schema marks quote required; a missing key is a broken reply,
        # while an explicitly null one means the symbol resolved to nothing.
        raise RobinhoodShapeError(f"{tool} results[0] has no quote")
    quote = entry.get("quote")
    if quote is None:
        return None
    if not isinstance(quote, dict):
        raise RobinhoodShapeError(f"{tool} results[0].quote is not an object")
    missing = sorted(QUOTE_FIELDS - set(quote))
    if missing:
        raise RobinhoodShapeError(f"{tool} quote is missing {', '.join(missing)}")
    unexpected = sorted(set(quote) - QUOTE_FIELDS)
    if unexpected:
        raise RobinhoodShapeError(f"{tool} quote has unmapped fields: {', '.join(unexpected)}")
    has_traded = quote.get("has_traded")
    if not isinstance(has_traded, bool):
        raise RobinhoodShapeError(f"{tool} quote has_traded is not a boolean")
    state = quote.get("state")
    if not isinstance(state, str) or not state.strip():
        raise RobinhoodShapeError(f"{tool} quote state is missing")
    symbol = quote.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise RobinhoodShapeError(f"{tool} quote symbol is missing")
    symbol = symbol.strip().upper()
    if expected_symbol and symbol != expected_symbol.strip().upper():
        raise RobinhoodShapeError(f"{tool} quote symbol does not match the requested symbol")
    usable = has_traded and state.strip() == "active"
    prices: dict[str, str | None] = {"last": None, "bid": None, "ask": None}
    price_as_of = None
    if usable:
        for field, name in (("last_trade_price", "last"), ("bid_price", "bid"), ("ask_price", "ask")):
            prices[name] = _decimal_text(quote.get(field), f"quote.{field}", tool)
        trades = []
        for price_field, time_field in (
            ("last_trade_price", "venue_last_trade_time"),
            ("last_non_reg_trade_price", "venue_last_non_reg_trade_time"),
        ):
            raw_price = quote.get(price_field)
            if raw_price is None:
                continue
            price = _decimal_text(raw_price, f"quote.{price_field}", tool)
            if Decimal(price) <= 0:
                continue
            stamp = quote.get(time_field)
            try:
                parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
                if parsed.utcoffset() is None:
                    raise ValueError("missing timezone")
            except (AttributeError, TypeError, ValueError):
                raise RobinhoodShapeError(f"{tool} quote.{time_field} is not a timestamp with timezone") from None
            trades.append((parsed, price, stamp))
        if trades:
            _, prices["last"], price_as_of = max(trades, key=lambda item: item[0])
    close_price = None
    close = entry.get("close")
    if close is not None:
        if not isinstance(close, dict):
            raise RobinhoodShapeError(f"{tool} results[0].close is not an object")
        missing = sorted(CLOSE_FIELDS - set(close))
        if missing:
            raise RobinhoodShapeError(f"{tool} close is missing {', '.join(missing)}")
        unexpected = sorted(set(close) - CLOSE_FIELDS)
        if unexpected:
            raise RobinhoodShapeError(f"{tool} close has unmapped fields: {', '.join(unexpected)}")
        if not isinstance(close.get("symbol"), str) or close["symbol"].strip().upper() != symbol:
            raise RobinhoodShapeError(f"{tool} close symbol does not match quote symbol")
        if usable and close.get("price") is not None:
            close_price = _decimal_text(close.get("price"), "close.price", tool)
            if not price_as_of and Decimal(close_price) > 0:
                price_as_of = close.get("date")
    return {
        "symbol": symbol,
        "last": prices["last"],
        "close": close_price,
        "bid": prices["bid"],
        "ask": prices["ask"],
        "has_traded": has_traded,
        "state": state.strip(),
        "price_as_of": price_as_of,
        "price_error": None if usable else f"Robinhood quote has_traded={has_traded}, state={state.strip()}",
    }


def position_rows(envelope: Any) -> list[dict[str, Any]]:
    """Map a ``get_equity_positions`` reply to generic position rows.

    Args:
        envelope: ``MCPServerAdapter.call_tool`` result for
            ``get_equity_positions``.

    Returns:
        One ``{"symbol", "quantity", "average_cost", "broker_type"}`` row per
        position. ``average_cost`` is ``None`` when Robinhood omitted
        ``average_buy_price`` (it does while a position reconciles); it is
        never read as zero. ``broker_type`` carries the row's ``type``
        verbatim — the schema describes it as the position direction
        (``'long' (positive quantity), 'short' (negative), 'boxed' (both,
        rare), or 'empty' (zero)``) — it is not an enum constraint and is
        never interpreted here.

    Raises:
        RobinhoodShapeError: If the reply is not a complete, single-page
            position list, or a row has no symbol, no ``type``, or a
            non-numeric quantity.
    """
    tool = "get_equity_positions"
    rows = []
    for index, row in enumerate(records(envelope, "positions", tool)):
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not symbol.strip():
            raise RobinhoodShapeError(f"{tool} positions[{index}] has no symbol")
        broker_type = row.get("type")
        if not isinstance(broker_type, str) or not broker_type.strip():
            raise RobinhoodShapeError(f"{tool} positions[{index}] has no type")
        average = row.get("average_buy_price")
        rows.append(
            {
                "symbol": symbol.strip().upper(),
                "quantity": _decimal_text(row.get("quantity"), f"positions[{index}].quantity", tool),
                "average_cost": (
                    _decimal_text(average, f"positions[{index}].average_buy_price", tool)
                    if average is not None
                    else None
                ),
                "broker_type": broker_type,
            }
        )
    return rows


def account_choices(envelope: Any) -> list[dict[str, Any]]:
    """Map a ``get_accounts`` reply to the rows an account picker shows.

    Args:
        envelope: ``MCPServerAdapter.call_tool`` result for ``get_accounts``.

    Returns:
        One row per account: ``account_ref`` (the account number, which is
        what the account-scoped tools take), a ``label`` built from the
        nickname or account type plus the last four digits, and the
        ``is_default`` / ``agentic_allowed`` / ``deactivated`` flags. Every
        other field Robinhood returns is dropped.

    Raises:
        RobinhoodShapeError: If the reply is not a complete account list, or an
            account has no account number.
    """
    tool = "get_accounts"
    choices = []
    for index, row in enumerate(records(envelope, "accounts", tool)):
        number = row.get("account_number")
        if not isinstance(number, str) or not number.strip():
            raise RobinhoodShapeError(f"{tool} accounts[{index}] has no account_number")
        number = number.strip()
        name = next(
            (
                str(row[key]).strip()
                for key in ("nickname", "brokerage_account_type", "type")
                if isinstance(row.get(key), str) and row[key].strip()
            ),
            "Robinhood account",
        )
        choices.append(
            {
                "account_ref": number,
                "label": f"{name} ····{number[-4:]}",
                "is_default": row.get("is_default") is True,
                "agentic_allowed": row.get("agentic_allowed") is True,
                "deactivated": row.get("deactivated") is True or row.get("permanently_deactivated") is True,
            }
        )
    return choices


def normalize_result(operation: str, envelope: dict[str, Any], *, expected_symbol: str | None = None) -> dict[str, Any]:
    """Add the generic account / positions / accounts / quote view to a Robinhood reply.

    The broker reply is kept as it came. A reply that does not match the mapped
    shape is not turned into an error here, because an agent reading the
    account can still use the raw text. It carries ``mapping_error`` instead,
    and a consumer that needs the mapped view (the portfolio) fails on it.

    Args:
        operation: Generic operation that produced the reply.
        envelope: ``MCPServerAdapter.call_tool`` result.
        expected_symbol: Symbol that a single-symbol quote must match.

    Returns:
        A shallow copy with ``account``, ``positions`` or ``accounts`` added, or
        with ``mapping_error`` when the shape did not match.
    """
    mappers = {
        "account": ("account", portfolio_summary),
        "positions": ("positions", position_rows),
        "accounts": ("accounts", account_choices),
        "quote": ("quote", lambda value: quote_summary(value, expected_symbol=expected_symbol)),
    }
    if operation not in mappers or str(envelope.get("status") or "").lower() != "ok":
        return envelope
    key, mapper = mappers[operation]
    normalized = dict(envelope)
    try:
        normalized[key] = mapper(envelope)
    except RobinhoodShapeError as exc:
        normalized["mapping_error"] = str(exc)
    return normalized
