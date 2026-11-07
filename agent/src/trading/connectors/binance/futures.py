"""Binance USD-M perpetual futures through the Binance connector's futures profiles.

Reached only from :mod:`src.trading.connectors.binance.sdk` when the selected
profile declares ``futures``. The client is ccxt's ``binanceusdm`` on the
futures testnet (paper) or ``fapi.binance.com`` (live), checked against the
profile before every call. Orders follow :mod:`src.trading.futures_guard`:
isolated margin, at most 5x, and a stop-loss plus take-profit on every new
position.

Binance cannot attach a stop-loss or take-profit to the opening order, so the
protection is two reduce-only conditional orders placed right after a market
entry fills. If either is refused, the connector cancels the other and closes
the entry at market rather than leave a position unprotected. Entries are
therefore market orders only — a resting limit entry would sit unprotected
until it filled — and so are reduce-only closes.
"""

from __future__ import annotations

from typing import Any

from src.trading.connectors.binance.shaping import (
    as_iter as _as_iter,
    nonzero_balances as _nonzero_balances,
    obj_get as _obj_get,
    ohlcv_to_dict as _ohlcv_to_dict,
    order_to_dict as _order_to_dict,
    to_float as _to_float,
)
from src.trading.futures_guard import MARGIN_MODE, FuturesOrderError, validate_order

#: Binance's "No need to change margin type" code: already isolated.
_MARGIN_UNCHANGED = "-4046"


def perp_symbol(symbol: str) -> str:
    """Return the ccxt unified USDT-perpetual symbol (``BTC/USDT:USDT``)."""
    from src.trading.connectors.binance.shaping import normalize_symbol

    clean = (symbol or "").strip().upper()
    if clean.endswith(":USDT"):
        clean = clean[: -len(":USDT")]
    pair = normalize_symbol(clean)
    if "/" not in pair:
        raise FuturesOrderError(f"could not resolve a USDT perpetual from symbol '{symbol}'.")
    base, quote = pair.split("/", 1)
    if quote != "USDT":
        raise FuturesOrderError(f"only USDT-margined perpetuals are supported; got '{symbol}'.")
    return f"{base}/USDT:USDT"


def _envelope(cfg: Any, **payload: Any) -> dict[str, Any]:
    return {
        "status": "ok",
        "profile": cfg.profile,
        "is_testnet": cfg.is_testnet,
        "paper_guard": "host_separated",
        "market": "usdt_perpetual",
        **payload,
    }


def get_account_snapshot(cfg: Any, exchange: Any) -> dict[str, Any]:
    """Return the non-zero futures wallet balances."""
    return _envelope(cfg, host=cfg.host, balances=_nonzero_balances(exchange.fetch_balance()))


def get_positions(cfg: Any, exchange: Any) -> dict[str, Any]:
    """Return open perpetual positions with signed size and USD notional.

    ``quantity`` is signed (short < 0) and ``value_usd`` is the absolute
    notional, the two fields the live mandate gate reads for exposure. A
    position without a readable notional keeps no ``value_usd`` so the gate
    fails closed on it.
    """
    rows = []
    for item in _as_iter(exchange.fetch_positions()):
        contracts = _to_float(_obj_get(item, "contracts"))
        if not contracts:
            continue
        size = abs(contracts) * (_to_float(_obj_get(item, "contractSize")) or 1.0)
        side = str(_obj_get(item, "side", "")).lower()
        row: dict[str, Any] = {
            "symbol": _obj_get(item, "symbol"),
            "side": side,
            "quantity": -size if side == "short" else size,
            "entry_price": _to_float(_obj_get(item, "entryPrice")),
            "mark_price": _to_float(_obj_get(item, "markPrice")),
            "leverage": _to_float(_obj_get(item, "leverage")),
            "margin_mode": _obj_get(item, "marginMode"),
            "liquidation_price": _to_float(_obj_get(item, "liquidationPrice")),
            "unrealized_pnl": _to_float(_obj_get(item, "unrealizedPnl")),
            "source": "usdt_perpetual",
        }
        notional = _to_float(_obj_get(item, "notional"))
        if notional is not None and notional == notional:
            row["value_usd"] = abs(notional)
        rows.append(row)
    return _envelope(cfg, positions=rows)


def get_open_orders(cfg: Any, exchange: Any) -> dict[str, Any]:
    """Return open regular perpetual orders.

    The reduce-only stop-loss and take-profit orders live in Binance's
    separate conditional (algo) book and are not listed here.
    """
    return _envelope(cfg, open_orders=[_order_to_dict(item) for item in _as_iter(exchange.fetch_open_orders())])


def get_quote(cfg: Any, exchange: Any, symbol: str) -> dict[str, Any]:
    """Return the latest perpetual ticker for ``symbol``."""
    clean = perp_symbol(symbol)
    ticker = exchange.fetch_ticker(clean)
    return {
        "status": "ok",
        "symbol": clean,
        "quote": {
            "bid": _obj_get(ticker, "bid"),
            "ask": _obj_get(ticker, "ask"),
            "last": _obj_get(ticker, "last"),
            "mark": _obj_get(ticker, "markPrice"),
            "high": _obj_get(ticker, "high"),
            "low": _obj_get(ticker, "low"),
            "volume": _obj_get(ticker, "baseVolume"),
            "time": str(_obj_get(ticker, "timestamp", "")),
        },
    }


def get_historical_bars(
    cfg: Any, exchange: Any, symbol: str, timeframe: str, limit: int, period: str
) -> dict[str, Any]:
    """Return perpetual OHLCV bars."""
    clean = perp_symbol(symbol)
    bars = exchange.fetch_ohlcv(clean, timeframe=timeframe, limit=int(limit))
    return {"status": "ok", "symbol": clean, "period": period, "bars": [_ohlcv_to_dict(b) for b in _as_iter(bars)]}


def place_order(
    cfg: Any,
    exchange: Any,
    *,
    symbol: str,
    side: str,
    quantity: float,
    order_type: str,
    stop_loss: Any,
    take_profit: Any,
    leverage: Any,
    reduce_only: Any,
) -> dict[str, Any]:
    """Place a protected perpetual order; any failure is an error envelope.

    The caller has already checked side and size. An opening order must be a
    market order; its symbol is put on isolated margin at the requested
    leverage (both read back first), the entry is sent, and the stop-loss and
    take-profit follow as reduce-only conditional orders for the filled size.
    """
    try:
        clean = perp_symbol(symbol)
        if order_type != "market":
            raise FuturesOrderError(
                "Binance futures orders are market orders only, so protection is placed "
                "as soon as a position exists."
            )
        reference = None
        if reduce_only is not True:
            reference = _to_float(_obj_get(exchange.fetch_ticker(clean), "last"))
        order = validate_order(
            side=side,
            reference_price=reference,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage,
            reduce_only=reduce_only,
        )
    except FuturesOrderError as exc:
        return {"status": "error", "error": str(exc)}

    if order.reduce_only:
        try:
            placed = exchange.create_order(clean, order_type, order.side, quantity, None, {"reduceOnly": True})
        except Exception as exc:  # noqa: BLE001 - surface any ccxt/auth/network error as fail-closed
            return {"status": "error", "error": str(exc)}
        return _envelope(cfg, **_order_fields(placed, clean, order.side, order_type), reduce_only=True)

    try:
        _ensure_isolated(exchange, clean, order.leverage)
    except Exception as exc:  # noqa: BLE001 - never trade on unverified margin settings
        return {"status": "error", "error": f"isolated {order.leverage}x margin could not be confirmed: {exc}"}

    try:
        entry = exchange.create_order(clean, "market", order.side, quantity, None, {})
    except Exception as exc:  # noqa: BLE001 - nothing was opened
        return {"status": "error", "error": str(exc)}

    filled = _to_float(_obj_get(entry, "filled")) or quantity
    exit_side = "sell" if order.side == "buy" else "buy"
    protection: dict[str, Any] = {}
    try:
        protection["stop_loss_order_id"] = str(_obj_get(exchange.create_order(
            clean, "market", exit_side, filled, None,
            {"stopLossPrice": order.stop_loss, "reduceOnly": True, "workingType": "MARK_PRICE"},
        ), "id", ""))
        protection["take_profit_order_id"] = str(_obj_get(exchange.create_order(
            clean, "market", exit_side, filled, None,
            {"takeProfitPrice": order.take_profit, "reduceOnly": True, "workingType": "MARK_PRICE"},
        ), "id", ""))
    except Exception as exc:  # noqa: BLE001 - an unprotected position must not stay open
        return _unwind(cfg, exchange, clean, exit_side, filled, entry, protection, exc)

    return _envelope(
        cfg,
        **_order_fields(entry, clean, order.side, "market"),
        reduce_only=False,
        leverage=order.leverage,
        margin_mode=MARGIN_MODE,
        stop_loss=order.stop_loss,
        take_profit=order.take_profit,
        **protection,
    )


def cancel_order(cfg: Any, exchange: Any, order_id: str, symbol: str) -> dict[str, Any]:
    """Cancel an open regular perpetual order.

    Only the regular order book is reachable: the reduce-only stop-loss and
    take-profit orders live in Binance's conditional book, which this call never
    addresses, so cancelling cannot strip a position's protection.
    """
    try:
        clean = perp_symbol(symbol)
        cancelled = exchange.cancel_order(order_id, clean)
    except Exception as exc:  # noqa: BLE001 - surface any ccxt/auth/network error as fail-closed
        return {"status": "error", "error": str(exc)}
    return _envelope(
        cfg,
        order_id=str(_obj_get(cancelled, "id", order_id)),
        symbol=_obj_get(cancelled, "symbol", clean),
        side=str(_obj_get(cancelled, "side", "")),
        order_status=str(_obj_get(cancelled, "status", "")),
    )


def _order_fields(placed: Any, symbol: str, side: str, order_type: str) -> dict[str, Any]:
    return {
        "order_id": str(_obj_get(placed, "id", "")),
        "symbol": _obj_get(placed, "symbol", symbol),
        "side": str(_obj_get(placed, "side", side)),
        "order_type": str(_obj_get(placed, "type", order_type)),
        "order_status": str(_obj_get(placed, "status", "")),
        "amount": _obj_get(placed, "amount"),
        "filled": _obj_get(placed, "filled"),
        "price": _obj_get(placed, "price"),
    }


def _unwind(
    cfg: Any,
    exchange: Any,
    symbol: str,
    exit_side: str,
    filled: float,
    entry: Any,
    protection: dict[str, Any],
    cause: BaseException,
) -> dict[str, Any]:
    """Close an entry whose protection failed; report loudly if that fails too."""
    for key in ("stop_loss_order_id", "take_profit_order_id"):
        if protection.get(key):
            try:
                exchange.cancel_order(protection[key], symbol, {"trigger": True})
            except Exception:  # noqa: BLE001 - the reduce-only close below still flattens
                pass
    entry_id = str(_obj_get(entry, "id", ""))
    try:
        exchange.create_order(symbol, "market", exit_side, filled, None, {"reduceOnly": True})
    except Exception as close_exc:  # noqa: BLE001 - the user must act now
        return {
            "status": "error",
            "error": (
                f"UNPROTECTED POSITION on {symbol}: protection failed ({cause}) and the "
                f"emergency close failed ({close_exc}). Close it on Binance now."
            ),
            "unprotected_position": True,
            "entry_order_id": entry_id,
            "profile": cfg.profile,
        }
    return {
        "status": "error",
        "error": f"protection could not be placed ({cause}); the entry was closed at market.",
        "entry_order_id": entry_id,
        "unwound": True,
        "profile": cfg.profile,
    }


def _ensure_isolated(exchange: Any, symbol: str, leverage: int) -> None:
    """Put ``symbol`` on isolated margin at ``leverage`` and read both back."""
    try:
        exchange.set_margin_mode(MARGIN_MODE, symbol)
    except Exception as exc:  # noqa: BLE001 - only "already isolated" is benign
        if _MARGIN_UNCHANGED not in str(exc):
            raise
    mode = str(_obj_get(exchange.fetch_margin_mode(symbol), "marginMode", "")).lower()
    if mode != MARGIN_MODE:
        raise FuturesOrderError(f"margin mode is '{mode or 'unknown'}', not isolated")
    exchange.set_leverage(leverage, symbol)
    reported = _obj_get(exchange.fetch_leverage(symbol), "longLeverage")
    if _to_float(reported) != float(leverage):
        raise FuturesOrderError(f"leverage reads back as {reported}, not {leverage}x")
