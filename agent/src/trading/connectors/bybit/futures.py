"""Bybit USDT perpetual futures through the Bybit connector's futures profiles.

Reached only from :mod:`src.trading.connectors.bybit.sdk` when the selected
profile declares ``futures``. The client, its host check and the testnet/live
split are the spot connector's; only the ccxt category changes (``swap`` /
``linear``). Orders follow :mod:`src.trading.futures_guard`: isolated margin,
at most 5x, and a stop-loss plus take-profit attached to the opening order
itself, so Bybit never holds a new position without both.
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

#: Bybit's "leverage not modified" retCode: the symbol already uses it.
_LEVERAGE_UNCHANGED = "110043"


def perp_symbol(symbol: str) -> str:
    """Return the ccxt unified USDT-perpetual symbol (``BTC/USDT:USDT``)."""
    from src.trading.connectors.bybit.sdk import normalize_symbol

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
    """Return the non-zero wallet balances that margin the perpetuals."""
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
            "stop_loss": _to_float(_obj_get(item, "stopLossPrice")),
            "take_profit": _to_float(_obj_get(item, "takeProfitPrice")),
            "unrealized_pnl": _to_float(_obj_get(item, "unrealizedPnl")),
            "source": "usdt_perpetual",
        }
        notional = _to_float(_obj_get(item, "notional"))
        if notional is not None and notional == notional:
            row["value_usd"] = abs(notional)
        rows.append(row)
    return _envelope(cfg, positions=rows)


def get_open_orders(cfg: Any, exchange: Any) -> dict[str, Any]:
    """Return open perpetual orders."""
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
    limit_price: float | None,
    time_in_force: str | None,
    stop_loss: Any,
    take_profit: Any,
    leverage: Any,
    reduce_only: Any,
) -> dict[str, Any]:
    """Place a protected perpetual order; any failure is an error envelope.

    The caller has already checked side, type, size and limit price. An
    opening order is validated against the protection rules, its symbol is put
    on isolated margin at the requested leverage (both read back before the
    order), and the stop-loss and take-profit are attached to the order so the
    exchange creates the position and its protection together.
    """
    try:
        clean = perp_symbol(symbol)
        reference = limit_price
        if reference is None and not reduce_only:
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

    params: dict[str, Any] = {}
    if time_in_force:
        params["timeInForce"] = time_in_force
    if order.reduce_only:
        params["reduceOnly"] = True
    else:
        try:
            _ensure_isolated(exchange, clean, order.leverage)
        except Exception as exc:  # noqa: BLE001 - never trade on unverified margin settings
            return {"status": "error", "error": f"isolated {order.leverage}x margin could not be confirmed: {exc}"}
        params["stopLoss"] = {"triggerPrice": order.stop_loss}
        params["takeProfit"] = {"triggerPrice": order.take_profit}
        params["tpslMode"] = "Full"

    try:
        placed = exchange.create_order(clean, order_type, order.side, quantity, limit_price, params)
    except Exception as exc:  # noqa: BLE001 - surface any ccxt/auth/network error as fail-closed
        return {"status": "error", "error": str(exc)}

    return _envelope(
        cfg,
        order_id=str(_obj_get(placed, "id", "")),
        symbol=_obj_get(placed, "symbol", clean),
        side=str(_obj_get(placed, "side", order.side)),
        order_type=str(_obj_get(placed, "type", order_type)),
        order_status=str(_obj_get(placed, "status", "")),
        amount=_obj_get(placed, "amount"),
        price=_obj_get(placed, "price"),
        reduce_only=order.reduce_only,
        leverage=order.leverage,
        margin_mode=None if order.reduce_only else MARGIN_MODE,
        stop_loss=order.stop_loss,
        take_profit=order.take_profit,
    )


def cancel_order(cfg: Any, exchange: Any, order_id: str, symbol: str) -> dict[str, Any]:
    """Cancel an open perpetual order.

    Bybit keeps an attached stop-loss and take-profit on the position, not as
    separate orders, so cancelling an order can never strip a position's
    protection.
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


def _ensure_isolated(exchange: Any, symbol: str, leverage: int) -> None:
    """Put ``symbol`` on isolated margin at ``leverage`` and read both back.

    On a unified trading account Bybit sets the margin mode per account, so
    switching to isolated applies to the whole account; Bybit refuses the
    switch while cross positions are open, and the order is then refused.
    """
    mode = str(_obj_get(exchange.fetch_margin_mode(symbol), "marginMode", "")).lower()
    if mode != MARGIN_MODE:
        exchange.set_margin_mode(MARGIN_MODE, symbol, {"leverage": leverage})
        mode = str(_obj_get(exchange.fetch_margin_mode(symbol), "marginMode", "")).lower()
        if mode != MARGIN_MODE:
            raise FuturesOrderError(f"margin mode is '{mode or 'unknown'}', not isolated")
    try:
        exchange.set_leverage(leverage, symbol)
    except Exception as exc:  # noqa: BLE001 - only "already at this leverage" is benign
        if _LEVERAGE_UNCHANGED not in str(exc):
            raise
    reported = _obj_get(exchange.fetch_leverage(symbol), "longLeverage")
    if _to_float(reported) != float(leverage):
        raise FuturesOrderError(f"leverage reads back as {reported}, not {leverage}x")
