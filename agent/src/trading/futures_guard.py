"""Shared protection rules for USDT-margined perpetual futures orders.

Every futures connector applies these checks before it touches the exchange,
on paper and live alike. They are hard limits of the connector, not mandate
settings, so neither a mandate nor a tool argument can widen them:

* leverage is a whole multiple from 1 to :data:`MAX_LEVERAGE`;
* margin is isolated per position, never cross;
* every order that opens or adds to a position carries both a stop-loss and
  a take-profit on the correct side of the entry reference price.

An order flagged ``reduce_only`` only shrinks an existing position, so it
needs no protection and changes no leverage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: Highest leverage any futures order may use.
MAX_LEVERAGE = 5
#: The only margin mode a futures position may use.
MARGIN_MODE = "isolated"


class FuturesOrderError(ValueError):
    """Raised when a futures order breaks a protection rule."""


@dataclass(frozen=True)
class FuturesOrder:
    """A validated futures order intent.

    Attributes:
        side: ``"buy"`` (long / close short) or ``"sell"`` (short / close long).
        leverage: Whole leverage multiple for an opening order, ``None`` for a
            reduce-only order.
        stop_loss: Stop-loss trigger price, ``None`` for a reduce-only order.
        take_profit: Take-profit trigger price, ``None`` for a reduce-only order.
        reduce_only: Whether the order may only shrink an existing position.
    """

    side: str
    leverage: int | None
    stop_loss: float | None
    take_profit: float | None
    reduce_only: bool


def _price(value: Any, name: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FuturesOrderError(f"{name} must be a positive number.") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise FuturesOrderError(f"{name} must be a positive number.")
    return parsed


def parse_leverage(value: Any) -> int:
    """Return a whole leverage between 1 and :data:`MAX_LEVERAGE`.

    A missing value means 1x. Fractional leverage is rejected rather than
    rounded, so the multiple sent to the exchange is the one the caller wrote.
    """
    if value is None or value == "":
        return 1
    if isinstance(value, bool):
        raise FuturesOrderError("leverage must be a whole number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FuturesOrderError("leverage must be a whole number.") from exc
    if not math.isfinite(parsed) or parsed != int(parsed):
        raise FuturesOrderError("leverage must be a whole number.")
    leverage = int(parsed)
    if not 1 <= leverage <= MAX_LEVERAGE:
        raise FuturesOrderError(f"leverage must be between 1 and {MAX_LEVERAGE}x; got {leverage}x.")
    return leverage


def validate_order(
    *,
    side: str,
    reference_price: float | None,
    stop_loss: Any = None,
    take_profit: Any = None,
    leverage: Any = None,
    reduce_only: Any = False,
) -> FuturesOrder:
    """Validate one futures order against the protection rules.

    Args:
        side: ``"buy"`` or ``"sell"``.
        reference_price: The entry reference — the limit price, or the current
            market price for a market order. Required for an opening order.
        stop_loss: Stop-loss trigger price; required unless ``reduce_only``.
        take_profit: Take-profit trigger price; required unless ``reduce_only``.
        leverage: Whole multiple from 1 to :data:`MAX_LEVERAGE`; default 1.
        reduce_only: The order may only shrink an existing position.

    Returns:
        The validated order.

    Raises:
        FuturesOrderError: Any rule is broken; nothing may be sent.
    """
    side_clean = str(side or "").strip().lower()
    if side_clean not in ("buy", "sell"):
        raise FuturesOrderError("side must be 'buy' or 'sell'.")
    if not isinstance(reduce_only, bool):
        raise FuturesOrderError("reduce_only must be true or false.")

    stop = _price(stop_loss, "stop_loss")
    target = _price(take_profit, "take_profit")

    if reduce_only:
        if stop is not None or target is not None:
            raise FuturesOrderError(
                "a reduce_only order closes exposure; it takes no stop_loss or take_profit."
            )
        if leverage not in (None, ""):
            raise FuturesOrderError("a reduce_only order does not change leverage.")
        return FuturesOrder(side=side_clean, leverage=None, stop_loss=None, take_profit=None, reduce_only=True)

    if stop is None or target is None:
        raise FuturesOrderError(
            "every futures position needs both stop_loss and take_profit; "
            "pass both, or set reduce_only to close exposure."
        )
    reference = _price(reference_price, "the entry reference price")
    if reference is None:
        raise FuturesOrderError("the entry reference price is unavailable; the order cannot be protected.")
    if side_clean == "buy" and not stop < reference < target:
        raise FuturesOrderError(
            f"a long needs stop_loss < entry < take_profit; got {stop} / {reference} / {target}."
        )
    if side_clean == "sell" and not target < reference < stop:
        raise FuturesOrderError(
            f"a short needs take_profit < entry < stop_loss; got {target} / {reference} / {stop}."
        )
    return FuturesOrder(
        side=side_clean,
        leverage=parse_leverage(leverage),
        stop_loss=stop,
        take_profit=target,
        reduce_only=False,
    )
