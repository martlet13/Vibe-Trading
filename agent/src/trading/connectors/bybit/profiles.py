"""Built-in Bybit connector profiles.

Read-only paper (testnet) and live profiles plus order-placing paper and live
profiles for spot, and order-placing paper and live profiles for USDT
perpetuals (``futures``). Paper and live use different key pairs and different
hosts, so they are distinct profiles. The order-placing live profile carries an
``orders.place.requires_mandate`` capability — the higher trading layer must
gate it behind the user's mandate before any order reaches the connector.
"""

from __future__ import annotations

from src.trading.types import READ_CAPABILITIES, TradingProfile

BYBIT_PROFILES: tuple[TradingProfile, ...] = (
    TradingProfile(
        id="bybit-paper-sdk",
        connector="bybit",
        label="Bybit Spot Testnet · ccxt",
        environment="paper",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES,
        readonly=True,
        config={"profile": "paper"},
        notes=(
            "Reads a Bybit spot testnet account (api-testnet.bybit.com) via ccxt. "
            "Testnet keys cannot reach the live host."
        ),
    ),
    TradingProfile(
        id="bybit-live-sdk-readonly",
        connector="bybit",
        label="Bybit Live · ccxt Read-Only",
        environment="live",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES,
        readonly=True,
        config={"profile": "live-readonly"},
        notes=(
            "Reads a Bybit spot live account (api.bybit.com, or the regional host set "
            "as hostname). Order placement is not exposed in this profile."
        ),
    ),
    TradingProfile(
        id="bybit-paper-trade",
        connector="bybit",
        label="Bybit Spot Testnet · ccxt Trading",
        environment="paper",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES + ("orders.place",),
        readonly=False,
        config={"profile": "paper"},
        notes=(
            "Places and cancels spot orders on a Bybit testnet account "
            "(api-testnet.bybit.com) via ccxt. Testnet keys cannot reach the live host, "
            "so no order from this profile can touch real funds."
        ),
    ),
    TradingProfile(
        id="bybit-live-trade",
        connector="bybit",
        label="Bybit Spot Live · ccxt Trading",
        environment="live",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES + ("orders.place.requires_mandate",),
        readonly=False,
        config={"profile": "live"},
        notes=(
            "Places and cancels spot orders on a Bybit live account via ccxt. Live order "
            "placement must be gated by the user's mandate; the "
            "orders.place.requires_mandate capability signals that requirement upstream. "
            "Derivatives are not traded."
        ),
    ),
    TradingProfile(
        id="bybit-futures-paper-trade",
        connector="bybit",
        label="Bybit USDT Perpetuals Testnet · ccxt Trading",
        environment="paper",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES + ("orders.place",),
        readonly=False,
        config={"profile": "paper", "futures": True},
        notes=(
            "Trades Bybit USDT perpetuals on the testnet (api-testnet.bybit.com). "
            "USDT-margined perpetuals only, on isolated margin at 1-5x; every new "
            "position carries a stop-loss and a take-profit or the order is refused."
            " The stop-loss and take-profit are attached to the opening order itself."
        ),
    ),
    TradingProfile(
        id="bybit-futures-live-trade",
        connector="bybit",
        label="Bybit USDT Perpetuals Live · ccxt Trading",
        environment="live",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES + ("orders.place.requires_mandate",),
        readonly=False,
        config={"profile": "live", "futures": True},
        notes=(
            "Trades Bybit USDT perpetuals on the live account (api.bybit.com) behind the "
            "user's mandate and kill switch. USDT-margined perpetuals only, on isolated "
            "margin at 1-5x; every new position carries a stop-loss and a take-profit or "
            "the order is refused."
            " The stop-loss and take-profit are attached to the opening order itself."
        ),
    ),
)
