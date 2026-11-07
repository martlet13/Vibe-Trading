"""Built-in Binance connector profiles.

Read-only paper (testnet) and live profiles plus order-placing paper and live
profiles for spot, and order-placing paper and live profiles for USD-M
perpetuals (``futures``). Paper and live use different key pairs and different hosts, so they
are distinct profiles. The order-placing live profile carries an
``orders.place.requires_mandate`` capability — the higher trading layer must
gate it behind the user's mandate before any order reaches the connector.
"""

from __future__ import annotations

from src.trading.types import READ_CAPABILITIES, TradingProfile

BINANCE_PROFILES: tuple[TradingProfile, ...] = (
    TradingProfile(
        id="binance-paper-sdk",
        connector="binance",
        label="Binance Spot Testnet · ccxt",
        environment="paper",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES,
        readonly=True,
        config={"profile": "paper"},
        notes=(
            "Reads a Binance spot testnet account (testnet.binance.vision) via ccxt. "
            "Testnet keys cannot reach the live host."
        ),
    ),
    TradingProfile(
        id="binance-live-sdk-readonly",
        connector="binance",
        label="Binance Live · ccxt Read-Only",
        environment="live",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES,
        readonly=True,
        config={"profile": "live-readonly"},
        notes=(
            "Reads a Binance spot live account (api.binance.com). Account and "
            "position reads may explicitly select USD-M Shadow Account mode "
            "(fapi.binance.com); that mode exposes no order or market-data surface."
        ),
    ),
    TradingProfile(
        id="binance-paper-trade",
        connector="binance",
        label="Binance Spot Testnet · ccxt Trading",
        environment="paper",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES + ("orders.place",),
        readonly=False,
        config={"profile": "paper"},
        notes=(
            "Places and cancels orders on a Binance spot testnet account "
            "(testnet.binance.vision) via ccxt. Testnet keys cannot reach the live host, "
            "so no order from this profile can touch real funds."
        ),
    ),
    TradingProfile(
        id="binance-live-trade",
        connector="binance",
        label="Binance Spot Live · ccxt Trading",
        environment="live",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES + ("orders.place.requires_mandate",),
        readonly=False,
        config={"profile": "live"},
        notes=(
            "Places and cancels orders on a Binance spot live account (api.binance.com) "
            "via ccxt. Live order placement must be gated by the user's mandate; the "
            "orders.place.requires_mandate capability signals that requirement upstream."
        ),
    ),
    TradingProfile(
        id="binance-futures-paper-trade",
        connector="binance",
        label="Binance USDT Perpetuals Testnet · ccxt Trading",
        environment="paper",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES + ("orders.place",),
        readonly=False,
        config={"profile": "paper", "futures": True},
        notes=(
            "Trades Binance USDT perpetuals on the testnet (testnet.binancefuture.com). "
            "USDT-margined perpetuals only, on isolated margin at 1-5x; every new "
            "position carries a stop-loss and a take-profit or the order is refused."
            " Entries are market orders; protection follows as reduce-only conditional "
            "orders and the entry is closed at market if either is refused."
        ),
    ),
    TradingProfile(
        id="binance-futures-live-trade",
        connector="binance",
        label="Binance USDT Perpetuals Live · ccxt Trading",
        environment="live",
        transport="broker_sdk",
        capabilities=READ_CAPABILITIES + ("orders.place.requires_mandate",),
        readonly=False,
        config={"profile": "live", "futures": True},
        notes=(
            "Trades Binance USDT perpetuals on the live account (fapi.binance.com) behind the "
            "user's mandate and kill switch. USDT-margined perpetuals only, on isolated "
            "margin at 1-5x; every new position carries a stop-loss and a take-profit or "
            "the order is refused."
            " Entries are market orders; protection follows as reduce-only conditional "
            "orders and the entry is closed at market if either is refused."
        ),
    ),
)
