"""Bybit spot trading connector via ccxt.

Account and market reads go through the ``ccxt`` library's unified ``bybit``
exchange client (REST, Bybit V5). A ``broker_sdk`` transport. Order placement
is exposed only through the order-placing profiles: paper orders reach the
testnet directly, live orders pass the user's mandate gate first.

Paper-vs-live separation is structural: the testnet (paper) and mainnet use
DIFFERENT API keys and DIFFERENT hosts (``api-testnet.bybit.com`` vs
``api.<hostname>``), so a testnet key cannot reach the live host. The
configured ``profile`` selects the host (via ``set_sandbox_mode``) and is the
authoritative discriminator; the client's resolved endpoints are checked
against it before every call. The connector trades spot only — derivatives
categories are never selected — and spot has no positions, so
``get_positions`` shapes the non-zero balances from ``fetch_balance`` as rows.
"""
