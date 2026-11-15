"""CLI for the TradingView (AI Agent Lens) webhook receiver and its packets."""

import json
import secrets

from src.config.accessor import get_env_config
from src.tradingview.receiver import DEFAULT_HOST, DEFAULT_PORT, ReceiverConfigError, build_server
from src.tradingview.store import list_charts, load_latest, store_dir


def add_parser(subparsers):
    parser = subparsers.add_parser(
        "tradingview", help="Receive TradingView AI Agent Lens packets; never places orders"
    )
    commands = parser.add_subparsers(dest="tradingview_command", required=True)
    serve = commands.add_parser("serve", help="Run the webhook receiver")
    serve.add_argument("--host", default=DEFAULT_HOST, help="Bind address (default: loopback)")
    serve.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve.add_argument(
        "--token", default="", help="Webhook token; defaults to TRADINGVIEW_WEBHOOK_TOKEN"
    )
    commands.add_parser("new-token", help="Print a random webhook token")
    commands.add_parser("charts", help="List charts that have sent a packet")
    latest = commands.add_parser("latest", help="Show the newest packet of one chart")
    latest.add_argument("symbol")
    latest.add_argument("timeframe")


def _serve(args) -> int:
    token = args.token or get_env_config().api.tradingview_webhook_token
    try:
        server = build_server(
            token,
            host=args.host,
            port=args.port,
            on_packet=lambda packet: print(
                f"[{packet.get('time')}] {packet.get('event')} {packet.get('state_code')}", flush=True
            ),
        )
    except ReceiverConfigError as exc:
        print(f"error: {exc}")
        return 2
    host, port = server.server_address[:2]
    print(f"TradingView receiver on http://{host}:{port}/webhook  (saving to {store_dir()})")
    print("Expose it through an HTTPS tunnel and use  https://<tunnel>/webhook?token=<token>  in the alert.")
    if host not in ("127.0.0.1", "localhost", "::1"):
        print("warning: bound to a non-loopback address; anyone who can reach it can try tokens.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def dispatch(args) -> int:
    if args.tradingview_command == "serve":
        return _serve(args)
    if args.tradingview_command == "new-token":
        print(secrets.token_urlsafe(32))
        return 0
    if args.tradingview_command == "charts":
        print(json.dumps(list_charts(), indent=2))
        return 0
    packet = load_latest(args.symbol, args.timeframe)
    if packet is None:
        print(f"error: no packet for {args.symbol} on {args.timeframe}")
        return 1
    print(json.dumps(packet, indent=2))
    return 0
