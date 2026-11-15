"""Standalone webhook receiver for AI Agent Lens packets.

Serves exactly two paths: ``GET /health`` and ``POST /webhook?token=...``.
TradingView cannot send custom headers, so the shared token travels in the
query string; the handler never logs a request line for that reason. The
server refuses to start without a token of at least :data:`MIN_TOKEN_LENGTH`
characters and binds to loopback by default — reach it through an HTTPS
tunnel rather than by opening the port.
"""

from __future__ import annotations

import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

from src.tradingview.store import MAX_BODY_BYTES, PacketError, parse_packet, save_packet

MIN_TOKEN_LENGTH = 16
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8787


class ReceiverConfigError(ValueError):
    """Raised when the receiver cannot be started safely."""


def _handler(token: str, on_packet: Callable[[dict], None] | None) -> type[BaseHTTPRequestHandler]:
    expected = token.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        server_version = "VibeTradingTV"
        sys_version = ""

        def _reply(self, code: int, message: str) -> None:
            body = json.dumps({"ok": code == 200, "message": message}).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - http.server naming
            if urlparse(self.path).path == "/health":
                self._reply(200, "TradingView receiver is running")
            else:
                self._reply(404, "not found")

        def do_POST(self) -> None:  # noqa: N802 - http.server naming
            url = urlparse(self.path)
            if url.path != "/webhook":
                return self._reply(404, "not found")
            supplied = parse_qs(url.query).get("token", [""])[0].encode("utf-8")
            if not hmac.compare_digest(supplied, expected):
                return self._reply(401, "bad token")
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                return self._reply(400, "invalid Content-Length")
            if length <= 0 or length > MAX_BODY_BYTES:
                return self._reply(400, "empty or oversized body")
            try:
                packet = parse_packet(self.rfile.read(length))
            except PacketError as exc:
                return self._reply(400, str(exc))
            save_packet(packet)
            if on_packet is not None:
                on_packet(packet)
            self._reply(200, "saved")

        def log_message(self, *args: object) -> None:
            """Stay silent: the request line carries the token."""

    return Handler


def build_server(
    token: str,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    on_packet: Callable[[dict], None] | None = None,
) -> ThreadingHTTPServer:
    """Create the receiver without starting it.

    Args:
        token: Shared secret TradingView sends as ``?token=``.
        host: Bind address; loopback by default.
        port: Listen port; ``0`` picks a free one.
        on_packet: Called with each stored packet (used for console output).

    Returns:
        A server ready for ``serve_forever()``.

    Raises:
        ReceiverConfigError: The token is missing or too short to be a secret.
    """
    clean = (token or "").strip()
    if len(clean) < MIN_TOKEN_LENGTH:
        raise ReceiverConfigError(
            f"a webhook token of at least {MIN_TOKEN_LENGTH} characters is required; "
            "generate one with `vibe-trading tradingview new-token`."
        )
    return ThreadingHTTPServer((host, port), _handler(clean, on_packet))
