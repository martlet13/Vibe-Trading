"""Validate AI Agent Lens packets and keep them under the runtime root.

Layout (``~/.vibe-trading/tradingview/``)::

    latest/<SYMBOL>_<TF>.json   newest packet per chart
    history.jsonl               every packet, one per line (rotated once)

Validation is strict because a packet arrives from the internet and is later
shown to a model. Only the schema's own top-level sections are kept, numbers
must be finite, and a string may hold only the characters the schema's enums,
symbols and timestamps use — no whitespace — so a packet cannot carry prose.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.paths import get_runtime_root

SCHEMA = "ai-agent-lens/v1"
STORE_DIRNAME = "tradingview"
MAX_BODY_BYTES = 64 * 1024
#: history.jsonl is rotated to history.1.jsonl once it passes this size.
MAX_HISTORY_BYTES = 8 * 1024 * 1024

EVENTS = frozenset({"BAR_CLOSE", "LONG_SETUP", "SHORT_SETUP", "TRADE_UPDATE"})
_SECTIONS = frozenset({
    "candle", "recent_candles", "regime", "trend", "momentum", "strength", "volatility",
    "volume", "structure", "levels", "scores", "trade", "last_setup",
})
_HEADER = ("schema", "event", "symbol", "timeframe", "time", "state_code")

_STRING = re.compile(r"[A-Za-z0-9_.:|/+\-]{0,96}")
_SYMBOL = re.compile(r"[A-Za-z0-9_.:!/\-]{1,48}")
_TIMEFRAME = re.compile(r"[0-9]{0,4}[SDWM]?")
_MAX_DEPTH = 4
_MAX_ITEMS = 64

_write_lock = threading.Lock()


class PacketError(ValueError):
    """Raised when a packet is not a valid AI Agent Lens packet."""


def store_dir() -> Path:
    """Return the directory packets are kept in."""
    return get_runtime_root() / STORE_DIRNAME


def chart_key(symbol: str, timeframe: str) -> str:
    """Return the file-safe key for one chart (symbol on one timeframe)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{symbol}_{timeframe}")[:80]


def _clean(value: Any, depth: int) -> Any:
    """Return ``value`` if it holds only schema-shaped data; raise otherwise."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise PacketError("packet contains a non-finite number")
        return value
    if isinstance(value, str):
        if not _STRING.fullmatch(value):
            raise PacketError("packet contains a string outside the schema's character set")
        return value
    if depth >= _MAX_DEPTH:
        raise PacketError("packet is nested too deeply")
    if isinstance(value, list):
        if len(value) > _MAX_ITEMS:
            raise PacketError("packet list is too long")
        return [_clean(item, depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > _MAX_ITEMS:
            raise PacketError("packet object has too many fields")
        cleaned = {}
        for key, item in value.items():
            if not isinstance(key, str) or not _STRING.fullmatch(key) or not key:
                raise PacketError("packet contains an invalid field name")
            cleaned[key] = _clean(item, depth + 1)
        return cleaned
    raise PacketError("packet contains an unsupported value")


def parse_packet(raw: bytes) -> dict[str, Any]:
    """Parse and validate one webhook body.

    Args:
        raw: The request body as received.

    Returns:
        The packet restricted to the schema's header fields and sections.

    Raises:
        PacketError: The body is not a well-formed ``ai-agent-lens/v1`` packet.
    """
    if not raw or len(raw) > MAX_BODY_BYTES:
        raise PacketError("empty or oversized body")
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise PacketError("body is not valid JSON") from exc
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise PacketError(f"expected schema {SCHEMA}")

    symbol, timeframe, event = data.get("symbol"), data.get("timeframe"), data.get("event")
    if not isinstance(symbol, str) or not _SYMBOL.fullmatch(symbol):
        raise PacketError("symbol is missing or malformed")
    if not isinstance(timeframe, str) or not timeframe or not _TIMEFRAME.fullmatch(timeframe):
        raise PacketError("timeframe is missing or malformed")
    if event not in EVENTS:
        raise PacketError("event is missing or unknown")
    if packet_time(data) is None:
        raise PacketError("time is missing or not an ISO 8601 timestamp")

    packet = {key: _clean(data.get(key), 1) for key in _HEADER}
    for key in sorted(_SECTIONS & data.keys()):
        packet[key] = _clean(data[key], 1)
    return packet


def packet_time(packet: dict[str, Any]) -> datetime | None:
    """Return the packet's candle close time as an aware UTC datetime."""
    value = packet.get("time")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def save_packet(packet: dict[str, Any], *, now: datetime | None = None) -> Path:
    """Store a validated packet as its chart's latest and append it to history.

    Args:
        packet: A packet returned by :func:`parse_packet`.
        now: Receipt time; defaults to the current UTC time.

    Returns:
        The path of the chart's ``latest`` file.
    """
    stamped = dict(packet)
    stamped["_received_at"] = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    root = store_dir()
    latest_dir = root / "latest"
    target = latest_dir / f"{chart_key(packet['symbol'], packet['timeframe'])}.json"
    with _write_lock:
        latest_dir.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(stamped, indent=2), encoding="utf-8")
        os.replace(tmp, target)
        history = root / "history.jsonl"
        if history.exists() and history.stat().st_size > MAX_HISTORY_BYTES:
            os.replace(history, root / "history.1.jsonl")
        with history.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(stamped) + "\n")
    return target


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def list_charts() -> list[dict[str, Any]]:
    """Return one summary row per chart that has sent a packet, newest first."""
    rows = []
    for path in sorted((store_dir() / "latest").glob("*.json")):
        packet = _read(path)
        if packet is None:
            continue
        rows.append({
            "symbol": packet.get("symbol"),
            "timeframe": packet.get("timeframe"),
            "event": packet.get("event"),
            "time": packet.get("time"),
            "received_at": packet.get("_received_at"),
            "state_code": packet.get("state_code"),
        })
    return sorted(rows, key=lambda row: str(row.get("received_at") or ""), reverse=True)


def load_latest(symbol: str, timeframe: str) -> dict[str, Any] | None:
    """Return the newest packet for one chart, or ``None`` when there is none."""
    return _read(store_dir() / "latest" / f"{chart_key(symbol, timeframe)}.json")


def load_history(symbol: str, timeframe: str, limit: int) -> list[dict[str, Any]]:
    """Return up to ``limit`` most recent packets for one chart, oldest first."""
    path = store_dir() / "history.jsonl"
    if not path.exists() or limit <= 0:
        return []
    matches: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                packet = json.loads(line)
            except ValueError:
                continue
            if isinstance(packet, dict) and packet.get("symbol") == symbol and packet.get("timeframe") == timeframe:
                matches.append(packet)
    return matches[-limit:]
