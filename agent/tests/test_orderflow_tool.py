"""The ``orderflow`` tool: venue routing, coverage honesty, validation, grounding.

The network functions are monkeypatched; no test opens a socket.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.agent.grounding import GroundingLedger
from src.tools import orderflow_tool
from src.tools.orderflow_tool import OrderFlowTool

T0 = 1_794_830_400_000  # 2026-11-16T12:00:00Z


def _book(bid=64000.0, ask=64000.5, wall=0.5):
    bids = [[bid - i * 0.5, 0.5] for i in range(10)]
    bids[5][1] = wall
    asks = [[ask + i * 0.5, 0.5] for i in range(10)]
    return {"bids": bids, "asks": asks}


def _trades(n=30):
    rows = []
    for i in range(n):
        rows.append({"id": str(i), "timestamp": T0 + i * 20_000, "price": 64000 + (i % 7) * 3,
                     "amount": 0.1 + (i % 3) * 0.05, "side": "buy" if i % 3 else "sell"})
    return rows


@pytest.fixture
def venue(monkeypatch):
    calls: dict[str, list] = {"book": [], "trades": [], "stream": []}

    def fetch_book(exchange, market, symbol, limit):
        calls["book"].append((exchange, market, symbol, limit))
        return _book(wall=5.0)

    def fetch_trades(exchange, market, symbol, limit):
        calls["trades"].append((exchange, market, symbol, limit))
        return _trades()

    def stream(exchange, market, symbol, seconds):
        calls["stream"].append((exchange, market, symbol, seconds))
        return _trades(), [_book(wall=5.0), _book(wall=5.0)]

    monkeypatch.setattr(orderflow_tool, "_fetch_book", fetch_book)
    monkeypatch.setattr(orderflow_tool, "_fetch_trades", fetch_trades)
    monkeypatch.setattr(orderflow_tool, "_stream", stream)
    monkeypatch.setattr(orderflow_tool.time, "sleep", lambda _s: None)
    return calls


def _run(**kwargs: Any) -> dict[str, Any]:
    return json.loads(OrderFlowTool().execute(**kwargs))


def test_defaults_read_bybit_linear_perp_over_rest(venue):
    out = _run(symbol="btc-usdt")
    assert out["ok"] is True
    assert venue["book"] == [("bybit", "perp", "BTC/USDT:USDT", 200)] * 3
    assert venue["trades"] == [("bybit", "perp", "BTC/USDT:USDT", 1000)]
    data = out["data"]
    assert data["coverage"]["source"] == "rest"
    assert data["coverage"]["trades_used"] == 30
    assert data["coverage"]["seconds_covered"] == 580.0
    assert data["book"]["mid_price"] == pytest.approx(64000.25)
    assert data["book"]["walls"][0]["side"] == "bid"
    assert data["book"]["walls"][0]["persistence"] == 1.0
    flow = data["flow"]
    assert flow["delta"] == pytest.approx(flow["buy_volume"] - flow["sell_volume"])
    assert flow["bars"][-1]["cvd"] == pytest.approx(flow["delta"])
    assert sum("footprint" in bar for bar in flow["bars"]) == 3
    assert data["profile"]["value_area_low"] <= data["profile"]["poc_price"] < data["profile"]["value_area_high"]


@pytest.mark.parametrize(("exchange", "market", "symbol", "limit"), [
    ("bybit", "spot", "ETH/USDT", 60),
    ("binance", "perp", "ETH/USDT:USDT", 1000),
    ("binance", "spot", "ETH/USDT", 1000),
])
def test_venue_and_market_routing(venue, exchange, market, symbol, limit):
    assert _run(symbol="ETH-USDT", exchange=exchange, market=market, snapshots=1)["ok"]
    assert venue["trades"] == [(exchange, market, symbol, limit)]


def test_a_full_rest_batch_is_flagged_as_incomplete(venue, monkeypatch):
    monkeypatch.setattr(orderflow_tool, "_fetch_trades", lambda *a: _trades(60))
    out = _run(symbol="BTC-USDT", market="spot", snapshots=1)
    assert "maximum batch" in out["data"]["coverage"]["note"]


def test_collect_seconds_streams_instead_of_rest(venue):
    out = _run(symbol="BTC-USDT", collect_seconds=45)
    assert venue["stream"] == [("bybit", "perp", "BTC/USDT:USDT", 45)]
    assert venue["book"] == [] and venue["trades"] == []
    assert out["data"]["coverage"]["source"] == "websocket"
    assert out["data"]["coverage"]["book_snapshots"] == 2


def test_no_trades_keeps_the_book_and_says_so(venue, monkeypatch):
    monkeypatch.setattr(orderflow_tool, "_fetch_trades", lambda *a: [])
    out = _run(symbol="BTC-USDT", snapshots=1)
    assert out["ok"] and out["data"]["flow"] is None and out["data"]["profile"] is None
    assert "no trades" in out["data"]["coverage"]["note"]


@pytest.mark.parametrize("book", [{"bids": [], "asks": [[1, 1]]}, {"bids": [[2, 1]], "asks": [[1, 1]]}])
def test_one_sided_or_crossed_books_are_refused(venue, monkeypatch, book):
    monkeypatch.setattr(orderflow_tool, "_fetch_book", lambda *a: book)
    assert _run(symbol="BTC-USDT", snapshots=1)["ok"] is False


@pytest.mark.parametrize("kwargs", [
    {"symbol": "BTCUSDT"}, {"symbol": "BTC-USDT", "exchange": "okx"}, {"symbol": "BTC-USDT", "market": "option"},
    {"symbol": "BTC-USDT", "bar_minutes": 2}, {"symbol": "BTC-USDT", "collect_seconds": 999},
    {"symbol": "BTC-USDT", "snapshots": 0}, {"symbol": "BTC-USDT", "footprint_bars": 11},
    {"symbol": "BTC-USDT", "wall_multiplier": 0.5}, {"symbol": "BTC-USDT", "collect_seconds": True},
])
def test_invalid_arguments_are_refused_before_any_request(venue, kwargs):
    assert _run(**kwargs)["ok"] is False
    assert venue == {"book": [], "trades": [], "stream": []}


def test_request_errors_become_an_error_envelope(venue, monkeypatch):
    def boom(*_a):
        raise RuntimeError("451 restricted location")

    monkeypatch.setattr(orderflow_tool, "_fetch_book", boom)
    out = _run(symbol="BTC-USDT")
    assert out["ok"] is False and "451" in out["error"]


def test_tool_metadata():
    tool = OrderFlowTool()
    assert tool.name == "orderflow" and tool.is_readonly and tool.repeatable


def test_figures_quoted_from_the_tool_are_grounded(venue, tmp_path: Path):
    result = OrderFlowTool().execute(symbol="BTC-USDT")
    data = json.loads(result)["data"]
    ledger = GroundingLedger(run_dir=tmp_path, user_message="Покажи ордерфлоу BTC-USDT")
    ledger.ingest_tool_result(tool_name="orderflow", arguments={"symbol": "BTC-USDT"},
                              result=result, call_id="of-1", success=True)
    poc = data["profile"]["poc_price"]
    delta = round(data["flow"]["delta"], 2)
    answer = (
        f"POC {poc:.0f} USDT, найкращий bid {data['book']['best_bid']:.1f} USDT, дельта {delta} BTC.\n\n"
        "```figures\n"
        f"{poc:.0f} | observed | POC | of-1\n"
        f"{data['book']['best_bid']:.1f} | observed | best bid | of-1\n"
        f"{delta} | observed | delta | of-1\n"
        "```"
    )
    outcome = ledger.validate_final_answer(answer)
    assert outcome.valid, outcome.issues
    wrong = answer.replace(f"POC {poc:.0f}", f"POC {poc + 500:.0f}").replace(f"{poc:.0f} | observed", f"{poc + 500:.0f} | observed")
    assert not ledger.validate_final_answer(wrong).valid
