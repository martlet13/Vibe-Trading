"""Figures an MCP tool returned as a markdown table ground the answer quoting them.

Regression: tvscreener's MCP server answers ``get_top_movers`` with
``df.to_markdown()``. The numeric-leaf walk only saw one string, so every price,
change and volume the model quoted from the table had no evidence behind it and
the gate replaced each with "(omitted※)".

What stays closed: only cells of a well-formed pipe table count, only MCP tools
are read this way, a wrong value is still refused, a volume never reads as
money, and a figure is still not evidence before the call that showed it.

Every value here is synthetic; no market data lives in the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.agent.grounding import GroundingLedger
from src.agent.grounding.figures import scan_figures

TABLE = (
    "Top gainers:\n\n"
    "| Symbol | Name | Price | Change % | Volume | Volume 24h in USD | Market Capitalization |\n"
    "|:--|:--|--:|--:|--:|--:|--:|\n"
    "| COINBASE:SOLUSD | SOLUSD | 184.23 | 7.41 | 3.21e+08 | 5.9e+10 | 9.81234e+10 |\n"
    "| BINANCE:BNBBTC | BNBBTC | 0.00921 | 2.05 | 1,234 | 1.1e+07 | |\n"
    "\nPrices refresh every 10.5 seconds.\n"
)


def _ledger(tmp_path: Path, tool_name: str = "mcp_tvscreener_get_top_movers") -> GroundingLedger:
    ledger = GroundingLedger(run_dir=tmp_path, user_message="Топ-5 криптовалют за зростанням за день")
    ledger.ingest_tool_result(
        tool_name=tool_name,
        arguments={"asset_type": "crypto"},
        result=json.dumps({"status": "ok", "data": TABLE, "text": TABLE}),
        call_id="call_1",
        success=True,
    )
    return ledger


def _answer(prose: str, *declarations: str) -> str:
    lines = "\n".join(f"{value} | observed | figure | call_1" for value in declarations)
    return f"{prose}\n\n```figures\n{lines}\n```"


@pytest.mark.parametrize(
    ("prose", "declared"),
    [
        ("SOL коштує 184.23 USD.", "184.23"),
        ("SOL зріс на +7.41% за день.", "7.41%"),
        ("Ціна в парі до BTC: 0.00921.", "0.00921"),
        ("Обсяг SOL 321 млн монет.", "321 млн"),
        ("Капіталізація SOL 98.12 млрд.", "98.12 млрд"),
        ("Капіталізація SOL 98.12B.", "98.12B"),
        ("Обсяг SOL за добу 59 млрд USD.", "59 млрд"),
        ("BNB: обсяг 1,234 монет.", "1,234"),
    ],
)
def test_a_table_cell_grounds_the_figure_that_quotes_it(tmp_path, prose, declared):
    result = _ledger(tmp_path).validate_final_answer(_answer(prose, declared))
    assert result.valid, result.issues


@pytest.mark.parametrize(
    ("prose", "declared"),
    [
        ("SOL коштує 185.40 USD.", "185.40"),
        ("SOL зріс на +9.10% за день.", "9.10%"),
        ("Капіталізація SOL 99.5 млрд.", "99.5 млрд"),
        # The Volume column counts coins, so it is no dollar amount.
        ("Обсяг SOL 321 млн USD.", "321 млн"),
        # The refresh interval sits in prose around the table, not in a cell.
        ("Оновлення кожні 10.5 секунд.", "10.5"),
    ],
)
def test_a_value_the_table_does_not_hold_is_still_refused(tmp_path, prose, declared):
    result = _ledger(tmp_path).validate_final_answer(_answer(prose, declared))
    assert not result.valid


def test_only_mcp_tools_have_their_text_tables_read(tmp_path):
    ledger = _ledger(tmp_path, tool_name="web_search")
    result = ledger.validate_final_answer(_answer("SOL коштує 184.23 USD.", "184.23"))
    assert not result.valid


def test_a_table_is_not_evidence_before_it_is_read(tmp_path):
    ledger = GroundingLedger(run_dir=tmp_path, user_message="Топ-5 криптовалют")
    result = ledger.validate_final_answer(_answer("SOL коштує 184.23 USD.", "184.23"))
    assert not result.valid


@pytest.mark.parametrize(
    ("text", "scale"),
    [("321 млн", 1e6), ("98.12 млрд", 1e9), ("2 трлн", 1e12), ("5 тис.", 1e3), ("7 млнх", 1.0)],
)
def test_cyrillic_magnitude_words_scale_a_figure(text, scale):
    figures = [figure for figure in scan_figures(text, None) if figure.value is not None]
    assert figures and float(figures[0].scale or 1.0) == scale
