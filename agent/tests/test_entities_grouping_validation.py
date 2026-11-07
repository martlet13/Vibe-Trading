"""Grouping inference must not turn malformed amounts into invented money."""

import pytest

from src.entities.ingest import CashFlowIngestError, load_cashflows


@pytest.mark.parametrize("text", ["1..234", "1.23.45", "1.234.56", "1,,234", "1,23,45", "1,23,45.67"])
def test_malformed_grouping_is_refused_by_file_ingestion(tmp_path, text):
    path = tmp_path / "flows.csv"
    path.write_text(f'date,amount,kind,currency\n2026-01-01,"{text}",coupon,EUR\n')
    with pytest.raises(CashFlowIngestError, match="grouping"):
        load_cashflows(path)


@pytest.mark.parametrize(
    "text,separator,expected",
    [
        ("1.234.567", None, 1234567),
        ("(1.234.567,89)", None, -1234567.89),
        ("12,34,567.89", None, 1234567.89),
        ("1.234.567e-3", None, 1234.567),
        ("1,234", ".", 1234),
        ("1,234", ",", 1.234),
        ("1.234", ",", 1234),
        ("1.234", ".", 1.234),
    ],
)
def test_valid_grouping_and_explicit_ambiguity_resolution(tmp_path, text, separator, expected):
    path = tmp_path / "flows.csv"
    path.write_text(f'date,amount,kind,currency\n2026-01-01,"{text}",recallable_distribution,EUR\n')
    assert load_cashflows(path, decimal_separator=separator)[0].amount == pytest.approx(expected)


@pytest.mark.parametrize("text,separator", [("1,23,45", "."), ("1.23.45", ","), ("1,234.5,6", ".")])
def test_declaring_decimal_does_not_authorize_malformed_grouping(tmp_path, text, separator):
    path = tmp_path / "flows.csv"
    path.write_text(f'date,amount,kind,currency\n2026-01-01,"{text}",coupon,EUR\n')
    with pytest.raises(CashFlowIngestError, match="grouping"):
        load_cashflows(path, decimal_separator=separator)
