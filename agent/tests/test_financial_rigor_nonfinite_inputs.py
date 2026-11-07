"""Regression tests: financial_rigor rejects NaN / infinite / boolean inputs.

Before the fix, ``verify_market_cap`` and ``verify_valuation`` returned
``status="ok"`` with NaN or Infinity in the payload, which is not valid JSON.
"""

import json

import pytest

from src.tools.financial_rigor_tool import FinancialRigorTool


def _run(**kwargs):
    return json.loads(FinancialRigorTool().execute(**kwargs))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("field", ["price", "shares", "reported_cap"])
def test_verify_market_cap_rejects_non_finite(field, bad):
    args = {"command": "verify_market_cap", "price": 10, "shares": 100, "reported_cap": 1000}
    args[field] = bad
    result = _run(**args)
    assert result["status"] == "error"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["price", "eps", "bvps", "fcf_per_share", "dividend", "revenue_per_share"])
def test_verify_valuation_rejects_non_finite(field, bad):
    args = {"command": "verify_valuation", "price": 10, "eps": 2, "bvps": 5}
    args[field] = bad
    result = _run(**args)
    assert result["status"] == "error"


def test_boolean_is_not_a_number():
    result = _run(command="verify_valuation", price=10, eps=True)
    assert result["status"] == "error"


def test_overflowing_result_is_an_error_not_invalid_json():
    raw = FinancialRigorTool().execute(
        command="verify_market_cap", price=1e308, shares=100, reported_cap=1000,
    )
    assert "Infinity" not in raw and "NaN" not in raw
    assert json.loads(raw)["status"] == "error"


def test_finite_inputs_still_pass():
    result = _run(command="verify_market_cap", price=10, shares=100, reported_cap=1000)
    assert result["status"] == "ok"
    assert result["verdict"] == "pass"
