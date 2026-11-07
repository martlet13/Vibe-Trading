"""Boolean strings never silently turn ambiguous action flags into writes."""

import json

import pytest

from src.tools._arguments import boolean_argument
from src.tools.trading_connector_tool import EtoroEditPositionStopsTool


@pytest.mark.parametrize("value", [False, None, 0, 0.0, "false", " False ", "0", "no", "OFF"])
def test_false_values(value):
    assert boolean_argument(value, field="flag") is False


@pytest.mark.parametrize("value", [True, 1, 1.0, "true", " True ", "1", "yes", "ON"])
def test_true_values(value):
    assert boolean_argument(value, field="flag") is True


@pytest.mark.parametrize("value", ["", "maybe", "flase", 2, -1, [], {}, float("nan")])
def test_ambiguous_values_are_errors(value):
    with pytest.raises(ValueError, match="dry_run must be a boolean"):
        boolean_argument(value, field="dry_run")


@pytest.mark.parametrize("field", ["clear_stop_loss", "clear_take_profit", "trailing_stop_loss"])
def test_invalid_stop_flag_cannot_call_the_broker(monkeypatch, field):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid flag reached the broker service")

    monkeypatch.setattr("src.tools.trading_connector_tool.edit_position_stops", forbidden)
    result = json.loads(EtoroEditPositionStopsTool().execute(position_id="123", **{field: "maybe"}))
    assert result["status"] == "error"
    assert field in result["error"]
