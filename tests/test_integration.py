import pytest


@pytest.mark.skip(reason="covered by tests/test_api.py::test_health_returns_200")
def test_health_endpoint():
    """API returns healthy status."""
    ...


@pytest.mark.skip(reason="covered by tests/test_api.py::test_predict_valid")
def test_predict_endpoint_valid_schema():
    """Valid 30×17 window returns a PredictResponse."""
    ...


@pytest.mark.skip(reason="covered by tests/test_api.py::test_predict_wrong_shape_422")
def test_invalid_input_422():
    """Malformed input returns HTTP 422."""
    ...


@pytest.mark.skip(reason="Day 9 wires the coverage regression test")
def test_rmse_under_threshold():
    """Test RMSE on CMAPSS test set is ≤ 30 cycles (rule C37)."""
    ...


@pytest.mark.skip(reason="Day 9 wires the anti-leakage test")
def test_train_val_engines_disjoint():
    """Train and val engine_id sets are disjoint (rule C33)."""
    ...
