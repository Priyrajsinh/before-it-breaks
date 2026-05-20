import numpy as np
import pytest


def test_logger_returns_logger():
    """Logger returns a Logger with the correct name."""
    from src.logger import get_logger

    assert get_logger("x").name == "x"


def test_exceptions_hierarchy():
    """Exception classes form the correct inheritance chain."""
    from src.exceptions import (
        ChecksumError,
        DataLoadError,
        DriftAlarmError,
        PredictionError,
        ProjectBaseError,
    )

    assert issubclass(DataLoadError, ProjectBaseError)
    assert issubclass(ChecksumError, DataLoadError)
    assert issubclass(PredictionError, ProjectBaseError)
    assert issubclass(DriftAlarmError, ProjectBaseError)


def test_base_model_is_abstract():
    """BaseMLModel cannot be instantiated directly."""
    from src.model.base import BaseMLModel

    with pytest.raises(TypeError):
        BaseMLModel()  # noqa: E501


def test_predict_request_valid_shape():
    """PredictRequest accepts a valid 30×17 window."""
    from src.data.schemas import PredictRequest

    PredictRequest(engine_id=1, sensor_window=[[0.0] * 17] * 30)


def test_predict_request_wrong_rows():
    """PredictRequest rejects a 29-row window."""
    import pydantic

    from src.data.schemas import PredictRequest

    with pytest.raises(pydantic.ValidationError):
        PredictRequest(engine_id=1, sensor_window=[[0.0] * 17] * 29)


def test_predict_request_wrong_cols():
    """PredictRequest rejects a window with 16 features per row."""
    import pydantic

    from src.data.schemas import PredictRequest

    with pytest.raises(pydantic.ValidationError):
        PredictRequest(engine_id=1, sensor_window=[[0.0] * 16] * 30)


def test_safe_predict_shape_guard():
    """safe_predict raises PredictionError on shape mismatch."""
    from src.data.safe_predict import safe_predict
    from src.exceptions import PredictionError

    with pytest.raises(PredictionError):
        safe_predict(lambda x: x, np.zeros((2, 3)), expected_shape=(2, 4))


def test_safe_predict_nan_guard():
    """safe_predict raises PredictionError when input contains NaN."""
    from src.data.safe_predict import safe_predict
    from src.exceptions import PredictionError

    x = np.zeros((2, 3))
    x[0, 0] = np.nan
    with pytest.raises(PredictionError):
        safe_predict(lambda x: x, x, expected_shape=(2, 3))


def test_config_load_valid(tmp_path):
    """load_config parses a valid YAML with all required keys."""
    import yaml

    from src.config import load_config

    cfg = {
        "seed": 42,
        "data": {},
        "model": {},
        "evaluation": {},
        "monitoring": {},
        "api": {},
    }
    p = tmp_path / "config.yaml"
    with open(str(p), "w") as fh:
        yaml.dump(cfg, fh)
    loaded = load_config(p)
    assert loaded["seed"] == 42


def test_config_load_missing_key(tmp_path):
    """load_config raises ConfigError when required keys are absent."""
    import yaml

    from src.config import load_config
    from src.exceptions import ConfigError

    cfg = {"seed": 42}
    p = tmp_path / "config.yaml"
    with open(str(p), "w") as fh:
        yaml.dump(cfg, fh)
    with pytest.raises(ConfigError):
        load_config(p)
