"""Unit tests for RULPredictor (rules C38, C47, C49)."""

import numpy as np
import torch

from src.model.base import BaseMLModel
from src.model.lstm import RULPredictor


def test_is_basemlmodel(mock_config: dict) -> None:
    """RULPredictor implements the BaseMLModel ABC (rule C38)."""
    model = RULPredictor(mock_config)
    assert isinstance(model, BaseMLModel)


def test_forward_output_shape(mock_config: dict) -> None:
    """Input [4, 30, 17] -> output shape [4], dtype float32 (rule C49)."""
    model = RULPredictor(mock_config)
    x = torch.zeros(4, 30, 17, dtype=torch.float32)
    out = model(x)
    assert out.shape == (4,)
    assert out.dtype == torch.float32


def test_gradients_flow(mock_config: dict) -> None:
    """A backward pass produces non-None gradients on every parameter."""
    model = RULPredictor(mock_config)
    x = torch.randn(4, 30, 17)
    y = torch.randn(4)
    loss = torch.nn.functional.mse_loss(model(x), y)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"no gradient for {name}"


def test_save_load_roundtrip(mock_config: dict, tmp_path) -> None:
    """save state_dict -> fresh load -> predictions match within 1e-6 (rule C47)."""
    model = RULPredictor(mock_config)
    x = np.random.randn(2, 30, 17).astype(np.float32)
    before = model.predict(x)

    model.save(tmp_path)
    assert (tmp_path / "lstm_rul.pt").exists()

    reloaded = RULPredictor.load(tmp_path)
    after = reloaded.predict(x)
    np.testing.assert_allclose(before, after, atol=1e-6)


def test_predict_method(mock_config: dict) -> None:
    """predict on numpy input returns a 1-D RUL vector of the right length."""
    model = RULPredictor(mock_config)
    out = model.predict(np.zeros((2, 30, 17), dtype=np.float32))
    assert out.shape == (2,)
    assert isinstance(out, np.ndarray)
