"""Unit tests for src/explainability/shap_explainer.py (rule C50).

Real ``shap.GradientExplainer`` is mocked so unit tests stay fast and do not
trip the ``error::DeprecationWarning`` pytest filter.
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.data.skew_check import FEATURE_COLS
from src.explainability.shap_explainer import RULExplainer
from src.model.lstm import RULPredictor

CFG = {"data": {"sequence_length": 30}}


def _model() -> RULPredictor:
    """Tiny untrained LSTM for explainer tests."""
    return RULPredictor(
        {
            "model": {
                "input_size": 17,
                "hidden_size": 8,
                "num_layers": 2,
                "dropout": 0.2,
            }
        }
    )


def _fake_shap_values(x: torch.Tensor) -> np.ndarray:
    """Deterministic abs attribution shaped exactly like the input tensor."""
    rng = np.random.default_rng(0)
    return np.abs(rng.standard_normal(tuple(x.shape)))


def _patched_explainer(model: RULPredictor, background: torch.Tensor) -> RULExplainer:
    """Build a RULExplainer with shap.GradientExplainer mocked out."""
    fake = MagicMock()
    fake.shap_values.side_effect = _fake_shap_values
    with patch(
        "src.explainability.shap_explainer.shap.GradientExplainer",
        return_value=fake,
    ):
        return RULExplainer(model, background, CFG)


def test_explainer_background_shape_check() -> None:
    """Wrong-shape background tensor raises ValueError (rule C50)."""
    bad = torch.zeros(5, 10, 17)  # seq_len 10 != 30
    with pytest.raises(ValueError):
        _patched_explainer(_model(), bad)


def test_explain_engine_keys() -> None:
    """explain_engine returns shap_values, top_features, predicted_rul."""
    explainer = _patched_explainer(_model(), torch.zeros(5, 30, 17))
    out = explainer.explain_engine(torch.randn(1, 30, 17))
    assert set(out) == {"shap_values", "top_features", "predicted_rul"}
    assert len(out["shap_values"]) == len(FEATURE_COLS)


def test_shap_values_normalized() -> None:
    """Per-feature normalised SHAP values sum to ~1.0."""
    explainer = _patched_explainer(_model(), torch.zeros(5, 30, 17))
    out = explainer.explain_engine(torch.randn(1, 30, 17))
    assert abs(sum(out["shap_values"].values()) - 1.0) < 1e-6


def test_top_features_subset() -> None:
    """top_features is 3 names drawn from FEATURE_COLS."""
    explainer = _patched_explainer(_model(), torch.zeros(5, 30, 17))
    out = explainer.explain_engine(torch.randn(1, 30, 17))
    assert len(out["top_features"]) == 3
    assert set(out["top_features"]).issubset(set(FEATURE_COLS))


def test_waterfall_plot_saved(tmp_path) -> None:
    """plot_waterfall writes a non-empty PNG."""
    explainer = _patched_explainer(_model(), torch.zeros(5, 30, 17))
    out = explainer.explain_engine(torch.randn(1, 30, 17))
    png = tmp_path / "waterfall.png"
    explainer.plot_waterfall(out, engine_id=7, predicted_rul=42.0, output=png)
    assert png.exists()
    assert png.stat().st_size > 0


def test_save_baseline_writes_json(tmp_path) -> None:
    """save_baseline writes a 17-key JSON file (one per feature)."""
    explainer = _patched_explainer(_model(), torch.zeros(5, 30, 17))
    out = tmp_path / "shap_baseline.json"
    explainer.save_baseline(torch.zeros(5, 30, 17), out)
    assert out.exists()
    baseline = json.loads(out.read_text())
    assert len(baseline) == len(FEATURE_COLS)
    assert set(baseline) == set(FEATURE_COLS)
