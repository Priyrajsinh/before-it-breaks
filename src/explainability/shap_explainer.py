"""SHAP GradientExplainer for the 2-layer LSTM (rule C50).

GradientExplainer (NOT TreeExplainer — this is a deep model). Background = 100
random training windows as ``torch.Tensor[100, 30, 17]``. The explainer output
for a single window ``[1, 30, 17]`` is shape ``[1, 30, 17]`` (some SHAP versions
append a trailing output axis) — we collapse any output axis, then average the
absolute attribution over the seq_len axis to a per-feature importance vector.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import shap
import torch
from torch import nn

from src.data.skew_check import FEATURE_COLS
from src.logger import get_logger
from src.model.lstm import RULPredictor

logger = get_logger(__name__)


class _TwoDimOutput(nn.Module):
    """Wrap RULPredictor so its output is ``[batch, 1]`` for GradientExplainer.

    ``RULPredictor.forward`` returns ``[batch]`` (rule C49), but SHAP indexes the
    model output as ``outputs[:, idx]`` and needs a 2-D tensor. This adapter is
    used only inside the explainer; the real model is untouched.
    """

    def __init__(self, model: RULPredictor) -> None:
        """Store the wrapped RUL model."""
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward through the model, promoting a 1-D output to ``[batch, 1]``."""
        out = self.model(x)
        if out.dim() == 1:
            out = out.unsqueeze(-1)
        return out


def _to_feature_array(shap_out: object) -> np.ndarray:
    """Normalise SHAP output to a ``[N, seq_len, n_features]`` float array.

    Handles the list-wrapped multi-output form and a trailing singleton output
    axis that some SHAP versions append for a scalar-output model.
    """
    if isinstance(shap_out, list):
        shap_out = shap_out[0]
    arr = np.asarray(shap_out)
    if arr.ndim == 4:  # [N, seq_len, n_features, n_outputs] -> drop output axis
        arr = arr[..., 0]
    return arr


class RULExplainer:
    """SHAP GradientExplainer wrapper for ``RULPredictor`` (rule C50)."""

    def __init__(
        self, model: RULPredictor, background_data: torch.Tensor, config: dict
    ) -> None:
        """Build the explainer; validate the background tensor shape (rule C50)."""
        seq = config["data"]["sequence_length"]
        if background_data.shape[1:] != (seq, len(FEATURE_COLS)):
            raise ValueError(
                f"background shape {tuple(background_data.shape)} "
                f"!= [N, {seq}, {len(FEATURE_COLS)}]"
            )
        model.eval()
        self.model = model
        self.explainer = shap.GradientExplainer(_TwoDimOutput(model), background_data)
        self.feature_names = FEATURE_COLS

    def explain_engine(self, window: torch.Tensor) -> dict:
        """window: [1, 30, 17] -> {shap_values, top_features, predicted_rul}."""
        with torch.no_grad():
            pred = float(self.model(window).item())
        arr = _to_feature_array(self.explainer.shap_values(window))
        # average abs over seq_len -> [n_features], then normalise to sum 1
        per_feature = np.abs(arr[0]).mean(axis=0)
        per_feature_norm = per_feature / max(per_feature.sum(), 1e-9)
        shap_values = {
            f: float(v) for f, v in zip(self.feature_names, per_feature_norm)
        }
        top_features = [
            f
            for f, _ in sorted(shap_values.items(), key=lambda kv: kv[1], reverse=True)[
                :3
            ]
        ]
        logger.info(f"explain: predicted_rul={pred:.1f} top={top_features}")
        return {
            "shap_values": shap_values,
            "top_features": top_features,
            "predicted_rul": pred,
        }

    def plot_waterfall(
        self, shap_dict: dict, engine_id: int, predicted_rul: float, output: Path
    ) -> None:
        """Horizontal bar of normalised |SHAP| contributions per feature."""
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.switch_backend("Agg")
        items = sorted(
            shap_dict["shap_values"].items(), key=lambda kv: kv[1], reverse=True
        )
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        colors = ["#0ea5e9" if v > 0 else "#ef4444" for v in vals]
        output.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 0.4 * len(labels) + 1))
        ax.barh(labels, vals, color=colors)
        ax.set_xlabel("normalized |SHAP| contribution")
        ax.set_title(f"Engine {engine_id} — predicted RUL = {predicted_rul:.0f} cycles")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(output, dpi=120)
        plt.close(fig)

    def save_baseline(self, background_data: torch.Tensor, output: Path) -> None:
        """Mean abs SHAP per feature across the background — Streamlit comparison."""
        arr = _to_feature_array(self.explainer.shap_values(background_data))
        per_feature = np.abs(arr).mean(axis=(0, 1))
        per_feature_norm = per_feature / max(per_feature.sum(), 1e-9)
        baseline = {f: float(v) for f, v in zip(self.feature_names, per_feature_norm)}
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as fh:
            json.dump(baseline, fh, indent=2)
