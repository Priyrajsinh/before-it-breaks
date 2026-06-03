"""ModelServer — artefact loader + inference pipeline (rules C36, C42, C45, C48)."""

import json
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from src.data.safe_predict import safe_predict
from src.data.skew_check import FEATURE_COLS, check_skew, psi
from src.exceptions import ModelNotFoundError, PredictionError
from src.logger import get_logger
from src.model.lstm import RULPredictor

logger = get_logger(__name__)


def _health_status(rul: float, thresholds: dict[str, int]) -> str:
    """Map a predicted RUL to a health label using configured thresholds."""
    if rul >= thresholds["healthy_min"]:
        return "Healthy"
    if rul >= thresholds["warning_min"]:
        return "Warning"
    return "Critical"


def _nl_summary(
    engine_id: int, rul: float, status: str, top_feature: str | None
) -> str:
    """Build a plain-English maintenance recommendation (rule C45)."""
    action = {
        "Healthy": "No action needed — continue normal operation.",
        "Warning": ("Monitor closely — plan maintenance within the next window."),
        "Critical": "Schedule maintenance immediately — failure is imminent.",
    }[status]
    base = (
        f"Engine {engine_id} is predicted to fail in approximately"
        f" {rul:.0f} cycles. Status: {status.upper()}. {action}"
    )
    if top_feature:
        base += f" Top warning signal: {top_feature}."
    return base


class ModelServer:
    """Model server — loads artefacts once at startup (rules C36, C42, C45, C48)."""

    def __init__(self, config: dict) -> None:
        """Load model, scaler, and training stats.

        Raises ModelNotFoundError if any required artefact is missing.
        """
        self.config = config
        models_dir = Path(config["paths"]["models_dir"])
        scaler_path = models_dir / "scaler.pkl"
        stats_path = models_dir / "training_stats.json"
        for p in (scaler_path, stats_path):
            if not p.exists():
                raise ModelNotFoundError(f"missing {p} — run /train first")
        self.model = RULPredictor.load(models_dir)
        with open(str(scaler_path), "rb") as fh:
            self.scaler = joblib.load(fh)
        with open(str(stats_path)) as fh:
            self.training_stats: dict = json.load(fh)
        self.stats_path = stats_path
        self.started = time.time()
        self.total_predictions = 0
        self.drift_events = 0
        self._explainer: Any = None

    def _scale_window(self, sensor_window: list[list[float]]) -> np.ndarray:
        """Scale a raw 30×17 sensor window using the training scaler (rule C48)."""
        raw = np.array(sensor_window, dtype=np.float32)
        seq_len = self.config["data"]["sequence_length"]
        n_features = len(FEATURE_COLS)
        if raw.shape != (seq_len, n_features):
            raise PredictionError(
                f"sensor_window shape {raw.shape} != ({seq_len}, {n_features})"
            )
        return self.scaler.transform(raw)

    def _drift_flags(self, scaled: np.ndarray) -> dict[str, bool]:
        """Return per-feature out-of-range flags for the last cycle (rule C42)."""
        return check_skew(scaled[-1], self.training_stats)

    def predict(self, engine_id: int, sensor_window: list[list[float]]) -> dict:
        """Run the full inference pipeline for one engine window."""
        scaled = self._scale_window(sensor_window)
        flags = self._drift_flags(scaled)
        if any(flags.values()):
            self.drift_events += 1
            logger.warning(
                "drift flagged for engine %s", engine_id, extra={"flags": flags}
            )
        x = np.expand_dims(scaled, 0).astype(np.float32)
        rul = float(
            safe_predict(
                self.model.predict,
                x,
                expected_shape=(1, scaled.shape[0], scaled.shape[1]),
            )[0]
        )
        self.total_predictions += 1
        status = _health_status(
            rul, self.config["evaluation"]["health_status_thresholds"]
        )
        return {
            "engine_id": engine_id,
            "predicted_rul": rul,
            "health_status": status,
            "confidence": 1.0 - min(abs(rul) / 125.0, 1.0),
            "nl_summary": _nl_summary(engine_id, rul, status, None),
            "drift_flags": flags,
        }

    def predict_batch(
        self, requests: Iterable[tuple[int, list[list[float]]]]
    ) -> list[dict]:
        """Run predict() for each (engine_id, window) pair in the iterable."""
        return [self.predict(eid, w) for eid, w in requests]

    def uptime_seconds(self) -> float:
        """Return wall-clock seconds since the server was initialised."""
        return time.time() - self.started

    def sensor_psi(self, recent_window: np.ndarray) -> dict[str, float]:
        """PSI per feature: training distribution vs recent samples (rule C42)."""
        out: dict[str, float] = {}
        for i, c in enumerate(FEATURE_COLS):
            s = self.training_stats[c]
            expected = np.random.default_rng(0).normal(
                s["mean"], max(s["std"], 1e-6), 1000
            )
            out[c] = psi(expected, recent_window[:, i])
        return out
