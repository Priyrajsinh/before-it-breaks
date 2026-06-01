"""Day 4 evaluation: RMSE / MAE / NASA Score on the official CMAPSS test set.

Predicts the last-window RUL per test engine and writes ``reports/results.json``
with per-engine predictions for downstream UI. Rule C37 — the test RMSE must
stay <= 30 cycles (wired as a CI gate on Day 9).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.safe_predict import safe_predict
from src.data.skew_check import FEATURE_COLS
from src.logger import get_logger
from src.model.lstm import RULPredictor

logger = get_logger(__name__)


def nasa_score(pred: np.ndarray, actual: np.ndarray) -> float:
    """Asymmetric CMAPSS score — late predictions penalised harder than early.

    s_late = exp(err/10) - 1 (err > 0) vs s_early = exp(-err/13) - 1 (err < 0).
    Encodes the business reality: warning too late is worse than too early.
    """
    err = pred - actual
    s = np.where(err < 0, np.exp(-err / 13) - 1, np.exp(err / 10) - 1)
    return float(s.sum())


def evaluate(model: RULPredictor, test_df: pd.DataFrame, config: dict) -> dict:
    """Predict last-window RUL per test engine; compute RMSE/MAE/NASA (rule C37)."""
    seq = config["data"]["sequence_length"]
    preds: list[float] = []
    actuals: list[float] = []
    engine_ids: list[int] = []
    for engine_id, eng in test_df.groupby("engine_id"):
        eng = eng.sort_values("cycle").reset_index(drop=True)
        if len(eng) < seq:
            continue
        last_window = eng[FEATURE_COLS].values[-seq:]
        x = np.expand_dims(last_window, 0).astype(np.float32)
        rul_pred = safe_predict(model.predict, x, expected_shape=(1, seq, 17))[0]
        preds.append(float(rul_pred))
        actuals.append(float(eng["rul"].iloc[-1]))
        engine_ids.append(int(engine_id))
    preds_arr = np.array(preds)
    actuals_arr = np.array(actuals)
    rmse = float(np.sqrt(((preds_arr - actuals_arr) ** 2).mean()))
    mae = float(np.abs(preds_arr - actuals_arr).mean())
    score = nasa_score(preds_arr, actuals_arr)
    result: dict = {
        "rmse": rmse,
        "mae": mae,
        "nasa_score": score,
        "test_set_size": len(preds_arr),
        "per_engine_predictions": [
            {
                "engine_id": e,
                "predicted_rul": p,
                "actual_rul": a,
                "abs_error": float(abs(p - a)),
            }
            for e, p, a in zip(engine_ids, preds, actuals)
        ],
    }
    Path(config["paths"]["reports_dir"]).mkdir(parents=True, exist_ok=True)
    with open(config["paths"]["results_json"], "w") as fh:
        json.dump(result, fh, indent=2)
    logger.info(f"evaluation: RMSE={rmse:.2f} MAE={mae:.2f} NASA={score:.1f}")
    return result
