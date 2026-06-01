"""Day 4 evaluation: RMSE / MAE / NASA Score + the README money visuals.

Predicts the last-window RUL per test engine, writes ``reports/results.json``,
and renders the predicted-vs-actual scatter (rule C37 — RMSE must stay <= 30)
plus per-engine degradation curves with the critical zone shaded.
"""

import json
from pathlib import Path

import matplotlib
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


def plot_rul_scatter(result: dict, output: Path) -> None:
    """Predicted-vs-actual RUL scatter colour-coded by abs error (README visual)."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    preds = np.array([p["predicted_rul"] for p in result["per_engine_predictions"]])
    actuals = np.array([p["actual_rul"] for p in result["per_engine_predictions"]])
    errs = np.abs(preds - actuals)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(actuals, preds, c=errs, cmap="RdYlBu_r", s=60, edgecolor="black")
    mx = max(actuals.max(), preds.max()) + 5
    ax.plot([0, mx], [0, mx], "k--", alpha=0.5, label="perfect prediction")
    ax.set_xlabel("Actual RUL (cycles)")
    ax.set_ylabel("Predicted RUL (cycles)")
    ax.set_title(f"RUL prediction vs actual — RMSE={result['rmse']:.1f}")
    fig.colorbar(sc, ax=ax, label="absolute error")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)


def plot_degradation_curves(
    test_df: pd.DataFrame,
    model: RULPredictor,
    config: dict,
    output: Path,
    n: int = 5,
) -> None:
    """5 random test engines: true RUL solid blue, predicted dashed, zone shaded."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    seq = config["data"]["sequence_length"]
    rng = np.random.default_rng(config["data"]["seed"])
    chosen = rng.choice(test_df["engine_id"].unique(), size=n, replace=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=False)
    for ax, engine_id in zip(axes, chosen):
        eng = (
            test_df[test_df["engine_id"] == engine_id]
            .sort_values("cycle")
            .reset_index(drop=True)
        )
        if len(eng) < seq:
            continue
        feats = eng[FEATURE_COLS].values.astype(np.float32)
        preds = []
        for i in range(seq - 1, len(eng)):
            x = np.expand_dims(feats[i - seq + 1 : i + 1], 0)
            preds.append(float(model.predict(x)[0]))
        cycles = eng["cycle"].values[seq - 1 :]
        true_rul = eng["rul"].values[seq - 1 :]
        ax.axhspan(0, 40, color="red", alpha=0.1, label="critical zone")
        ax.plot(cycles, true_rul, color="#0ea5e9", label="true RUL")
        ax.plot(cycles, preds, color="#f97316", linestyle="--", label="predicted RUL")
        ax.set_title(f"Engine {int(engine_id)}")
        ax.set_xlabel("cycle")
        ax.set_ylabel("RUL")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
