"""Unit tests for src/model/evaluate.py — metrics + plots (rule C37 path)."""

import json

import numpy as np
import pandas as pd

from src.model.evaluate import (
    evaluate,
    nasa_score,
    plot_degradation_curves,
    plot_rul_scatter,
)
from src.model.lstm import RULPredictor


def _model_cfg() -> dict:
    """Tiny LSTM config for an untrained model under test."""
    return {
        "input_size": 17,
        "hidden_size": 8,
        "num_layers": 2,
        "dropout": 0.2,
    }


def _two_engine_df() -> pd.DataFrame:
    """Two engines, 40 cycles each (>= sequence_length 30)."""
    rng = np.random.default_rng(0)
    rows = []
    for engine in (1, 2):
        for cycle in range(1, 41):
            row: dict = {
                "engine_id": engine,
                "cycle": cycle,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "setting_3": 0.0,
            }
            for s in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]:
                row[f"sensor_{s}"] = float(rng.standard_normal())
            row["rul"] = float(min(40 - cycle, 125))
            rows.append(row)
    return pd.DataFrame(rows)


def _eval_config(tmp_path) -> dict:
    """Config with reports paths pointing into tmp_path for evaluate()."""
    return {
        "data": {"sequence_length": 30, "seed": 42},
        "paths": {
            "reports_dir": str(tmp_path / "reports"),
            "results_json": str(tmp_path / "reports" / "results.json"),
        },
    }


def test_nasa_score_asymmetric() -> None:
    """For the same |err|, a late prediction (+10) scores higher than early (-10)."""
    actual = np.array([50.0])
    late = nasa_score(np.array([60.0]), actual)  # err = +10
    early = nasa_score(np.array([40.0]), actual)  # err = -10
    assert late > early


def test_rmse_formula(tmp_path) -> None:
    """evaluate() RMSE matches the closed-form sqrt(mean(sq err))."""
    model = RULPredictor({"model": _model_cfg()})
    result = evaluate(model, _two_engine_df(), _eval_config(tmp_path))
    preds = np.array([p["predicted_rul"] for p in result["per_engine_predictions"]])
    actuals = np.array([p["actual_rul"] for p in result["per_engine_predictions"]])
    expected = float(np.sqrt(((preds - actuals) ** 2).mean()))
    assert abs(result["rmse"] - expected) < 1e-9


def test_evaluate_writes_results_json(tmp_path) -> None:
    """evaluate() writes results.json with the required keys."""
    model = RULPredictor({"model": _model_cfg()})
    result = evaluate(model, _two_engine_df(), _eval_config(tmp_path))
    out = tmp_path / "reports" / "results.json"
    assert out.exists()
    saved = json.loads(out.read_text())
    for key in ("rmse", "mae", "nasa_score", "per_engine_predictions"):
        assert key in saved
    assert saved["test_set_size"] == 2
    assert result["test_set_size"] == 2


def test_scatter_plot_saved(tmp_path) -> None:
    """plot_rul_scatter writes a non-empty PNG."""
    result = {
        "rmse": 12.3,
        "per_engine_predictions": [
            {"predicted_rul": 30.0, "actual_rul": 28.0},
            {"predicted_rul": 50.0, "actual_rul": 60.0},
        ],
    }
    out = tmp_path / "scatter.png"
    plot_rul_scatter(result, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_degradation_plot_saved(tmp_path) -> None:
    """plot_degradation_curves writes a non-empty PNG for 2 sample engines."""
    model = RULPredictor({"model": _model_cfg()})
    cfg = {"data": {"sequence_length": 30, "seed": 42}}
    out = tmp_path / "degradation.png"
    plot_degradation_curves(_two_engine_df(), model, cfg, out, n=2)
    assert out.exists()
    assert out.stat().st_size > 0
