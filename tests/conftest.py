import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def dummy_processed_df() -> pd.DataFrame:
    """Tiny well-formed processed CMAPSS DataFrame for unit tests."""
    rows = []
    for engine in [1, 2]:
        for cycle in range(1, 41):
            row: dict = {
                "engine_id": engine,
                "cycle": cycle,
                "setting_1": 0.0,
                "setting_2": 0.0,
                "setting_3": 100.0,
            }
            for s in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]:
                row[f"sensor_{s}"] = float(np.random.randn())
            row["rul"] = float(min(40 - cycle, 125))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def mock_config() -> dict:
    """Minimal config dict for unit tests."""
    return {
        "seed": 42,
        "data": {
            "sequence_length": 30,
            "max_rul": 125,
            "train_engine_ids": [1, 1],
            "val_engine_ids": [2, 2],
        },
        "model": {
            "input_size": 17,
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.2,
            "batch_size": 4,
            "epochs": 2,
            "learning_rate": 0.01,
            "clip_grad_norm": 1.0,
            "early_stopping_patience": 1,
            "scheduler_patience": 1,
            "scheduler_factor": 0.5,
        },
        "evaluation": {
            "rmse_threshold": 30.0,
            "health_status_thresholds": {"healthy_min": 80, "warning_min": 40},
        },
        "shap": {"n_background_samples": 5},
        "monitoring": {"psi_threshold": 0.2, "drift_window_size": 100},
    }
