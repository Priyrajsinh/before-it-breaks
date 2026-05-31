"""Unit tests for the training loop (src/model/train.py)."""

import math

import mlflow
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from src.model.train import _epoch, _save_loss_curve, train_model


def _tiny_loader(n: int = 16) -> DataLoader:
    """Synthetic [n, 30, 17] -> [n] loader — no real CMAPSS data."""
    X = torch.randn(n, 30, 17)
    y = torch.randn(n)
    return DataLoader(TensorDataset(X, y), batch_size=8)


def _train_config(tmp_path, dummy_processed_df: pd.DataFrame, **overrides) -> dict:
    """Build a tiny end-to-end training config backed by synthetic parquet."""
    train_p = tmp_path / "train.parquet"
    val_p = tmp_path / "val.parquet"
    test_p = tmp_path / "test.parquet"
    dummy_processed_df.to_parquet(train_p)
    dummy_processed_df.to_parquet(val_p)
    dummy_processed_df.to_parquet(test_p)
    model_cfg = {
        "input_size": 17,
        "hidden_size": 8,
        "num_layers": 2,
        "dropout": 0.2,
        "batch_size": 8,
        "epochs": 2,
        "learning_rate": 0.01,
        "clip_grad_norm": 1.0,
        "early_stopping_patience": 1,
        "scheduler_patience": 1,
        "scheduler_factor": 0.5,
    }
    model_cfg.update(overrides)
    return {
        "data": {
            "seed": 42,
            "sequence_length": 30,
            "max_rul": 125,
            "processed_train": str(train_p),
            "processed_val": str(val_p),
            "processed_test": str(test_p),
        },
        "model": model_cfg,
        "paths": {
            "models_dir": str(tmp_path / "models"),
            "figures_dir": str(tmp_path / "figures"),
        },
        "mlflow": {
            "experiment_name": "test_pm1",
            "tracking_uri": (tmp_path / "mlruns").as_uri(),
        },
    }


def test_epoch_train_returns_float(mock_config: dict) -> None:
    """One training epoch on a synthetic loader returns a finite loss."""
    from src.model.lstm import RULPredictor

    model = RULPredictor(mock_config)
    loss = _epoch(
        model, _tiny_loader(), nn.MSELoss(), optim.Adam(model.parameters()), 1.0
    )
    assert isinstance(loss, float)
    assert math.isfinite(loss)


def test_epoch_eval_mode_no_optimizer(mock_config: dict) -> None:
    """With optimizer=None, _epoch runs in eval mode and returns a finite loss."""
    from src.model.lstm import RULPredictor

    model = RULPredictor(mock_config)
    loss = _epoch(model, _tiny_loader(), nn.MSELoss(), None, 1.0)
    assert math.isfinite(loss)
    assert not model.training


def test_save_loss_curve_creates_png(tmp_path) -> None:
    """_save_loss_curve writes a PNG to the requested path."""
    out = tmp_path / "figures" / "training_loss_curve.png"
    _save_loss_curve([1.0, 0.5, 0.3], [1.1, 0.6, 0.4], 2, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_train_model_runs_and_logs_mlflow(tmp_path, dummy_processed_df) -> None:
    """train_model trains, saves a state_dict, writes the curve, logs MLflow."""
    cfg = _train_config(tmp_path, dummy_processed_df)
    result = train_model(cfg)

    assert (tmp_path / "models" / "lstm_rul.pt").exists()
    assert (tmp_path / "figures" / "training_loss_curve.png").exists()
    assert math.isfinite(result["best_val_loss"])

    runs = mlflow.search_runs(experiment_names=["test_pm1"])
    for param in ("params.hidden_size", "params.num_layers", "params.dropout"):
        assert param in runs.columns


def test_early_stopping_triggers(tmp_path, dummy_processed_df) -> None:
    """A diverging LR makes val loss stop improving -> early stop before max."""
    cfg = _train_config(
        tmp_path,
        dummy_processed_df,
        epochs=12,
        early_stopping_patience=2,
        learning_rate=5.0,
    )
    result = train_model(cfg)
    assert result["early_stopped"] is True
    assert result["epochs_run"] < cfg["model"]["epochs"]
