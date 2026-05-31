"""Training loop for RULPredictor.

Gradient clipping (rule), early stopping (patience), ReduceLROnPlateau, and
MLflow experiment tracking. Saves the best checkpoint as a state_dict (rule
C47) and writes reports/figures/training_loss_curve.png (rule C15).
"""

import argparse
from pathlib import Path

import matplotlib
import mlflow
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.config import load_config
from src.data.dataset import get_dataloaders
from src.logger import get_logger
from src.model.lstm import RULPredictor

logger = get_logger(__name__)


def _epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    clip: float,
) -> float:
    """Run one epoch. If optimizer is None, run in eval mode (no grad/step)."""
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    for X, y in loader:
        if training:
            assert optimizer is not None
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        else:
            with torch.no_grad():
                preds = model(X)
                loss = criterion(preds, y)
        total += loss.item() * X.size(0)
        n += X.size(0)
    return total / max(n, 1)


def train_model(config: dict) -> dict:
    """Train RULPredictor end-to-end. Returns a small training-summary dict."""
    from utils.seed import set_seed  # rule C11 — import inside main path

    set_seed(config["data"]["seed"])

    train_df = pd.read_parquet(config["data"]["processed_train"])
    val_df = pd.read_parquet(config["data"]["processed_val"])
    test_df = pd.read_parquet(config["data"]["processed_test"])
    train_loader, val_loader, _ = get_dataloaders(train_df, val_df, test_df, config)

    model = RULPredictor(config)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=config["model"]["scheduler_patience"],
        factor=config["model"]["scheduler_factor"],
    )

    models_dir = Path(config["paths"]["models_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    curve_path = figures_dir / "training_loss_curve.png"

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    with mlflow.start_run():
        mlflow.log_params({**config["model"], "seed": config["data"]["seed"]})

        best_val, patience, best_state = float("inf"), 0, None
        train_losses: list[float] = []
        val_losses: list[float] = []
        max_epochs = config["model"]["epochs"]
        early_stop_epoch = max_epochs
        early_stopped = False

        for epoch in range(max_epochs):
            tr = _epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                config["model"]["clip_grad_norm"],
            )
            vl = _epoch(
                model,
                val_loader,
                criterion,
                None,
                config["model"]["clip_grad_norm"],
            )
            scheduler.step(vl)
            train_losses.append(tr)
            val_losses.append(vl)
            mlflow.log_metrics({"train_loss": tr, "val_loss": vl}, step=epoch)
            logger.info(f"epoch {epoch}: train_loss={tr:.4f} val_loss={vl:.4f}")

            if vl < best_val:
                best_val = vl
                patience = 0
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
            else:
                patience += 1
                if patience >= config["model"]["early_stopping_patience"]:
                    early_stop_epoch = epoch
                    early_stopped = True
                    logger.info(f"early stopping at epoch {epoch}")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.save(models_dir)
        mlflow.log_artifact(str(models_dir / "lstm_rul.pt"))
        stats_path = models_dir / "training_stats.json"
        if stats_path.exists():
            mlflow.log_artifact(str(stats_path))

        _save_loss_curve(train_losses, val_losses, early_stop_epoch, curve_path)
        mlflow.log_artifact(str(curve_path))
        mlflow.log_metric("best_val_loss", best_val)

    return {
        "best_val_loss": best_val,
        "epochs_run": len(train_losses),
        "early_stopped": early_stopped,
    }


def _save_loss_curve(tr: list[float], vl: list[float], stop: int, out: Path) -> None:
    """Plot train/val loss with the early-stop marker. rule C15 (Agg backend)."""
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tr, label="train", color="#0ea5e9")
    ax.plot(vl, label="val", color="#ef4444")
    ax.axvline(stop, color="gray", linestyle="--", label=f"early stop @ {stop}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title("RULPredictor training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    """CLI entry point: python -m src.model.train --config config/config.yaml."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    train_model(load_config(args.config))


if __name__ == "__main__":
    main()  # rule C11 — set_seed is called inside train_model, not at import
