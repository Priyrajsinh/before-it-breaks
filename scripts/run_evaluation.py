"""Day 4 evaluation entrypoint — runs the full evaluation pipeline."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import load_config
from src.data.dataset import CMAPSSDataset
from src.data.skew_check import FEATURE_COLS
from src.explainability.shap_explainer import RULExplainer
from src.logger import get_logger
from src.model.evaluate import evaluate, plot_degradation_curves, plot_rul_scatter
from src.model.lstm import RULPredictor

logger = get_logger(__name__)


def main(config_path: str) -> None:
    """Evaluate the trained LSTM and write all Day-4 artefacts."""
    cfg = load_config(config_path)
    model = RULPredictor.load(Path(cfg["paths"]["models_dir"]))
    test_df = pd.read_parquet(cfg["data"]["processed_test"])
    train_df = pd.read_parquet(cfg["data"]["processed_train"])

    result = evaluate(model, test_df, cfg)
    plot_rul_scatter(result, Path("reports/figures/rul_prediction_vs_actual.png"))
    plot_degradation_curves(
        test_df, model, cfg, Path("reports/figures/degradation_curves.png")
    )

    train_ds = CMAPSSDataset(train_df, cfg["data"]["sequence_length"])
    n = cfg["shap"]["n_background_samples"]
    rng = np.random.default_rng(cfg["data"]["seed"])
    idx = rng.choice(len(train_ds), size=min(n, len(train_ds)), replace=False)
    background = torch.stack([train_ds[i][0] for i in idx])  # [n, 30, 17]
    explainer = RULExplainer(model, background, cfg)

    demo_engine = sorted(test_df["engine_id"].unique())[0]
    eng = test_df[test_df["engine_id"] == demo_engine].sort_values("cycle")
    last = eng[FEATURE_COLS].values[-cfg["data"]["sequence_length"] :]
    window = torch.tensor(last, dtype=torch.float32).unsqueeze(0)
    exp = explainer.explain_engine(window)
    explainer.plot_waterfall(
        exp,
        int(demo_engine),
        exp["predicted_rul"],
        Path("reports/figures/shap_waterfall_sample.png"),
    )
    explainer.save_baseline(background, Path("models/shap_baseline.json"))
    logger.info("evaluation done — reports/figures and shap_baseline.json updated")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    main(ap.parse_args().config)
