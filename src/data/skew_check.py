"""Training-stats persistence, per-feature skew check, and PSI metric."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_COLS = ["setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
]


def save_training_stats(X: pd.DataFrame | np.ndarray, path: Path) -> None:
    """Save per-feature mean/std/min/max for the 17 features (rule C23)."""
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=FEATURE_COLS)
    stats = {
        c: {
            "mean": float(X[c].mean()),
            "std": float(X[c].std()),
            "min": float(X[c].min()),
            "max": float(X[c].max()),
        }
        for c in X.columns
        if c in FEATURE_COLS
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(str(path), "w") as fh:
        json.dump(stats, fh, indent=2)


def check_skew(x_row: np.ndarray, stats_path: Path) -> dict[str, bool]:
    """Per-feature out-of-range bool for a single 17-vector (rule C42)."""
    with open(str(stats_path)) as fh:
        stats = json.load(fh)
    out: dict[str, bool] = {}
    for i, c in enumerate(FEATURE_COLS):
        s = stats[c]
        out[c] = bool(x_row[i] < s["min"] or x_row[i] > s["max"])
    return out


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index between two 1D arrays (rule C42)."""
    eps = 1e-6
    breaks = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breaks[0] -= eps
    breaks[-1] += eps
    e_hist, _ = np.histogram(expected, bins=breaks)
    a_hist, _ = np.histogram(actual, bins=breaks)
    e_p = e_hist / max(e_hist.sum(), 1)
    a_p = a_hist / max(a_hist.sum(), 1)
    e_p = np.clip(e_p, eps, None)
    a_p = np.clip(a_p, eps, None)
    return float(np.sum((a_p - e_p) * np.log(a_p / e_p)))
