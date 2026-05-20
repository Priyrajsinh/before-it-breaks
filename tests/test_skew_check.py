"""Tests for training-stats persistence, skew detection, and PSI (rules C23, C42)."""

import json
from pathlib import Path

import numpy as np

from src.data.skew_check import FEATURE_COLS, check_skew, psi, save_training_stats


def test_save_and_load_training_stats(tmp_path: Path) -> None:
    """save_training_stats writes a JSON file with 17 feature entries."""
    import pandas as pd

    X = pd.DataFrame(np.random.randn(50, 17), columns=FEATURE_COLS)
    stats_path = tmp_path / "training_stats.json"
    save_training_stats(X, stats_path)
    assert stats_path.exists()
    with open(str(stats_path)) as fh:
        stats = json.load(fh)
    assert len(stats) == 17
    assert "mean" in stats[FEATURE_COLS[0]]


def test_save_training_stats_numpy_input(tmp_path: Path) -> None:
    """save_training_stats accepts a numpy array."""
    X = np.random.randn(20, 17)
    stats_path = tmp_path / "training_stats.json"
    save_training_stats(X, stats_path)
    assert stats_path.exists()


def test_check_skew_all_in_range(tmp_path: Path) -> None:
    """check_skew returns all False when the vector is within training bounds."""
    import pandas as pd

    X = pd.DataFrame(np.zeros((10, 17)), columns=FEATURE_COLS)
    stats_path = tmp_path / "stats.json"
    save_training_stats(X, stats_path)
    x_row = np.zeros(17)
    flags = check_skew(x_row, stats_path)
    assert not any(flags.values())


def test_check_skew_detects_outlier(tmp_path: Path) -> None:
    """check_skew returns True for a feature far outside training range."""
    import pandas as pd

    X = pd.DataFrame(np.zeros((10, 17)), columns=FEATURE_COLS)
    stats_path = tmp_path / "stats.json"
    save_training_stats(X, stats_path)
    x_row = np.zeros(17)
    x_row[0] = 999.0
    flags = check_skew(x_row, stats_path)
    assert flags[FEATURE_COLS[0]] is True


def test_psi_identical_distributions() -> None:
    """PSI is near zero when expected and actual are from the same distribution."""
    rng = np.random.default_rng(0)
    arr = rng.normal(size=500)
    result = psi(arr, arr)
    assert result < 0.01


def test_psi_very_different_distributions() -> None:
    """PSI is large when distributions are very different."""
    rng = np.random.default_rng(0)
    expected = rng.normal(loc=0, size=500)
    actual = rng.normal(loc=10, size=500)
    result = psi(expected, actual)
    assert result > 0.2
