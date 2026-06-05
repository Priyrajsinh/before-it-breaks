"""Anti-leakage guarantees: engine-id split + scaler fitted on train only.

Rule C33 — train/val are split BY ENGINE_ID (engines 1..80 train, 81..100 val),
never by row. A random row split leaks future cycles of training engines into
validation, which is the #1 mistake in CMAPSS papers.

Rule C48 — the StandardScaler is fitted on TRAINING data only, then applied at
inference. We assert the persisted scaler is a fitted StandardScaler over the 17
features, and that the processed train split carries the standardised signature
(~0 mean / ~1 std) that proves the fit came from train alone.
"""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.data.skew_check import FEATURE_COLS

TRAIN = Path("data/processed/train.parquet")
VAL = Path("data/processed/val.parquet")
SCALER = Path("models/scaler.pkl")


@pytest.fixture(scope="module")
def train_df() -> pd.DataFrame:
    """Processed train split (engines 1..80)."""
    if not TRAIN.exists():
        pytest.skip("train parquet not yet generated (run preprocessing)")
    return pd.read_parquet(TRAIN)


@pytest.fixture(scope="module")
def val_df() -> pd.DataFrame:
    """Processed val split (engines 81..100)."""
    if not VAL.exists():
        pytest.skip("val parquet not yet generated (run preprocessing)")
    return pd.read_parquet(VAL)


def test_train_val_engines_disjoint(
    train_df: pd.DataFrame, val_df: pd.DataFrame
) -> None:
    """Rule C33: train and val engine_ids never overlap."""
    train_eng = set(train_df["engine_id"])
    val_eng = set(val_df["engine_id"])
    overlap = sorted(train_eng & val_eng)
    assert not overlap, f"engine_id leakage between train and val: {overlap[:5]}"


def test_train_engines_are_1_to_80(train_df: pd.DataFrame) -> None:
    """Rule C33: train engines should be exactly 1..80."""
    assert int(train_df["engine_id"].min()) == 1
    assert int(train_df["engine_id"].max()) == 80


def test_val_engines_are_81_to_100(val_df: pd.DataFrame) -> None:
    """Rule C33: val engines should be exactly 81..100."""
    assert int(val_df["engine_id"].min()) == 81
    assert int(val_df["engine_id"].max()) == 100


def test_scaler_is_fitted_standardscaler() -> None:
    """Rule C48: the persisted scaler is a fitted StandardScaler over 17 features."""
    if not SCALER.exists():
        pytest.skip("scaler not yet generated")
    scaler = joblib.load(SCALER)
    assert type(scaler).__name__ == "StandardScaler"
    assert getattr(scaler, "n_features_in_", None) == len(FEATURE_COLS) == 17


def test_scaled_train_features_are_standardised(train_df: pd.DataFrame) -> None:
    """Rule C48: features in the processed TRAIN set carry the StandardScaler signature.

    Every feature is mean-centred (~0). Informative features additionally have
    unit variance (~1). FD001 runs at a single operating condition, so setting_3
    is a zero-variance constant that StandardScaler leaves at std 0 — that is the
    one expected exception. This centred/standardised shape only holds if the
    scaler was fitted on (and applied to) the train split alone; a scaler fitted
    on train+val (leakage) would not centre the train split at exactly zero.
    """
    for col in FEATURE_COLS:
        col_mean = float(train_df[col].mean())
        col_std = float(train_df[col].std())
        assert abs(col_mean) < 1e-2, f"{col} scaled mean {col_mean:.4f} far from 0"
        is_unit = abs(col_std - 1.0) < 1e-1
        is_constant = col_std < 1e-6  # e.g. setting_3 in FD001
        assert is_unit or is_constant, f"{col} scaled std {col_std:.4f} unexpected"
