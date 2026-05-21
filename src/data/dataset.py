"""CMAPSS FD001 data loading, preprocessing, and sliding-window dataset."""

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.data.skew_check import FEATURE_COLS, save_training_stats
from src.data.validation import validate_processed_cmapss, validate_raw_cmapss
from src.exceptions import ChecksumError, DataLoadError
from src.logger import get_logger

logger = get_logger(__name__)

RAW_COLUMNS = ["engine_id", "cycle", "setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in range(1, 22)
]


def verify_checksum(path: Path) -> None:
    """Verify a raw CMAPSS file matches its .sha256 sidecar (rule C41)."""
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.exists():
        raise DataLoadError(f"missing checksum sidecar for {path}")
    expected = sidecar.read_text().strip().split()[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ChecksumError(f"checksum mismatch for {path}")


def _load_cmapss_raw(path: Path) -> pd.DataFrame:
    """Read a raw CMAPSS space-separated file, drop trailing NaN columns."""
    df = pd.read_csv(path, sep=r"\s+", header=None, index_col=False)
    df = df.dropna(axis=1, how="all")
    if df.shape[1] != 26:
        raise DataLoadError(f"expected 26 cols after NaN drop, got {df.shape[1]}")
    df.columns = RAW_COLUMNS
    float_cols = {c: float for c in RAW_COLUMNS if c not in ("engine_id", "cycle")}
    df = df.astype({"engine_id": int, "cycle": int, **float_cols})
    return validate_raw_cmapss(df)


def _compute_train_rul(df: pd.DataFrame, max_rul: int) -> pd.Series:
    """Piecewise-linear RUL: clip (max_cycle - cycle) at max_rul (rule C35)."""
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    return (max_cycle - df["cycle"]).clip(upper=max_rul).astype(float)


def _compute_test_rul(df: pd.DataFrame, rul_file: Path, max_rul: int) -> pd.Series:
    """RUL for test engines: final_rul + remaining cycles, clipped at max_rul."""
    final_rul = pd.read_csv(rul_file, header=None, names=["final_rul"])
    final_rul["engine_id"] = np.arange(1, len(final_rul) + 1)
    max_cycle = df.groupby("engine_id")["cycle"].transform("max")
    df = df.merge(final_rul, on="engine_id", how="left")
    return (
        (df["final_rul"] + (max_cycle - df["cycle"])).clip(upper=max_rul).astype(float)
    )


def load_cmapss(
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Load + preprocess CMAPSS FD001. Returns (train, val, test, scaler).

    Pipeline:
    1. Verify SHA-256 checksums (rule C41).
    2. pandera-validate raw frames (rule C34).
    3. Drop 7 constant/near-constant sensors (rule C40).
    4. Compute piecewise-linear RUL capped at max_rul=125 (rule C35).
    5. Split BY ENGINE_ID: 1-80 train, 81-100 val (rule C33).
    6. Fit StandardScaler on TRAIN only (rule C48). Save to models/scaler.pkl.
    7. Transform train, val, test.
    8. Save processed parquets + training_stats.json (rule C23).
    9. pandera-validate processed frames (rule C34).
    """
    raw_train = Path(config["data"]["raw_train"])
    raw_test = Path(config["data"]["raw_test"])
    raw_rul = Path(config["data"]["raw_rul"])
    for p in (raw_train, raw_test, raw_rul):
        verify_checksum(p)

    train_full = _load_cmapss_raw(raw_train)
    test = _load_cmapss_raw(raw_test)

    train_full["rul"] = _compute_train_rul(train_full, config["data"]["max_rul"])
    test["rul"] = _compute_test_rul(test, raw_rul, config["data"]["max_rul"])

    drop_cols = config["data"]["drop_columns"]
    train_full = train_full.drop(columns=drop_cols)
    test = test.drop(columns=drop_cols)

    tlo, thi = config["data"]["train_engine_ids"]
    vlo, vhi = config["data"]["val_engine_ids"]
    train = train_full[
        (train_full["engine_id"] >= tlo) & (train_full["engine_id"] <= thi)
    ].copy()
    val = train_full[
        (train_full["engine_id"] >= vlo) & (train_full["engine_id"] <= vhi)
    ].copy()

    if set(train["engine_id"]) & set(val["engine_id"]):
        raise DataLoadError("train and val engine_ids overlap — rule C33 violation")

    scaler = StandardScaler()
    train.loc[:, FEATURE_COLS] = scaler.fit_transform(train[FEATURE_COLS])
    val.loc[:, FEATURE_COLS] = scaler.transform(val[FEATURE_COLS])
    test.loc[:, FEATURE_COLS] = scaler.transform(test[FEATURE_COLS])

    Path("models/").mkdir(exist_ok=True)
    with open("models/scaler.pkl", "wb") as fh:
        joblib.dump(scaler, fh)
    save_training_stats(train[FEATURE_COLS], Path("models/training_stats.json"))

    for df in (train, val, test):
        validate_processed_cmapss(df)

    Path("data/processed/").mkdir(exist_ok=True)
    train.to_parquet(config["data"]["processed_train"])
    val.to_parquet(config["data"]["processed_val"])
    test.to_parquet(config["data"]["processed_test"])

    logger.info(
        "CMAPSS loaded",
        extra={
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_train_engines": train["engine_id"].nunique(),
            "n_val_engines": val["engine_id"].nunique(),
        },
    )
    return train, val, test, scaler


class CMAPSSDataset(Dataset):
    """Sliding-window dataset. Window shape [seq_len, n_features] (rule C49)."""

    def __init__(self, df: pd.DataFrame, sequence_length: int) -> None:
        """Build sliding windows of length sequence_length over each engine's cycles."""
        self.windows: list[np.ndarray] = []
        self.targets: list[float] = []
        for _, eng in df.groupby("engine_id"):
            eng = eng.sort_values("cycle").reset_index(drop=True)
            feats = eng[FEATURE_COLS].values
            ruls = eng["rul"].values
            for i in range(len(eng) - sequence_length + 1):
                w = feats[i : i + sequence_length]
                assert w.shape == (
                    sequence_length,
                    len(FEATURE_COLS),
                ), f"window shape {w.shape} != ({sequence_length}, {len(FEATURE_COLS)})"
                self.windows.append(w)
                self.targets.append(float(ruls[i + sequence_length - 1]))

    def __len__(self) -> int:
        """Return total number of sliding windows."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (window tensor [seq_len, n_features], rul scalar tensor)."""
        return (
            torch.tensor(self.windows[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32),
        )


def get_dataloaders(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config: dict,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build CMAPSSDataset objects and wrap in DataLoaders."""
    seq = config["data"]["sequence_length"]
    bs = config["model"]["batch_size"]
    return (
        DataLoader(CMAPSSDataset(train, seq), batch_size=bs, shuffle=True),
        DataLoader(CMAPSSDataset(val, seq), batch_size=bs, shuffle=False),
        DataLoader(CMAPSSDataset(test, seq), batch_size=bs, shuffle=False),
    )
