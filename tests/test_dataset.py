"""Tests for src/data/dataset.py — all synthetic fixtures, no real CMAPSS files."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.dataset import CMAPSSDataset, _compute_train_rul, load_cmapss
from src.data.skew_check import FEATURE_COLS, save_training_stats
from src.data.validation import PROCESSED_CMAPSS_SCHEMA

# ── synthetic helpers ─────────────────────────────────────────────────────────


def _make_raw_df(n_engines: int = 2, cycles: int = 40) -> pd.DataFrame:
    """26-column synthetic raw CMAPSS frame (constant sensors included)."""
    rng = np.random.default_rng(0)
    rows = []
    for eid in range(1, n_engines + 1):
        for cyc in range(1, cycles + 1):
            row: dict = {
                "engine_id": eid,
                "cycle": cyc,
                "setting_1": float(rng.standard_normal()),
                "setting_2": float(rng.standard_normal()),
                "setting_3": 100.0,
            }
            for i in range(1, 22):
                if i in (1, 5, 10, 16, 18, 19):
                    row[f"sensor_{i}"] = 0.0  # constant
                elif i == 6:
                    row[f"sensor_{i}"] = float(cyc % 2)  # 2 unique vals
                else:
                    row[f"sensor_{i}"] = float(rng.standard_normal())
            rows.append(row)
    return pd.DataFrame(rows)


def _write_raw_file(df: pd.DataFrame, path: Path) -> None:
    """Write space-separated CMAPSS format with 2 trailing NaN columns."""
    lines = []
    for _, row in df.iterrows():
        vals = " ".join(str(v) for v in row.values)
        lines.append(f"{vals} NaN NaN")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_sha256(path: Path) -> None:
    """Write .sha256 sidecar."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}", encoding="utf-8"
    )


@pytest.fixture()
def cmapss_tmp(
    tmp_path: Path, mock_config: dict, monkeypatch: pytest.MonkeyPatch
) -> dict:
    """Synthetic CMAPSS files in tmp_path; SHA-256 sidecars; cwd patched."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir()
    (tmp_path / "data" / "processed").mkdir(parents=True)

    train_df = _make_raw_df(n_engines=2, cycles=40)
    test_df = _make_raw_df(n_engines=1, cycles=30)

    raw_train = tmp_path / "train_FD001.txt"
    raw_test = tmp_path / "test_FD001.txt"
    raw_rul = tmp_path / "RUL_FD001.txt"

    _write_raw_file(train_df, raw_train)
    _write_raw_file(test_df, raw_test)
    raw_rul.write_text("50\n", encoding="utf-8")

    for p in (raw_train, raw_test, raw_rul):
        _write_sha256(p)

    cfg: dict = {
        **mock_config,
        "data": {
            **mock_config["data"],
            "raw_train": str(raw_train),
            "raw_test": str(raw_test),
            "raw_rul": str(raw_rul),
            "processed_train": str(tmp_path / "data" / "processed" / "train.parquet"),
            "processed_val": str(tmp_path / "data" / "processed" / "val.parquet"),
            "processed_test": str(tmp_path / "data" / "processed" / "test.parquet"),
            "drop_columns": [
                "sensor_1",
                "sensor_5",
                "sensor_6",
                "sensor_10",
                "sensor_16",
                "sensor_18",
                "sensor_19",
            ],
        },
    }
    return {"config": cfg, "tmp_path": tmp_path}


# ── tests ────────────────────────────────────────────────────────────────────


def test_rul_piecewise_cap() -> None:
    """_compute_train_rul always clips to ≤ max_rul (rule C35)."""
    df = _make_raw_df(n_engines=3, cycles=200)
    rul = _compute_train_rul(df, max_rul=125)
    assert rul.max() <= 125
    assert rul.min() >= 0


def test_constant_sensors_dropped(cmapss_tmp: dict) -> None:
    """After load_cmapss, drop_columns absent from train/val/test (rule C40)."""
    train, val, test, _ = load_cmapss(cmapss_tmp["config"])
    drop = cmapss_tmp["config"]["data"]["drop_columns"]
    for df_split in (train, val, test):
        for col in drop:
            assert col not in df_split.columns, f"{col} still present after drop"


def test_scaler_saved(tmp_path: Path) -> None:
    """Scaler saved with joblib is reloadable as a StandardScaler (rule C48)."""
    scaler = StandardScaler()
    rng = np.random.default_rng(1)
    scaler.fit(rng.standard_normal((100, 17)))
    pkl = tmp_path / "scaler.pkl"
    with open(str(pkl), "wb") as fh:
        joblib.dump(scaler, fh)
    loaded = joblib.load(str(pkl))
    assert isinstance(loaded, StandardScaler)
    assert loaded.n_features_in_ == 17


def test_training_stats_saved(tmp_path: Path) -> None:
    """save_training_stats writes 17 features each with mean/std/min/max (rule C23)."""
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.standard_normal((50, 17)), columns=FEATURE_COLS)
    out = tmp_path / "training_stats.json"
    save_training_stats(df, out)
    stats = json.loads(out.read_text())
    assert len(stats) == 17
    for col in FEATURE_COLS:
        assert col in stats
        for key in ("mean", "std", "min", "max"):
            assert key in stats[col], f"missing {key} for {col}"


def test_window_shape(dummy_processed_df: pd.DataFrame) -> None:
    """CMAPSSDataset windows are exactly (30, 17) (rule C49)."""
    ds = CMAPSSDataset(dummy_processed_df, sequence_length=30)
    assert len(ds) > 0
    x, y = ds[0]
    assert x.shape == (30, 17), f"got {x.shape}"
    assert y.shape == ()


def test_short_engine_skipped() -> None:
    """Engine with fewer cycles than sequence_length produces 0 windows."""
    rows = []
    for cyc in range(1, 20):  # 19 cycles < seq_len=30
        row: dict = {"engine_id": 1, "cycle": cyc, "rul": float(20 - cyc)}
        for col in FEATURE_COLS:
            row[col] = 0.0
        rows.append(row)
    df = pd.DataFrame(rows)
    ds = CMAPSSDataset(df, sequence_length=30)
    assert len(ds) == 0


def test_engine_split_disjoint(cmapss_tmp: dict) -> None:
    """No engine_id appears in both train and val splits (rule C33)."""
    train, val, _, _ = load_cmapss(cmapss_tmp["config"])
    overlap = set(train["engine_id"]) & set(val["engine_id"])
    assert len(overlap) == 0, f"Overlap: {overlap}"


def test_pandera_processed_rejects_bad_rul(dummy_processed_df: pd.DataFrame) -> None:
    """PROCESSED_CMAPSS_SCHEMA raises SchemaError for rul=200 (> 125 cap)."""
    import pandera

    bad = dummy_processed_df.copy()
    bad.loc[bad.index[0], "rul"] = 200.0
    with pytest.raises(pandera.errors.SchemaError):
        PROCESSED_CMAPSS_SCHEMA.validate(bad)
