"""Tests for the Streamlit 4-tab dashboard (Day 7).

The pure compute helpers are tested directly (no Streamlit runtime). The UI is
exercised end-to-end with Streamlit's ``AppTest`` against *synthetic* artefacts
written to a tmp dir, so the whole module is covered without the real (git-
ignored) model files being present in CI.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from streamlit.testing.v1 import AppTest

from src.dashboard import streamlit_app as app

_APP_PATH = str(Path(app.__file__).resolve())


def _stats() -> dict[str, dict[str, float]]:
    """Synthetic scaled training stats (mean 0, std 1; setting_3 constant)."""
    stats = {
        c: {"mean": 0.0, "std": 1.0, "min": -3.0, "max": 3.0} for c in app.FEATURE_COLS
    }
    stats["setting_3"] = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return stats


def _importance() -> dict[str, float]:
    """Synthetic per-feature importance baseline."""
    return {c: 0.05 for c in app.FEATURE_COLS}


def _ref_bins(x: np.ndarray, bins: int = 10) -> tuple[np.ndarray, np.ndarray]:
    """Build (ref_pct, edges) from a sample the way drift_reference.json does."""
    q = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    edges = q.astype(float).copy()
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(x, bins=edges)
    return np.clip(counts / counts.sum(), 1e-6, None), edges


def _drift_ref() -> dict[str, dict]:
    """Synthetic per-feature training reference (standard-normal; setting_3 const)."""
    rng = np.random.default_rng(7)
    ref: dict[str, dict] = {}
    for c in app.FEATURE_COLS:
        if c == "setting_3":
            ref[c] = {"edges": [], "ref_pct": [1.0]}
            continue
        x = rng.standard_normal(5000)
        q = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        edges = q.astype(float).copy()
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(x, bins=edges)
        pct = np.clip(counts / counts.sum(), 1e-6, None)
        ref[c] = {
            "edges": [float(v) for v in q[1:-1]],
            "ref_pct": [float(v) for v in pct],
        }
    return ref


def _synthetic_test_df(engines: int = 3, cycles: int = 40) -> pd.DataFrame:
    """Build a small pre-scaled test frame with the dashboard's exact columns."""
    rng = np.random.default_rng(0)
    rows = []
    for engine in range(1, engines + 1):
        for cycle in range(1, cycles + 1):
            row: dict = {"engine_id": engine, "cycle": cycle}
            for col in app.FEATURE_COLS:
                row[col] = 0.0 if col == "setting_3" else float(rng.standard_normal())
            row["rul"] = float(min(cycles - cycle, app.MAX_RUL))
            rows.append(row)
    return pd.DataFrame(rows)


def _write_artifacts(root: Path) -> None:
    """Write a complete set of synthetic artefacts under ``root``."""
    (root / "models").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)
    (root / "data" / "processed").mkdir(parents=True)
    torch.save(  # nosec B614 — state_dict save, not an untrusted load
        app.RULPredictor().state_dict(), root / "models" / "lstm_rul.pt"
    )
    with open(root / "models" / "training_stats.json", "w", encoding="utf-8") as fh:
        json.dump(_stats(), fh)
    with open(root / "models" / "shap_baseline.json", "w", encoding="utf-8") as fh:
        json.dump(_importance(), fh)
    with open(root / "models" / "drift_reference.json", "w", encoding="utf-8") as fh:
        json.dump(_drift_ref(), fh)
    with open(root / "reports" / "results.json", "w", encoding="utf-8") as fh:
        json.dump(
            {"rmse": 15.8, "mae": 11.1, "nasa_score": 674.0, "test_set_size": 3}, fh
        )
    _synthetic_test_df().to_parquet(root / "data" / "processed" / "test.parquet")


# --------------------------------------------------------------------------- #
# Pure compute helpers
# --------------------------------------------------------------------------- #


def test_health_status_bands() -> None:
    """Health bands respect the 80 / 40 cycle thresholds at their edges."""
    assert app.health_status(125) == "HEALTHY"
    assert app.health_status(80) == "HEALTHY"
    assert app.health_status(79.9) == "WARNING"
    assert app.health_status(40) == "WARNING"
    assert app.health_status(39.9) == "CRITICAL"
    assert app.health_status(0) == "CRITICAL"


def test_last_window_shape(dummy_processed_df: pd.DataFrame) -> None:
    """The final window is exactly [30, 17] in feature order."""
    window = app.last_window(dummy_processed_df, 1)
    assert window.shape == (app.SEQ_LEN, app.N_FEATURES)
    assert window.dtype == np.float32


def test_predict_rul_is_finite_and_non_negative() -> None:
    """A forward pass returns a finite, clamped-non-negative float."""
    model = app.RULPredictor()
    window = np.zeros((app.SEQ_LEN, app.N_FEATURES), dtype=np.float32)
    rul = app.predict_rul(model, window)
    assert isinstance(rul, float)
    assert np.isfinite(rul)
    assert rul >= 0.0


def test_warning_scores_cover_all_features() -> None:
    """Warning scores are produced for every feature and are non-negative."""
    window = np.ones((app.SEQ_LEN, app.N_FEATURES), dtype=np.float32) * 2.0
    scores = app.warning_scores(window, _stats(), _importance())
    assert set(scores) == set(app.FEATURE_COLS)
    assert all(v >= 0.0 for v in scores.values())


def test_psi_near_zero_for_same_distribution() -> None:
    """PSI of a fresh sample against its own training bins is ~0."""
    rng = np.random.default_rng(1)
    ref_pct, edges = _ref_bins(rng.standard_normal(20000))
    actual = rng.standard_normal(20000)
    assert app.population_stability_index(ref_pct, edges, actual) < 0.05


def test_psi_positive_for_shifted_distribution() -> None:
    """A clear mean shift drives PSI past the alarm threshold."""
    rng = np.random.default_rng(2)
    ref_pct, edges = _ref_bins(rng.standard_normal(20000))
    actual = rng.standard_normal(20000) + 3.0
    assert app.population_stability_index(ref_pct, edges, actual) > app.PSI_THRESHOLD


def test_compute_drift_shape_and_constant_handling() -> None:
    """Drift frame has one row per feature, is sorted, and constant feature -> 0."""
    drift = app.compute_drift(_synthetic_test_df(), _drift_ref())
    assert list(drift.columns) == ["sensor", "PSI"]
    assert len(drift) == app.N_FEATURES
    assert (drift["PSI"].to_numpy()[:-1] >= drift["PSI"].to_numpy()[1:]).all()
    throttle_psi = drift.loc[drift["sensor"] == "Throttle Setting", "PSI"].iloc[0]
    assert throttle_psi == 0.0


def test_compute_drift_single_engine_scope() -> None:
    """Restricting the cohort to one engine still returns a full drift frame."""
    drift = app.compute_drift(_synthetic_test_df(), _drift_ref(), scope_engine=2)
    assert len(drift) == app.N_FEATURES


def test_result_card_html_is_plain_english() -> None:
    """The result card carries the status + RUL and never leaks model jargon."""
    html = app._result_card_html(42, 38.0, "CRITICAL", 30.4, "HPC Static Pressure")
    assert "CRITICAL" in html
    assert "38" in html
    assert "HPC Static Pressure" in html
    for jargon in ("SHAP", "LSTM", "tensor"):
        assert jargon not in html


# --------------------------------------------------------------------------- #
# Chart builders return Figures (thread-safe OO API)
# --------------------------------------------------------------------------- #


def test_chart_builders_return_figures() -> None:
    """Every chart builder returns a matplotlib Figure without using pyplot."""
    df = _synthetic_test_df()
    scores = app.warning_scores(app.last_window(df, 1), _stats(), _importance())
    assert app.warning_chart(scores, 1).axes
    top3 = sorted(scores, key=lambda k: scores[k], reverse=True)[:3]
    assert app.trend_chart(df[df["engine_id"] == 1], top3).axes
    assert app.drift_chart(app.compute_drift(df, _drift_ref())).axes


# --------------------------------------------------------------------------- #
# End-to-end UI via AppTest (synthetic artefacts)
# --------------------------------------------------------------------------- #


def test_app_renders_four_tabs(tmp_path: Path, monkeypatch) -> None:
    """The dashboard runs cleanly and renders the four named tabs (rule C19)."""
    _write_artifacts(tmp_path)
    monkeypatch.setenv("BIB_ARTIFACT_ROOT", str(tmp_path))
    st.cache_resource.clear()
    at = AppTest.from_file(_APP_PATH).run(timeout=30)
    assert not at.exception
    assert len(at.tabs) == 4
    assert any("Analyze" in b.label for b in at.button)
    assert len(at.selectbox) >= 3
    # Drift tab always renders an alarm state (success or error) — rule C42.
    assert len(at.success) + len(at.error) >= 1


def test_app_reacts_to_selector_and_button(tmp_path: Path, monkeypatch) -> None:
    """Changing the engine selector and clicking Analyze re-runs without error."""
    _write_artifacts(tmp_path)
    monkeypatch.setenv("BIB_ARTIFACT_ROOT", str(tmp_path))
    st.cache_resource.clear()
    at = AppTest.from_file(_APP_PATH).run(timeout=30)
    at.selectbox[0].set_value(3).run(timeout=30)
    assert not at.exception
    at.button[0].click().run(timeout=30)
    assert not at.exception


def test_app_handles_missing_artifacts(tmp_path: Path, monkeypatch) -> None:
    """With no artefacts the app shows a graceful error, not a crash."""
    monkeypatch.setenv("BIB_ARTIFACT_ROOT", str(tmp_path))
    st.cache_resource.clear()
    at = AppTest.from_file(_APP_PATH).run(timeout=30)
    assert not at.exception
    assert len(at.error) >= 1
    assert "not found" in at.error[0].value.lower()
