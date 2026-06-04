"""Streamlit 4-tab dashboard for *Before It Breaks* (rules C19, C42, C44).

Single, self-contained module — Streamlit Cloud deploys from one file. The four
tabs are: **Engine Health Monitor** (recruiter UX, plain English — rule C19),
**Sensor Analysis** (per-engine sensor trends + warning contributions),
**Drift Monitoring** (per-sensor PSI vs the training distribution — rule C42),
and **How It Works** (the piecewise-linear-RUL + engine-id-split narrative).

Design: a "Light Console" look — clean white surfaces, ink text, a single indigo
accent, soft status tints, Inter-ish system sans. The base palette is driven by
``.streamlit/config.toml`` (a real theme, not ``!important`` patching); this file
adds a thin layer of CSS for the result card, status pill and life bar. This is
the same direction shipped on the Gradio HF Space and supersedes the literal
glassmorphism of rule C44 at the owner's request for a production-grade look.

Two non-obvious correctness points baked in here:

* ``data/processed/test.parquet`` is **already StandardScaler-scaled** (the
  scaler was fitted on training engines, rule C48, then applied when the
  processed parquet was written). It is therefore fed to the model **without
  re-scaling** — re-scaling would double-standardise the window and wreck the
  forecast. Verified to reproduce ``reports/results.json`` to the cycle.
* The strongest-warning-signal output is computed **per engine** (rule C28) from
  each engine's own last window — abnormality vs the healthy baseline weighted by
  the model's reliance on that channel — so the engine selector genuinely drives
  the result rather than showing one global ranking.

No "SHAP" / "LSTM" / "tensor" jargon appears on the user-facing surface (rule
C45); raw ``sensor_11``-style names are mapped to plain English everywhere shown.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, NamedTuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import nn

matplotlib.use("Agg")
plt.switch_backend("Agg")  # rule C15 — headless backend after all imports

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# 17 features in the exact order the scaler/model expect (rule C40 — 7 constant
# sensors already dropped): 3 operational settings + 14 informative sensors.
FEATURE_COLS: list[str] = ["setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
]
SEQ_LEN = 30  # sliding-window length (rule C49)
N_FEATURES = 17
MAX_RUL = 125  # piecewise-linear cap (rule C35) — also the "100% life" reference
HEALTHY_MIN = 80  # config evaluation.health_status_thresholds.healthy_min
WARNING_MIN = 40  # config evaluation.health_status_thresholds.warning_min
PSI_THRESHOLD = 0.2  # rule C42 — config monitoring.psi_threshold
PSI_EPS = 1e-6  # proportion floor so empty bins never divide by zero

# Light Console palette.
INK = "#0F172A"
MUTED = "#64748B"
ACCENT = "#4F46E5"

# Plain-English channel names so the UI never shows raw "sensor_11" (rule C45).
SENSOR_LABELS: dict[str, str] = {
    "setting_1": "Altitude Setting",
    "setting_2": "Mach Setting",
    "setting_3": "Throttle Setting",
    "sensor_2": "LPC Outlet Temp",
    "sensor_3": "HPC Outlet Temp",
    "sensor_4": "LPT Outlet Temp",
    "sensor_7": "HPC Outlet Pressure",
    "sensor_8": "Fan Speed",
    "sensor_9": "Core Speed",
    "sensor_11": "HPC Static Pressure",
    "sensor_12": "Fuel-to-Pressure Ratio",
    "sensor_13": "Corrected Fan Speed",
    "sensor_14": "Corrected Core Speed",
    "sensor_15": "Bypass Ratio",
    "sensor_17": "Bleed Enthalpy",
    "sensor_20": "HPT Coolant Bleed",
    "sensor_21": "LPT Coolant Bleed",
}

# Status design tokens (text colour / soft tint / border / progress-bar fill).
STATUS: dict[str, dict[str, str]] = {
    "HEALTHY": {
        "color": "#047857",
        "tint": "#ECFDF5",
        "border": "#A7F3D0",
        "bar": "#10B981",
    },
    "WARNING": {
        "color": "#B45309",
        "tint": "#FFFBEB",
        "border": "#FDE68A",
        "bar": "#F59E0B",
    },
    "CRITICAL": {
        "color": "#BE123C",
        "tint": "#FFF1F2",
        "border": "#FECDD3",
        "bar": "#F43F5E",
    },
}
ACTION: dict[str, str] = {
    "HEALTHY": "Status is nominal — continue normal operation.",
    "WARNING": "Plan maintenance within the next service window.",
    "CRITICAL": "Schedule maintenance immediately — failure is imminent.",
}

_DEPLOY_HINT = (
    "Commit the small model artefacts (models/, reports/results.json, "
    "data/processed/test.parquet) or download them at startup — see "
    "MANUAL_TASKS.md (Day 7 — Streamlit Cloud)."
)


# --------------------------------------------------------------------------- #
# Model + artefact loading
# --------------------------------------------------------------------------- #


class RULPredictor(nn.Module):
    """2-layer LSTM regressor — layer names match the saved ``state_dict``."""

    def __init__(self) -> None:
        """Build the LSTM/dropout/linear stack with the trained hyper-parameters."""
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a ``[batch, 30, 17]`` window batch to ``[batch]`` predicted RUL."""
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


class Artifacts(NamedTuple):
    """Everything the dashboard needs, loaded once and cached."""

    model: RULPredictor
    stats: dict[str, dict[str, float]]
    importance: dict[str, float]
    drift_ref: dict[str, dict[str, Any]]
    results: dict[str, Any]
    test_df: pd.DataFrame
    engine_ids: list[int]


def _artifact_root() -> Path:
    """Return the dir holding ``models/``, ``data/`` and ``reports/``.

    Defaults to the repository root (two levels above this file) so a plain
    ``streamlit run src/dashboard/streamlit_app.py`` works; ``BIB_ARTIFACT_ROOT``
    overrides it (used by the test-suite to point at synthetic artefacts).
    """
    override = os.environ.get("BIB_ARTIFACT_ROOT")
    return Path(override) if override else Path(__file__).resolve().parents[2]


@st.cache_resource(show_spinner="Loading model and data…")
def load_artifacts(root_str: str) -> Artifacts:
    """Load the model, training stats, importance, drift reference, metrics, data.

    ``root_str`` is part of the cache key so different artefact roots (e.g. in
    tests) never collide. The LSTM weights are loaded with ``weights_only=True``
    (no arbitrary-object deserialisation — keeps bandit's B614 happy).
    """
    root = Path(root_str)
    model = RULPredictor()
    state = torch.load(
        root / "models" / "lstm_rul.pt", map_location="cpu", weights_only=True
    )
    model.load_state_dict(state)
    model.eval()
    with open(root / "models" / "training_stats.json", encoding="utf-8") as fh:
        stats = json.load(fh)
    with open(root / "models" / "shap_baseline.json", encoding="utf-8") as fh:
        importance = json.load(fh)
    with open(root / "models" / "drift_reference.json", encoding="utf-8") as fh:
        drift_ref = json.load(fh)
    with open(root / "reports" / "results.json", encoding="utf-8") as fh:
        results = json.load(fh)
    test_df = pd.read_parquet(root / "data" / "processed" / "test.parquet")
    engine_ids = sorted(int(e) for e in test_df["engine_id"].unique())
    return Artifacts(model, stats, importance, drift_ref, results, test_df, engine_ids)


# --------------------------------------------------------------------------- #
# Pure compute helpers (unit-tested directly, no Streamlit runtime needed)
# --------------------------------------------------------------------------- #


def health_status(rul: float) -> str:
    """Map a predicted RUL in cycles to HEALTHY / WARNING / CRITICAL."""
    if rul >= HEALTHY_MIN:
        return "HEALTHY"
    if rul >= WARNING_MIN:
        return "WARNING"
    return "CRITICAL"


def last_window(test_df: pd.DataFrame, engine_id: int) -> np.ndarray:
    """Return an engine's final 30-cycle window as a pre-scaled ``[30, 17]`` array."""
    eng = test_df[test_df["engine_id"] == engine_id].sort_values("cycle")
    return eng[FEATURE_COLS].to_numpy(dtype=np.float32)[-SEQ_LEN:]


def predict_rul(model: RULPredictor, window: np.ndarray) -> float:
    """Forward a single pre-scaled ``[30, 17]`` window -> non-negative RUL."""
    with torch.no_grad():
        value = float(model(torch.tensor(window).unsqueeze(0)).item())
    return max(value, 0.0)


def warning_scores(
    window: np.ndarray,
    stats: dict[str, dict[str, float]],
    importance: dict[str, float],
) -> dict[str, float]:
    """Per-sensor warning contribution for one engine (rule C28).

    Combines how abnormal each sensor's latest reading is (distance from the
    healthy training baseline, in std units — stats are in scaled space) with how
    much the model relies on that channel. A transparent heuristic, not a SHAP
    attribution, so it stays self-contained and varies per engine.
    """
    last = window[-1]
    scores: dict[str, float] = {}
    for i, col in enumerate(FEATURE_COLS):
        ref = stats[col]
        std = ref["std"] if ref["std"] > 1e-6 else 1.0
        deviation = abs(float(last[i]) - ref["mean"]) / std
        scores[col] = deviation * float(importance.get(col, 0.0))
    return scores


def _result_card_html(
    engine_id: int, rul: float, status: str, pct: float, top_label: str
) -> str:
    """Build the Light-Console result-card HTML for the health tab."""
    s = STATUS[status]
    return f"""<div class="bib-card">
  <span class="bib-pill" style="color:{s['color']};background:{s['tint']};
        border-color:{s['border']}">
    <span class="bib-dot" style="background:{s['color']}"></span>{status}</span>
  <div class="bib-metric">
    <span class="bib-value" style="color:{s['color']}">{rul:.0f}</span>
    <span class="bib-unit">cycles<br>remaining</span></div>
  <div class="bib-track">
    <div class="bib-fill" style="width:{pct:.0f}%;background:{s['bar']}"></div></div>
  <div class="bib-sub">{pct:.0f}% of expected service life remaining</div>
  <p class="bib-summary">Engine <b>{engine_id}</b> is predicted to reach
     end-of-life in about <b>{rul:.0f} cycles</b>. {ACTION[status]}</p>
  <div class="bib-signal"><span>Strongest warning signal</span>
    <span class="bib-chip">{top_label}</span></div>
</div>"""


# --------------------------------------------------------------------------- #
# Matplotlib charts (thread-safe OO API — never the global pyplot state machine)
# --------------------------------------------------------------------------- #


def _style_axes(ax: Axes, xlabel: str) -> None:
    """Apply the shared light-theme axis styling to a chart."""
    ax.set_facecolor("#FFFFFF")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#E5E9F0")
    ax.tick_params(axis="y", colors=INK, length=0, labelsize=10)
    ax.tick_params(axis="x", colors="#94A3B8", labelsize=9)
    ax.set_xlabel(xlabel, color=MUTED, fontsize=10)
    ax.xaxis.grid(True, color="#EEF2F7", zorder=0)
    ax.set_axisbelow(True)


def warning_chart(scores: dict[str, float], engine_id: int) -> Figure:
    """Horizontal bar chart of per-sensor warning contributions for one engine."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1])  # ascending → top wins
    labels = [SENSOR_LABELS.get(k, k) for k, _ in ordered]
    values = [v for _, v in ordered]
    top = max(values) if values else 1.0
    fig = Figure(figsize=(7.6, 0.42 * len(labels) + 1.0), dpi=120)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#FFFFFF")
    ax = fig.subplots()
    colours = [ACCENT if v == top else "#C7CBF7" for v in values]
    ax.barh(labels, values, color=colours, height=0.62, zorder=3)
    _style_axes(ax, "warning contribution  ·  abnormality × model reliance")
    ax.set_title(
        f"What is driving the forecast for engine {engine_id}",
        color=INK,
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    fig.tight_layout()
    return fig


def trend_chart(eng_df: pd.DataFrame, sensors: list[str]) -> Figure:
    """Stacked line charts of the top warning sensors over the engine's life."""
    fig = Figure(figsize=(7.6, 5.4), dpi=120)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor("#FFFFFF")
    axes = fig.subplots(len(sensors), 1, sharex=True)
    axes = np.atleast_1d(axes)
    for ax, sensor in zip(axes, sensors):
        ax.plot(eng_df["cycle"], eng_df[sensor], color=ACCENT, linewidth=1.8, zorder=3)
        ax.set_facecolor("#FFFFFF")
        ax.set_ylabel(SENSOR_LABELS.get(sensor, sensor), color=INK, fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.grid(True, color="#EEF2F7", zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors="#94A3B8", labelsize=9)
    axes[-1].set_xlabel("operational cycle", color=MUTED, fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Tab renderers
# --------------------------------------------------------------------------- #


def _render_health_tab(art: Artifacts) -> None:
    """Tab 1 — recruiter-facing engine health monitor (plain English, rule C19)."""
    left, right = st.columns([1, 2], gap="large")
    with left:
        engine = st.selectbox(
            "Select engine",
            art.engine_ids,
            key="health_engine",
            help="100 unseen NASA test engines from the CMAPSS benchmark.",
        )
        st.button("Analyze  →", type="primary", key="health_btn")
        st.caption("One cycle ≈ one full flight, from takeoff to landing.")
    with right:
        window = last_window(art.test_df, engine)
        rul = predict_rul(art.model, window)
        status = health_status(rul)
        pct = min(rul / MAX_RUL * 100.0, 100.0)
        scores = warning_scores(window, art.stats, art.importance)
        top_label = SENSOR_LABELS.get(max(scores, key=lambda k: scores[k]), "—")
        st.markdown(
            _result_card_html(engine, rul, status, pct, top_label),
            unsafe_allow_html=True,
        )
        st.markdown("**Top contributing signals**")
        peak = max(scores.values()) or 1.0
        for col, val in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]:
            st.progress(
                min(val / peak, 1.0),
                text=f"{SENSOR_LABELS.get(col, col)} — contribution score {val:.2f}",
            )


def _render_sensor_tab(art: Artifacts) -> None:
    """Tab 2 — per-engine sensor trends + warning contributions + headline metrics."""
    engine = st.selectbox("Engine", art.engine_ids, key="sensor_engine")
    window = last_window(art.test_df, engine)
    scores = warning_scores(window, art.stats, art.importance)
    top3 = [
        k for k, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
    ]
    eng_df = art.test_df[art.test_df["engine_id"] == engine].sort_values("cycle")

    st.markdown("**Top warning sensors over this engine's life**")
    st.pyplot(trend_chart(eng_df, top3))
    st.markdown("**Why the model forecast this — per-sensor contributions**")
    st.pyplot(warning_chart(scores, engine))

    st.markdown("**Held-out test-set performance**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test RMSE", f"{art.results['rmse']:.1f} cycles")
    c2.metric("Test MAE", f"{art.results['mae']:.1f} cycles")
    c3.metric("NASA Score", f"{art.results['nasa_score']:.0f}")


# --------------------------------------------------------------------------- #
# CSS + entry point
# --------------------------------------------------------------------------- #

CSS = """
<style>
.block-container { max-width: 1080px; padding-top: 2.2rem; }
#MainMenu, footer, header [data-testid="stToolbar"] { visibility: hidden; }

.bib-header { display:flex; align-items:center; gap:10px; font-size:24px;
  font-weight:800; color:#0F172A; letter-spacing:-.3px; }
.bib-mark { width:30px; height:30px; border-radius:9px; color:#fff; font-size:14px;
  display:inline-flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#6366F1,#4F46E5);
  box-shadow:0 2px 6px rgba(79,70,229,.35); }
.bib-slash { color:#CBD5E1; font-weight:400; }
.bib-role { font-weight:600; color:#334155; }
.bib-tag { margin:6px 0 4px; color:#64748B; font-size:14px; }

.bib-card { background:#FFFFFF; border:1px solid #E5E9F0; border-radius:16px;
  padding:22px 24px; display:flex; flex-direction:column; gap:14px;
  box-shadow:0 1px 3px rgba(16,24,40,.05); }
.bib-pill { display:inline-flex; align-items:center; gap:8px; align-self:flex-start;
  font-weight:700; font-size:12.5px; letter-spacing:.5px; padding:6px 14px;
  border-radius:999px; border:1px solid; }
.bib-dot { width:8px; height:8px; border-radius:50%; }
.bib-metric { display:flex; align-items:baseline; gap:14px; }
.bib-value { font-size:72px; line-height:.9; font-weight:800; letter-spacing:-3px;
  font-variant-numeric:tabular-nums; }
.bib-unit { font-size:15px; color:#64748B; font-weight:600; line-height:1.25; }
.bib-track { height:10px; border-radius:999px; background:#EEF2F7; overflow:hidden; }
.bib-fill { height:100%; border-radius:999px;
  transition:width .7s cubic-bezier(.4,0,.2,1); }
.bib-sub { font-size:13px; color:#64748B; margin-top:-6px; }
.bib-summary { font-size:15px; line-height:1.6; color:#334155; margin:0; }
.bib-signal { display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  font-size:13px; color:#64748B; }
.bib-chip { font-weight:600; font-size:13px; color:#4F46E5; background:#EEF0FE;
  border:1px solid #E0E2FB; padding:5px 12px; border-radius:8px; }
</style>
"""


def _inject_css() -> None:
    """Inject the thin Light-Console CSS layer (cards / pill / life bar)."""
    st.markdown(CSS, unsafe_allow_html=True)


def _render_header() -> None:
    """Render the brand bar shared with the Gradio Space."""
    st.markdown(
        '<div class="bib-header"><span class="bib-mark">◆</span>'
        '<span class="bib-name">Before It Breaks</span>'
        '<span class="bib-slash">/</span>'
        '<span class="bib-role">Engine Health Dashboard</span></div>'
        '<div class="bib-tag">Predictive maintenance for industrial turbofan '
        "engines · NASA CMAPSS FD001</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the dashboard shell — tabs are layered in over the next commits."""
    st.set_page_config(page_title="Before It Breaks", page_icon="🔧", layout="wide")
    _inject_css()
    _render_header()
    try:
        art = load_artifacts(str(_artifact_root()))
    except FileNotFoundError as exc:
        st.error(f"Model artefacts not found. {_DEPLOY_HINT}")
        st.caption(str(exc))
        return
    tabs = st.tabs(
        ["Engine Health Monitor", "Sensor Analysis", "Drift Monitoring", "How It Works"]
    )
    with tabs[0]:
        _render_health_tab(art)
    with tabs[1]:
        _render_sensor_tab(art)
    for tab in tabs[2:]:
        with tab:
            st.info("This view is coming online shortly.")


if __name__ == "__main__":
    main()
