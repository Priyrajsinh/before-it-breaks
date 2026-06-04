"""HF Space — Engine Health Monitor (Gradio).

Self-contained per rule C12 (NEVER import from ``src/``). Streaming
yield-generator per rule C43. 3-tab recruiter UX per rule C19. Plain-English
output, no "SHAP"/"LSTM"/"tensor" jargon on the user-facing surface (rule C45).

Design: a "Light Console" dashboard (clean white surfaces, ink text, a single
indigo accent, soft status tints) built on a custom ``gr.themes.Base`` light
theme. The theme's ``_dark`` variants are pinned to the light values so the UI
renders identically regardless of the visitor's system dark-mode preference —
this is what guarantees text is always legible. (Supersedes the literal
glassmorphism of rule C44 at the user's request for a production-grade look.)

Gradio 6: ``theme`` and ``css`` are passed to ``demo.launch(...)`` (both moved
off the ``Blocks`` constructor in v6).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import gradio as gr
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gradio.themes.utils import colors, fonts
from torch import nn

matplotlib.use("Agg")
plt.switch_backend("Agg")  # rule C15 — headless backend after imports

# Artefacts sit next to this file on the HF Space; resolve relative to __file__
# so both `cd hf_space && python app.py` and `python -m hf_space.app` work.
_HERE = Path(__file__).resolve().parent

# 17 features in the exact order the scaler was fitted (rule C40 — 7 constant
# sensors already dropped): 3 settings + 14 sensors.
FEATURE_COLS = ["setting_1", "setting_2", "setting_3"] + [
    f"sensor_{i}" for i in [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
]
SEQ_LEN = 30
N_FEATURES = 17
MAX_RUL = 125  # piecewise-linear cap (rule C35) — also the "100% life" reference
HEALTHY_MIN = 80  # config evaluation.health_status_thresholds.healthy_min
WARNING_MIN = 40  # config evaluation.health_status_thresholds.warning_min

# Plain-English names so the UI never shows raw "sensor_11" (rule C45).
# Standard NASA CMAPSS FD001 channel descriptions.
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
ACTION = {
    "HEALTHY": "Status is nominal — continue normal operation.",
    "WARNING": "Plan maintenance within the next service window.",
    "CRITICAL": "Schedule maintenance immediately — failure is imminent.",
}
STEPS = ["Load window", "Scale sensors", "Forecast life", "Find warning"]


class RULPredictor(nn.Module):
    """2-layer LSTM — layer names match the saved state_dict (lstm/dropout/fc)."""

    def __init__(self) -> None:
        """Build the LSTM stack with the trained hyper-parameters."""
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
        """x: [batch, 30, 17] -> [batch] predicted RUL."""
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


def _load_artefacts() -> tuple:
    """Load model weights, scaler, training stats, importance baseline, windows."""
    model = RULPredictor()
    state = torch.load(_HERE / "lstm_rul.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    with open(_HERE / "scaler.pkl", "rb") as fh:
        scaler = joblib.load(fh)
    with open(_HERE / "training_stats.json") as fh:
        stats = json.load(fh)
    with open(_HERE / "shap_baseline.json") as fh:
        importance = json.load(fh)
    # Raw (unscaled) last-30 windows per engine, built from the NASA test file.
    # Scaling happens in _scale() so the pipeline mirrors production (rule C48).
    test_df = pd.read_parquet(_HERE / "test_windows.parquet")
    return model, scaler, stats, importance, test_df


MODEL, SCALER, TRAIN_STATS, IMPORTANCE, TEST_DF = _load_artefacts()
ENGINE_IDS = sorted(int(e) for e in TEST_DF["engine_id"].unique())


def _health(rul: float) -> str:
    """Map a predicted RUL to a health-status key."""
    if rul >= HEALTHY_MIN:
        return "HEALTHY"
    if rul >= WARNING_MIN:
        return "WARNING"
    return "CRITICAL"


def _last_window(engine_id: int) -> np.ndarray:
    """Return the engine's final 30-cycle window as a raw [30, 17] array."""
    eng = TEST_DF[TEST_DF["engine_id"] == engine_id].sort_values("cycle")
    if len(eng) < SEQ_LEN:
        raise gr.Error(
            f"Engine {engine_id} has only {len(eng)} cycles (need {SEQ_LEN})."
        )
    return eng[FEATURE_COLS].to_numpy(dtype=np.float32)[-SEQ_LEN:]


def _scale(window: np.ndarray) -> np.ndarray:
    """Scale a raw window with the training scaler (rule C48).

    A named DataFrame is passed so the scaler (fitted with feature names) does
    not emit a missing-feature-names warning.
    """
    framed = pd.DataFrame(window, columns=FEATURE_COLS)
    return SCALER.transform(framed).astype(np.float32)


def _warning_scores(scaled_window: np.ndarray) -> dict[str, float]:
    """Per-sensor warning contribution for THIS engine.

    Combines how abnormal each sensor's latest reading is (its distance from the
    healthy training baseline, in std units — training_stats is in scaled space)
    with how much the model relies on that sensor. Varies per engine, so the
    engine selector genuinely drives the output (rule C28). Honest and
    self-contained — a transparent heuristic, not a SHAP attribution.
    """
    last = scaled_window[-1]
    scores: dict[str, float] = {}
    for i, col in enumerate(FEATURE_COLS):
        st = TRAIN_STATS[col]
        std = st["std"] if st["std"] > 1e-6 else 1.0
        deviation = abs(float(last[i]) - st["mean"]) / std
        scores[col] = deviation * float(IMPORTANCE.get(col, 0.0))
    return scores


def _predict(engine_id: int) -> tuple[float, str, float, str]:
    """Run the full pipeline once -> (rul, status, pct_life, top_sensor_label)."""
    scaled = _scale(_last_window(engine_id))
    with torch.no_grad():
        rul = max(float(MODEL(torch.tensor(scaled).unsqueeze(0)).item()), 0.0)
    status = _health(rul)
    pct = min(rul / MAX_RUL * 100.0, 100.0)
    scores = _warning_scores(scaled)
    top_label = SENSOR_LABELS.get(max(scores, key=lambda k: scores[k]), "—")
    return rul, status, pct, top_label


# --------------------------------------------------------------------------- #
# HTML fragments
# --------------------------------------------------------------------------- #

HEADER_HTML = """
<div class="app-header">
  <div class="brand">
    <span class="brand-mark">◆</span>
    <span class="brand-name">Before It Breaks</span>
    <span class="brand-sep">/</span>
    <span class="brand-sub">Engine Health Monitor</span>
  </div>
  <div class="brand-tagline">
    Predictive maintenance for industrial turbofan engines · NASA CMAPSS FD001
  </div>
</div>
"""

PLACEHOLDER_HTML = """
<div class="result placeholder">
  <div class="ph-mark">◇</div>
  <div>Select an engine and press <b>Analyze</b><br>
       to forecast its remaining life.</div>
</div>
"""


def _stepper_html(active: int) -> str:
    """Render the 4-step pipeline tracker; steps < active are done."""
    cells = []
    for i, name in enumerate(STEPS):
        cls = "done" if i < active else ("active" if i == active else "")
        cells.append(f'<div class="step {cls}"><span class="sdot"></span>{name}</div>')
    return '<div class="stepper">' + "".join(cells) + "</div>"


def _loading_html(active: int, msg: str) -> str:
    """Render the animated loading card for one pipeline step."""
    return (
        '<div class="result loading">'
        '<div class="loader-row"><span class="dot-pulse"></span>'
        f"<span>{msg}</span></div>"
        f"{_stepper_html(active)}"
        "</div>"
    )


def _result_html(engine_id: int, rul: float, status: str, pct: float, top: str) -> str:
    """Render the final result card."""
    s = STATUS[status]
    return f"""
<div class="result">
  <div class="pill" style="color:{s['color']};background:{s['tint']};
       border-color:{s['border']}">
    <span class="pdot" style="background:{s['color']}"></span>{status}
  </div>
  <div class="metric">
    <span class="metric-value" style="color:{s['color']}">{rul:.0f}</span>
    <span class="metric-unit">cycles<br>remaining</span>
  </div>
  <div class="lifebar">
    <div class="lifebar-track">
      <div class="lifebar-fill" style="width:{pct:.0f}%;background:{s['bar']}"></div>
    </div>
    <div class="lifebar-label">{pct:.0f}% of expected service life remaining</div>
  </div>
  <p class="summary">Engine <b>{engine_id}</b> is predicted to reach end-of-life
     in about <b>{rul:.0f} cycles</b>. {ACTION[status]}</p>
  <div class="signal">
    <span class="signal-label">Strongest warning signal</span>
    <span class="signal-chip">{top}</span>
  </div>
</div>
"""


def analyze_static(engine_id: int) -> str:
    """Instant (non-streaming) result — used for the on-load default view."""
    engine_id = int(engine_id)
    rul, status, pct, top = _predict(engine_id)
    return _result_html(engine_id, rul, status, pct, top)


def predict_streaming(engine_id: int) -> Iterator[str]:
    """Streaming generator endpoint (rule C43) — animates the pipeline stepper."""
    engine_id = int(engine_id)
    yield _loading_html(0, "Loading the engine's last 30 operational cycles…")
    window = _last_window(engine_id)
    time.sleep(0.4)

    yield _loading_html(1, "Scaling the sensor channels with the training scaler…")
    scaled = _scale(window)
    time.sleep(0.4)

    yield _loading_html(2, "Reading the degradation trend and forecasting life…")
    time.sleep(0.4)
    with torch.no_grad():
        rul = max(float(MODEL(torch.tensor(scaled).unsqueeze(0)).item()), 0.0)
    status = _health(rul)

    yield _loading_html(3, "Pinpointing the strongest warning signal…")
    time.sleep(0.4)
    scores = _warning_scores(scaled)
    top = SENSOR_LABELS.get(max(scores, key=lambda k: scores[k]), "—")
    pct = min(rul / MAX_RUL * 100.0, 100.0)
    yield _result_html(engine_id, rul, status, pct, top)


def warning_chart(engine_id: int):
    """Tab 2 — per-engine warning-contribution bar chart (light-themed)."""
    engine_id = int(engine_id)
    scaled = _scale(_last_window(engine_id))
    scores = _warning_scores(scaled)
    ordered = sorted(scores.items(), key=lambda kv: kv[1])  # ascending -> top wins
    labels = [SENSOR_LABELS.get(k, k) for k, _ in ordered]
    values = [v for _, v in ordered]
    top_val = max(values) if values else 1.0

    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(labels) + 1.2), dpi=120)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    bar_colors = ["#4F46E5" if v == top_val else "#C7CBF7" for v in values]
    ax.barh(labels, values, color=bar_colors, height=0.62, zorder=3)
    ax.set_title(
        f"What is driving the forecast for engine {engine_id}",
        color="#0F172A",
        fontsize=13,
        fontweight="bold",
        loc="left",
        pad=12,
    )
    ax.set_xlabel(
        "warning contribution  ·  sensor abnormality × model reliance",
        color="#64748B",
        fontsize=10,
    )
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#E5E9F0")
    ax.tick_params(axis="y", colors="#0F172A", length=0, labelsize=10)
    ax.tick_params(axis="x", colors="#94A3B8", labelsize=9)
    ax.xaxis.grid(True, color="#EEF2F7", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


STATS_HTML = """
<div class="stats">
  <div class="stat"><div class="stat-val">15.8</div>
    <div class="stat-key">Test RMSE</div>
    <div class="stat-sub">cycles · held-out engines</div></div>
  <div class="stat"><div class="stat-val">11.1</div>
    <div class="stat-key">Test MAE</div>
    <div class="stat-sub">cycles · held-out engines</div></div>
  <div class="stat"><div class="stat-val">674</div>
    <div class="stat-key">NASA Score</div>
    <div class="stat-sub">asymmetric · lower is better</div></div>
</div>
"""

HOWITWORKS_MD = """
### What this does
An unplanned engine failure mid-operation can cost hundreds of thousands of
dollars per hour. This tool predicts how many more operational cycles an engine
has before failure — its **Remaining Useful Life** — so maintenance can be
scheduled *before it breaks*.

### The pipeline
1. Take the engine's last **30 cycles** of sensor readings.
2. Drop 7 constant sensors → **17 informative features**.
3. Scale them with the **training scaler** (fitted on training engines only).
4. Read the degradation trend → forecast RUL (**0** = failure, **125** = fully healthy).
5. Translate to a health status + the strongest warning signal.

### Dataset & method
NASA CMAPSS FD001 · 100 train + 100 test engines · 21 sensor channels per cycle.
Trained on engines 1–80, validated on 81–100, tested on the held-out NASA set —
split **by engine**, never by row, so no future cycle ever leaks into training.

Built by **Priyrajsinh Parmar** ·
[github.com/Priyrajsinh/before-it-breaks](https://github.com/Priyrajsinh/before-it-breaks)
"""


# Pick a dramatic default (the most critical engine) for the on-load view.
def _default_engine() -> int:
    """Return the engine with the lowest predicted RUL (most critical)."""
    worst, worst_rul = ENGINE_IDS[0], float("inf")
    for e in ENGINE_IDS:
        rul, _, _, _ = _predict(e)
        if rul < worst_rul:
            worst, worst_rul = e, rul
    return worst


DEFAULT_ENGINE = _default_engine()


# --------------------------------------------------------------------------- #
# Theme + CSS
# --------------------------------------------------------------------------- #

THEME = gr.themes.Base(
    primary_hue=colors.indigo,
    secondary_hue=colors.indigo,
    neutral_hue=colors.slate,
    font=(fonts.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
    font_mono=(fonts.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
).set(
    body_background_fill="#F7F9FC",
    body_background_fill_dark="#F7F9FC",
    block_background_fill="#FFFFFF",
    block_background_fill_dark="#FFFFFF",
    block_border_width="1px",
    block_border_color="#E5E9F0",
    block_border_color_dark="#E5E9F0",
    block_radius="16px",
    block_label_background_fill="#FFFFFF",
    block_label_text_color="#64748B",
    body_text_color="#0F172A",
    body_text_color_dark="#0F172A",
    body_text_color_subdued="#64748B",
    body_text_color_subdued_dark="#64748B",
    border_color_primary="#E5E9F0",
    border_color_primary_dark="#E5E9F0",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#E5E9F0",
    input_border_color_dark="#E5E9F0",
    button_primary_background_fill="#4F46E5",
    button_primary_background_fill_hover="#4338CA",
    button_primary_background_fill_dark="#4F46E5",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_large_radius="12px",
    button_small_radius="10px",
)

CSS = """
.gradio-container { max-width: 1080px !important; margin: 0 auto !important; }
.gradio-container * { -webkit-font-smoothing: antialiased; }

/* header */
.app-header { padding: 6px 2px 2px; }
.brand { display:flex; align-items:center; gap:10px; font-size:22px;
         font-weight:800; color:#0F172A; letter-spacing:-.2px; }
.brand-mark { width:28px; height:28px; border-radius:9px; color:#fff; font-size:13px;
  display:inline-flex; align-items:center; justify-content:center;
  background:linear-gradient(135deg,#6366F1,#4F46E5);
  box-shadow:0 2px 6px rgba(79,70,229,.35); }
.brand-sep { color:#CBD5E1; font-weight:400; }
.brand-sub { font-weight:600; color:#334155; }
.brand-tagline { margin-top:6px; color:#64748B; font-size:14px; }

/* card panels (applied to gradio columns via elem_classes) */
.panel { background:#FFFFFF !important; border:1px solid #E5E9F0 !important;
  border-radius:16px !important; padding:22px !important;
  box-shadow:0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.05) !important; }

.section-title { font-size:16px; font-weight:700; color:#0F172A; margin:0 0 6px; }
.muted-text { color:#64748B; font-size:13.5px; line-height:1.6; }

/* result card */
.result { min-height:300px; display:flex; flex-direction:column;
  justify-content:center; gap:18px; padding:4px 2px; }
.result.loading { justify-content:flex-start; gap:18px; }
.placeholder { align-items:center; text-align:center; color:#94A3B8;
  justify-content:center; gap:12px; }
.placeholder .ph-mark { font-size:34px; opacity:.55; }

/* status pill */
.pill { display:inline-flex; align-items:center; gap:8px; align-self:flex-start;
  font-weight:700; font-size:12.5px; letter-spacing:.5px; padding:6px 14px;
  border-radius:999px; border:1px solid; }
.pill .pdot { width:8px; height:8px; border-radius:50%; }

/* hero metric */
.metric { display:flex; align-items:baseline; gap:14px; }
.metric-value { font-size:78px; line-height:.9; font-weight:800;
  letter-spacing:-3px; font-variant-numeric:tabular-nums; }
.metric-unit { font-size:15px; color:#64748B; font-weight:600; line-height:1.25; }

/* life bar */
.lifebar-track { height:10px; border-radius:999px; background:#EEF2F7;
  overflow:hidden; }
.lifebar-fill { height:100%; border-radius:999px;
  transition:width .7s cubic-bezier(.4,0,.2,1); }
.lifebar-label { margin-top:8px; font-size:13px; color:#64748B; }

.summary { font-size:15px; line-height:1.6; color:#334155; margin:0; }

/* warning-signal chip */
.signal { display:flex; align-items:center; gap:10px; flex-wrap:wrap; }
.signal-label { font-size:13px; color:#64748B; }
.signal-chip { font-weight:600; font-size:13px; color:#4F46E5; background:#EEF0FE;
  border:1px solid #E0E2FB; padding:5px 12px; border-radius:8px; }

/* loading + stepper */
.loader-row { display:flex; align-items:center; gap:12px; color:#0F172A;
  font-weight:600; font-size:15px; }
.dot-pulse { width:12px; height:12px; border-radius:50%; background:#4F46E5;
  animation:pulse 1.2s infinite; }
@keyframes pulse {
  0% { box-shadow:0 0 0 0 rgba(79,70,229,.45); }
  70% { box-shadow:0 0 0 10px rgba(79,70,229,0); }
  100% { box-shadow:0 0 0 0 rgba(79,70,229,0); } }
.stepper { display:flex; gap:8px; }
.step { flex:1; display:flex; align-items:center; gap:8px; font-size:12px;
  color:#94A3B8; padding:10px 12px; border:1px solid #E5E9F0; border-radius:10px;
  background:#FBFCFE; transition:all .25s ease; }
.step .sdot { width:13px; height:13px; border-radius:50%; border:2px solid #CBD5E1;
  display:inline-block; flex:0 0 auto; }
.step.active { color:#4F46E5; border-color:#C7CBF7; background:#F2F3FE; }
.step.active .sdot { border-color:#4F46E5; background:#4F46E5;
  box-shadow:0 0 0 3px #E5E7FB; }
.step.done { color:#047857; border-color:#BBF7D0; background:#F0FDF6; }
.step.done .sdot { border-color:#10B981; background:#10B981; }

/* metrics strip (How It Works) */
.stats { display:flex; gap:14px; flex-wrap:wrap; margin:2px 0 6px; }
.stat { flex:1; min-width:150px; background:#FFFFFF; border:1px solid #E5E9F0;
  border-radius:14px; padding:16px 18px; }
.stat-val { font-size:30px; font-weight:800; color:#0F172A;
  font-variant-numeric:tabular-nums; letter-spacing:-1px; }
.stat-key { font-size:11.5px; color:#64748B; font-weight:700; text-transform:uppercase;
  letter-spacing:.7px; margin-top:4px; }
.stat-sub { font-size:12px; color:#94A3B8; margin-top:2px; }

footer { display:none !important; }
"""


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #

with gr.Blocks(title="Before It Breaks — Engine Health Monitor") as demo:
    gr.HTML(f"<style>{CSS}</style>")  # belt-and-suspenders alongside launch(css=)
    gr.HTML(HEADER_HTML)

    with gr.Tabs():
        with gr.Tab("Engine Health Monitor"):
            with gr.Row():
                with gr.Column(scale=2, min_width=230, elem_classes="panel"):
                    gr.HTML('<div class="section-title">Select engine</div>')
                    engine_dd = gr.Dropdown(
                        choices=ENGINE_IDS, value=DEFAULT_ENGINE, label="Test engine"
                    )
                    analyze_btn = gr.Button("Analyze  →", variant="primary")
                    gr.HTML(
                        '<div class="muted-text" style="margin-top:12px">'
                        "100 unseen NASA test engines. One cycle ≈ one full flight, "
                        "from takeoff to landing.</div>"
                    )
                with gr.Column(scale=5, elem_classes="panel"):
                    result_out = gr.HTML(PLACEHOLDER_HTML)
            analyze_btn.click(predict_streaming, engine_dd, result_out)

        with gr.Tab("Sensor Analysis"):
            with gr.Column(elem_classes="panel"):
                gr.HTML(
                    '<div class="section-title">Sensor contributions</div>'
                    '<div class="muted-text">For the selected engine, each sensor’s '
                    "latest reading is compared with its healthy baseline and weighted "
                    "by how much the model relies on it. Taller bars are the strongest "
                    "warning signals.</div>"
                )
                with gr.Row():
                    sensor_dd = gr.Dropdown(
                        choices=ENGINE_IDS,
                        value=DEFAULT_ENGINE,
                        label="Engine",
                        scale=2,
                    )
                    sensor_btn = gr.Button(
                        "Show contributions", variant="primary", scale=1
                    )
                sensor_plot = gr.Plot()
                sensor_btn.click(warning_chart, sensor_dd, sensor_plot)

        with gr.Tab("How It Works"):
            with gr.Column(elem_classes="panel"):
                gr.HTML(STATS_HTML)
                gr.Markdown(HOWITWORKS_MD)

    # Populate both tabs on load so nothing is blank (rule C28 — driven by inputs).
    demo.load(analyze_static, engine_dd, result_out)
    demo.load(warning_chart, sensor_dd, sensor_plot)


if __name__ == "__main__":
    demo.queue().launch(theme=THEME, css=CSS)
