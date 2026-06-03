"""HF Space — Engine Health Monitor (Gradio).

Self-contained per rule C12 (NEVER import from ``src/``). Streaming
yield-generator per rule C43. Glassmorphism UI per rule C44. 3-tab recruiter
UX per rule C19. Plain-English output, no "SHAP"/"LSTM"/"tensor" jargon on the
user-facing surface per rule C45.

Gradio 6 note: ``gr.Blocks`` no longer accepts a ``css=`` argument (removed in
v6). Custom CSS is injected via a ``gr.HTML`` ``<style>`` block (reliable on HF
Spaces) and also passed to ``demo.launch(css=...)``.
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
    """Load model weights, scaler, training stats, importance baseline, test set."""
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


def _health(rul: float) -> tuple[str, str]:
    """Map a predicted RUL to a (status, colour) pair."""
    if rul >= HEALTHY_MIN:
        return "HEALTHY", "#22c55e"
    if rul >= WARNING_MIN:
        return "WARNING", "#f59e0b"
    return "CRITICAL", "#ef4444"


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
    self-contained — this is a transparent heuristic, not a SHAP attribution.
    """
    last = scaled_window[-1]
    scores: dict[str, float] = {}
    for i, col in enumerate(FEATURE_COLS):
        st = TRAIN_STATS[col]
        std = st["std"] if st["std"] > 1e-6 else 1.0
        deviation = abs(float(last[i]) - st["mean"]) / std
        scores[col] = deviation * float(IMPORTANCE.get(col, 0.0))
    return scores


def predict_streaming(engine_id: int) -> Iterator[tuple[str, str, str]]:
    """Streaming generator endpoint (rule C43) — animates the pipeline.

    Yields (badge_html, headline_html, body_html). Progress messages appear in
    the body slot while the badge/headline stay empty until the result is ready.
    """
    engine_id = int(engine_id)

    def progress(msg: str) -> tuple[str, str, str]:
        return "", "", f"<div class='step'>{msg}</div>"

    yield progress("Loading the engine's last 30 operational cycles…")
    window = _last_window(engine_id)
    time.sleep(0.45)

    yield progress("Scaling 17 sensor channels with the training scaler…")
    scaled = _scale(window)
    time.sleep(0.45)

    yield progress("Reading the degradation trend across all 30 cycles…")
    time.sleep(0.45)
    with torch.no_grad():
        rul = max(float(MODEL(torch.tensor(scaled).unsqueeze(0)).item()), 0.0)
    status, colour = _health(rul)

    yield progress("Comparing each sensor against its healthy baseline…")
    time.sleep(0.45)
    scores = _warning_scores(scaled)
    top_key = max(scores.items(), key=lambda kv: kv[1])[0]
    top_label = SENSOR_LABELS.get(top_key, top_key)

    action = {
        "HEALTHY": "No action needed — keep operating normally.",
        "WARNING": "Plan maintenance within the next service window.",
        "CRITICAL": "Schedule maintenance immediately — failure is imminent.",
    }[status]
    badge = f"<div class='badge' style='background:{colour}'>{status}</div>"
    headline = (
        f"<h2 class='headline'>Engine {engine_id} — about "
        f"{rul:.0f} cycles of life remaining</h2>"
    )
    summary = (
        "<div class='summary'>"
        f"<p>Engine <b>{engine_id}</b> is predicted to reach end-of-life in "
        f"approximately <b>{rul:.0f} cycles</b>. Status: <b>{status}</b>. {action}</p>"
        f"<p class='muted'>Strongest warning signal right now: <b>{top_label}</b>. "
        "One cycle is roughly a full flight, from takeoff to landing.</p>"
        "</div>"
    )
    yield badge, headline, summary


def warning_chart(engine_id: int):
    """Tab 2 — per-engine warning contribution bar chart (responds to selector)."""
    engine_id = int(engine_id)
    scaled = _scale(_last_window(engine_id))
    scores = _warning_scores(scaled)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    labels = [SENSOR_LABELS.get(k, k) for k, _ in ordered]
    values = [v for _, v in ordered]

    fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 1))
    ax.barh(labels, values, color="#0ea5e9")
    ax.set_xlabel("warning contribution  (sensor abnormality × model reliance)")
    ax.set_title(f"What is driving the forecast for engine {engine_id}")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


CSS = """
.gradio-container {
  background: linear-gradient(135deg, #0c4a6e 0%, #4338ca 100%) !important;
}
.glass {
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  background: rgba(255, 255, 255, 0.90) !important;
  border-radius: 16px !important;
  padding: 20px !important;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.18) !important;
}
.badge {
  display: inline-block; color: #fff; font-weight: 800; letter-spacing: .5px;
  padding: 8px 22px; border-radius: 999px; font-size: 1.05rem;
  animation: slideUp .4s ease;
}
.headline {
  margin: .5rem 0 0; font-size: 1.5rem; font-weight: 800;
  background: linear-gradient(135deg, #0ea5e9, #6366f1);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
  animation: slideUp .45s ease;
}
.summary { animation: slideUp .5s ease; line-height: 1.55; }
.summary .muted { color: #64748b; font-size: .92em; }
.step {
  color: #475569; font-style: italic; padding: 6px 0; animation: slideUp .3s ease;
}
@keyframes slideUp {
  from { transform: translateY(16px); opacity: 0; }
  to   { transform: translateY(0);    opacity: 1; }
}
"""

with gr.Blocks(title="Before It Breaks — Engine Health Monitor") as demo:
    gr.HTML(f"<style>{CSS}</style>")  # rule C44 — glassmorphism, Gradio-6 safe
    gr.Markdown("# Before It Breaks · Engine Health Monitor")
    gr.Markdown(
        "> Predictive maintenance for industrial turbofan engines · "
        "NASA CMAPSS FD001"
    )

    with gr.Tab("Engine Health Monitor"):
        with gr.Row():
            with gr.Column(scale=1, elem_classes="glass"):
                engine_dd = gr.Dropdown(
                    choices=ENGINE_IDS,
                    value=ENGINE_IDS[0],
                    label="Select an engine to analyse",
                )
                predict_btn = gr.Button("Predict remaining life", variant="primary")
            with gr.Column(scale=2, elem_classes="glass"):
                badge_out = gr.HTML()
                headline_out = gr.HTML()
                summary_out = gr.HTML()
        predict_btn.click(
            fn=predict_streaming,
            inputs=engine_dd,
            outputs=[badge_out, headline_out, summary_out],
        )

    with gr.Tab("Sensor Analysis"):
        with gr.Column(elem_classes="glass"):
            gr.Markdown(
                "For the selected engine, each sensor's latest reading is compared "
                "against its healthy baseline and weighted by how much the model "
                "relies on it. Taller bars are the strongest warning signals."
            )
            sensor_dd = gr.Dropdown(
                choices=ENGINE_IDS, value=ENGINE_IDS[0], label="Engine"
            )
            sensor_btn = gr.Button("Show sensor contributions")
            sensor_plot = gr.Plot()
            sensor_btn.click(fn=warning_chart, inputs=sensor_dd, outputs=sensor_plot)

    with gr.Tab("How It Works"):
        gr.Markdown(
            """
## What this does
An unplanned engine failure mid-operation can cost hundreds of thousands of
dollars per hour in downtime and lost revenue. This system learns from
historical sensor data and predicts how many more operational cycles an engine
has before failure — its **Remaining Useful Life (RUL)** — so maintenance can be
scheduled *before it breaks*.

## The pipeline
```
Sensor data (21 readings × 30 cycles)
    -> drop 7 constant sensors            (17 informative features remain)
    -> scale with the training scaler     (fitted on training engines only)
    -> read the 30-cycle degradation trend
    -> RUL forecast                       (0 = failure, 125 = fully healthy)
    -> health status + strongest warning signal
```

## Dataset
NASA CMAPSS FD001 · 100 train + 100 test engines · 21 sensor channels per cycle.
Trained on engines 1–80, validated on 81–100, tested on the held-out NASA test
set — split **by engine**, never by row, so no future cycle ever leaks into
training.

## Model performance (held-out test set)
- **RMSE ≈ 15.8 cycles**
- **MAE ≈ 11.1 cycles**
- NASA Score ≈ 674 (asymmetric — late predictions are penalised more than early)

## Built by
Priyrajsinh Parmar ·
[github.com/Priyrajsinh/before-it-breaks](https://github.com/Priyrajsinh/before-it-breaks)
"""
        )


if __name__ == "__main__":
    demo.queue().launch(css=CSS)
