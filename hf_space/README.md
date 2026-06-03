---
title: Before It Breaks — Engine Health Monitor
emoji: 🛠️
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.14.0
app_file: app.py
pinned: false
license: mit
---

# Before It Breaks · Engine Health Monitor

Predictive maintenance for industrial turbofan engines. Pick an engine and the
app forecasts its **Remaining Useful Life (RUL)** — how many operational cycles
it has left before failure — and names the sensor giving the strongest warning.

- **Model:** 2-layer LSTM reading sliding 30-cycle windows of 17 sensor signals.
- **Dataset:** NASA CMAPSS FD001 (100 train + 100 test engines), split by engine.
- **Performance (held-out test set):** RMSE ≈ 15.8 cycles · MAE ≈ 11.1 cycles.

The pipeline mirrors production: a raw sensor window is scaled with the training
scaler, run through the model, then translated into a plain-English maintenance
recommendation.

Source: [github.com/Priyrajsinh/before-it-breaks](https://github.com/Priyrajsinh/before-it-breaks)
