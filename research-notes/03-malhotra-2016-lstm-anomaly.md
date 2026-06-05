# 03 · Malhotra et al. (2016) — LSTM Encoder-Decoder for multi-sensor signals

**Reference:** Malhotra, P., Ramakrishnan, A., Anand, G., Vig, L., Agarwal, P., & Shroff, G. (2016). *LSTM-Based Encoder-Decoder for Multi-Sensor Anomaly Detection.* ICML 2016 Anomaly Detection Workshop.

---

## The core idea
Train an **LSTM encoder-decoder** to reconstruct *normal* multivariate sensor sequences. At inference, the **reconstruction error** becomes a health score: the model reconstructs healthy behaviour well and degraded behaviour badly, so a rising error signals an emerging fault. The follow-on work (Malhotra et al., *"Multi-Sensor Prognostics using an Unsupervised Health Index based on LSTM-ED"*) turns that health index directly into RUL estimation.

## Why LSTM beats feedforward on multi-sensor degradation
- **Memory across time.** Degradation is a *trajectory*, not a state. A feedforward net sees one cycle and must guess; an LSTM's hidden/cell state **carries context across the whole window**, so it can read *trends* — a pressure that is slowly drifting, variance that is slowly growing — rather than an instantaneous value.
- **Joint multi-sensor dynamics.** Faults show up as **changing correlations** between sensors (e.g. temperature rising relative to pressure), not just one channel crossing a threshold. The recurrent state mixes all 17 channels through time, capturing those coupled dynamics.
- **Robust to noise / phase.** Because the LSTM integrates evidence over many cycles, a single noisy reading doesn't derail it — it's the *accumulated* pattern that drives the output. A snapshot classifier has no such smoothing.
- **Empirically:** the paper shows LSTM-ED detecting anomalies in signals (engines, power plants, etc.) that are **predictable but non-periodic and noisy** — precisely the regime CMAPSS sensors live in.

## What we carried into this project
- The conviction that **sequence models are the right inductive bias** for run-to-failure sensor data — our 2-layer LSTM over a 30-cycle window is the supervised-regression cousin of Malhotra's reconstruction idea.
- The framing that *"which sensor is drifting"* is itself the actionable output — we surface it via SHAP instead of reconstruction-error-per-channel, but the spirit (per-sensor warning signal) is the same.
- The mental model behind our **drift monitor**: a model trained on "normal" must be told when the incoming distribution stops being normal (PSI > 0.2), echoing the reconstruction-error-as-health-score logic.
