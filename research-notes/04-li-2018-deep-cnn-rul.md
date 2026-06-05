# 04 · Li et al. (2018) — Deep CNN for RUL

**Reference:** Li, X., Ding, Q., & Sun, J.-Q. (2018). *Remaining Useful Life Estimation in Prognostics Using Deep Convolution Neural Networks.* Reliability Engineering & System Safety, 172, 1–11.

---

## The core idea
Instead of a recurrent network, treat the **sliding window `[time × sensors]` as a 2-D image** and run a **deep 1-D CNN** along the time axis. Stacked convolutions learn local degradation motifs (short-window slopes, bumps), and the receptive field grows with depth until it spans the window. The paper reports **state-of-the-art RMSE on FD001 (≈ 12.6)** and notably **also uses a piecewise-linear RUL with the 125 cap** — independent corroboration of note 02's labelling choice.

## The CNN-vs-LSTM trade-off
| | **CNN (Li 2018)** | **LSTM (this repo / Zheng 2017)** |
|---|---|---|
| Inductive bias | local, translation-invariant motifs | order-aware sequential memory |
| Parallelism / speed | highly parallel, fast to train | sequential, slower per step |
| Long-range dependence | needs depth/dilation to reach | native via cell state |
| Headline FD001 RMSE | ~12.6 (often lower) | ~16 (ours: 15.79) |
| Interpretability story | feature-map saliency, less intuitive per-cycle | "reads the last 30 cycles like a trend" — matches the operator's mental model |

**When CNNs win:** raw accuracy on fixed-length windows, and training throughput — convolutions parallelise where recurrence can't, so CNNs are often both faster *and* a point or two lower on RMSE for FD001.

## Why we still picked the LSTM
- **Sequential interpretability.** The LSTM's "read the last 30 cycles as a sequence" maps **one-to-one onto how a maintenance engineer reasons** about an engine's operational history — the window *is* the engine's recent flight history. That narrative clarity is worth a small RMSE premium for a portfolio/decision-support tool.
- **The sliding window matches the industrial-cycle mental model.** One cycle ≈ one flight; a 30-cycle window ≈ "the last 30 flights." Stakeholders grasp it instantly.
- **SHAP on a sequence is cleaner to *tell*.** `GradientExplainer` over `[1, 30, 17]` averaged across time yields a per-sensor warning signal that reads naturally as "this sensor has been the warning over the recent history."
- **Honest accounting:** we record the CNN's accuracy edge as a known trade-off, not a blind spot. The goal here is a *defensible, interpretable* RUL system, and ~15.8 RMSE is competitive while keeping the temporal-degradation story front and centre.
