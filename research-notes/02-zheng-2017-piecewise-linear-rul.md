# 02 · Zheng et al. (2017) — LSTM for RUL + the piecewise-linear label

**Reference:** Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017). *Long Short-Term Memory Network for Remaining Useful Life Estimation.* IEEE Int'l Conf. on Prognostics and Health Management (ICPHM), pp. 88–95.

---

## The core idea
Stack LSTM layers over a **sliding window** of recent cycles and regress RUL. The recurrence lets the network accumulate evidence of degradation across the window instead of judging each cycle in isolation. On FD001 the paper reports an **RMSE ≈ 16.14** — the baseline our **15.79** sits right alongside.

## The piecewise-linear cap (rule C35)
The paper's most-cited practical contribution is the **target-function** choice. Using raw `RUL = max_cycle − current_cycle` is a mistake because:

> A healthy engine shows **no measurable degradation** for a long initial period, so a label that keeps climbing linearly into the hundreds asks the model to predict a number the sensors give no evidence for.

The fix is a **piecewise-linear RUL**: RUL is **constant (clipped) during the healthy phase** and only **decreases linearly once degradation begins**. Concretely, choose a constant `R_early` and set

```
label = min(raw_RUL, R_early)
```

The classic figure in the paper shows the true (staircase-then-ramp) target overlaid on the naive straight line: flat at the cap, then a clean downward ramp toward zero. The model stops being graded on un-learnable early-life numbers and spends its capacity on the **degradation phase** — exactly where a maintenance decision is made.

## How we set `max_rul = 125`
- We use **`max_rul = 125`**, the value that has become the **de-facto standard for FD001** across the follow-on literature (e.g. Li et al. 2018 use 125; many CMAPSS papers converge on the 120–130 range).
- Implementation: every cycle with raw RUL > 125 is clipped to 125 (`config/config.yaml: data.max_rul: 125`); labels are validated `0 ≤ rul ≤ 125` by the pandera schema **before** training (rule C34).
- Why it's defensible to *fix* rather than tune: 125 is large enough to cover the whole degradation phase of FD001 engines and small enough to discard the meaningless flat-healthy tail — and using the literature-standard value keeps our RMSE directly comparable to published numbers.

## Takeaway for this repo
The piecewise-linear cap is **rule C35 — sacred**. It is the single label-engineering decision that most improves CMAPSS RUL models, and it is *why* our training focuses on the part of an engine's life where predictions actually matter.
