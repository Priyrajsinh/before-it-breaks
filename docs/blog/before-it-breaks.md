# How I Built a Production-Grade Predictive Maintenance System for German Manufacturing Interviews

*A walkthrough of "Before It Breaks" — an LSTM that forecasts how many cycles a jet engine has left, why every engineering decision was made, and how the whole thing maps to the EU AI Act.*

---

## The hook

Imagine you run a fleet of turbofan engines. Each one is healthy today. In 40 cycles, one of them will fail — but you don't know which, and you don't know when. You have two bad options: replace parts too early and burn money on perfectly good components, or wait too long and eat an unplanned failure. Predictive maintenance is the third option: **read the sensor history, predict the remaining useful life (RUL), and schedule the repair in the narrow window where it's neither wasteful nor too late.**

That is exactly what I built. The model takes the last 30 cycles of an engine's 17 sensor readings and answers one question a maintenance engineer actually asks: *"How many cycles do I have left, and which sensor is the warning signal?"*

## The problem: downtime is the most expensive number in manufacturing

In heavy industry, unplanned downtime is routinely quoted at **~$250,000 per hour** for a critical production line, and far more in aviation, where a grounded aircraft cascades into missed flights and crew reshuffling. The entire discipline of Prognostics and Health Management (PHM) exists to convert that surprise cost into a *scheduled* cost.

The catch: degradation is a **trajectory, not a snapshot**. A single sensor reading at one moment tells you almost nothing. What matters is the *trend* — a pressure slowly drifting, a temperature creeping up, variance widening across a window of operating cycles. Any model that classifies one cycle at a time is structurally blind to the thing that actually predicts failure.

## The dataset: NASA CMAPSS FD001

I used **NASA's C-MAPSS FD001** benchmark (Saxena & Goebel, 2008) — the industry-standard turbofan run-to-failure dataset. It contains:

- **100 training engines**, each run from healthy until failure.
- **100 test engines**, each cut off at some point before failure, with a separate file giving the true RUL at that cutoff.
- **21 sensors + 3 operational settings** recorded every cycle.

Each engine starts healthy and degrades over its operational life. The training data shows full run-to-failure trajectories; the test data asks the model to predict RUL from a partial history. It's the perfect benchmark because the answer is known and published baselines exist to compare against.

## The technical stack

The pipeline is deliberately boring in the best way — every stage is validated, typed, and tested:

1. **pandera schema validation** on raw and processed data, *before* any model sees it. Dtype or range errors get caught at the source, not after a 50-epoch training run.
2. **Drop 7 constant sensors** (sensor 1, 5, 6, 10, 16, 18, 19) that are flat across every engine's life. They add parameters and noise without signal. 17 features remain.
3. **A 2-layer LSTM** (hidden size 128, dropout 0.2) reading a sliding window of `[30 cycles × 17 features]`.
4. **SHAP GradientExplainer** to identify which sensor is the biggest warning signal for each individual engine.
5. **A drift monitor** computing per-sensor Population Stability Index (PSI) on incoming data versus the training distribution.
6. **FastAPI** serving `/predict` and `/explain` with Pydantic validation, rate limiting, and Prometheus metrics, plus two live front-ends (a streaming Gradio Space and a 4-tab Streamlit dashboard).

The result on the canonical test set: **RMSE 15.79 cycles, MAE 11.06 cycles, NASA Score 673.5** across all 100 engines — sitting right alongside Zheng et al.'s (2017) published LSTM baseline of ~16.14.

## The five sacred rules

Most of the engineering value isn't the model architecture — it's five decisions that separate a defensible system from a leaky demo.

### 1. Piecewise-linear RUL labelling (cap at 125)

The naive label is `RUL = max_cycle − current_cycle`. It's wrong. A healthy engine shows **no measurable degradation** for a long initial period, so a label that keeps climbing into the hundreds asks the model to predict a number the sensors give no evidence for. Following Zheng et al. (2017), I cap RUL at **125**: it's constant during the healthy phase and only ramps down once degradation begins. This focuses the model's capacity on the degradation phase — exactly where a maintenance decision is made.

### 2. Split by engine ID, never by row

This is the single most important anti-leakage decision in CMAPSS work. If you split the rows randomly, future cycles of a *training* engine leak into validation — the model "validates" on data whose recent history it has already memorised, and your metrics lie. I split by **engine ID**: engines 1–80 train, 81–100 validate. Disjoint units, no temporal leakage. There's a CI test that fails the build if those sets ever overlap.

### 3. Save the `state_dict`, not the pickled module

`torch.save(model, ...)` pickles the entire module object and breaks the moment you refactor `__init__`. I save `model.state_dict()` instead — just the learned weights — and reconstruct the architecture explicitly at load time. This is the difference between a model you can still load in six months and one you can't.

### 4. Fit the scaler on training data only

The `StandardScaler` is fitted on the **training split alone**, saved to disk, and `transform()`-applied to every inference window. Fitting on train+val (or worse, on the test set) leaks distributional information and inflates your numbers. The model server refuses to start without the saved scaler — there's no silent path to an unscaled prediction. A CI test asserts the processed train features carry the standardised signature that only a train-only fit produces.

### 5. GradientExplainer with a 100-window background

SHAP's TreeExplainer is exact — for tree ensembles. An LSTM is a differentiable deep model, so it needs **GradientExplainer**, which integrates input-gradients against a *background* dataset. I pass 100 random training windows as that background; the raw output of shape `[1, 30, 17]` is averaged over the sequence axis to get a per-feature importance vector. The result, translated to plain English, is the line a maintenance engineer wants: *"Top warning signal: HPC Static Pressure."*

## The EU AI Act angle

This is where the project stops being a Kaggle notebook and starts looking like something a German manufacturer can actually deploy. Under the **EU AI Act**, a predictive-maintenance system like this is **high-risk** on two counts:

- **Annex III §2 — Critical infrastructure.** When it manages or operates machinery in aviation, energy, or transport.
- **Annex III §4 — Employment & workers management.** When its outputs inform maintenance worker scheduling.

High-risk classification triggers concrete obligations, and the repo is built to satisfy them: **documented training-data lineage** (pandera schemas + DVC tracking + SHA-256 checksums), **risk management** (the drift monitor and a coverage-regression CI gate that fails if RMSE crosses 30 cycles), and **human oversight** (the output is a plain-English recommendation — informational, never an autonomous repair order). For a Bosch, Siemens, or Schaeffler interview, *that* compliance story is often more interesting than the architecture.

## The results

| Metric | This model | Benchmark |
|--------|-----------|-----------|
| RMSE | **15.79 cycles** | Zheng 2017 LSTM ≈ 16.14 |
| MAE | **11.06 cycles** | — |
| NASA Score | **673.5** | lower is better (asymmetric) |
| Test engines | **100 / 100** | full coverage |

The NASA Score is asymmetric on purpose: late predictions are penalised more harshly than early ones, because in the real world it is always better to warn too early than too late. That single asymmetry encodes the entire business logic of maintenance.

## See it live

- **Gradio (Hugging Face Space)** — a streaming "Engine Health Monitor" that animates the whole pipeline: input window → preprocess → LSTM forward pass → SHAP attribution → plain-English recommendation.
- **Streamlit Cloud** — a 4-tab dashboard: Engine Health, Sensor Analysis + SHAP, Drift Monitoring (per-sensor PSI), and a How-It-Works explainer.

Both are linked from the repository README, along with the Swagger docs you get from `make serve`.

## Lessons learned

- **The label is the model.** Switching from raw RUL to the piecewise-linear cap moved the needle more than any architecture tweak. Spend your time on the target function before you touch hidden sizes.
- **Leakage is silent and seductive.** A row-level split *looks* fine and gives you beautiful validation curves. Splitting by engine ID dropped my apparent performance — and made it real. Wire the anti-leakage assertion into CI so you can never regress on it by accident.
- **Interpretability is a feature, not an afterthought.** "RUL ≈ 38 cycles" is a number. "RUL ≈ 38 cycles, top warning signal HPC Static Pressure" is a *decision*. SHAP on the LSTM is what turns a regression output into something an engineer trusts.
- **Treat the demo as a deliverable.** The model is 5% of the perceived value; the live dashboards, the README narrative, and the compliance framing are the other 95% in an interview.
- **Honest accounting builds credibility.** A deep CNN (Li et al., 2018) can beat this LSTM on raw RMSE (~12.6). I documented that trade-off openly and explained why I still chose the LSTM — sequential interpretability that maps one-to-one onto how an engineer reasons about an engine's flight history. Naming your model's weakness is more convincing than hiding it.

---

*Built by Priyrajsinh Parmar. Code, model card, and research notes are in the [repository](https://github.com/Priyrajsinh/before-it-breaks).*
