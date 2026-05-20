# Model Card — Before It Breaks · Predictive Maintenance RUL

## Model details
- Name: LSTM Remaining Useful Life Forecaster
- Owner: Priyrajsinh Parmar
- Repository: https://github.com/Priyrajsinh/before-it-breaks
- License: MIT
- Date: TBD (Day 9 ship)
- Stack: PyTorch 2-layer LSTM + SHAP GradientExplainer + FastAPI

## Intended use
- TBD — fill on Day 8.

## Training data
- NASA CMAPSS FD001 (100 train engines, 100 test engines)
- Schema: see `CLAUDE.md` Dataset Schema section.
- Citation: Saxena & Goebel (2008), C-MAPSS Data Set, NASA Ames Prognostics Data Repository.

## Headline test results
TBD — fill from `reports/results.json` on Day 8.

## Limitations
TBD — fill on Day 8 (single fault mode, simulated data, generalisability beyond CMAPSS operating conditions, etc.).

## EU AI Act framing
- Annex III §2 (Critical infrastructure — management and operation) — predictive maintenance for industrial machinery falls within scope when used in safety-critical contexts (aviation, energy, transport).
- Annex III §4 (Employment, workers management) — if the system informs maintenance worker scheduling or workload, it falls in scope. See README Day 8.

## Citations
TBD — Saxena & Goebel 2008, Zheng et al. 2017, Malhotra et al. 2016, Li et al. 2018.
