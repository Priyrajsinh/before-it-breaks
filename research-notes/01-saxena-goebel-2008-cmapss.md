# 01 · Saxena & Goebel (2008) — The C-MAPSS Data Set

**Reference:** Saxena, A., & Goebel, K. (2008). *Turbofan Engine Degradation Simulation Data Set (C-MAPSS).* NASA Ames Prognostics Data Repository, NASA Ames Research Center.
Companion paper: Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation.* IEEE Int'l Conf. on Prognostics and Health Management (PHM).

---

## Why this dataset matters
C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) is the **de-facto benchmark for data-driven prognostics**. It gives a fleet of engines run **from healthy all the way to failure**, with per-cycle multivariate sensor traces — the exact shape RUL forecasting needs. Because almost every modern RUL paper reports on it, it's the dataset that lets us *compare* against published RMSE / NASA-Score numbers instead of inventing our own yardstick. It's also realistic without being proprietary: a high-fidelity thermodynamic simulator stands in for a real fleet you could never get failure data from.

## Dataset structure summary
- **Four sub-datasets** FD001–FD004, varying along two axes: number of **operating conditions** (1 or 6) and number of **fault modes** (1 or 2).
- Each row = one engine at one operational **cycle**: `engine_id`, `cycle`, **3 operational settings**, **21 sensor** readings.
- Each engine starts healthy and degrades; the **train** files run to failure, the **test** files stop some cycles *before* failure and ship a separate `RUL_*.txt` with the ground-truth remaining life at the last observed cycle.
- The signal is **gradual degradation** buried in noise — many sensors are flat until the engine enters its degradation phase.

| Sub-dataset | Operating conditions | Fault modes |
|-------------|----------------------|-------------|
| FD001 | 1 | 1 (HPC degradation) |
| FD002 | 6 | 1 |
| FD003 | 1 | 2 |
| FD004 | 6 | 2 |

## What we used (FD001 only)
- **100 train + 100 test** engines, **single operating condition, single fault mode** (HPC degradation).
- This is the cleanest setting: one regime means degradation is **directly visible** in the raw sensors without first having to disentangle operating-condition effects — ideal for demonstrating the *temporal-degradation* story this project is about.
- We drop 7 near-constant sensors → **17 features**, cap RUL at 125, split by engine ID. (See notes 02 and 05.)

## What we didn't use (FD002–FD004) and why
- **FD002 / FD004 (6 operating conditions):** the raw sensors are dominated by the *operating regime*, not by wear. You first need condition-clustering / regime-normalisation before degradation is learnable — a substantial extra pipeline that would dilute the core narrative.
- **FD003 / FD004 (2 fault modes):** two failure mechanisms mean the model must implicitly classify the fault before forecasting — more capacity, more ambiguity, harder to interpret per-sensor SHAP cleanly.
- **Decision:** FD001 keeps the spotlight on *temporal-degradation modelling + early warning + interpretability*. Generalisation to FD002–FD004 is explicitly listed as a **limitation** in `MODEL_CARD.md`, not a silent omission.
