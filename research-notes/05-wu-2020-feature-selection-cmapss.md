# 05 · Wu et al. (2020) — Feature selection for RUL

**Reference:** Wu, J., et al. (2020). *Feature Selection for Remaining Useful Life Prediction.* (CMAPSS sensor-selection study.) Related line: Wu et al., *Data-driven RUL prediction with feature/ sensor selection on C-MAPSS.*

---

## The core idea
Not every sensor carries degradation information. Feeding constant or uninformative channels into the model **adds parameters and noise without adding signal**, hurting both accuracy and interpretability. The paper's thesis: **select the sensors whose statistics actually evolve over an engine's life**, drop the rest, and the downstream RUL model gets simpler *and* better.

## Justification for dropping the 7 constant sensors (rule C40)
On **FD001**, seven sensors are effectively **flat across every engine's entire life** — they have near-zero variance and near-zero correlation with RUL, so they cannot encode degradation:

```
sensor_1, sensor_5, sensor_6, sensor_10, sensor_16, sensor_18, sensor_19   ->  DROPPED
```

Removing them leaves **17 features** = 3 operational settings + **14 informative sensors**. Benefits, straight from the feature-selection rationale:
- **Less noise into the LSTM** → cleaner gradients, less overfitting capacity wasted on dead channels.
- **Sharper SHAP attributions** → the per-sensor warning signal isn't diluted by channels that can never be the cause.
- **Smaller, faster model** with no measurable accuracy loss — the dropped channels had nothing to contribute.

This is enforced in `config/config.yaml` (`data.drop_columns`) and validated by the pandera schema, which expects exactly the 14 retained sensors after preprocessing.

## Which sensors the literature finds most informative
The monotonically-trending sensors on FD001 — the ones that carry the degradation signal and that our SHAP analysis consistently surfaces as warning signals — include:

| Sensor | Physical meaning (this repo's label) | Behaviour over life |
|--------|--------------------------------------|---------------------|
| sensor_11 | HPC Static Pressure | trends strongly — top SHAP signal in our apps |
| sensor_4 | LPT Outlet Temperature | rises with degradation |
| sensor_7 | HPC Outlet Pressure | drifts down |
| sensor_12 | Fuel-to-Pressure Ratio | trends |
| sensor_15 | Bypass Ratio | trends |
| sensor_2 / sensor_3 | LPC / HPC Outlet Temperature | rise with wear |
| sensor_21 / sensor_20 | LPT / HPT Coolant Bleed | drift |

The takeaway: the **14 retained sensors are the informative subset**, and the seven we drop are exactly the channels feature-selection studies flag as carrying no prognostic value. Sensor selection and the piecewise-linear label (note 02) are the two preprocessing decisions that do most of the heavy lifting before the LSTM ever sees the data.
