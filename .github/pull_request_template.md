## Summary
<!-- 1–3 bullets -->

## v3 + PM1 checklist (CLAUDE.md C1–C50)
- [ ] `make ci` green locally
- [ ] No `print()` — `get_logger(__name__)` only
- [ ] `pandera CMAPSS_SCHEMA.validate(df)` runs before any analysis (rule C34)
- [ ] `safe_predict()` wraps every model.predict() call (rule C36)
- [ ] RUL capped at max_rul=125 — verified (rule C35)
- [ ] Train/val split by engine_id, not row (rule C33)
- [ ] LSTM saved as state_dict, not full module (rule C47)
- [ ] StandardScaler applied to input before LSTM (rule C48)
- [ ] If model touched: test RMSE ≤ 30 cycles still holds (rule C37)
- [ ] No new secrets (detect-secrets clean)
- [ ] No new CVEs (pip-audit clean)
- [ ] Test coverage ≥ 70%
- [ ] No `Co-Authored-By:` trailers in commits or this PR body

## Test plan
- [ ] ...
