# SPEC task32 — value model: "move-type mean + panel residual" features, scale comparison

Follows EVAL_PLAN_overnight_2 Amendment 6 (result: R1 supported → keep the panel; feed the value
model the move-type mean and the panel's deviation from it). Module `forecaster/valuemodel.py`.

## one residual definition, shared
- Add `turn_residual(effect, idx, by_turn_moves, mu, table) -> float | None`:
  `(board._value_of(effect) − mu) − tagv(turn)` where `tagv(turn)` = mean `table.get(m, 0.0)` over
  the moves tagged for that turn index (0.0 when the turn has no moves); `None` when the effect
  has no readings (`n_readings` ≤ 0) or no value. This is exactly the per-turn term that
  `scalecheck.rows_for` averages into `row["resid"]`; make `scalecheck` call this helper so the
  two definitions cannot drift.

## features
- `FEATURES_LEGACY = ["rec_self", "rec_other", "tagv_self", "tagv_other", "n_self"]` (current).
- `FEATURES_RESID = ["tagv_self", "tagv_other", "resid_self", "resid_other", "n_self"]` where
  `resid_self` / `resid_other` = mean `turn_residual` over that side's effects with index ≤ t
  (0.0 if none).
- `prefix_features(item, t, side, mu, table, featureset="resid")`; `fit_value`, `value_curves`,
  `value_pivot`, `evaluate` take `featureset` and the model dict stores `"featureset"`.
  `ridge_predict` uses the model's feature list. Default everywhere: `"resid"`.

## CLI `uptake-value`
- `--features {legacy,resid,both}` (default `both`). With `both`, fit and evaluate each featureset
  and write `value_legacy.{json,md}` and `value_resid.{json,md}` per eval dir plus one
  `value_compare.md` per eval dir: a table with rows V1, V1−C1, V2, V3 and columns legacy / resid
  (point estimate and 95% CI), n, and the fit/eval dirs and their `scale`.
- Refuse (clear error) if fit and eval records are on different scales.

## tests `tests/test_task32.py` (Muse writes)
- `turn_residual` is 0 when the value equals the table value; `None` without readings.
- `scalecheck.rows_for` residuals equal the mean of `turn_residual` on toy records.
- legacy featureset reproduces the task25 toy results (regression).
- resid featureset: prefix ignores turns after t; `evaluate` returns V1/V2/V3 keys.
- `pytest -q` must stay green.

Reporting (Claude runs, not Muse): 2×2 = {legacy, resid} × {v2 fit/eval = overnight_1 →
overnight_2b, overnight_2; fine100 fit/eval = scale6/fit_fine100 → full_ho2_fine100,
full_ho1_fine100}. V1/V2 are reported with CIs; no new acceptance rule (Amendment 7 notes this).
