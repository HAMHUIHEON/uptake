# SPEC task25 — value model prototype: 형세 = predicted final result (offline, stdlib)

See EVAL_PLAN_overnight_2 Amendment 5. New module `forecaster/valuemodel.py`.

- `FEATURES = ["rec_self", "rec_other", "tagv_self", "tagv_other", "n_self"]`.
- `prefix_features(item, t, side, mu, table) -> dict`: using only effects/moves with index ≤ t:
  rec_self = mean (board._value_of(e) − mu) over effects spoken by `side` (0.0 if none); rec_other = same for the other side
  ("A"↔"B"); tagv_self/tagv_other = mean over that side's move entries (turn_index ≤ t) of the mean `table.get(m, 0.0)` of the entry's moves
  (0.0 if none); n_self = number of `side`'s effects so far.
- `ridge_fit(X, y, lam=1.0) -> dict`: standardize each column (mean/pop sd; sd 0 → 1), add intercept (not penalized), solve
  (XᵀX + lam·I)w = Xᵀy by Gaussian elimination with partial pivoting. Returns `{"w", "b", "means", "sds", "lam", "features"}`.
- `ridge_predict(model, x: dict) -> float`.
- `fit_value(fit_items, outcomes, mu, table, lam=1.0) -> model`: one row per (dialogue, side) using the FINAL turn's prefix features;
  y = likeness reported by the side's opponent (`outcome.load_outcomes` format).
- `value_curves(item, model, mu, table) -> {"A": [...], "B": [...]}`: prediction for each side after each turn index (one entry per effect).
- `value_pivot(curves, item) -> dict | None`: the turn index with the largest |Δ| of the SPEAKER's own curve (Δ at turn 0 = value − the
  prediction with no turns, i.e. ridge_predict on all-zero features); returns `{"index", "speaker", "delta"}`.
- `evaluate(model, eval_items, outcomes, mu, table, b=2000, seed=0) -> dict`:
  rows per (dialogue, side): pred_final, pred_mid (prediction after turn index (n_effects // 2) − 1; ≥ 0), rec (outcome.reception),
  opp_likeness. V1 = outcome.boot_spearman(rows, "pred_final", "opp_likeness"); V1_minus_C1 = outcome.boot_spearman_diff(rows, "pred_final",
  "rec", "opp_likeness"); V2 = boot_spearman(rows, "pred_mid", "opp_likeness"); V3 = boot_spearman over pivot rows
  {"dialogue", "delta", "opp_likeness" of the pivot speaker}. Returns these plus n and the model weights.
- CLI `uptake-value --fit DIR --eval DIR [DIR ...] --casino PATH [--lam 1.0]`: mu from calibrate.fit_board_norm(z) on fit negotiation items;
  table = outcome.move_value_table(fit items, mu); writes `DIR/value.json` and `DIR/value.md` per eval DIR (weights table + V1/V1−C1/V2/V3
  with CIs). Register in pyproject.

Tests `tests/test_task25.py` (Muse writes): ridge recovers y = 2x + 1 (lam ~0); prefix_features ignores turns after t; value_curves length;
value_pivot finds the planted largest jump; evaluate returns V1/V2/V3 keys on toy data.
