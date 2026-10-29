# SPEC task33 — commentary marks from the panel residual (z_resid), "!!" and "??"

Rationale (Amendment 6 result): the panel's information beyond the move-type mean is the
residual, so "!" / "?" should say "better / worse received than a move of this kind usually is",
and the 0–100 scale has a positive tail that makes "!!" possible.

## norm
- `calibrate.residual_norm(fit_items, mu, table) -> {"sd_resid": float, "n": int}`: population sd of
  `valuemodel.turn_residual` over all negotiation effects of the fit items with readings
  (sd 0 → 1.0). `commentary.main` computes `table = outcome.move_value_table(sel, mu)` and
  `sd_resid`, and stores `table`, `sd_resid`, `scale` in `norm` (JSON-serializable; `table` keys are
  move names).

## turn values and marks (`forecaster/commentary.py`)
- `turn_values(item, norm)`: when `norm` has `table` and `sd_resid`, add per turn `tagv`, `resid`
  (= `turn_residual`, 0.0 when None) and `z_resid = resid / sd_resid`. Keep `value` and `z`.
- `mark_for(z, surprise, z_resid=None)`: if `z_resid` is given: ≥ 2 → `"!!"`, ≥ 1 → `"!"`,
  ≤ −2 → `"??"`, ≤ −1 → `"?"`, else `"!?"` when `surprise`, else None. Without `z_resid` the legacy
  thresholds on `z` apply unchanged.
- `pick_turns` ranks by `|z_resid|` when present (fallback `|z|`); the chosen reference and the
  template/LLM lines use the same turn dict (they get `resid`/`z_resid` for free).
- Output JSON keeps `z` and adds `z_resid`, `tagv`, `resid` per turn; `norm` is written as above.

## viewer (`forecaster/viewer.py`)
- `_MARKS` accepts `"!!"` and `"??"`; CSS: `!!` stronger good colour, `??` stronger bad colour;
  legend lists all five marks. Hover shows `resid` and `z_resid` when present.

## tests `tests/test_task33.py` (Muse writes)
- a planted turn whose value is far above its move-type mean gets `"!!"` with the residual norm and
  `"!"` with the legacy norm (no table).
- a turn with high absolute value but value == table value gets no mark under the residual norm.
- `residual_norm` on toy items; `commentary.main` writes `sd_resid`/`table`/`scale` into `norm`.
- viewer renders `"!!"` and `"??"` badges.
- `pytest -q` must stay green.
