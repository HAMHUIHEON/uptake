# SPEC task29 — scale-resolution pilot, residual test and power (Amendment 6 analysis)

Context: Amendment 6 at the end of `specs/EVAL_PLAN_overnight_2.md`. New module `forecaster/scalecheck.py`, stdlib only, reusing
`outcome` (load_outcomes, participant_rows, move_value_table, dialogue_id_of, _cluster_pool), `calibrate.fit_board_norm`,
`board._value_of`, `batch.spearman`, `stats._percentile`, `usage.PRICES`. No model calls anywhere in this module.

Item records are the JSON files `casino-*.json` in a directory (format written by `uptake-batch` or `uptake-rescore`).
`load_dir(path, ids=None) -> dict[id, record]` (ids None → all `casino-*.json`).

## 1. Reading helpers
- `SCALE_BOUNDS = {"v2": ((-2, 2), (0, 2)), "fine100": ((0, 100), (0, 100)), "seven": ((-3, 3), (-3, 3))}` — (rel bounds, face bounds) of the raw values.
- `reading_raw(r) -> (rel_raw, face_raw)`: uses `rel_raw`/`face_raw` if present, else `relationship_shift`/`face_threat` (v2 records).
- `reading_norm(r) -> (rel_norm, face_norm)`: uses `rel_norm`/`face_norm` if present, else `(relationship_shift, -face_threat)`.
- `record_scale(record) -> str`: `record.get("scale", "v2")`.

## 2. Resolution of one condition: `resolution(records, base_n=3) -> dict`
Over all effects of all records:
- `turn_values` = `board._value_of(e)` for effects with `n_readings > 0`; `n_turns`, `distinct_turn_values` = number of distinct values after `round(v, 6)`.
- `reading_values` = rel_norm + face_norm per reading; `distinct_reading_values` (round 6).
- endpoint shares over readings, using SCALE_BOUNDS of the record's scale: `rel_min_share`, `rel_max_share`, `face_min_share`, `face_max_share`.
- z over turn values (population sd; sd 0 → all z 0): `z_ge2_share`, `z_le_minus2_share`, `z_max`, `z_min`, `mean`, `sd`.
- `parse_fail_rate` = 1 − Σn_readings / Σexpected, where expected per effect = `e["n_expected"]` if present else `base_n * len(e["listeners"])`
  (0 if expected sum is 0).
- `icc1` = one-way random-effects ICC(1) over turns with ≥ 2 readings (groups = turns, values = rel_norm + face_norm of each reading),
  unbalanced form: MSB = SSB/(g−1), MSW = SSW/(N−g), n0 = (N − Σn_i²/N)/(g−1), ICC = (MSB − MSW)/(MSB + (n0 − 1)·MSW);
  0.0 if g < 2 or the denominator is 0. `sb3`, `sb5` = Spearman–Brown m·ICC/(1+(m−1)·ICC) for m = 3, 5.
- `n_dialogues` = number of records.

`convergent(records, base_records) -> dict`: Spearman of turn values for (id, index) pairs present with readings in both → `{"rho", "n"}`.

## 3. Cost from a calls log: `board_cost(calls_path, ids) -> dict`
Read `calls.jsonl` lines; keep records whose `meta.layer == "board"` and `meta.item in ids` and `error` is null/absent.
usd = prompt_tokens/1e6·price_in + completion_tokens/1e6·price_out with `usage.PRICES[model]` (fallback `usage._FALLBACK`).
Return `{"calls", "usd", "usd_per_dialogue" (usd / len(ids) if ids else 0), "mean_completion_tokens"}`. Missing file → zeros.

## 4. Statistics
- `rank(values)` average ranks for ties (1-based).
- `ols(X_rows, y) -> (coef, residuals, r2)`: X_rows list of lists WITHOUT intercept; add intercept; solve normal equations (XᵀX + 1e-9·I) β = Xᵀy
  by Gaussian elimination with partial pivoting. r2 = 1 − SSres/SStot (0 if SStot = 0).
- `partial_spearman(x, y, controls) -> float`: rank x, y and every control column; residualise rank(x) and rank(y) on the control ranks with `ols`;
  return Pearson of the two residual vectors (0.0 if either residual variance is 0). With `controls == []` equals Spearman(x, y).
- `rank_regression(x, y, control) -> (beta_x, delta_r2)`: z-score the ranks of x, y, control; OLS y ~ control + x; beta_x = coefficient of x;
  delta_r2 = R²(control + x) − R²(control).
- Cluster bootstrap helper `boot(rows, stat_fns: dict[name -> fn(rows) -> float], b, seed) -> dict[name -> {"point", "lo", "hi", "se", "reps_n"}]`:
  resample dialogues (key `dialogue`) with replacement, SAME resamples for all statistics (one RNG `random.Random(seed)`), percentile 2.5/97.5 via
  `stats._percentile` on sorted reps, `se` = population sd of the reps. A stat that raises or returns NaN in a replicate is skipped for that replicate.

## 5. Participant rows with extra columns: `rows_for(eval_records, outcomes, fit_records, genre="negotiation") -> (rows, info)`
- `norm = calibrate.fit_board_norm([effects of fit records with genre], method="z")`; mu, sd; `table = outcome.move_value_table(fit list, mu)`.
- `rows = outcome.participant_rows(eval_records, outcomes, mu, sd, table=table)` (keeps reception, tag_value, turns, opp_likeness, dialogue, side).
- move types for shares = moves appearing in ≥ 1% of fit-set tagged turns (turn counted once per move type); sorted. For each row add
  `share_<move>` = share of X's tagged turns (moves of that speaker in the eval record) containing the move.
- `resid` (R4) = mean over X's effects with n_readings > 0 of [(value − mu) − tv(turn)], tv(turn) = mean of table values of that turn's moves
  (unknown move 0; no moves → 0); None when X has no such turns.
- info = {"mu", "sd", "share_moves": [...], "n_rows"}.

## 6. `analyze_eval(eval_records, outcomes, fit_records, compare_records=None, b=2000, b_r2=1000, seed=0, same_scale=False) -> dict`
On rows with reception, tag_value and opp_likeness not None:
- `C1` = Spearman(reception, opp_likeness); `C6_tag` = Spearman(tag_value, opp_likeness); `C6diff` = C1 − C6_tag.
- `R1` = partial_spearman(reception, opp_likeness, [tag_value]).
- `R3_beta`, `R3_dr2` from `rank_regression(reception, opp_likeness, tag_value)`.
- `R4` (only if same_scale) = Spearman(resid, opp_likeness).
- If compare_records given: rows for them too (same fit), matched by (dialogue, side) → `dC1` = C1(eval) − C1(compare) on the matched rows
  (column `reception_cmp`), and `R1_cmp` for the compare rows.
All of the above from one `boot(..., b, seed)` call. `R2` = partial_spearman(reception, opp_likeness, [tag_value, turns, share_*...]) from a
separate `boot(..., b_r2, seed)` call.
Verdicts: `C6diff` and `R1`: "SUPPORTED" if lo > 0 else "NOT SUPPORTED"; everything else "REPORTED".
Power for `C6diff` and `R1` (and `dC1` if present): N = number of distinct dialogues in rows; for δ in (0.03, 0.06):
`n80 = ceil(N·(se·(1.959964+0.841621)/δ)²)`, `n90` with 1.281552 instead of 0.841621. Return
`{"n_dialogues", "n_rows", "stats": {...}, "verdicts": {...}, "power": {name: {"se", "n80": {"0.03", "0.06"}, "n90": {...}}}, "info": info}`.

## 7. `pilot(conditions: dict[name -> records], base_name, outcomes, fit_records, calls_paths: dict[name -> path], ids, b=2000, seed=0) -> dict`
For every condition: `resolution`, `convergent` vs the base condition (skip for base), `board_cost(calls_paths[name], ids)`,
and `C1`, `C6diff` (with CI from `boot`) on its rows. Base condition `base_n=3`; others: base_n from records' effects `n_expected` (present).

## 8. CLI `uptake-scale` (`forecaster.scalecheck:main`, register in pyproject)
Subcommands:
- `pilot --fit DIR --casino PATH --ids-file PATH --cond NAME=DIR [--cond ...] --base NAME --out PREFIX [--b 2000] [--seed 0]`
  Calls log for each condition = `DIR/calls.jsonl`. Writes `PREFIX.json` and `PREFIX.md` (one markdown table, rows = conditions, columns:
  n_dialogues, distinct turn values, distinct reading values, rel min/max share, face min/max share, z≥2 share, z≤−2 share, z_max, parse fail,
  icc1, sb3, sb5, convergent rho, usd/dialogue, mean completion tokens, C1 [CI], C6diff [CI]). Prints the table.
- `residual --fit DIR --casino PATH --eval DIR [--compare DIR] [--ids-file PATH] [--same-scale] --out PREFIX [--b 2000] [--b-r2 1000] [--seed 0]`
  Writes `PREFIX.json` and `PREFIX.md` (table: metric | value [CI] | verdict | n, then a power table) and prints it.
  `--ids-file` restricts both eval and compare to those ids.

## Tests `tests/test_task29.py` (synthetic only, no network, no data/)
- rank with ties; ols recovers y = 1 + 2a − b exactly; r2 = 1 on exact data.
- partial_spearman: with no controls equals batch.spearman; when y = f(control) + noise independent of x, partial ≈ 0 while plain is large
  (construct deterministic data: x = c + e1, y = c + e2 with e1, e2 orthogonal patterns) — assert |partial| < 0.2 and plain > 0.5.
- rank_regression delta_r2 ≥ 0, and ≈ 0 when x duplicates control.
- ICC: identical readings within turns, different across → icc1 = 1; constant across turns → icc1 ≤ 0; Spearman–Brown values.
- resolution on hand-built v2 and fine100 records: distinct counts, endpoint shares, parse_fail_rate with n_expected and the base_n fallback, z shares.
- board_cost on a temp calls.jsonl with board and moves layers and an error line.
- boot: same resamples for all stats (two identical stat functions give identical lo/hi), se > 0 with varied data; deterministic for a seed.
- analyze_eval on a synthetic eval/fit set (≥ 30 dialogues, random.Random(1)) returns all keys; power n80 for δ 0.03 = 4× n80 for δ 0.06 (±1 for ceil).
- CLI smoke: `main(["residual", ...])` and `main(["pilot", ...])` on temp dirs with a tiny synthetic casino-format outcome file — reuse the
  structure that `outcome.load_outcomes` expects (read its code).
