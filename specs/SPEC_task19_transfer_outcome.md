# SPEC task19 — held-out transfer test and outcome validity (offline, no API calls)

Context: `specs/EVAL_PLAN_overnight_2.md` (Parts B and C). Stdlib only (no numpy). Reuse `calibrate`, `predict`,
`stats`, `board`, `batch.spearman`. Item JSON format = files written by `batch.run_item`
(keys id, genre, moves, effects, hits, gold). Item ids for CaSiNo are `casino-<dialogue_id>`.

## 1. Transfer evaluation — `forecaster/predict.py`
`transfer(fit_items, eval_items, genre="negotiation", b=2000, seed=0) -> dict`
- fit_items / eval_items: dict id → item (as returned by `calibrate._load_items`). Use only items with `genre == genre`.
- Fit on ALL fit items: `base = calibrate.base_rates`, `markov = fit_markov`, `lam_base = calibrate.fit_lambda`,
  `lam_markov = fit_linear(markov prior)`, `(a, b_pool) = fit_pool`. Hits come from `with_context`.
- Models evaluated on ALL eval items (hits from `with_context`):
  `raw_panel` = `calibrate.panel_prob(h)`; `base`; `markov`; `panel_base` = blend(panel_prob, base, lam_base);
  `panel_markov` = blend(panel_prob, markov_prob, lam_markov); `pool` = pool([markov_prob, panel_prob], [a, b_pool]).
- Returns `{"models": compare(eval_hits, models, b, seed), "fitted": {lam_base, lam_markov, a, b}, "n_fit_items", "n_eval_items",
  "n_eval_hits", "contrasts": {...}}`.
- contrasts (each `[point, lo, hi]` from `stats.paired_bootstrap_diff`, clusters = item id, before → after):
  `"B1_panel_base_vs_raw"` (before raw_panel, after panel_base), `"B2_markov_vs_base"` (before base, after markov),
  `"B3_pool_vs_markov"` (before markov, after pool), `"B4_panel_base_vs_markov"` (before markov, after panel_base).
  Each contrast also gets `"supported": lo > 0`.
- CLI `uptake-transfer --fit DIR --eval DIR [--genre negotiation] [--b 2000] [--seed 0]` writes `EVAL/transfer.json` and
  `EVAL/transfer.md` (one row per model: log score, skill vs base [CI], vs markov [CI], hit rate; then one row per contrast
  with point [lo, hi] and SUPPORTED / NOT SUPPORTED). Register in pyproject.
- `calibrate._is_skipped` must also skip `transfer.json`, `outcome.json`, `predict.json`, `commentary*.json`
  (they lack item keys anyway; make it explicit).

## 2. Outcome validity — new `forecaster/outcome.py`
Constants:
- `SATISFACTION = {"Extremely dissatisfied": -2, "Slightly dissatisfied": -1, "Undecided": 0, "Slightly satisfied": 1, "Extremely satisfied": 2}`
- `LIKENESS = {"Extremely dislike": -2, "Slightly dislike": -1, "Undecided": 0, "Slightly like": 1, "Extremely like": 2}`
- `WARM = {"empathize", "praise", "repair", "smalltalk"}`, `HOSTILE = {"pressure", "challenge", "reject"}`

Functions:
- `load_outcomes(casino_path) -> dict[int, dict]`: dialogue_id → `{"A": {"points": int, "satisfaction": int, "likeness": int},
  "B": {...}}` with A = mturk_agent_1, B = mturk_agent_2 (same mapping as `casino.SPEAKERS`). Unknown text → skip that dialogue.
- `dialogue_id_of(item_id) -> int | None`: "casino-123" → 123; otherwise None.
- `reception(item, mu) -> dict`: for side X in ("A", "B"): mean of (`board._value_of(e)` − mu) over effects with `e["speaker"] == X`;
  side with no effects → None.
- `dyad_value(item, mu) -> float | None`: mean (value − mu) over all effects; None if no effects.
- `flagged(item, mu, sd, z=2.0) -> dict`: X → True if any of X's effects has (value − mu) ≤ −z·sd (sd > 0), else False.
- `tag_score(item) -> dict`: X → (share of X's entries in `item["moves"]` whose moves intersect WARM) − (share intersecting HOSTILE);
  None if X has no entries.
- `turn_count(item) -> dict`: X → number of X's entries in `item["moves"]`.
- `participant_rows(items, outcomes, mu, sd) -> list[dict]`: one row per (dialogue, side X) where both reception and outcomes exist:
  `{"dialogue", "side", "reception", "tag", "turns", "flagged", "opp_likeness" (likeness reported by the OTHER side),
  "own_points", "own_satisfaction"}`.
- `dyad_rows(items, outcomes, mu) -> list[dict]`: one per dialogue: `{"dialogue", "dyad", "mean_satisfaction" (mean of A and B),
  "rec_diff" (reception A − B), "points_diff" (points A − B)}`; skip if any needed value is None.
- `boot_spearman(rows, x, y, b=2000, seed=0) -> [point, lo, hi]`: point = `batch.spearman` over all rows;
  bootstrap = resample dialogue keys with replacement (`random.Random(seed)`), pool their rows, recompute; percentile CI
  (2.5/97.5, same interpolation as `stats._percentile`). Fewer than 3 rows → [0.0, 0.0, 0.0].
- `boot_spearman_diff(rows, x1, x2, y, b=2000, seed=0) -> [point, lo, hi]`: ρ(x1,y) − ρ(x2,y) on the same resamples.
- `boot_mean_diff(rows, flag, y, b=2000, seed=0) -> [point, lo, hi]`: mean y where row[flag] is True minus mean y where False,
  dialogue-cluster bootstrap; a resample with an empty group is skipped.
- `analyze(fit_items, eval_items, outcomes, genre="negotiation", b=2000, seed=0) -> dict`:
  norm = `calibrate.fit_board_norm([effects of fit items of genre], method="z")`; mu = norm["mu"], sd = norm["sd"].
  Returns `{"mu", "sd", "n_dialogues", "n_participants",
  "C1": boot_spearman(prow, "reception", "opp_likeness"),
  "C2": boot_spearman(drow, "dyad", "mean_satisfaction"),
  "C3": boot_spearman(drow, "rec_diff", "points_diff"),
  "C4_tag": boot_spearman(prow, "tag", "opp_likeness"), "C4_turns": boot_spearman(prow, "turns", "opp_likeness"),
  "C4_board_minus_tag": boot_spearman_diff(prow, "reception", "tag", "opp_likeness"),
  "C5": boot_mean_diff(prow, "flagged", "opp_likeness"), "n_flagged": count}` plus `"verdicts"`:
  C1 / C2 / C4_board_minus_tag → "SUPPORTED" if lo > 0 else "NOT SUPPORTED"; C3, C5 → "REPORTED".
  Rows with tag None are dropped only from tag-based metrics.
- CLI `uptake-outcome --fit DIR --eval DIR [DIR ...] --casino PATH [--b 2000] [--seed 0]`: for each eval DIR writes
  `DIR/outcome.json` and `DIR/outcome.md` (table: metric, value [CI], verdict, n). Register in pyproject.

## 3. Unannotated CaSiNo dialogues (held-out 2)
- `casino.load_casino(path, limit=None, seed=0, exclude_ids=(), annotated=True)`: `annotated=True` → current behavior exactly.
  `annotated=False` → ONLY dialogues without annotations; turns = chat_logs minus DEAL_ACTIONS (same speaker mapping);
  `gold = None`. Sampling/exclusion as before.
- `uptake-manifest --unannotated` → uses `annotated=False`; items get `"gold": false` in addition to the usual fields.
- `batch.run_item` and `batch.genre_report` must work when the dialogue gold is None (record `"gold": null`; F1 only over items with gold).
  The batch casino index must contain the unannotated dialogues when the manifest has `"gold": false` items.

## Tests (`tests/test_task19.py`, written by Muse from this spec)
Must at least cover: transfer on synthetic items where the panel is pure noise (B3 not supported) and where markov is
informative (B2 supported); outcome mapping of all 5 Likert strings; reception sign and side split on a 4-turn toy item;
boot_spearman on perfectly monotone rows → point 1.0; boot_mean_diff sign; load_casino(annotated=False) returns only
unannotated dialogues with gold None (toy JSON written to tmp_path); existing behavior unchanged (all old tests pass).
