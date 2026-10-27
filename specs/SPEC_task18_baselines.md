# SPEC task18 — stronger baseline, learned forecast, z-score annotations (offline, no API calls)

## Why (task17 result, 10/28)
- Base-rate blend removed the −0.33 loss but the panel added almost nothing (λ≈0.1, skill +0.003…+0.008, CI>0 in 1/3 seeds).
- Question now: does the listener panel carry information about the next move **beyond what the current move already tells us**?
- Quantile annotation thresholds broke on discrete values (!!=19.8% vs target 2.5%).

## New module `forecaster/predict.py` (K = len(NEXT_ACTIONS), alpha 0.5 as in calibrate)
- `with_context(item) -> list[dict]`: copy of every `item["hits"]` entry plus `"item"` = item["id"] and
  `"cur_class"` = `board.actual_class(moves of turn hit["index"])`, moves taken from `item["moves"]` via `board._tag_info`
  (first tag per turn index). No tag for that turn → `"proceed"`.
- `fit_markov(hits, base, beta=2.0) -> dict`: for every cur in NEXT_ACTIONS and every nxt in NEXT_ACTIONS,
  p = (c(cur→nxt) + beta·base[nxt]) / (c(cur) + beta), counts over hits (cur_class → actual_class). Unseen cur → base. Rows sum to 1.
- `markov_prob(hit, markov) -> dict`: `markov[hit.get("cur_class", "proceed")]`.
- `pool(dists, weights) -> dict`: geometric pool ∝ Π dist_i[a] ** w_i over NEXT_ACTIONS, normalized to 1.
- `fit_pool(hits, markov, grid=None, alpha=0.5) -> tuple[float, float]`: grid default [0, 0.25, …, 2.0] for both a (markov) and b (panel);
  maximize mean ln pool([markov_prob, calibrate.panel_prob], [a, b])[actual]; ties → smaller b, then smaller a.
- `fit_linear(hits, prior_fn, grid=None, alpha=0.5) -> float`: λ maximizing mean ln(λ·panel_prob + (1−λ)·prior_fn(hit))[actual];
  grid default 0.00…1.00 step 0.05; ties → smaller λ.
- `compare(hits, models, b=2000, seed=0) -> dict`: models = {name: fn(hit) -> dist}; must include "base" and "markov".
  For each name: per-hit ln p(actual); clusters = hit["item"].
  `{"log_score": mean, "skill_vs_base": [point, lo, hi]` (stats.cluster_bootstrap_ci on ln p_model − ln p_base),
  `"vs_markov": [point, lo, hi]` (stats.paired_bootstrap_diff, before = markov logs, after = model logs),
  `"hit_rate":` share where argmax (ties alphabetical) == actual, `"n": len(hits)}`.
- `main(argv=None)` = CLI `uptake-predict --out DIR [--seeds 0 1 2] [--frac 0.5]`:
  load items like calibrate (`genre`, `hits`, `effects`, `moves`); per genre, per seed: split_items, fit on dev
  (base_rates, fit_markov, λ_base = calibrate.fit_lambda, λ_markov = fit_linear with markov prior, pool a/b,
  z-norm), evaluate on test with models:
  `base`, `markov`, `panel_base` (task17 blend), `panel_markov` (linear), `pool` (geometric).
  Board on test with z-norm: annotation shares and mean end value before→after.
  Writes DIR/predict.json and DIR/predict.md: one row per genre×seed×model (skill vs base [CI], vs markov [CI], hit rate,
  fitted params) + one board row per genre×seed + per genre a summary line
  "seeds with CI lower > 0: markov vs base X/3; best panel model vs markov Y/3".
  Register `uptake-predict = "forecaster.predict:main"`.

## Change in `forecaster/calibrate.py`
- `fit_board_norm(effects_lists, method="quantile")` — current behavior unchanged for "quantile".
  `method="z"`: thresholds = {"??": −2·sd, "?": −1·sd, "?!": −0.5·sd, "!": +1·sd, "!!": +2·sd} (centered units);
  sd == 0 → thresholds None.
- `board.annotate_turn(v, norm)` with `norm["thresholds"] is None` → "=".

## Pre-registered reading (written before running)
- Markov is a real baseline if skill vs base CI lower > 0 in ≥ 2/3 seeds (negotiation).
- Panel adds forecasting information if `panel_markov` or `pool` beats markov (vs_markov CI lower > 0) in ≥ 2/3 seeds.
  If not → stop selling "next-move forecast"; keep the panel for reading/understanding (effect gap, stakes) and
  use markov (+ learned) for the next-move hint.
- Board z annotations OK if on test "??"+"!!" ≤ 10% and "=" ≥ 40%.
- YouTube: report only, no verdict (13 clips across 6 genres).
