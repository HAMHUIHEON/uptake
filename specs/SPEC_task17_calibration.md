# SPEC task17 — forecast calibration and centered board (offline, no API calls)

## Why (overnight run 1, pre-registered Q4 = PROBLEM)
- Next-turn forecast skill vs climatology: negotiation −0.33 [−0.37, −0.29], youtube −0.47 nats.
- Panel over-predicts `proceed` (35% predicted vs 16% actual, negotiation) and `ask_clarification` (24% vs 11%);
  almost never predicts `retaliate`/`repair`/`refuse`. With n=3 readings the distribution is too sharp.
- Board: per-turn value = relationship + face, face ∈ [−2, 0]. Mean per-turn value −0.32 (negotiation, n=2330),
  −0.38 (youtube, n=862) → every curve drifts down. Annotation thresholds are hand-set.

## Fix 1 — blend panel with base rates (standard forecast calibration)
`forecaster/calibrate.py`, K = len(NEXT_ACTIONS), alpha default 0.5:
- `base_rates(hits, alpha=0.5) -> dict` over all NEXT_ACTIONS: (count(actual_class)+alpha)/(n+K·alpha). Sums to 1.
- `panel_prob(hit, alpha=0.5) -> dict` over all NEXT_ACTIONS: (share·nn+alpha)/(nn+K·alpha), share from
  `hit["action_dist"]`, nn = `hit["n_readings"]` (same smoothing as `board.forecast_skill`).
- `blend(panel_p, base, lam) -> dict`: lam·panel + (1−lam)·base, per action.
- `log_score(hits, base, lam, alpha=0.5) -> float`: mean ln p_blend(actual_class). Empty hits → 0.0.
- `fit_lambda(hits, base, grid=None, alpha=0.5) -> float`: grid default [0.00, 0.05, …, 1.00];
  argmax of log_score; ties → smaller lam.
- `blend_skill(hits, base, lam, b=1000, seed=0, alpha=0.5) -> dict`:
  gain_i = ln p_blend(actual_i) − ln base[actual_i]; returns
  `{"skill_mean", "skill_ci" (bootstrap 2.5/97.5 pct), "n", "lam", "hit_rate", "majority_hit_rate"}`;
  hit_rate = share where argmax of blend (ties → alphabetical) == actual; majority_hit_rate = share of actual ==
  argmax of base. Empty → zeros, ci (0.0, 0.0).
- `split_items(ids, frac=0.5, seed=0) -> (dev, test)`: sort ids, shuffle with random.Random(seed),
  dev = first round(len·frac). Deterministic, disjoint, union = ids.

## Fix 2 — centered board, thresholds learned from data
- `fit_board_norm(effects_lists) -> dict`: values = `board._value_of(e)` over all effects of all lists;
  mu = mean; centered c = v − mu; thresholds from percentiles of c (linear interpolation, same as `board._percentile`):
  `{"??": q0.025, "?": q0.10, "?!": q0.25, "!": q0.90, "!!": q0.975}`; return `{"mu", "sd" (population), "n", "thresholds"}`.
  Empty → `{"mu": 0.0, "sd": 0.0, "n": 0, "thresholds": None}`.
- `board.position_curve(effects, norm=None)`: when norm is given, cumulative sum of (value − norm["mu"]).
  norm=None keeps current behavior exactly.
- `board.annotate_turn(v, norm=None)`: when norm (with thresholds) is given, v is a centered value:
  v ≤ t["??"] → "??"; elif v ≤ t["?"] → "?"; elif v ≤ t["?!"] → "?!"; elif v ≥ t["!!"] → "!!"; elif v ≥ t["!"] → "!"; else "=".
  norm=None keeps current behavior exactly.

## CLI `uptake-calibrate --out DIR [--seed 0] [--frac 0.5]`
- Reads every item JSON in DIR that has `genre`, `hits`, `effects` (skip report.json, run.json, calibration.json,
  manifest, logs; skip subdirectories).
- Per genre: split item ids dev/test; on dev fit base_rates, lam, board norm; on test report blend_skill (CI),
  raw-panel skill (= blend_skill with lam=1.0 against the same dev base), annotation shares with the norm,
  mean final curve value before (norm=None) and after (norm).
- Writes DIR/calibration.json (all numbers above, plus seed, frac, dev/test ids) and DIR/calibration.md
  (one table row per genre: n dev/test items, lam, test skill [CI], raw skill [CI], blended hit vs majority,
  mu, mean end value before→after, annotation shares). No network calls.
- Register in pyproject: `uptake-calibrate = "forecaster.calibrate:main"`.

## Pre-registered reading (before running)
- Fix 1 works if test skill CI lower bound > 0 in negotiation. lam≈0 means the panel adds nothing beyond base rates.
- Fix 2 works if centered mean end value is near 0 (|·| < 1) and annotation shares on test are within ±5 pt of
  the dev quantile targets (2.5/7.5/15 low, 10/2.5 high).
