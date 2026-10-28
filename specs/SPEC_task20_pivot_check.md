# SPEC task20 — exploratory pivotal-turn check (offline)

Add to `forecaster/outcome.py` (see EVAL_PLAN_overnight_2 Amendment 2):
- `pivot_rows(items, outcomes, mu) -> list[dict]`: per item with outcomes, `t = board.turning_point(effects)`; skip if None.
  Find the effect with index t; X = its speaker; row = `{"dialogue", "side": X, "pivot": value − mu,
  "opp_likeness": likeness reported by the other side, "own_likeness_gap": opp_likeness(X) − opp_likeness(other)}`
  where opp_likeness(Y) = likeness reported by Y's opponent.
- `random_turn_rho(items, outcomes, mu, draws=200, seed=0) -> float`: for each draw, for each item that has a pivot row,
  pick one random effect of the same speaker X (`random.Random(seed)` shared across draws, items in sorted id order)
  and compute Spearman(value − mu, opp_likeness of X); return the mean over draws.
- `pivot_direction(rows) -> dict`: rows with pivot ≠ 0 and own_likeness_gap ≠ 0; agree = sign(pivot) == sign(gap);
  `{"agree": k, "n": n, "share": k/n, "ci": stats.wilson_ci(k, n)}` (n = 0 → share 0.0, ci (0.0, 1.0)).
- In `analyze`: add `"D1_pivot": boot_spearman(pivot_rows, "pivot", "opp_likeness")`, `"D1_random_turn_mean": random_turn_rho(...)`,
  `"D2": pivot_direction(...)`, verdicts "REPORTED". Add the rows to outcome.md.
Tests in `tests/test_task20.py` (written by Muse): toy items where pivot sign matches likeness gap → share 1.0;
random_turn_rho deterministic for a fixed seed; analyze output contains D1/D2 keys.
