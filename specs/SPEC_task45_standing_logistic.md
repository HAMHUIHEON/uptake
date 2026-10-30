# SPEC task45 — standings on the pairwise scale: logistic conversion instead of 50 + 25·z

Observed (demo regen on pairwise records, 2026-10-30): with BT ratings sd ≈ 2.29 the linear map `50 + 25·z` pins
standings to 0 or 100 for ordinary turns ("standing 51 → 0"). Replace it everywhere standings are shown for pairwise
records (commentary lines, viewer side labels, square summary, settlement sentence):

- `standing = round(100 · sigmoid(z))` with `z = (value − mu) / sd` from the pairwise table (mu/sd by move kind if the
  table has them, else global). sigmoid(z) = 1 / (1 + e^(−z)). So mean → 50, ±1σ → 73/27, ±2σ → 88/12, never pinned.
- 0–100 records keep the current behaviour (identity).
- Commentary JSON: `"standing_scale": "pairwise→0-100 via logistic(z)"`. Viewer badge text unchanged.
- Settlement sentence and hover must use the same function (one helper in commentary.py reused by viewer2.py).

tests (`tests/test_task45.py`): sigmoid mapping values (z=0→50, ±1→73/27, ±2→88/12), commentary standings on a toy
pairwise record no longer hit 0/100 for |z|<3, viewer2 side labels/settlement use the same numbers as the commentary
JSON, 0–100 records unchanged (compare to task43 fixtures). `pytest -q` stays green.
