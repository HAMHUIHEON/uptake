# SPEC task31 — make the 0–100 panel scale the default (batch, commentary, viewer)

Decision after EVAL_PLAN_overnight_2 Amendment 6 (2026-10-29): S* = `fine100` is the default
board scale from now on. `v2` stays available so old records still reproduce.

## batch (`forecaster/batch.py`)
- `uptake-batch run` gets `--scale {v2,fine100,seven}`, default `fine100`.
- When `--scale v2`: unchanged path (`board.evaluate_turns`).
- Otherwise call `rescore.evaluate_turns_scaled(client, turns, scale=..., n, k, model, ...)` with the
  same worker/effort settings. The record gains `"scale": <scale>` (top level) and the effects must
  carry every key the downstream readers use (`board._value_of`, `forecast_hits`, `position_curve`,
  `outcome.participant_rows`, `commentary.turn_values`, `viewer`). If `evaluate_turns_scaled` lacks a
  field that `forecast_hits` needs (e.g. next-action votes), add it there — do not change the
  `v2` output format.
- Manifest and cost logging unchanged. `calls.jsonl` meta already records `scale`.

## commentary (`forecaster/commentary.py`)
- `main`: after loading fit items, read their `scale` (missing → `"v2"`). If the item's scale differs
  from the fit items' scale, exit with a clear error (`scale mismatch: item=fine100 fit=v2`); the
  norm must be fitted on same-scale records.
- Put `"scale"` into the `norm` dict that goes into the result JSON.

## viewer (`forecaster/viewer.py`)
- Show the panel scale in the header line (`panel 0–100` or `panel −2..+2`).
- In the turn hover/detail, show the raw panel reading for `fine100` records (mean `rel_raw`,
  `face_raw` over readings) next to the normalized value. Marks are unchanged here (task33).
- No visual change for `v2` records.

## tests `tests/test_task31.py` (Muse writes)
- batch item function with a fake client on `fine100` and `v2`: both records have the keys listed
  above; `scale` field set; `forecast_hits` runs on both.
- commentary `main` refuses mismatched scales (SystemExit) and accepts matching ones.
- viewer renders a `fine100` record and the HTML contains the scale label and a raw reading.
- `pytest -q` must stay green (currently 207 tests).
