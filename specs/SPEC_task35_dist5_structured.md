# SPEC task35 — structured output for the panel; "dist5" distribution reading (Amendment 8, P8-A)

Touches `forecaster/reader.py`, `forecaster/rescore.py`, `forecaster/scalecheck.py`. Stdlib only.

## reader: structured output
- `MuseClient.complete(messages, model, temperature, reasoning_effort, response_format=None)`:
  when `response_format` is given, pass it through to the Chat Completions request (OpenAI-style
  `response_format={"type": "json_schema", "json_schema": {"name", "strict": true, "schema"}}`).
  Log `structured: true` in the call record. A 400 from the API must surface as a clear
  exception (`StructuredOutputError` with the server message), not be retried silently.
- Schemas live in `forecaster/schemas.py`: `panel_schema(scale)` for `v2`, `fine100`, `seven`,
  `dist5`. All strict-subset compliant: root object, `additionalProperties: false`, every property
  in `required`, `next_action` as an enum of `NEXT_ACTIONS`, numbers as `integer`/`number` with a
  one-line `description` stating the range (do not rely on min/max keywords).

## rescore: `--structured` and scale `dist5`
- `uptake-rescore --structured` switches every scale to schema-constrained output; default off
  (keeps Amendment 6 runs reproducible). Record top level gains `"structured": true/false`.
- New scale `dist5` (always structured). Prompt = the fine100 listener prompt, but the answer is
  `p_rel` and `p_face`: arrays of 5 numbers (bins 0–20, 20–40, 40–60, 60–80, 80–100) that sum to 1
  (normalize; if the sum is 0 or any value negative → parse failure), plus `next_action`,
  `understood`.
- `parse_scaled(raw, "dist5")` → `rel_raw` = EV over midpoints [10, 30, 50, 70, 90], `rel_norm` =
  (rel_raw − 50)/25, `rel_spread` = sd of the distribution in 0–100 units; same for face; keep
  `p_rel`/`p_face` in the reading. Effect level: existing keys plus `spread` = mean of readings'
  `rel_spread`.
- `SCALES = ("v2", "fine100", "seven", "dist5")`; `record_scale` works for the new one.

## scalecheck
- `resolution` handles continuous values: count distinct turn values rounded to 0.01 and also to
  0.1; report mean `spread` when present.
- `uptake-scale pilot` accepts a `dist5` dir alongside the others; `reliability-fine100` (task34)
  must work on any scale's readings (it reads `rel_norm`), rename the CLI to
  `uptake-pairwise reliability-split` with `--src` and keep the old name as an alias.

## tests `tests/test_task35.py` (Muse writes)
- schemas: every `panel_schema(scale)` satisfies the strict rules (checked by a small validator in
  the test); `next_action` enum equals `NEXT_ACTIONS`.
- `parse_scaled` dist5: EV/normalization math on a hand-made distribution; failure on zero or
  negative sums; spread of a point mass is 0.
- fake client records `response_format` when `--structured`; a 400 raises `StructuredOutputError`.
- `resolution` on toy dist5 records reports rounded distinct counts and mean spread.
- `pytest -q` must stay green.
