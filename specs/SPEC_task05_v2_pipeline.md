# SPEC — task05: v2 end-to-end, move annotation, control benchmark

## 1. `forecaster/v2_run.py`
`run_forecast_v2(client, message, goal, base_receiver, n=6, k=5, seed=0, model=DEFAULT_MODEL, judge_model="muse-spark-1.3", b=2000, max_workers=8) -> dict`
- Panel = `build_panel(base_receiver, n)`.
- For each (receiver, sample) call `client.complete(build_messages_v2(message, receiver), model, 1.0)`; parse with `parse_reading_v2`; retry once then skip (same as v0). Run concurrently with a ThreadPoolExecutor(max_workers); return readings ordered by panel order then sample.
- For each reading call `judge_same_meaning(client, goal.core_meaning, reading.understood, model=judge_model)` (concurrently).
- Returns `{"panel", "readings", "same": {(rid, sample): bool|None}, "score": score_v2(...), "annotation": annotate(score)}`.

## 2. Move annotation (chess style) — `annotate(score) -> str`
Based on the `any` point estimate p:
- p < 0.15 → "!" (good move)
- 0.15 ≤ p < 0.35 → "=" (sound)
- 0.35 ≤ p < 0.60 → "?!" (dubious)
- 0.60 ≤ p < 0.85 → "?" (mistake)
- p ≥ 0.85 → "??" (blunder)

## 3. Case files v2
`{"message", "goal": {acceptable_actions, relationship_floor, face_ceiling, core_meaning}, "receiver": {...}}`. Loader `load_case_v2(path) -> (message, Goal, Receiver)`.

## 4. CLI
- `forecast --case FILE --v2` prints: annotation, then one line per stake and `any` as `stake  point [low-high]` plus a 20-char bar (█ for the point share), then 3 example `understood` sentences from readings with any gap.
- `forecast-bench --dir DIR [--n --k --workers --model]` runs every v2 case in DIR as a **control** set and prints per-case annotation and `any`, then the false-alarm rate with Wilson CI (`false_alarm_rate`). Register the entry point in pyproject.toml.
