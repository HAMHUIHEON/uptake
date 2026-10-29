# SPEC task28 — re-read the board panel on existing records with a different scale / panel size

Context: `specs/EVAL_PLAN_overnight_2.md`, Amendment 6. We re-run ONLY the listener panel (board layer) on dialogues that were already
processed by `uptake-batch`. Move tags are reused from the existing record — no re-tagging, no casino dataset needed.

## New module `forecaster/rescore.py` (stdlib + existing project modules only)

### Constants
```python
SCALES = ("v2", "fine100", "seven")
```

### Prompt: `build_scaled_messages(turns, i, listener, receiver, scale, window=6) -> list[dict]`
- `scale == "v2"`: return exactly `board.build_listener_messages(turns, i, listener, receiver, window)` (unchanged prompt).
- Otherwise copy the v2 system prompt wording (listener / relationship / power / channel / state / stance lines, "You just heard the TARGET
  turn …", "You do not know what the speaker intended.", JSON-only instruction, `next_action` line with `labels.NEXT_ACTIONS`,
  `understood` line) but replace the two numeric keys:
  - `fine100`: keys `next_action, relationship_after, face_effect, understood`.
    "- relationship_after must be an integer from 0 to 100: how you feel toward the speaker of the TARGET turn now, where 50 = unchanged
    from before this turn, 0 = much colder, 100 = much warmer"
    "- face_effect must be an integer from 0 to 100: what the TARGET turn does to your face (dignity, standing), where 50 = no effect,
    0 = severe threat to your face, 100 = strongly affirms your face"
  - `seven`: keys `next_action, relationship_shift, face_effect, understood`.
    "- relationship_shift must be an integer from -3 (much colder) to 3 (much warmer) toward the speaker of the TARGET turn, 0 = unchanged"
    "- face_effect must be an integer from -3 (severe threat to your face) to 3 (strongly affirms your face), 0 = no effect"
  The user message (context lines + `TARGET turn {speaker}: {text}`) is built exactly like `board.build_listener_messages`.
- Unknown scale → `ValueError`.

### Parse: `parse_scaled(raw, scale) -> dict | None`
Returns `{"next_action": str, "rel_raw": int, "face_raw": int, "rel_norm": float, "face_norm": float}` or `None`.
- Extract the first JSON object in `raw` (same tolerant approach as `v2._extract_obj`; text before `{` allowed). Not a dict / no object → None.
- `next_action` lower-cased, must be in `labels.NEXT_ACTIONS`, else None.
- `v2`: rel key `relationship_shift` bounded −2..2, face key `face_threat` bounded 0..2; `rel_norm = rel`, `face_norm = -face`.
- `fine100`: rel key `relationship_after`, face key `face_effect`, both bounded 0..100; `rel_norm = (rel-50)/25`, `face_norm = (face-50)/25`.
- `seven`: rel key `relationship_shift`, face key `face_effect`, both bounded −3..3; `rel_norm = rel*2/3`, `face_norm = face*2/3`.
- Numeric coercion: accept int, float or numeric string; round half away from zero is NOT required — use `int(round(float(x)))`; then clamp
  to the bounds. Missing key, bool, or non-numeric → None (do NOT default to 0).

### Turn evaluation: `evaluate_turns_scaled(client, turns, scale="v2", n=3, k=1, model=DEFAULT_MODEL, stance_mix=DEFAULT_MIX, window=6,
b=1000, seed=0, max_workers=8, move_tags=None, item_id="") -> list[dict]`
Mirror `board.evaluate_turns` exactly for listener selection (addressed_to rule), panel construction (`build_panel(base, n, seed, stance_mix)`
with the same `base` Receiver), k samples, threading, and the empty / no-reading cases. Differences:
- messages from `build_scaled_messages(..., scale, window)`; each call through `board._call_board(client, messages, model, meta)` with
  `meta = {"layer": "board", "scale": scale, "item": item_id, "turn": i, "listener": listener, "member": receiver.id,
  "stance": receiver.stance, "sample": s}`; up to 2 attempts per reading (BudgetExceeded re-raised, other exceptions → next attempt,
  parse None → next attempt).
- `relationship` = `stats.cluster_bootstrap_ci` over `rel_norm` clustered by member id; `face` = same over `face_norm`
  (so `board._value_of(effect)` = mean rel_norm + mean face_norm).
- each reading row: `{"turn", "listener", "member", "stance", "sample", "next_action", "rel_raw", "face_raw", "rel_norm", "face_norm"}`;
  for `v2` also `"relationship_shift": rel_raw, "face_threat": face_raw`.
- extra effect keys: `"scale": scale`, `"n_expected": len(tasks)` (0 when no listeners).
- `action_dist`, `disagreement`, `n_readings` as in board.

### Record helper: `turns_from_record(record) -> list[Turn]`
Turns rebuilt from `record["effects"]` sorted by `index`: `Turn(speaker=e["speaker"], text=e["text"])` (`transcribe.Turn`).

### `select_ids(src_dir, sample=None, sample_seed=6, ids_file=None) -> list[str]`
All ids from files `casino-*.json` in `src_dir` (id = file stem), sorted as strings. If `ids_file` is given: one id per line, keep that order,
only ids that exist. Else if `sample` is an int: `random.Random(sample_seed).sample(sorted_ids, min(sample, len))`, returned in the sampled order.

### `rescore_item(client, record, out_dir, scale, n, k, model, inner_workers, window=6) -> dict`
Evaluates, then writes `out_dir/<id>.json` with keys `id, genre, model, moves (copied), effects, board (= effects), hits
(= board.forecast_hits(effects, moves)), gold (copied), scale, n`. Returns a summary
`{"id", "n_turns", "n_readings", "n_expected"}`.

## CLI `uptake-rescore` (`forecaster.rescore:main`, register in pyproject)
Args: `--src DIR` (required), `--out DIR` (required), `--scale {v2,fine100,seven}` (required), `--n 3`, `--k 1`, `--sample INT`,
`--sample-seed 6`, `--ids-file PATH`, `--budget FLOAT` (required), `--workers 6`, `--inner-workers 8`, `--effort low`,
`--model muse-spark-1.3-contributor`, `--window 6`.
- Ids from `select_ids`. Write the selected ids to `out/ids.txt` (one per line).
- Resume: skip ids whose `out/<id>.json` already exists.
- Client: `MuseClient(meter=Meter(budget_usd=budget), default_effort=effort or None)` wrapped by `trace.LoggingClient(inner,
  CallLog(out/"calls.jsonl"), run_id)` — same as `batch.batch_main`.
- Items run in a `ThreadPoolExecutor(max_workers=workers)`. On `usage.BudgetExceeded` from any item: stop submitting new items, let running
  ones finish or fail, record the stop.
- At the end write `out/run.json` (run_id, argv, settings, git commit via `batch._git_commit()`) and `out/report.json`:
  `{"n_selected", "n_done" (files present for the selected ids), "n_skipped_existing", "n_errors", "errors": [...],
  "readings": sum n_readings, "expected": sum n_expected, "parse_fail_rate": 1 - readings/expected (0 if expected 0),
  "spend": meter.by_model(), "cost": meter.cost(), "budget_stop": bool}`; print a one-line summary incl. `spend: $x.xxxx`.
- Never print environment variables or the API key.

## Tests `tests/test_task28.py` (synthetic only, fake client, no network, no data/)
- prompts: v2 equals `board.build_listener_messages` output; fine100 / seven system prompt contains the new keys and ranges and NOT
  `face_threat`; unknown scale raises.
- parse: each scale, valid JSON → correct raw and norm (e.g. fine100 rel 75 face 50 → 1.0, 0.0; seven 3,-3 → 2.0,-2.0;
  v2 rel -1 face 2 → -1, -2); clamping (fine100 150 → 100); string numbers; missing key → None; bad next_action → None; prose before JSON ok.
- evaluate_turns_scaled with a fake client returning fixed JSON: value per turn = rel_norm + face_norm; listeners rule matches
  `board.evaluate_turns` on the same input (same listeners / n_readings for v2); `n_expected` = n·k·listeners; a fake client that returns
  garbage for one member gives n_readings < n_expected; meta passed includes scale when the client has `accepts_meta=True`.
- select_ids: sort, sampling deterministic for a seed and equal to `random.Random(seed).sample(sorted_ids, k)`, ids_file order.
- rescore_item writes the record with copied moves/gold and `board` == `effects`.
- `outcome.reception` on a rescored record equals the mean of (value − mu) per speaker (format compatibility).
