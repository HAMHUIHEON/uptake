# SPEC — task03: parallel reads, run logs, rewrite comparison

1. `read_panel(..., max_workers: int = 8)`: run calls concurrently with `concurrent.futures.ThreadPoolExecutor(max_workers)`. Retry-once-then-skip behaviour stays the same per (receiver, sample). Return readings ordered by panel order, then sample index, regardless of completion order. `run_forecast` gets a `max_workers` parameter passed through. CLI gets `--workers` (default 8).
2. `forecaster/runlog.py`:
   - `save_readings(path, readings, meta: dict)` writes JSONL: first line `{"meta": meta}`, then one line per reading (dataclass fields as JSON).
   - `load_readings(path) -> (meta, list[Reading])`.
   - CLI gets `--save PATH` to write the run (meta: message, intent, receiver, model, n, k, seed, timestamp ISO-8601 UTC).
3. `forecaster/rewrite.py`: `run_rewrite(client, original, rewrite, intent, base_receiver, n=6, k=5, seed=0, b=2000, model=DEFAULT_MODEL, max_workers=8) -> dict` with keys `panel`, `before` (Forecast), `after` (Forecast), `diff` = `paired_bootstrap_diff(before_values, after_values, b=b, seed=seed)` where values are per-receiver lists of 1.0 (miss) / 0.0 (hit). The SAME panel object is used for both runs. If a receiver has no readings in one run, drop it from both before diffing.
   - CLI: `forecast --case FILE --rewrite "new text"` prints before/after miss rates with CIs and the paired difference with its CI.
