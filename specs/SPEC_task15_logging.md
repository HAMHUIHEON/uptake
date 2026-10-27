# SPEC — task15: per-call trace log, run metadata, health check, morning analysis

Implements the logging and the report required by specs/EVAL_PLAN_overnight_1.md.

## 1. Trace log (`forecaster/trace.py`)
- `class CallLog(path)`: thread-safe JSONL appender (`write(record: dict)`).
- `class LoggingClient(inner, log: CallLog, run_id: str)`: has `accepts_meta = True`. `complete(messages, model, temperature, reasoning_effort=None, meta=None)` calls `inner.complete(...)` (pass reasoning_effort only if not None), measures latency, and writes one record:
  `{"run_id","ts","model","effort","prompt_hash","meta": meta or {},"latency_s","raw","error"}` where `prompt_hash` = first 12 hex chars of sha1 of the system message content, `raw` = returned text (or "" on exception) and `error` = exception class name or null. If `inner` exposes `last_stats` (dict with "retries", "status_codes", "prompt_tokens", "completion_tokens") after a call, merge it into the record. Re-raise exceptions after logging.
- `MuseClient` sets `self.last_stats` per call (thread-local is fine): retries, list of status codes seen, prompt_tokens, completion_tokens.
- Callers pass `meta` only when `getattr(client, "accepts_meta", False)`: `tag_conversation` → `{"layer":"tag","item":item_id,"turn":i}`; `evaluate_turns` → `{"layer":"board","item":item_id,"turn":i,"listener":l,"member":receiver.id,"stance":receiver.stance,"sample":s}`. Add an optional `item_id=""` parameter to both.

## 2. Run metadata
`uptake-batch` writes `DIR/run.json` at start: run_id (UTC timestamp), git commit (`git rev-parse HEAD`, or "unknown"), argv, settings, models; and wraps its client in LoggingClient writing `DIR/calls.jsonl`. Board output files must also store per-reading values (`readings`: list of {turn, listener, member, stance, sample, next_action, relationship_shift, face_threat}) so variance can be split later.

## 3. Health check
`run_batch(..., health_after=20, max_failed_rate=0.20, max_error_rate=0.10)`: after `health_after` items have finished, if the mean failed_rate of finished items > max_failed_rate or the share of items with "error" > max_error_rate, stop submitting new items and record `{"health_stop": true, ...}` in the report.

## 4. Analysis (`forecaster/analyze.py`, CLI `uptake-analyze --out DIR`)
Pure helpers (tested):
- `verdict(value, ok, direction)`: "OK" if (direction == ">=" and value >= ok) or (direction == "<=" and value <= ok) else "PROBLEM".
- `percentile(values, q)`: linear interpolation, q in [0,100].
- `confusion(pred_labels, gold_labels) -> dict[(gold, pred), count]` over single labels.
- `variance_split(readings) -> dict`: for each (item, turn) group of numeric values v = relationship_shift - face_threat, split total variance into `between_members` (variance of per-member means) and `within_member` (mean of per-member variances; 0 when one sample). Return overall means of both and their ratio.
The report (`DIR/report.md`) follows the plan: Q1–Q5 table with value, threshold, verdict; for each PROBLEM the listed drill-down output; Q5 latency p50/p95 and retries by status code from calls.jsonl; then 30 random raw traces per genre (seeded).
