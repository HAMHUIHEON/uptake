# SPEC — task13: metered client, model routing by data rights, overnight batch runner, genre report

Measured: per call ≈ 1,000 input + 900–2,400 output tokens (mostly reasoning; effort "low" halves it and is ~2x faster). Prices per 1M tokens (dev.meta.ai pricing page): muse-spark-1.3 / 1.2 $1.25 in, $4.25 out; *-contributor $0.10 in, $0.20 out. Contributor mode lets Meta train on prompts and responses, so it may be used ONLY for openly licensed research data. Contributor rate limit: 150 requests/min per team.

## 1. Usage meter (`forecaster/usage.py`)
- `PRICES = {"muse-spark-1.3": (1.25, 4.25), "muse-spark-1.2": (1.25, 4.25), "muse-spark-1.3-contributor": (0.10, 0.20), "muse-spark-1.2-contributor": (0.10, 0.20)}` ($ per 1M input, output). Unknown model → standard 1.3 prices.
- `class BudgetExceeded(RuntimeError)`.
- `class Meter(budget_usd: float | None = None)`: thread-safe. `add(model, prompt_tokens, completion_tokens)`, `cost() -> float`, `by_model() -> dict[model, {"calls","prompt_tokens","completion_tokens","usd"}]`, `check()` raises BudgetExceeded when budget is set and cost() >= budget.

## 2. MuseClient changes (`forecaster/reader.py`)
`MuseClient(api_key=None, base_url=BASE_URL, meter=None, default_effort=None, max_retries=5, create=None, sleep=time.sleep)`
- `create` is an injectable callable(**kwargs) returning an OpenAI-style response (default: the SDK's chat.completions.create).
- `complete(...)`: call `meter.check()` first (if meter). If `reasoning_effort` is None use `default_effort`. On an exception whose `status_code` attribute is 429 or >= 500, or whose class name contains "RateLimit", "Timeout" or "Connection", sleep 2**attempt seconds (1, 2, 4, 8, 16) and retry up to max_retries; then re-raise. After success add `resp.usage.prompt_tokens` / `completion_tokens` to the meter.

## 3. Batch runner (`forecaster/batch.py`)
Manifest JSONL, one conversation per line: `{"id", "genre", "kind": "turns"|"casino", "path"?: turns jsonl, "dialogue_id"?: int, "public": bool}`.
- `model_for(item, public_model="muse-spark-1.3-contributor", private_model="muse-spark-1.3") -> str`: contributor only when `public` is true.
- `run_item(client, item, casino_index, out_dir, n=3, k=1, effort="low", inner_workers=8) -> dict`: load turns (casino items: from `casino_index[dialogue_id]` built with `load_casino`); tag moves (`tag_conversation`, same model/effort); `evaluate_turns(..., move_tags=tags)`; `forecast_hits` and `forecast_skill`. Write `out_dir/<id>.json` with id, genre, model, moves (list of tag dicts), board effects, hits, and `gold` (casino only). Return summary `{"id","genre","n_turns","other_rate","failed_rate","hits","pred","gold"}` where other_rate = share of turns whose moves are exactly ["other"], failed_rate = share of tags with failed=True, pred = list of move sets, gold = list of gold sets or None.
- `run_batch(manifest_items, client, out_dir, casino_path=None, workers=4, **item_kwargs) -> list[dict]`: items whose output file already exists are loaded from disk instead of re-run (resume). Run remaining items concurrently (ThreadPoolExecutor(workers)). If BudgetExceeded is raised, stop submitting new items, keep finished ones, and record `{"id", "error": "budget"}` for the item. Other exceptions → `{"id", "error": str(e)}`; the batch continues.
- `genre_report(summaries, b=1000, seed=0) -> dict`: per genre: `n_conv`, `other_rate` (mean over conversations with bootstrap CI over conversations), `failed_rate`, `hit_rate`, `skill` (= forecast_skill over the genre's pooled hits), and for casino items `micro_f1` (projected onto CASINO_SCHEME, `eval_with_ci` over dialogues). Error items are counted in `n_errors`.
- `cohen_kappa(a: list[str], b: list[str]) -> float` (1.0 when both constant and equal); `spearman(x, y) -> float` (average ranks for ties).
- Stability: `repeat_check(items, client, out_dir, sample=5, seed=0, **kw)` re-runs a random sample into `out_dir/repeat/` and returns per-item kappa on the FIRST move of each turn and spearman between the two position curves, plus their means.

## 4. CLIs
- `uptake-manifest --casino PATH --n 60 --seed 2 [--exclude results.jsonl ...] [--turns-glob GLOB --turns-genre youtube]` prints manifest JSONL: casino items `public: true, genre: "negotiation"`; turns files `public: false`.
- `uptake-batch --manifest M --out DIR --casino PATH --budget 100 --workers 4 --inner-workers 8 --n 3 --k 1 --effort low --repeat-sample 5` runs the batch with a metered client, then the repeat check, writes `DIR/report.json` (genre report, stability, spend by model) and prints a compact report.
