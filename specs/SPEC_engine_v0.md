# SPEC — Forecast engine v0

Goal: given a message, the sender's intent and what we know about the receiver, simulate a panel of plausible receivers, collect K readings from each with a Meta Muse model, and report the **misreading rate** with an honest 95% confidence interval.

Everything below must be implemented in Python 3.11+ as the package `forecaster/` with a CLI. The tests in `tests/` are the contract: make them pass without editing them.

## 0. Ground rules
- No network calls in tests. All model calls go through a client object that tests replace with a fake.
- Read the API key from the environment variable `META_API_KEY` (fallback `MODEL_API_KEY`). Never print or log it.
- Model API: OpenAI-compatible, base URL `https://api.meta.ai/v1`, endpoint `chat/completions`. Default reader model `muse-spark-1.3-contributor`; overridable.
- Use the `openai` Python SDK for the real client. Put dependencies in `pyproject.toml` (`openai`, `pytest`). Create the virtualenv in `.venv` inside the repo.
- Deterministic where seeded: the same seed must give the same panel and the same bootstrap result.

## 1. Labels (`forecaster/labels.py`)
```python
SPEECH_ACTS = ["request", "command", "refusal", "acceptance", "warning", "threat",
               "praise", "criticism", "apology", "promise", "disclosure",
               "information", "question", "probe"]
NEXT_ACTIONS = ["comply", "refuse", "negotiate", "ask_clarification", "escalate",
                "withdraw", "retaliate", "repair", "proceed", "ignore"]
TONE_MIN, TONE_MAX = -2, 2
```
`probe` = testing the other side's reaction ("I'm thinking of stepping down"). `disclosure` = revealing a feeling or fact about oneself.

## 2. Data types (`forecaster/types.py`, dataclasses)
- `Intent(speech_act: str, next_action: str, tone: int)` — what the sender means and hopes the receiver will do. Validate against the label sets and tone range; raise `ValueError` otherwise.
- `Receiver(id: str, relationship: str, power: str, channel: str, state: str, stance: str = "neutral")`
  - `power` in {"sender_higher", "receiver_higher", "equal"}
  - `state` = free text: what the receiver wants, believes, fears right now.
  - `stance` in {"trusting", "neutral", "suspicious"}.
- `Reading(receiver_id: str, sample: int, speech_act: str, tone: int, next_action: str, trigger: str, gloss: str)`
  - `trigger` = the exact phrase in the message that drove this reading (must be a substring of the message; if not, set to "").
  - `gloss` = one sentence: what the receiver thinks the sender meant.
- `Forecast(message, intent, readings: list[Reading], miss_rate: float, ci_low: float, ci_high: float, per_receiver: dict[str, float], top_triggers: list[tuple[str,int]])`

## 3. Panel (`forecaster/panel.py`)
`build_panel(base: Receiver, n: int = 6, seed: int = 0) -> list[Receiver]`
- Keeps `relationship`, `power`, `channel`, `state` from `base`.
- Varies `stance` so the panel spans uncertainty about the receiver: cycle through ["neutral", "suspicious", "trusting"] in that order, starting with "neutral".
- ids: `f"{base.id}-{i}"` for i in 0..n-1.
- `n` must be >= 1. Same inputs give an identical list (seed reserved for later variation; do not use randomness in v0 beyond what is specified).

## 4. Reader (`forecaster/reader.py`)
- `build_messages(message: str, receiver: Receiver) -> list[dict]` returns chat messages. The system prompt casts the model as **this receiver** (relationship, power, channel, state, stance) reading the message **without knowing the sender's intent**, and asks for JSON only with keys `speech_act`, `tone`, `next_action`, `trigger`, `gloss`, listing the allowed labels. The sender's intent must NOT appear anywhere in the prompt.
- `parse_reading(raw: str, receiver_id: str, sample: int, message: str) -> Reading | None`
  - Accept raw JSON or JSON inside a ```json fenced block or surrounded by prose (take the first `{...}` object).
  - Lowercase and strip labels. Map these synonyms: "order"→"command", "instruction"→"command", "ask"→"request", "rejection"→"refusal", "compliment"→"praise", "complaint"→"criticism", "test"→"probe", "clarify"→"ask_clarification".
  - `tone`: coerce numeric strings; clamp to [-2, 2]; round floats to nearest int.
  - Return `None` if speech_act or next_action is not in the label sets after mapping, or JSON cannot be parsed.
  - If `trigger` is not a substring of `message`, set it to "".
- `read_panel(client, message, panel, k=5, model=DEFAULT_MODEL, temperature=1.0) -> list[Reading]`
  - For each receiver, call the client k times. `client.complete(messages, model, temperature) -> str`.
  - If a sample fails to parse, retry that sample once; if it fails again, skip it.
- `MuseClient` class implementing `complete` with the openai SDK.

## 5. Scoring (`forecaster/score.py`)
`is_miss(reading: Reading, intent: Intent) -> bool` — True if ANY of:
1. `reading.speech_act != intent.speech_act`
2. `reading.next_action != intent.next_action`
3. `abs(reading.tone - intent.tone) >= 2`

`miss_rate(readings, intent) -> float` — share of misses (0.0 if no readings).
`per_receiver_rates(readings, intent) -> dict[str, float]`.
`top_triggers(readings, intent, n=3) -> list[tuple[str, int]]` — most frequent non-empty triggers among **missed** readings, ties broken alphabetically.

## 6. Statistics (`forecaster/stats.py`)
- `wilson_ci(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]` — Wilson score interval; (0.0, 1.0) when n == 0.
- `cluster_bootstrap_ci(values_by_cluster: dict[str, list[float]], b: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]`
  - Returns (point, low, high). point = mean of all values pooled.
  - Each replicate resamples **clusters** with replacement (same number of clusters), pools their values, takes the mean. Percentile interval.
  - Use `random.Random(seed)`; identical inputs+seed → identical output.
  - Single cluster → return (point, point, point).
- `paired_bootstrap_diff(before: dict[str, list[float]], after: dict[str, list[float]], b=2000, seed=0, alpha=0.05) -> tuple[float, float, float]`
  - Clusters are matched by key (same receiver before/after). Replicate: resample cluster keys, diff = mean(after pooled) − mean(before pooled). Returns (point, low, high).
  - Keys must match exactly; otherwise raise `ValueError`.

## 7. Forecast (`forecaster/forecast.py`)
`run_forecast(client, message, intent, base_receiver, n=6, k=5, seed=0, model=..., b=2000) -> Forecast`
- panel → readings → miss values per receiver (1.0 miss / 0.0 hit) → `cluster_bootstrap_ci` → fill `Forecast`.

## 8. CLI (`forecaster/cli.py`, entry point `forecast`)
`forecast --case examples/<file>.json [--n 6] [--k 5] [--model ...] [--seed 0]`
Case file: `{"message": "...", "intent": {...}, "receiver": {...}}`. Prints a short report: miss rate with CI, per-receiver rates, top triggers, three example glosses from missed readings. `--json` prints the full Forecast as JSON.

## 9. Done when
- `.venv/bin/pytest -q` passes.
- `forecast --case examples/deadline_slack.json --n 3 --k 2` runs against the real API (manual check, not in tests).
