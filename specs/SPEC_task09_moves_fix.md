# SPEC — task09: fix move evaluation and definitions (after first CaSiNo run)

First run (30 dialogues, 347 turns): micro F1 0.338 vs majority baseline 0.300. Cause: CaSiNo labels deal-making utterances (offers, accepts) as `non-strategic`, while our tagger answered propose/counter/accept, which do not exist in the gold scheme and were all counted as errors. Projecting predictions onto the gold scheme gives micro F1 0.572 [0.531–0.614]. Weak labels: cooperate 0.23, appeal_fairness recall 0.20, concede 0.13 (definition mismatch with CaSiNo `no-need`). Speed: 16.6 min for 347 turns.

## 1. Moves
- Add `reveal_preference` right before `other` (MOVES has 21 entries, `other` stays last). Definition: "reveals what one values or does not need (e.g. 'we have plenty of water'), signalling room to trade". Map CaSiNo `no-need` → `reveal_preference`.
- `MOVE_EXAMPLES: dict[str, list[str]]` — at least one short example utterance for: cooperate ("Let's find a split that works for both of us"), appeal_fairness ("It's only fair we each get two"), ask ("What matters most to you?"), concede ("Okay, you can have the firewood"), reveal_preference ("We don't really need water"), propose ("I'll take 2 food and you take 3 water"). Include examples in the prompt next to the definitions.
- Prompt guidance lines: "Label ask only when the speaker actually seeks the other's preferences or information, not for rhetorical questions." and "One utterance can carry a deal move AND a strategy (e.g. propose + appeal_need)."

## 2. Projection onto a gold scheme (`forecaster/eval_moves.py`)
- `project(pred: list[set], scheme: set[str], other: str = "other") -> list[set]`: keep only labels in scheme; if nothing remains → {other}.
- `CASINO_SCHEME = set(CASINO_MAP.values())`.
- `uptake-eval-moves` projects by default; `--no-project` disables. Raw predictions are still saved unprojected in `--out`.

## 3. Hold-out sampling (`forecaster/casino.py`)
- `load_casino(path, limit=None, seed=0, exclude_ids=())`: drop dialogues whose `dialogue_id` is in `exclude_ids` BEFORE sampling.
- CLI `--exclude RESULTS.jsonl` reads dialogue_ids from a previous results file and excludes them.

## 4. Speed and model
- CLI `--model` default for tagging becomes `muse-spark-1.3-contributor`.
- `MuseClient.complete(messages, model, temperature, reasoning_effort=None)`: when `reasoning_effort` is set, pass it to the API as `extra_body={"reasoning_effort": value}`. Tagging CLI flag `--effort` (default "low"). `tag_conversation(..., reasoning_effort=None)` passes it through; fake clients in tests that do not accept the keyword must still work (only pass the keyword when not None).
