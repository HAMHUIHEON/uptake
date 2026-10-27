# SPEC — task08: L1 game record (기보) — classify each turn into moves

A conversation is a game; each turn is one or more **moves**. L1 turns a list of speaker turns into a move record that later layers (board, turning points, strategy) build on. It must be measurable: CaSiNo (1,030 negotiation dialogues, CC BY 4.0) has per-utterance strategy labels we map onto our moves.

## 1. Move set (`forecaster/moves.py`)
```python
MOVES = ["smalltalk", "inform", "ask", "propose", "counter", "concede", "accept", "reject",
         "probe", "appeal_need", "appeal_fairness", "empathize", "cooperate", "pressure",
         "praise", "challenge", "repair", "deflect", "joke", "other"]
CASINO_MAP = {"small-talk": "smalltalk", "self-need": "appeal_need", "other-need": "appeal_need",
              "no-need": "concede", "elicit-pref": "ask", "uv-part": "challenge",
              "vouch-fair": "appeal_fairness", "showing-empathy": "empathize",
              "promote-coordination": "cooperate", "non-strategic": "other"}
```
Short definitions for the prompt (one line each) — e.g. `probe`: tests the other side's reaction without committing; `pressure`: deadline, threat or ultimatum; `deflect`: avoids the question or changes topic; `cooperate`: frames the exchange as joint problem-solving; `appeal_need`: justifies a claim by one's own (or one's group's) needs.

## 2. Types
`@dataclass MoveTag(turn_index: int, speaker: str, moves: list[str], addressed_to: str = "", quote: str = "", failed: bool = False)`

## 3. Prompt and parsing
- `build_move_messages(turns, i, window=4) -> list[dict]`: system prompt lists MOVES with definitions; user content shows up to `window` PREVIOUS turns as context (speaker: text), then the target turn clearly marked. **Never include turns after i** (the tagger must be causal so it can run live later). Ask for JSON only: `{"moves": [1-3 labels, most important first], "addressed_to": "<speaker label or 'all'>", "quote": "<short exact phrase from the target turn>"}`.
- `parse_moves(raw, turn_text) -> dict | None`: tolerant JSON (fenced / prose), lowercase labels, keep only labels in MOVES, dedupe preserving order, cap at 3; None if no valid label remains. `quote` must be a substring of `turn_text`, else "".

## 4. Tagging (`tag_conversation(client, turns, model=DEFAULT_MODEL, window=4, max_workers=8) -> list[MoveTag]`)
One call per turn (concurrent), retry once on parse failure; if it still fails → `MoveTag(i, speaker, ["other"], failed=True)`. Output ordered by turn index. `turns` are `forecaster.transcribe.Turn` objects.

## 5. CaSiNo loader (`forecaster/casino.py`)
`load_casino(path, limit=None, seed=0) -> list[dict]` each `{"dialogue_id", "turns": list[Turn], "gold": list[set[str]]}`:
- Walk `chat_logs` in order; skip entries whose `text` is a deal action (`Submit-Deal`, `Accept-Deal`, `Reject-Deal`, `Walk-Away`).
- Speaker = `id` (`mturk_agent_1` → "A", `mturk_agent_2` → "B").
- Gold labels: `annotations` is a list of `[text, "label1,label2"]` in utterance order; match each kept chat turn to the next annotation with identical text; map labels through CASINO_MAP into a set. Turns without a matching annotation get gold `None` and are excluded from scoring. Dialogues with no annotations are skipped.
- `limit`: random sample of that many dialogues using `random.Random(seed)`, then sorted by dialogue_id.

## 6. Evaluation (`forecaster/eval_moves.py`)
- `prf(pred: list[set], gold: list[set]) -> dict`: per-label {"p","r","f1","support"}, plus "micro_f1" and "macro_f1" (macro over labels that appear in gold). Precision with no predictions = 0; F1 = 0 when p+r = 0.
- `eval_with_ci(per_dialogue: list[tuple[list[set], list[set]]], b=1000, seed=0) -> dict`: micro_f1 and macro_f1 point estimates plus 95% bootstrap CI **resampling dialogues** (cluster = dialogue).
- Baseline: `majority_baseline(gold_all) -> set` = {most frequent label}; report its micro_f1 next to the model's.
- CLI `uptake-eval-moves --casino PATH --n 30 --seed 0 [--model ...] [--workers 8] [--out results.jsonl]` prints per-label table, micro/macro F1 with CI and the baseline.
- CLI `uptake-moves --turns turns.jsonl --out moves.jsonl` tags a transcript.
