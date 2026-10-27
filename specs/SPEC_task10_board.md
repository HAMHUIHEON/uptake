# SPEC — task10: L5 board — turn-by-turn position curve, turning points, forecast check

Picture a Go broadcast: after every move the commentators show who is ahead. Here, after every turn, simulated listeners (the other participants) react; we plot how the position moves, mark the move that turned the game, and **check our own forecast against what the listener actually said next**.

## 1. Listener reading in context (`forecaster/board.py`)
- `build_listener_messages(turns, i, listener: str, receiver: Receiver, window=6) -> list[dict]`: the model plays `listener` (a speaker label in the conversation) who just heard turn i. Show up to `window` previous turns and turn i marked with the literal word TARGET. Never include turns after i. Ask for JSON: `next_action`, `relationship_shift` (-2..2 toward the speaker of turn i), `face_threat` (0..2 to the listener), `understood` (one sentence). Same label sets as v2. Receiver framing as v2 (stance etc.).
- Parse with `parse_reading_v2` (message argument = turn i text).

## 2. Turn effects (`evaluate_turns(client, turns, n=3, k=1, model=DEFAULT_MODEL, stance_mix=DEFAULT_MIX, window=6, b=1000, seed=0, max_workers=8) -> list[dict]`)
For each turn i, listeners = every other speaker label present in `turns` except "unknown" (if a conversation has only one speaker, the turn gets no effect). For each listener, panel = `build_panel(Receiver(id=f"{listener}@{i}", relationship="participant in this conversation", power="equal", channel="conversation", state=""), n, seed, stance_mix)`, k samples each (concurrent; retry once then skip).
Per turn returns:
```
{"index": i, "speaker": s, "text": ..., "listeners": [...],
 "relationship": (mean, low, high),   # mean relationship_shift over readings, cluster bootstrap CI (cluster = panel member)
 "face": (mean, low, high),           # NEGATIVE mean face_threat (higher = better for the listener's face)
 "action_dist": {next_action: share}, # over all readings of this turn
 "disagreement": float,               # normalized entropy of action_dist in [0,1] (0 if <= 1 reading)
 "n_readings": int}
```
Effect "value" of a turn `v_i = relationship_mean + face_mean` (both already oriented so higher = better for the relationship).

## 3. Curve, annotations and turning point
- `position_curve(effects) -> list[float]`: cumulative sum of v_i (turns with no readings add 0).
- `annotate_turn(v) -> str`: v ≤ -1.5 "??", ≤ -0.75 "?", ≤ -0.25 "?!", ≥ 1.5 "!!", ≥ 0.75 "!", otherwise "=".
- `turning_point(effects, min_turn=2) -> int | None`: index (≥ min_turn) of the turn with the largest |v_i|, tie → earliest; None if all v_i are 0.
- `disagreement_spike(effects, z=2.0, min_turn=3) -> int | None`: first turn (index ≥ min_turn) whose disagreement exceeds mean + z·std of disagreement of all EARLIER turns (std floor 0.05). None if never.

## 4. Forecast check — the conversation is its own answer key
`ACTION_TO_MOVES = {"comply": {"accept", "concede"}, "proceed": {"inform", "propose", "accept", "smalltalk"}, "refuse": {"reject"}, "negotiate": {"counter", "propose", "appeal_need", "appeal_fairness"}, "ask_clarification": {"ask", "probe"}, "escalate": {"pressure", "challenge"}, "retaliate": {"challenge", "pressure"}, "withdraw": {"deflect", "other"}, "ignore": {"deflect", "other"}, "repair": {"repair", "empathize", "praise"}}`
- `forecast_hits(effects, move_tags) -> list[dict]`: for each turn i whose NEXT turn (i+1) is by one of turn i's listeners and has move tags: predicted action = argmax of action_dist (ties → alphabetical). `hit` = actual moves of turn i+1 intersect ACTION_TO_MOVES[predicted]. Also `p_actual` = total share in action_dist of actions whose move set intersects the actual moves (a probability the forecast gave to what happened). Returns `{"index", "predicted", "actual_moves", "hit", "p_actual"}`.
- `forecast_score(hits, b=1000, seed=0) -> dict`: hit rate with Wilson CI, mean p_actual with bootstrap CI over turns, and the baseline hit rate of always predicting the most common predicted action ("always_<action>").

## 5. Output and CLI
`uptake-board --turns turns.jsonl [--moves moves.jsonl] --out board.json [--n 3 --k 1 --workers 8 --model M]` writes:
`{"turns": [effects + "value", "annotation", "position"], "turning_point": idx, "disagreement_spike": idx, "forecast": forecast_score(...) or null}` and prints a compact text board: one line per turn `#i speaker annotation value position ▁▂▃▅▇ text[:60]`.
