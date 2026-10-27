# SPEC — scoring v2: effect gap by stake, not label match

Why: v0 counts a reading as a miss when ANY label differs. On the first live run ("by Friday? No pressure.") every reading was a "miss", yet the new hire still delivers on Friday — the sender's goal is met; what differs is felt pressure. Label boundaries (request vs command, comply vs proceed) inflate misses. v2 measures the **gap between the intended effect and the likely effect**, separately for each stake, and reports how often a well-communicated message is wrongly flagged.

Theory: Austin — what was said (locution), what the speaker tried to do (illocution), what actually happened to the listener (perlocution). A communication gap is intended perlocution ≠ likely perlocution. Each party pursues stakes: outcome, relationship, face, understanding.

## 1. Types (`forecaster/v2.py`)
```python
STAKES = ["outcome", "relationship", "face", "understanding"]

@dataclass
class Goal:                        # what the sender wants this message to achieve
    acceptable_actions: list[str]  # subset of NEXT_ACTIONS that count as success
    relationship_floor: int = 0    # minimum acceptable relationship shift (-2..2)
    face_ceiling: int = 1          # maximum acceptable face threat felt by receiver (0..2)
    core_meaning: str = ""         # one sentence the receiver should understand

@dataclass
class ReadingV2:
    receiver_id: str
    sample: int
    next_action: str               # in NEXT_ACTIONS
    relationship_shift: int        # -2..2 how the receiver now feels toward the sender
    face_threat: int               # 0..2 how much the message threatens the receiver's standing/dignity
    understood: str                # one sentence: what the receiver thinks the message means
    speech_act: str = ""           # kept only as a diagnostic label (not scored)
    trigger: str = ""
```
Validation: `Goal.acceptable_actions` non-empty and all in NEXT_ACTIONS; ranges enforced → `ValueError`.

## 2. Parsing (`parse_reading_v2(raw, receiver_id, sample, message) -> ReadingV2 | None`)
Same tolerance as v0 (fenced/prose JSON, synonym map for next_action, clamp ints, round floats, trigger must be a substring else ""). Clamp relationship_shift to [-2,2], face_threat to [0,2]. Return None if next_action invalid or `understood` missing/empty.

## 3. Gaps (`gaps(reading, goal, same_meaning: bool | None) -> dict[str, bool]`)
- outcome: `next_action not in goal.acceptable_actions`
- relationship: `relationship_shift < goal.relationship_floor`
- face: `face_threat > goal.face_ceiling`
- understanding: `same_meaning is False` (None → False, i.e. not judged = no gap)
`any_gap(g) = any(g.values())`

## 4. Meaning judge (`judge_same_meaning(client, core_meaning, understood, model) -> bool | None`)
Asks a model (default `muse-spark-1.3`) whether `understood` preserves the core meaning of `core_meaning`. Prompt asks for JSON `{"same": true|false, "why": "..."}`. Return the bool, or None if unparsable. Skip (return None without calling) when `core_meaning` is empty.

## 5. Report (`score_v2(readings, goal, same: dict[tuple[str,int], bool|None], b=2000, seed=0) -> dict`)
For each stake and for `any`: point and 95% cluster-bootstrap CI (cluster = receiver), using values 1.0 gap / 0.0 no gap. Keys: `"outcome"`, `"relationship"`, `"face"`, `"understanding"`, `"any"` → `(point, low, high)`. Also `"n_readings"`.

## 6. False-alarm check (`false_alarm_rate(results: list[dict]) -> tuple[float, float, float]`)
Input: one `score_v2` result per **control case** (a message known to communicate well). A control case is a false alarm if its `any` point estimate > 0.5. Returns (rate, wilson_low, wilson_high).

## 7. Reader prompt v2 (`build_messages_v2(message, receiver)`)
Same receiver framing as v0 (sender intent never shown). Asks for JSON keys: next_action, relationship_shift, face_threat, understood, speech_act, trigger, with allowed values listed.
