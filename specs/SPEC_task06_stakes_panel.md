# SPEC — task06: hierarchical stakes, situational panel mix

## 1. Panel stance mix (`forecaster/panel.py`)
- `STANCE_ORDER = ["neutral", "trusting", "suspicious"]`, `DEFAULT_MIX = {"neutral": 0.5, "trusting": 0.3, "suspicious": 0.2}`.
- `allocate(n, mix) -> dict[str, int]`: largest-remainder rounding of n*weight per stance (weights normalized to sum 1). Ties in remainder go to the stance earlier in STANCE_ORDER. Sum equals n. Unknown stance keys → ValueError.
- `build_panel(base, n=6, seed=0, stance_mix=None)`: if `stance_mix is None` keep the legacy cycle (unchanged behaviour). Otherwise build receivers grouped in STANCE_ORDER with counts from `allocate`, ids `f"{base.id}-{i}"` in that order.
- Why: a fixed one-third "suspicious" panel makes even praise look doubtful (observed false alarm). The mix should reflect the situation; case files may set `"stance_mix"` in `receiver`.

## 2. Hierarchical stakes (`forecaster/v2.py`)
```python
STAKE_TREE = {
  "outcome": ["money", "time", "deliverable", "commitment"],
  "relationship": ["trust", "warmth", "continuity"],
  "face": ["status", "autonomy", "competence"],
  "understanding": ["facts", "intent", "obligations"],
}
SUB_STAKES = [s for subs in STAKE_TREE.values() for s in subs]
```
- `Goal` gains `sub_stakes: list[str] = []` (each must be in SUB_STAKES) and `sub_floor: int = -1`.
- `ReadingV2` gains `sub_effects: dict[str, int] = {}` (each value clamped to -2..2; unknown keys dropped by the parser).
- `build_messages_v2(message, receiver, subs=())`: when `subs` is non-empty, also ask for `"sub_effects"`: an object with exactly these keys, each -2..2 = how the message moves that stake for the receiver.
- `parse_reading_v2` reads optional `sub_effects` (missing → {}).
- `sub_gaps(reading, goal) -> dict[str, bool]`: for each sub in goal.sub_stakes, gap if the reading has a value for it and value < goal.sub_floor; missing value → False.
- `score_v2(...)` adds key `"subs"`: {sub: (point, low, high)} for each sub in goal.sub_stakes (cluster bootstrap, same as others). Top-level keys unchanged.

## 3. v2 pipeline
- `run_forecast_v2(..., stance_mix=None)` passes it to build_panel and passes `goal.sub_stakes` to `build_messages_v2`.
- `load_case_v2` reads optional `receiver.stance_mix`, `goal.sub_stakes`, `goal.sub_floor`; returns stance_mix as a 4th value: `(message, goal, receiver, stance_mix)`.
