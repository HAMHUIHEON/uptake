# SPEC — task12: forecast skill vs climatology, addressed listeners

First board run (football panel, 23 turns): next-turn hit rate 0.91 [0.72–0.97], but "always proceed" also scored 0.91 — no skill shown. Two causes: an agreeable conversation is easy to predict, and `proceed` mapped to many moves. Fix the measurement the way weather forecasts are judged: score against **climatology** with a proper scoring rule, and use one actual class per turn.

## 1. One actual class per next turn (`forecaster/board.py`)
`MOVE_PRIORITY` (first match wins) maps moves → action class:
reject→refuse, accept→comply, concede→comply, counter→negotiate, propose→negotiate, pressure→escalate, challenge→retaliate, repair→repair, ask→ask_clarification, probe→ask_clarification, appeal_need→negotiate, appeal_fairness→negotiate, empathize→repair, praise→repair, deflect→withdraw; anything else (inform, smalltalk, joke, cooperate, reveal_preference, other) → proceed.
`actual_class(moves: list[str]) -> str` walks MOVE_PRIORITY in the order listed above and returns the first class whose move is present; default "proceed".

## 2. Hits
`forecast_hits` now adds `actual_class` and `n_readings`, and `hit = (predicted == actual_class)`. `p_actual` = share of `actual_class` in action_dist (0 if absent). Predicted = argmax of action_dist, ties alphabetical (unchanged).

## 3. Skill (`forecast_skill(hits, b=1000, seed=0, alpha=0.5) -> dict`)
- Model probability with add-alpha smoothing over the 10 NEXT_ACTIONS: `p_model = (share*n + alpha) / (n + 10*alpha)` where n = n_readings of that turn.
- Climatology, leave-one-out: counts of `actual_class` over all OTHER hits, `p_clim = (count + alpha) / (N-1 + 10*alpha)`.
- Per turn `gain = ln p_model − ln p_clim`. Report `skill_mean` with percentile bootstrap CI over turns, `n`, and `majority_class` with its hit rate (always predicting the most frequent actual class).
- Interpretation in CLI output: "skill > 0 with CI above 0 = better than climatology".
`uptake-board` prints the skill block next to the hit rate.

## 4. Addressed listeners
`evaluate_turns(..., move_tags=None)`: if move tags are given and the tag for turn i has `addressed_to` equal to another speaker label present in the conversation, the listeners for turn i are just that speaker; otherwise all other speakers (unchanged). CLI passes the loaded move tags. Order of returned effects unchanged.
