# SPEC task57 — put the original frame back on the human-facing pages: 의도 vs 해석 (misreading), three dials

Owner (10/30 21:40): the board/game drifted to a single "relationship score up/down". The original design is a
*misreading* reading: the speaker's move (의도) vs how the listener took it (해석), on three dials — next action (세상),
relationship (마음), face (체면). The pairwise favourability rating stays as the validated relationship dial only.

## Inputs
`uptake-board2 --item PAIRWISE_RECORD --panel PANEL_RECORD ...` and `uptake-game ... --panel PANEL_RECORD`: the panel record
is the 0–100/dist5 record of the same dialogue (effects carry `action_dist`, `face`, `relationship`, `readings`, and
the record has `moves` and `hits`). If `--panel` is missing, pages fall back to the current single-dial view with a badge
"관계 축만".

## Per-turn reading (module `forecaster/misread.py`, reused by board2 and game)
- 의도 = the speaker's move kinds (MOVE_KO).
- 해석 = majority `next_action` across readings (labels.NEXT_ACTIONS → Korean: comply 따른다, refuse 거절한다, negotiate 흥정한다,
  ask_clarification 되묻는다, escalate 맞선다, withdraw 물러난다, … fill from labels.py), with its share.
- 어긋남 = reuse `board.py`'s expected-action mapping (move → the actions it invites); 어긋남 if the majority next_action is not
  invited by any of the turn's moves. Severity = share of readings outside the invited set.
- Three dials for the listener: 관계 (pairwise z→standing if available, else mean relationship_shift), 체면 (mean face_threat,
  shown as 위협 없음/약간/큼), 다음 행동 (the majority label + share).

## Board2 changes
- Verdict line stays. Add a "오해 지점" strip above the tide: the turns with 어긋남 (max 5 flags): "#1 B 거절 → A는 '압박'으로 읽음".
- Each decisive card: "의도: 거절·사정 호소 → 상대가 읽은 것: 맞선다(67%) · 체면 위협 큼 · 관계 50 → 8". Blunder notice (task55)
  unchanged underneath.
- Score cards: keep the relationship cards; add one small line "어긋난 턴 3 / 16".

## Game changes (question = what the original design asked)
- Question: "상대는 이 말을 어떻게 받을까?" Buttons = the Korean next-action labels (5–6). Key = majority next_action; a pick
  counts if it equals the key or lies in the invited set when the key is in it too (be lenient only that way).
- Reveal: "상대 판정: 거절한다(67%) · 체면 위협 약간 · 관계 50 → 8" + the 49b reason line. Ask only turns whose majority share ≥ 0.5.
- End card: hit rate + "오해 지점 n곳 중 k곳을 미리 봤다".

tests (`tests/test_task57.py`): misread reading on a toy panel record (majority, invited set, 어긋남 severity); board2 with
--panel shows the strip and three-dial lines and without it shows the badge; game with --panel asks next-action labels and
keys match; `pytest -q` green.
