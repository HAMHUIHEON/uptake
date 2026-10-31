# SPEC task58 — alert timeline on board2 (예보·경보): when could this have been seen coming?

Owner (10/30 22:00): the app is an analysis of how a misreading in a high-stakes conversation could have been predicted and
prevented, not a game. Add to `uptake-board2` (module `forecaster/alerts.py`):

1. Per-turn alert level from (a) misread severity when `--panel` is given (task57: share of readings outside the invited set) or
   else the pairwise z-drop (z ≤ −1 → 0.5, z ≤ −2 → 1.0), times (b) a stakes weight by genre: `STAKES_WEIGHT = {"aviation_atc":
   1.0, "medical": 1.0, "meeting": 0.6, "negotiation": 0.4, default 0.5}` (constant, documented). Level: < 0.25 관심, < 0.5 주의,
   < 0.75 경계, else 심각.
2. **First alert turn** = first turn at 경계 or above; **lead time** = turns (and clock time when effects carry `time`) from
   the first alert to the outcome turn (`--outcome-turn N` or the last turn). Render a strip above the tide: colored ticks per
   turn, the first-alert flag with "#19 경계 — 결과까지 15턴 / 8분 31초".
3. Section "의도 vs 전달 vs 해석" for turns that carry `cockpit_before` notes (or a `--intent` JSON {turn: text}): three
   columns: 뜻한 것(the note), 말한 것(the turn text), 읽힌 것(the task57 majority next-action if panel, else the pairwise
   verdict word 잘/보통/나쁘게 with the standing change).
4. Header: "사건과 결과" block from `--case JSON` {title, outcome, source_url, note}; the footer keeps the model disclaimer and
   adds the source link.
5. `--no-game` is the default for board2 links; the game link is removed from the board footer.
tests (`tests/test_task58.py`): alert levels and thresholds; first-alert and lead time with and without `time`; intent table
rows from cockpit_before; case header rendering; `pytest -q` green.
