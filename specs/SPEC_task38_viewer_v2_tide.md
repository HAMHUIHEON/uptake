# SPEC task38 — viewer v2: the tide (momentum) board, standings, replay, guess mode

Design source: owner's sketch (2026-10-29) — "red and blue surge back and forth like a tide; sometimes
red floods into the other side's home base and then retreats; like a voice waveform but with area".
Static mock that was approved for direction: `runs/demo_1029_fine100` + scratch `wave_mock2.py`
(the mock is not repo code; re-implement cleanly here). Keep viewer v1 (`uptake-viewer`) untouched;
add `uptake-viewer2` (`forecaster/viewer2.py`, stdlib, one self-contained HTML with inline JS/CSS,
no external assets). Input: a record (any scale, `fine100` expected) + its commentary JSON (task33
format, with `norm.table`/`sd_resid`) + optional `--casino` outcomes, optional `--audio-record`
(task36) and `--audio FILE` for synced playback.

## 1. Tide chart (hero)
- One moving boundary between A's region (top, red) and B's region (bottom, blue). Momentum
  `M_t = 0.55·M_{t−1} + s_t` where `s_t = z_resid_t` if the speaker is A else `−z_resid_t`
  (fallback to `z` when `z_resid` is absent); boundary y = mid + clamp(M_t, −3, 3)·amp, so A's
  well-received move pushes the boundary DOWN into B's base and vice versa. Smooth (cubic) between
  turns. Dot per turn (bigger when it carries a mark), mark glyph above the dot (`!!`, `!`, `!?`,
  `?`, `??`), hover shows the turn text, the raw panel reading (0–100) and the one-line comment.
- Side labels show each side's **standing** (잔고) = 50 + 25·rel_norm of that side's latest turn,
  updated as the replay advances: "A · 잔고 39".
- Multi-party (3+ speakers): stacked streams, one band per speaker, band width ∝ max(0.2, 1 + 0.5·EMA
  of that speaker's received z_resid); same colours as the turn list. Two speakers → the tide above.

## 2. Replay
- Play/pause, speed ×1/×2/×4, scrubber. The boundary grows turn by turn (eased); the turn list
  scrolls in sync; comments appear when their turn is reached. If `--audio` is given and the record
  has turn `start`/`end`, play the audio and advance the chart by the audio clock (seek on scrub).

## 3. Guess mode ("다음 수 맞히기")
- Toggle. Before each turn is revealed, show the turn text and three buttons `!` `=` `?`
  (better / about as expected / worse than a move of this kind usually is → z_resid ≥ 1, in between,
  ≤ −1). Reveal the panel's reading, score +1 on a match, keep a running tally "나 7 / 해설자 10"
  (the commentator is always right by definition; the point is the player's hit rate). Store nothing
  server-side; everything in page JS.

## 4. Square summary (end state)
- x = A's standing, y = B's standing, axes zoomed to 20–80 with a note; dotted trail of all turns;
  last point large. Corner labels: 둘 다 잘됨 / 둘 다 잃음 / 한쪽만. Shown under the tide, with the
  plain-language settlement "A 잔고 50 → 39 · B 잔고 50 → 51" and, when `--casino` outcomes exist,
  the actual post-conversation likeness of each side next to it (the ground truth the chart is
  compared against).

## 5. Commentary wording fixes (in `commentary.py`, small)
- Mark names: `!!` 묘수 / `!` 호수(좋은 수) / `!?` 특이수 / `?` 실수 / `??` 악수. English: brilliant /
  good / interesting / mistake / blunder.
- Never call a side "흑/백"; use the speaker label ("B") or the display name when provided.
- Numbers quoted in lines use standings on the 0–100 scale ("잔고 53 → 45"), not cumulative values.
  Template lines and the LLM prompt both.

## tests `tests/test_task38.py` (Muse writes)
- momentum series on a toy record (direction: A's good move lowers the boundary); multi-party bands
  width; standings per side; guess-mode scoring function (pure JS logic mirrored in Python for the
  data prep, or test the emitted data table); HTML contains play controls, guess buttons, the square
  with 20–80 axes, the five mark labels; commentary wording tests (no "흑", labels per mark).
- `pytest -q` must stay green.
