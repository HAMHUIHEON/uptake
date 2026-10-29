# SPEC task40 — viewer v2 polish: hero moment, confidence band, square fixes, audio path, genre refs

Follows task38 (`forecaster/viewer2.py`). Owner review of the first build (2026-10-29 20:00) and the
design note. Keep the single-file HTML; no external assets.

## 1. Hero moment (결정적 순간)
- Pick the turn with the largest |ΔM| (momentum change, task38 definition). Show a banner above
  the tide: "판이 갈린 수 · #13 A — <first 80 chars of the turn>" + its commentary line if any, and
  a button "그 장면 보기" that scrubs the replay to that turn and pulses the dot. Secondary: the two
  next-largest swings listed small under it.

## 2. Confidence band
- When readings carry `rel_spread` (dist5 records) or, failing that, effect `disagreement`, draw a
  translucent band around the boundary whose half-width ∝ the speaker's spread at that turn
  (map spread 10→2px, 25→14px; clamp). Hover text adds "확신도: 심사위원 분포 퍼짐 18".
  Without either field, no band (unchanged look).

## 3. Square fixes
- Fixed font sizes in px via CSS (axis ticks 12px, corner labels 13px, settlement 14px) — no viewBox
  scaling of text. Smooth trail (Catmull-Rom → cubic) instead of right-angle steps; keep per-turn dots
  but small (r=2.5) and the last point large. Axis titles "A 잔고 →" / "B 잔고 ↑".

## 4. Audio path
- `--audio FILE`: by default copy the file next to the HTML (`<out stem>.audio.<ext>`) and reference it
  relatively; `--no-copy-audio` writes `os.path.relpath(audio, out_dir)` instead. Never emit the raw
  CLI string.

## 5. Speaker colours and legend
- One palette for chips, dots, bands, turn list (A red, B blue, C green, D amber, E purple, …).
  3+ speakers: legend with display names from `--names`. For > 40 turns hide unmarked dots until
  hover of the chart (marked dots always shown).

## 6. Genre-aware references (`forecaster/commentary.py`, `forecaster/references.json`)
- Each reference entry gains `genres: ["negotiation"] | ["meeting"] | ["any"]`; existing entries
  default to `["negotiation"]`; add meeting-genre entries already in the file (Grice, Sacks/Schegloff,
  Goffman face-work → `["any"]`) and no new citations beyond what is in the file.
  `choose_reference` filters by the item's genre, falling back to `any`.

## 7. Motion
- Replay: boundary eases (300 ms) per turn; mark glyph pops in; the hero banner highlights when its
  turn is reached. Keep `prefers-reduced-motion` respected.

## tests `tests/test_task40.py` (Muse writes)
- hero pick on a toy record; confidence band present only with spread/disagreement; square path uses
  `C` segments and CSS font sizes present; audio copy/relpath behaviour (tmp dirs, no real audio);
  legend for 3 speakers; `choose_reference` filters by genre. Existing task38/21/33 tests unchanged.
- `pytest -q` must stay green.
