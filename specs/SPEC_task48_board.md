# SPEC task48 — layer 2 "관계 보드": a one-page story for first-time viewers

Layer 1 is the tide board (viewer2, engine dashboard). Layer 2 is what a person who has never seen the project reads in
30 seconds. New module `forecaster/board2.py`, entry point `uptake-board2 --item RECORD --commentary JSON --out HTML
[--casino PATH] [--names JSON] [--audio FILE]`. Single self-contained HTML (inline CSS/JS, no CDN), light theme, mobile
width works (max-width 720px column), Korean UI with English subtitles in small type. Stdlib only. Everything is drawn
from the record + commentary JSON produced by task43/45/47 (standings on 0–100, decisive turns, caption data).

Sections, top to bottom:
1. Verdict (한 문장): the task47 caption, set large. Under it one line "관계 점수는 상대 눈에 비친 점수입니다 (50 = 보통)".
2. Two score cards side by side: "A · 50 → 21", "B · 50 → 14", each with an arrow and the words 잃음/지킴/얻음 by sign,
   and under each the actual outcome if `--casino` is given: 점수 17, 상대를 약간 좋아함 (map CaSiNo likeness/satisfaction
   labels to Korean: Extremely like → 매우 좋아함 … Undecided → 미정; points_scored → 점수).
3. Simplified tide: same momentum series as viewer2 but no bands, no mark glyphs, no hover; only the boundary curve, the
   midline, and up to 3 numbered flags at the decisive turns (the caption's turns). Height 220px. A tiny legend: 빨강 A ·
   파랑 B · 선이 내려오면 A 우세.
4. Decisive moments (up to 3 cards): "#1 B — "No. That will not be possible…"" + one plain-language line from the
   commentary (reuse task40 wording but replace "잔고 50 → 8" with "관계 점수 50 → 8") + the move kind in Korean.
5. One-line lesson: from the commentary's reference line (e.g. "입장이 아니라 이해(관심사)에 집중하고…") as a quote block.
6. Footer: "모델이 발언을 읽은 것이지 사람을 판정한 것이 아닙니다" + link text to the dashboard file name.

tests (`tests/test_task48.py`): HTML contains verdict text, both score cards with correct start/end numbers, outcome
labels mapped to Korean, ≤3 flags, no `.band` element, mobile meta viewport; works without --casino (cards show "결과 없음");
`pytest -q` stays green.
