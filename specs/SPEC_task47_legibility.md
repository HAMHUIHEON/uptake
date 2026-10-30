# SPEC task47 — viewer v2 legibility for first-time viewers

Owner feedback (2026-10-30, looking at casino-878 on the pairwise scale): even the owner could not read the tide at
first glance. Fix the page so a person who has never seen it can read it in 10 seconds. Keep all existing behaviour and
tests; add the following.

1. "How to read" card at the top of the tide panel (collapsible, open by default; Korean + English one-liners):
   - 빨간 면 = A의 영역, 파란 면 = B의 영역. 경계선이 아래로 내려오면 A가 밀고 있는 것, 올라오면 B가 되찾는 것.
   - 점 하나 = 발언 하나(빨간 점 A, 파란 점 B). 가로 점선 = 50:50.
   - 표시: !! 묘수 · ! 호수 · ? 실수 · ?? 악수 (그 발언이 보통 그런 종류의 발언이 받아들여지는 정도보다 얼마나 더/덜 받아들여졌나)
2. Axis label under the numbers: "발언 순서 (turn)". Legend adds "● 빨간 점 = A 발언 · ● 파란 점 = B 발언".
3. Start the tide at the midline: prepend a virtual point at x = −0.5 with M = 0 so the first real turn moves away
   from 50:50 instead of the chart opening already tilted. The replay also starts there.
4. Disagreement band: draw it only when the record's `scale` is a 0–100 panel scale AND effects carry `disagreement`
   from that same reading; for `scale == "pairwise"` hide it (and do not read stale `disagreement` fields).
5. Auto caption above the chart (plain language, from the commentary JSON): the decisive turn and the biggest later
   reversal, e.g. "1수에서 B가 거절하며 A 쪽으로 기울었고, 14수에서 B가 많이 되찾았다. 마지막은 A가 조금 앞선 채 끝." Build it
   from: first |ΔM| ≥ 1.5 turn (who, move kind in plain words), the largest opposite-sign ΔM after it, and the sign of the
   final M. Move-kind glossary in Korean (reject → 거절, appeal_need → 사정 호소, counter → 역제안, concession → 양보, …;
   fall back to the raw label).
6. Square summary: replace the corner words with one sentence under the square:
   "A 잔고 50 → 21 · B 잔고 50 → 14 — 둘 다 평균(50) 아래, A가 덜 잃음" (or 둘 다 위 / 한쪽만 위). Keep the actual-likeness line.
7. Rename "잔고" in the UI to "관계 점수(잔고)" on first mention; keep "잔고" afterwards.

tests (`tests/test_task47.py`): how-to card and axis label present; virtual start point at x = −0.5 with M = 0 in the data
table; band absent for a pairwise fixture and present for a 0–100 fixture with disagreement; caption contains the decisive
turn number and both speakers; square sentence matches the standings; `pytest -q` stays green.
