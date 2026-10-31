# SPEC task49b — game answer key must match the question (owner scored 50% on casino-878)

Problem: the buttons ask "잘/보통/나쁘게 받아들여질까" but the key used z_resid (relative to the move-kind mean), and
|z_resid| < 1 ("보통") covers most turns. Fix:
1. Key = absolute reception: z_abs = (value − mu) / sd from the record's scale table (pairwise: BT rating; 0–100: panel
   value). Thresholds: ≥ +0.75 잘, ≤ −0.75 나쁘게, else 보통. Keep z_resid only for the "해설" line ("거절치고는 부드러웠다"
   when z_resid ≥ 1 while z_abs is 나쁘게, and the mirror case).
2. Ask only where the reader is confident: skip turns with |z_abs| < 0.5 (show them as context with "해설 없음", no
   buttons, auto-advance). Also skip a turn whose `disagreement` (if present) is above the record's 75th percentile.
   Counter shows "문제 k / n" over asked turns only.
3. On a miss show one line: the reader's reason (commentary line if any; else "{move}이(가) 평균보다 {잘/나쁘게} 읽힘").
4. Record the player's picks: in-page array, `localStorage['uptake-game:<id>']` (try/catch), and an "내 답 내보내기" button on
   the end card that downloads JSON {dialogue, picks:[{turn, pick, key, z_abs}], hits, asked}.
5. End card: hit rate over asked turns, plus "자가 확신한 수만 출제했습니다" footnote.
tests (`tests/test_task49.py` update + new cases): key thresholds on z_abs; skip rule (|z|<0.5 → no buttons); the per-turn
data table includes z_abs and z_resid; export JSON shape present in page JS; counter counts asked turns only.
