# SPEC task49c — game question = "does the speaker's score go up or down?" (after 49b on casino-878)

49b's absolute key degenerates on a cold dialogue: casino-878 asks 11 turns and 9 keys are 나쁘게 (both sides sit around
20/100 all along). The player already sees the score line "A의 관계 점수 50 → 17", so ask exactly that:

1. Question text: "이 말로 {speaker}의 관계 점수가 오를까, 그대로일까, 내릴까?" Buttons: 오른다 / 그대로 / 내린다.
2. Key = direction of the speaker's standing vs. that speaker's previous standing (first turn: vs 50):
   Δ = standing_after − standing_before (0–100 standings from the logistic map, same numbers as the score line).
   Δ ≥ +6 → 오른다, Δ ≤ −6 → 내린다, else 그대로. (6 points ≈ 0.25 sd on the logistic scale around 50; keep as a constant.)
3. Ask every turn except: |Δ| in (3, 6) ("애매", skip with 해설 없음) or disagreement above the 75th percentile when present.
   The first turn of each speaker is asked (vs 50).
4. Reveal line: "{speaker}의 관계 점수 {before} → {after} ({오름/그대로/내림})" + the 49b reason line. Keep z_abs/z_resid in the
   data table for the end-card tide and export.
5. End card unchanged except the footnote: "질문은 '직전보다 오르나 내리나'입니다".
tests: key thresholds on Δ; first-turn baseline 50; skip band; question text per turn names the speaker; on a toy cold
dialogue (all z < 0) the keys are not all the same. `pytest -q` green.
