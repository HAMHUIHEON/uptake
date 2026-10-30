# SPEC task47b — tide caption: numbering and direction fixes

Observed on casino-878 (pairwise): caption read "1수에서 A가 제안하며 B 쪽으로 기울었고, 2수에서 B가 많이 되찾았다" while
the decisive banner says "#1 B — 악수(reject)" and the all-turns list numbers turns #0, #1 … Two defects:
1. Numbering: the caption is 1-based while the banner and turn list are 0-based (#n). Use the same `#n` (index) everywhere
   — caption says "#0에서 A가 …", "#1에서 B가 …".
2. Direction wording: at #1 B rejected and ΔM = +6.48 (toward A). The caption attributes the move to B and says B 되찾았다,
   but the side that gained is A. Rule: the clause names the *speaker* and the *side that gained*; "되찾았다" only when the
   gaining side had been behind. Templates:
   - first swing: "#{n}에서 {speaker}가 {move}했고 판은 {gainer} 쪽으로 기울었다" (if speaker == gainer: "…하며 {gainer} 쪽으로
     가져갔다"; if speaker != gainer: "…했지만 되려 {gainer} 쪽으로 넘어갔다").
   - later reversal: "#{n}에서 {speaker}가 {move}해 {gainer}가 {much} 되찾았다" (speaker == gainer) / "…했지만 {gainer}가 {much}
     되찾았다" (speaker != gainer).
   - ending unchanged.
3. Keep the virtual start point at the midline visible: the x-axis must start at −0.5 so the curve is seen leaving 50:50.

tests: update `tests/test_task47.py` — caption uses `#n` numbering; a toy where B speaks and A gains yields "되려 A 쪽으로";
a toy where A speaks and A gains yields "A 쪽으로 가져갔다"; x-axis min −0.5. `pytest -q` green.
