# SPEC task48b — relationship board polish (after seeing casino-878)

1. Tide fill: the legend says 빨강 A · 파랑 B but the chart is a bare line. Fill above the curve light red (A's side) and
   below light blue (B's side), same convention as viewer2 (boundary down = A pushing). Keep it flat (no marks, no bands).
2. Decisive moments must be diverse, not three of the same kind: pick up to 3 turns by |ΔM| subject to (a) at least one
   reversal (opposite sign to the first big swing) if any |ΔM| ≥ 1.0 reversal exists, (b) not three turns by the same
   speaker when the other speaker has any |ΔM| ≥ 1.0 turn. Flags on the tide are these same turns.
3. Card header uses the turn number as everywhere else: "#3 B — "…"" (not the moment ordinal). Body: plain-language line
   only ("거절하며 사정을 호소했지만 관계 점수 50 → 8"), no theory sentence, no raw move labels (Korean only; raw label in a
   title attribute). The theory/reference line appears once, in the 한 줄 교훈 block only.
4. Verdict: take the caption from the commentary/viewer2 helper (task47b fixes its numbering/direction); do not rebuild.
5. Leave an import point for task50: `scene_card_html(turn) -> str` called under each decisive card (returns "" for now).

tests (`tests/test_task48.py` additions): fills present (two path elements with the A/B colors); decisive picks on a toy with
A-swing then B-reversal include the reversal; header text uses `#<index>`; no raw move label in card body text;
`pytest -q` green.
