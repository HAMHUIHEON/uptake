# SPEC task26 — viewer: show both sides' outcome stake (L2) next to the relational curve

`uptake-viewer` gains `--stakes FILE` (a `stakes.json` written by `uptake-stakes`: {item_id: {"offers": {turn: {"A": {...}, "B": {...}}},
"track": [{"A": int|None, "B": int|None, "offer_turn": int|None}, ...]}}).
- `build_view(..., stakes=None)`: when the item id is in stakes, add `"stakes": {"track": [...], "offers": {...}}` to the view; else null.
- Second small chart under the 형세 chart, titled "판돈: 테이블 위 점수 / Stake: points on the table": A and B lines of `track` points
  (gaps where None), y-axis 0–36 with tick labels 0, 12, 24, 36, same x positions as the main chart, same colors as A/B.
  A short caption: "points each side would get if the latest offer were accepted (CaSiNo point values)".
- Add y-axis tick labels (min, 0, max, rounded to 1 decimal) to the main 형세 chart.
- Score panel: when stakes present, show "on-table points at the end: A x / B y".
- Each turn card where an offer was made shows a chip "offer A x · B y".
Keep all task22 guarantees (no external resources, escaping, textContent only). Tests in `tests/test_task26.py` (Muse writes): stakes
present/absent in view, second chart only when stakes present, offer chip text, no external URL.
