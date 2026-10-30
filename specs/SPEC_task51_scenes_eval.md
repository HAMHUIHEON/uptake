# SPEC task51 — `uptake-scenes-eval`: famous-misreading validation (Amendment 12)

New module `forecaster/scenes_eval.py`, entry `uptake-scenes-eval`, subcommands:
- `prepare --src JSONL --out DIR`: keep rows with `입력.상황`; write `scenes.json` with id, 계열, 작품, 상황, 상황_가림, 발화,
  발화_가림, 말한사람, 받은사람, 맥락.관계/권력거리/채널, 해석, 근거, 오해발생. Print counts.
- `read --scenes FILE --out DIR --model M --effort low --budget B --masked/--real`: for each scene, three stances (reuse the
  stance definitions and API wrapper from `pairwise.py`), structured output {reception: 잘|보통|나쁘게, reading: str}. Save
  `pred_<real|masked>.jsonl` and calls/cost.
- `judge --preds FILE --scenes FILE --out FILE --model M --budget B`: per prediction, judge {match, partial, miss} against
  해석 (+근거). The judge prompt is a module constant; the judge never sees the reader's prompt or the scene outcome text
  beyond 해석/근거. Save `judge_<version>.jsonl`.
- `report --out FILE.md`: M1, M2, M3 with Wilson 95% intervals, per-계열 breakdown, reception vs 오해발생 cross-table, the
  verdict per Amendment 12, and a table of the 30 scenes (작품 · 발화 ≤ 80 chars · predicted reading ≤ 100 · judgment real/masked).
Rules: stdlib only; tests mock the API (`tests/test_task51.py`: prepare filters; read builds 3 stances × 2 versions; judge
parsing; report metrics on a fixture with a known answer and Wilson interval check). `pytest -q` stays green.
