# SPEC task56 — `uptake-pairwise run`: `--ids-file` and `--stances N` (Amendment 14)
- `--ids-file FILE`: only dialogues whose id is listed (one per line) are read from `--src`.
- `--stances 3|5`: 3 = current 중립/우호/의심; 5 adds 피로한 (tired, wants it over) and 급한 (in a hurry, impatient) as fixed
  prompt constants next to the existing ones. Votes/records carry the stance name; fit/reliability-split work unchanged.
tests: ids filter on a toy src; 5-stance run issues 5 votes per presentation (mocked API); existing tests unchanged.
