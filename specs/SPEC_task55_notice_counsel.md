# SPEC task55 — blunder notice + alternative wording + one line of counsel from the classics

Owner (10/30 21:20): "악수로 판단되는 부분에서 노티를 주고 다른 방식으로 말하는 걸 제안, 그 밑에 코퍼스로 근거를 짧게 — 손자병법·사기로
다독여 주듯이". Counsel, not analogy: never claim the situations are the same; offer perspective in one line.

## A. `uptake-scenes build` (from SPEC_task50 §build) + `uptake-scenes counsel --scenes forecaster/scenes.json --model M --budget B`
- build: as task50 §build, plus `문화`: 동 (분야/작품 in 동양고전·실록·한국·중국·일본) / 서 (otherwise), inferred from 계열·작품·출처 domain
  (zh.wikisource, sillok, ko.wikisource → 동; gutenberg, en.wikipedia → 서), and `상황_한줄` (장면 trimmed to one sentence ≤ 90 chars,
  outcome included — this is the told scene).
- counsel: for each scene ask the model once for `교훈` — one sentence ≤ 60 Korean chars, plain, no imperative "하라" tone
  (prefer "…은 …에 달려 있다" / "…는 …에서 갈린다"), no modern jargon; and `교훈_en` same in English ≤ 110 chars. Cache in
  scenes.json with `교훈_검토: false`. Idempotent (skip rows that have 교훈). Prompt is a module constant.

## B. Picking (module constant `TYPE_FOR_BLUNDER`, auditable)
For a blunder turn (z_resid ≤ −1.5, or standing drop ≥ 15 for the speaker): move kind → candidate 주유형 as in SPEC_task50
"received badly" column. Choose deterministically (hash of dialogue id + turn), prefer 데모적합 상, and the `--culture` option:
east | west | mix (mix = alternate east/west across the page's notices). Never repeat a scene on a page.

## C. Alternatives: `uptake-suggest --item RECORD --commentary JSON --out JSON --model M --budget B`
For each blunder turn: two rewrites that keep the proposition (same ask/offer/numbers) and change only the interpersonal
wording — one "soft" (acknowledge + ask), one "firm but fair" (state need + offer a trade). ≤ 40 words each, same language as
the dialogue. Then score each rewrite against the original with the pairwise vote (reuse `perturb.py` vote: 6 votes) →
p_better. Output {dialogue, notices:[{turn, speaker, original, standing_before, standing_after, alternatives:[{text, p_better,
votes}], scene_id}]}. Budget-capped; without `--model` produce the file with alternatives [] (so pages still build).

## D. Rendering (board2 after each decisive blunder card; play page under the original turn)
```
⚠ 이 말은 B에게 50 → 8로 읽혔습니다
이렇게 말해 볼 수도
  · "I hear you — I need a lot too. Could we find a split that works for both?"  (원문보다 잘 읽힐 확률 83%)
  · "Water is my priority as well; I can give you more firewood for it."         (67%)
사마천은 범증이 옥을 세 번 들어 보여도 항우가 보지 못한 장면을 적었다. 신호는 받는 사람의 눈높이에서 전해진다.
『사기』 항우본기 · 사마천
```
Korean page: 교훈; English page (`--lang en`): 교훈_en and the scene line in English if present else Korean. Scene line is
상황_한줄; the counsel line is 교훈; the source line small. No type codes, no "닮은 장면" claim.

tests (`tests/test_task55.py`): build adds 문화/상황_한줄 on a fixture; counsel is idempotent and respects the length cap
(mock); pick returns east/west alternately under mix and never repeats; suggest output shape with mocked votes; board2 renders
the notice block with p_better percentages and the counsel line; without a suggestions file the block shows only the notice
and counsel. `pytest -q` green.
