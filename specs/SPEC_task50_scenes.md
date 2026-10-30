# SPEC task50 — "닮은 장면": literature/history scene seasoning for the board and the game

The owner collected 181 scenes of misreading from literature, East-Asian classics, history and the Joseon annals
(`코퍼스_v1.jsonl`, fields: id, 계열, 분야, 작품, 작가, 위치, 원문출처, 장면, 발화, 말한사람, 받은사람, 의도, 해석, 주유형, 맥락,
오해발생, 대조종류, 데모적합, 검증). Use them as the human-readable "seasoning" on layer 2/3 pages instead of theory
citations (book titles mean nothing to most viewers; a told scene does).

## `uptake-scenes build --src JSONL --out forecaster/scenes.json`
Keep only 검증 == "확인" and 데모적합 in {"상","중"}. Copy: id, 계열, 작품, 작가, 위치, 원문출처 (list), 장면, 발화, 말한사람,
받은사람, 의도, 해석, 주유형, 맥락.관계, 맥락.권력거리, 오해발생, 대조종류. Sort by id. Print counts per 계열 and 주유형.

## Pairing rule (module constant `MOVE_TO_TYPES`, auditable) — move kind × how it was received → scene types
| turn | received badly (z ≤ −1) | received well (z ≥ +1) |
|---|---|---|
| reject / pressure / challenge | C2 체면 위협, C1 권력거리 | 대조(통함) with 의도 containing 거절/경고 |
| propose / counter / ask (short, < 8 words) | B1 양의 격률, B4 간접화행 | 대조(바로잡힘) |
| deflect / joke / smalltalk | B3 반어·풍자, E3 침묵·무응답 | 대조(통함) |
| appeal_need / appeal_fairness | E1 확증 편향, D2 가치 충돌 | 대조(통함) |
| probe / reveal_preference | F1 전략적 모호함, F2 설계된 오독 | 대조(한쪽정확) |
| empathize / repair / concession / accept | E2 감정 투사 (the rare bad case) | 대조(바로잡힘), D1 고/저맥락 |
| other | B4 | 대조(통함) |
Pick deterministically: candidates filtered by type (대조 rows are the 오해발생 == false rows with that 대조종류), then
`hash(dialogue_id + turn_index) % len(candidates)`. Prefer 데모적합 상. One scene per decisive turn, at most 3 per page,
never the same scene twice on a page.

## Rendering (board2 decisive cards; game reveal card) — readable without knowing the work
```
닮은 장면
코딜리어가 "Nothing, my lord."라고만 답하자, 리어 왕은 그것을 사랑이 없다는 모욕으로 읽었다.   ← 장면 (+해석 if 장면 lacks the outcome)
같은 점: 짧은 거절이 체면 위협으로 읽혔다.                                                      ← from the type name + move kind
『리어 왕』 1막 1장 · 셰익스피어                                                                 ← small type, source link (원문출처[0])
```
Korean only in the card except the quoted 발화 (keep original language, ≤ 120 chars, ellipsis). "같은 점" templates per type
in a module constant (23 + 3 대조 entries), e.g. C2 → "{move}이(가) 체면을 건드려 내용보다 감정으로 읽혔다",
통함 → "{move}이(가) 뜻 그대로 받아들여졌다".

## Wiring
- `forecaster/scenes.py`: build, load_scenes(), pick_scene(move, z, dialogue_id, turn_index, used=set()) → dict|None,
  render_scene_card(scene, move) → html string.
- board2 (task48) and game (task49) call pick_scene for decisive/revealed turns; viewer2 unchanged (theory refs stay there).
- `forecaster/scenes.json` is committed with the package (public-domain sources; the owner's own notes). The 입력/정답구조
  fields are NOT copied.

tests (`tests/test_task50.py`): build filters and copies fields on a 5-row fixture; pick_scene is deterministic and never
repeats within `used`; a reject with z = −2 yields a C2/C1 scene; a concession with z = +1.5 yields a 대조 scene; the card
contains 장면 text, "같은 점", title in 『』 and no raw type code. `pytest -q` stays green.
