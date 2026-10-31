"""Contract for SPEC_task49_game + SPEC_task49b_game_key + SPEC_task49c_game_direction (synthetic data only)."""
import json
import re
from pathlib import Path

from forecaster import board2 as B2
from forecaster import commentary as CM
from forecaster import game as G
from forecaster import viewer2 as V2


def _eff(idx, speaker, value, rel=0.0, text="", dis=0.0):
    return {"index": idx, "speaker": speaker, "text": text or f"t{idx}",
            "value": value, "disagreement": dis, "n_readings": 2,
            "relationship": [rel, rel, rel], "face": [0.0, 0.0, 0.0]}


def _item(cid, effects, moves, scale="fine100"):
    return {"id": cid, "genre": "negotiation", "scale": scale,
            "moves": moves, "effects": effects, "hits": []}


def _resid_norm(table, sd_resid=1.0):
    return {"mu": 0.0, "sd": 1.0, "scale": "fine100",
            "table": dict(table), "sd_resid": sd_resid}


def _story_fixture():
    # z_abs == value (zero table, mu=0/sd=1).
    # standings (50 + 25·rel, half up): A 50→63, B 50→53, B 53→20, A 63→55.
    # Δ = +13 / +3 / −33 / −8 → 오른다 / 그대로 / 내린다 / 내린다; all asked.
    effects = [
        _eff(0, "A", 1.2, rel=0.5, text="A opens warmly"),
        _eff(1, "B", -0.3, rel=0.1, text="B hedges a little"),
        _eff(2, "B", -1.5, rel=-1.2, text="B rejects coldly"),
        _eff(3, "A", 0.4, rel=0.2, text="A tweaks the offer"),
    ]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
        {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
        {"turn_index": 2, "speaker": "B", "moves": ["reject"]},
        {"turn_index": 3, "speaker": "A", "moves": ["counter"]},
    ]
    item = _item("casino-878", effects, moves)
    norm = _resid_norm({"propose": 0.0, "reject": 0.0, "counter": 0.0})
    return item, CM.comment(item, norm, None, k=3)


def _skip_band_fixture():
    # Δ per turn: 0 / 0 / +4 / +5 / +3 / +6.
    # |Δ| in (3, 6) turns (2, 3) are context-only, so asked == 4.
    effects = [
        _eff(0, "A", 1.0, rel=0.0, text="A opens"),
        _eff(1, "B", 1.0, rel=0.0, text="B replies"),
        _eff(2, "A", 1.0, rel=0.16, text="A nudges"),
        _eff(3, "B", 1.0, rel=0.2, text="B nudges"),
        _eff(4, "A", 1.0, rel=0.28, text="A moves on"),
        _eff(5, "B", 1.0, rel=0.44, text="B moves on"),
    ]
    moves = [{"turn_index": i, "speaker": s, "moves": ["propose"]}
             for i, s in enumerate(["A", "B", "A", "B", "A", "B"])]
    item = _item("skipband", effects, moves)
    return item, CM.comment(item, _resid_norm({"propose": 0.0}), None, k=3)


def _cold_fixture():
    # all z_abs < 0, but standings swing both ways vs each speaker's previous.
    effects = [
        _eff(0, "A", -1.0, rel=-1.0, text="A cold open"),
        _eff(1, "B", -0.5, rel=-0.1, text="B cold reply"),
        _eff(2, "A", -0.8, rel=-0.2, text="A stays cold"),
        _eff(3, "B", -1.2, rel=-1.0, text="B coldest"),
    ]
    moves = [{"turn_index": i, "speaker": s, "moves": ["propose"]}
             for i, s in enumerate(["A", "B", "A", "B"])]
    item = _item("cold-1", effects, moves)
    return item, CM.comment(item, _resid_norm({"propose": 0.0}), None, k=3)


def _run_main(tmp_path, item, com, extra=()):
    item_p = tmp_path / "record.json"
    com_p = tmp_path / "commentary.json"
    out_p = tmp_path / "game.html"
    item_p.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
    com_p.write_text(json.dumps(com, ensure_ascii=False), encoding="utf-8")
    G.main(["--item", str(item_p), "--commentary", str(com_p),
            "--out", str(out_p), *list(extra)])
    return out_p.read_text(encoding="utf-8")


def _data_of(html):
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>',
                  html, re.S)
    assert m, "per-turn data table missing"
    return json.loads(m.group(1))


def test_direction_thresholds():
    # task49c: direction of the speaker's standing vs their previous one.
    assert G.direction_verdict(13) == "오른다"
    assert G.direction_verdict(6) == "오른다"
    assert G.direction_verdict(5) == "그대로"
    assert G.direction_verdict(3) == "그대로"
    assert G.direction_verdict(0) == "그대로"
    assert G.direction_verdict(-3) == "그대로"
    assert G.direction_verdict(-5) == "그대로"
    assert G.direction_verdict(-6) == "내린다"
    assert G.direction_verdict(-33) == "내린다"
    assert G.direction_verdict(None) == "그대로"
    assert G.direction_verdict("x") == "그대로"


def test_reader_verdict_absolute():
    # task49b: the reason line still uses absolute reception, ±0.75.
    assert G.reader_verdict(1.2) == "잘"
    assert G.reader_verdict(0.75) == "잘"
    assert G.reader_verdict(0.74) == "보통"
    assert G.reader_verdict(0.0) == "보통"
    assert G.reader_verdict(-0.3) == "보통"
    assert G.reader_verdict(-0.74) == "보통"
    assert G.reader_verdict(-0.75) == "나쁘게"
    assert G.reader_verdict(-1.5) == "나쁘게"
    assert G.reader_verdict(None) == "보통"
    assert G.reader_verdict("x") == "보통"


def test_intro_and_three_buttons(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    assert ("두 사람이 캠핑 물품을 나누는 협상입니다. 발언을 하나씩 보고, "
            "상대가 이 말을 어떻게 받아들일지 맞혀 보세요." in html)
    assert "해설자는 정의상 항상 맞습니다" in html
    assert '>시작</button>' in html
    assert ">오른다</button>" in html
    assert ">그대로</button>" in html
    assert ">내린다</button>" in html
    assert 'data-guess="오른다"' in html
    assert 'data-guess="그대로"' in html
    assert 'data-guess="내린다"' in html
    assert ">잘 받아들여짐</button>" not in html
    assert ">나쁘게</button>" not in html
    assert 'id="question"' in html
    assert "나 0 / 해설자 0" in html
    assert ('<meta name="viewport" '
            'content="width=device-width, initial-scale=1">') in html
    assert "max-width:480px" in html
    assert "#0d1117" not in html


def test_per_turn_data_with_direction_and_score(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    data = _data_of(html)
    assert data["total"] == 4
    assert data["asked"] == 4
    assert [t["verdict"] for t in data["turns"]] == \
        ["오른다", "그대로", "내린다", "내린다"]
    assert [t["asked"] for t in data["turns"]] == \
        [True, True, True, True]
    assert [t["delta"] for t in data["turns"]] == [13, 3, -33, -8]
    for t in data["turns"]:
        assert isinstance(t["z_abs"], float)
        assert isinstance(t["z_resid"], float)
        assert t["text"]
        assert "관계 점수" in t["score_line"] and "→" in t["score_line"]
        assert t["reason"]
        assert t["question"] == (
            f"이 말로 {t['display']}의 관계 점수가 "
            f"오를까, 그대로일까, 내릴까?")
    assert [t["z_abs"] for t in data["turns"]] == \
        [1.2, -0.3, -1.5, 0.4]
    assert data["turns"][0]["score_line"] == "A의 관계 점수 50 → 63 (오름)"
    assert data["turns"][1]["score_line"] == "B의 관계 점수 50 → 53 (그대로)"
    assert data["turns"][2]["score_line"] == "B의 관계 점수 53 → 20 (내림)"
    assert data["turns"][3]["score_line"] == "A의 관계 점수 63 → 55 (내림)"
    assert "B의 관계 점수" in html
    assert "잔고" not in html


def test_skip_band_ambiguous(tmp_path):
    item, com = _skip_band_fixture()
    html = _run_main(tmp_path, item, com)
    data = _data_of(html)
    assert [t["delta"] for t in data["turns"]] == [0, 0, 4, 5, 3, 6]
    assert [t["asked"] for t in data["turns"]] == \
        [True, True, False, False, True, True]
    assert data["asked"] == 4
    # skipped turns still carry a key; the band edges stay asked.
    assert [t["verdict"] for t in data["turns"]] == ["그대로"] * 5 + ["오른다"]
    skipped = [t for t in data["turns"] if not t["asked"]]
    assert all(3 < abs(t["delta"]) < 6 for t in skipped)
    # skipped turns render as context: no guess buttons, auto-advance.
    assert "해설 없음" in html
    assert "if(!t.asked)" in html
    assert '$("guessbtns").hidden=true' in html


def test_first_turn_baseline_50(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    data = _data_of(html)
    seen = set()
    for t in data["turns"]:
        if t["speaker"] not in seen:
            seen.add(t["speaker"])
            assert t["score_before"] == 50
            assert t["asked"] is True
            assert t["delta"] == t["score_after"] - 50
    assert seen == {"A", "B"}


def test_first_turn_asked_despite_ambiguous_delta(tmp_path):
    effects = [
        _eff(0, "A", 1.0, rel=0.0, text="A opens"),
        _eff(1, "B", 1.0, rel=0.16, text="B nudges"),  # Δ +4, first turn → asked
    ]
    moves = [{"turn_index": i, "speaker": s, "moves": ["propose"]}
             for i, s in enumerate(["A", "B"])]
    item = _item("first-ask", effects, moves)
    com = CM.comment(item, _resid_norm({"propose": 0.0}), None, k=3)
    html = _run_main(tmp_path, item, com)
    data = _data_of(html)
    assert [t["delta"] for t in data["turns"]] == [0, 4]
    assert [t["asked"] for t in data["turns"]] == [True, True]


def test_question_text_names_speaker(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    data = _data_of(html)
    for t in data["turns"]:
        assert t["display"] in t["question"]
        assert t["question"].endswith("오를까, 그대로일까, 내릴까?")
    assert "오를까, 그대로일까, 내릴까?" in html
    assert '$("question")' in html


def test_cold_dialogue_keys_not_all_same(tmp_path):
    item, com = _cold_fixture()
    html = _run_main(tmp_path, item, com)
    data = _data_of(html)
    assert all(t["z_abs"] < 0 for t in data["turns"])
    keys = [t["verdict"] for t in data["turns"]]
    assert len(set(keys)) >= 2


def test_counter_counts_asked_only(tmp_path):
    item, com = _skip_band_fixture()
    html = _run_main(tmp_path, item, com)
    data = _data_of(html)
    assert data["asked"] == 4
    assert "문제 " in html
    assert "askedTotal" in html
    assert '"문제 "+' in html
    assert "4수 중" in html
    assert "6수 중" not in html


def test_export_json_shape(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    assert 'id="export"' in html
    assert ">내 답 내보내기</button>" in html
    assert "uptake-game:" in html
    assert "dialogue" in html
    assert "picks" in html
    assert "z_abs" in html
    assert "turn:t.index,pick:g,key:t.verdict,z_abs:t.z_abs" in html
    assert "dialogue:dialogueId,picks:picks,hits:hits,asked:askedTotal" in html
    assert "localStorage" in html and "try" in html and "catch" in html


def test_disagreement_skip(tmp_path):
    effects = [
        _eff(0, "A", 1.5, text="t0", dis=0.1),
        _eff(1, "B", -1.2, text="t1", dis=0.1),
        _eff(2, "A", 1.0, text="t2", dis=0.1),
        _eff(3, "B", -1.4, text="t3", dis=0.9),
    ]
    moves = [{"turn_index": i, "speaker": s, "moves": ["propose"]}
             for i, s in enumerate(["A", "B", "A", "B"])]
    item = _item("d1", effects, moves)
    com = CM.comment(item, _resid_norm({"propose": 0.0}), None, k=3)
    html = _run_main(tmp_path, item, com)
    data = _data_of(html)
    # flat standings (Δ == 0 everywhere), but the high-disagreement turn skips.
    assert [t["asked"] for t in data["turns"]] == \
        [True, True, True, False]
    assert data["asked"] == 3


def test_mismatch_note_and_fallback_reason():
    assert G._special_note("나쁘게", 1.2, ["reject"]) == "거절치고는 부드러웠다"
    assert G._special_note("잘", -1.3, ["propose"]) == "제안치고는 싸늘했다"
    assert G._special_note("나쁘게", 0.5, ["reject"]) is None
    assert G._special_note("보통", 2.0, ["reject"]) is None
    assert G._reason_of({"moves": ["reject"]}, 50, 40, z_abs=-1.0,
                        z_resid=1.2, verdict="나쁘게") == "거절치고는 부드러웠다"
    assert G._fallback_reason(["reject"], -1.0) == "거절이 평균보다 나쁘게 읽힘"
    assert G._fallback_reason(["propose"], 1.0) == "제안이 평균보다 잘 읽힘"
    assert G._reason_of({"moves": ["counter"]}, 50, 50, z_abs=0.6,
                        z_resid=0.0, verdict="보통") == "역제안이 평균보다 잘 읽힘"


def test_end_card_tide_outcome_share(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    # tide present but hidden until the end
    assert '<section id="end" class="card" hidden>' in html
    assert 'id="tide"' in html
    assert 'class="tide-a"' in html and 'class="tide-b"' in html
    assert V2.SPEAKER_COLORS[0] in html
    assert "실제 결과:" in html
    assert "A 50 →" in html and "B 50 →" in html
    # direction question footnote
    assert "질문은 '직전보다 오르나 내리나'입니다" in html
    assert "자가 확신한 수만 출제했습니다" not in html
    assert "나는 협상의 흐름을 " in html
    assert "4수 중" in html
    assert "수 맞혔다 — uptake" in html
    assert 'id="share"' in html and '>복사</button>' in html
    assert ">내 답 내보내기</button>" in html
    assert "localStorage" in html and "try" in html and "catch" in html
    assert "fetch(" not in html


def test_names_mapping(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com,
                     extra=["--names", '{"A": "Alice", "B": "Bob"}'])
    data = _data_of(html)
    assert data["turns"][0]["score_line"].startswith("Alice의 관계 점수")
    assert data["turns"][0]["question"].startswith("이 말로 Alice의 관계 점수가")
    assert "Bob의 관계 점수" in html


def test_reuses_board2_and_commentary_helpers():
    assert G.tide_svg is B2.tide_svg
    assert G.decisive_of is B2.decisive_of
    assert G.mark_for is CM.mark_for
    assert G.standing_round is CM.standing_round
    assert G.build_view2 is V2.build_view2
    assert G.MOVE_KO is B2.MOVE_KO


def test_pyproject_entry_point():
    root = Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'uptake-game = "forecaster.game:main"' in text
    assert callable(G.main)
