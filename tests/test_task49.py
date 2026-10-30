"""Contract for SPEC_task49_game + SPEC_task49b_game_key (synthetic data only)."""
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
    # z_abs == value (zero table, mu=0/sd=1): 잘 / 보통 / 나쁘게 / 보통.
    # |z| < 0.5 turns (1, 3) are context-only, so asked == 2.
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


def test_verdict_thresholds():
    # task49b: absolute reception, ±0.75.
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
    assert ">잘 받아들여짐</button>" in html
    assert ">보통</button>" in html
    assert ">나쁘게</button>" in html
    assert 'data-guess="잘"' in html
    assert "나 0 / 해설자 0" in html
    assert ('<meta name="viewport" '
            'content="width=device-width, initial-scale=1">') in html
    assert "max-width:480px" in html
    assert "#0d1117" not in html


def test_per_turn_data_with_verdict_and_score(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    data = _data_of(html)
    assert data["total"] == 4
    assert data["asked"] == 2
    assert [t["verdict"] for t in data["turns"]] == \
        ["잘", "보통", "나쁘게", "보통"]
    assert [t["asked"] for t in data["turns"]] == \
        [True, False, True, False]
    for t in data["turns"]:
        assert isinstance(t["z_abs"], float)
        assert isinstance(t["z_resid"], float)
        assert t["text"]
        assert "관계 점수" in t["score_line"] and "→" in t["score_line"]
        assert t["reason"]
    assert [t["z_abs"] for t in data["turns"]] == \
        [1.2, -0.3, -1.5, 0.4]
    assert data["turns"][0]["score_line"].startswith("A의 관계 점수 50 →")
    assert "B의 관계 점수" in html
    assert "잔고" not in html


def test_skip_rule_no_buttons(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    data = _data_of(html)
    skipped = [t for t in data["turns"] if not t["asked"]]
    assert len(skipped) == 2
    assert all(abs(t["z_abs"]) < 0.5 for t in skipped)
    # skipped turns render as context: no guess buttons, auto-advance.
    assert "해설 없음" in html
    assert "if(!t.asked)" in html
    assert '$("guessbtns").hidden=true' in html


def test_counter_counts_asked_only(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    data = _data_of(html)
    assert data["asked"] == 2
    assert "문제 " in html
    assert "askedTotal" in html
    assert '"문제 "+' in html
    assert "2수 중" in html
    assert "4수 중" not in html


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
    # strong |z| everywhere, but the high-disagreement turn is skipped.
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
    # hit rate over asked turns only, with the confident-only footnote
    assert "자가 확신한 수만 출제했습니다" in html
    assert "나는 협상의 흐름을 " in html
    assert "2수 중" in html
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
