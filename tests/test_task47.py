"""Contract for SPEC_task47_legibility (synthetic data only)."""
import json

import pytest

from forecaster import commentary as CM
from forecaster import viewer2 as V2


def _eff(idx, speaker, value, rel=0.0, text="", **kw):
    e = {"index": idx, "speaker": speaker, "text": text or f"t{idx}",
         "value": value, "disagreement": 0.0, "n_readings": 2,
         "relationship": [rel, rel, rel], "face": [0.0, 0.0, 0.0]}
    e.update(kw)
    return e


def _item(cid, effects, moves, scale="fine100"):
    return {"id": cid, "genre": "negotiation", "scale": scale,
            "moves": moves, "effects": effects, "hits": []}


def _resid_norm(table, sd_resid=1.0):
    return {"mu": 0.0, "sd": 1.0, "scale": "fine100",
            "table": dict(table), "sd_resid": sd_resid}


def _blob_data(html):
    tag = '<script type="application/json" id="data">'
    start = html.index(tag) + len(tag)
    return json.loads(html[start:html.index("</script>", start)])


def _caption_fixture():
    # z_resid: A +0.3, B -2.0, B +2.5, A +0.2 (top speaker A).
    # M = [0.3, 2.165, -1.30925, -0.5200875].
    effects = [
        _eff(0, "A", 0.3, rel=0.2, text="a opens mildly"),
        _eff(1, "B", -2.0, rel=-0.4, text="b rejects the offer"),
        _eff(2, "B", 2.5, rel=0.1, text="b counters well"),
        _eff(3, "A", 0.2, rel=0.0, text="a proposes a tweak"),
    ]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["inform"]},
        {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
        {"turn_index": 2, "speaker": "B", "moves": ["counter"]},
        {"turn_index": 3, "speaker": "A", "moves": ["propose"]},
    ]
    item = _item("cap-1", effects, moves)
    norm = _resid_norm({"inform": 0.0, "reject": 0.0, "counter": 0.0,
                        "propose": 0.0})
    return item, CM.comment(item, norm, None, k=3)


# --- how-to card + axis label + dot legend ---------------------------------

def test_howto_card_axis_label_and_dot_legend():
    item, com = _caption_fixture()
    html = V2.render_html(V2.build_view2(item, com))
    assert '<details class="howto" open>' in html
    assert "읽는 법 · How to read" in html
    assert ("빨간 면 = A의 영역, 파란 면 = B의 영역. 경계선이 아래로 내려오면 "
            "A가 밀고 있는 것, 올라오면 B가 되찾는 것.") in html
    assert "점 하나 = 발언 하나(빨간 점 A, 파란 점 B). 가로 점선 = 50:50." in html
    assert ("표시: !! 묘수 · ! 호수 · ? 실수 · ?? 악수 (그 발언이 보통 그런 "
            "종류의 발언이 받아들여지는 정도보다 얼마나 더/덜 받아들여졌나)"
            ) in html
    assert "발언 순서 (turn)" in html
    assert "● 빨간 점 = A 발언 · ● 파란 점 = B 발언" in html


def test_first_mention_uses_full_term():
    item, com = _caption_fixture()
    html = V2.render_html(V2.build_view2(item, com))
    assert html.count("관계 점수(잔고)") == 1
    first = html.index("잔고")
    assert html[first - len("관계 점수("):first] == "관계 점수("
    assert html.count("잔고") > 1  # later mentions keep the short form


# --- virtual start point ----------------------------------------------------

def test_virtual_start_point_in_data_table():
    item, com = _caption_fixture()
    view = V2.build_view2(item, com)
    assert view["tide_start"] == {"x": -0.5, "m": 0.0}
    # momentum itself is unchanged (task38 behaviour)
    assert view["momentum"] == pytest.approx([0.3, 2.165, -1.30925,
                                              -0.5200875])
    data = _blob_data(V2.render_html(view))
    assert data["tide_start"] == {"x": -0.5, "m": 0.0}
    # the chart and the replay both grow from that midline point
    assert "tX(-0.5)" in V2._JS
    assert "(i+0.5)/(N-0.5)" in V2._JS
    # x-axis starts at -0.5 (task47b): virtual start stays visible,
    # including the single-turn case (no N<=1 collapse to centre).
    assert V2._TIDE_START_X == -0.5
    assert "if(!N)" in V2._JS
    assert "N<=1" not in V2._JS
    single = _item("single", [_eff(0, "A", 1.0, text="a")],
                   [{"turn_index": 0, "speaker": "A",
                     "moves": ["propose"]}])
    sv = V2.build_view2(single, CM.comment(
        single, _resid_norm({"propose": 0.0}), None, k=3))
    assert sv["tide_start"] == {"x": -0.5, "m": 0.0}


# --- disagreement band ------------------------------------------------------

def test_band_hidden_for_pairwise_shown_for_panel_scales():
    pw_moves = [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
                {"turn_index": 1, "speaker": "B", "moves": ["counter"]}]
    pw = _item("pw-band",
               [{"index": 0, "speaker": "A", "text": "a", "value": 0.6,
                 "disagreement": 0.5, "n_readings": 3,
                 "listeners": ["X"]},
                {"index": 1, "speaker": "B", "text": "b", "value": -0.6,
                 "disagreement": 0.5, "n_readings": 3,
                 "listeners": ["X"]}],
               pw_moves, scale="pairwise")
    pw_norm = {"mu": 0.0, "sd": 0.6, "scale": "pairwise",
               "table": {"propose": 0.0, "counter": 0.0}, "sd_resid": 0.5}
    v_pw = V2.build_view2(pw, CM.comment(pw, pw_norm, None, k=3))
    assert v_pw["spread_src"] is None
    assert v_pw["spread"] == [None, None]

    f100 = _item("f-band",
                 [_eff(0, "A", 1.0, text="a", disagreement=0.42),
                  _eff(1, "B", 0.5, text="b", disagreement=0.10)],
                 pw_moves)
    norm = _resid_norm({"propose": 0.0, "counter": 0.0})
    v_100 = V2.build_view2(f100, CM.comment(f100, norm, None, k=3))
    assert v_100["spread_src"] == "disagreement"
    assert v_100["spread"] == [0.42, 0.10]


def test_spread_scale_gate_unit():
    e = {"index": 0, "speaker": "A", "text": "a", "value": 1.0,
         "disagreement": 0.5}
    assert V2.spread_of_effect(e, "pairwise") == (None, None)
    assert V2.spread_of_effect(e, "fine100") == (0.5, "disagreement")
    assert V2.spread_of_effect(e, "dist5") == (0.5, "disagreement")
    assert V2.spread_of_effect(e) == (0.5, "disagreement")
    r = dict(e, readings=[{"rel_spread": 18.0}])
    assert V2.spread_of_effect(r, "pairwise") == (18.0, "rel_spread")


# --- auto caption -----------------------------------------------------------

def test_caption_decisive_reversal_and_leader():
    item, com = _caption_fixture()
    view = V2.build_view2(item, com)
    cap = view["caption"]
    assert cap == ("#1에서 B가 거절했지만 되려 A 쪽으로 넘어갔다, "
                   "#2에서 B가 역제안해 B가 많이 되찾았다. "
                   "마지막은 B가 조금 앞선 채 끝.")
    assert "#1" in cap and "#2" in cap and "되려 A 쪽으로" in cap
    assert "2수" not in cap and "3수" not in cap
    html = V2.render_html(view)
    assert 'id="caption"' in html
    assert "#1에서 B가 거절했지만 되려 A 쪽으로 넘어갔다" in html


def test_caption_falls_back_to_raw_move_label():
    item = _item("raw-mv", [_eff(0, "A", 2.0, text="a")],
                 [{"turn_index": 0, "speaker": "A",
                   "moves": ["mystery_move"]}])
    com = CM.comment(item, _resid_norm({}), None, k=3)
    cap = V2.build_view2(item, com)["caption"]
    assert "mystery_move" in cap
    assert "#0에서 A가 mystery_move하며 A 쪽으로 가져갔다" in cap
    assert "A 쪽으로 가져갔다" in cap


def test_move_glossary_covers_spec_labels():
    assert V2.MOVE_KO["reject"] == "거절"
    assert V2.MOVE_KO["appeal_need"] == "사정 호소"
    assert V2.MOVE_KO["counter"] == "역제안"
    assert V2.MOVE_KO["concession"] == "양보"


# --- square sentence --------------------------------------------------------

def test_square_note_variants():
    assert V2.square_note([21.0], [14.0], "A", "B") == \
        "둘 다 평균(50) 아래, A가 덜 잃음"
    assert V2.square_note([60.0], [70.0], "A", "B") == \
        "둘 다 평균(50) 위, B가 더 앞섬"
    assert V2.square_note([75.0], [40.0], "A", "B") == \
        "한쪽만 평균(50) 위, A가 앞섬"
    assert V2.square_note([], [], "A", "B") == ""


def test_square_sentence_matches_standings():
    item = _item("sq-1",
                 [_eff(0, "A", -1.0, rel=-1.16, text="a loses"),
                  _eff(1, "B", -1.2, rel=-1.44, text="b loses more")],
                 [{"turn_index": 0, "speaker": "A", "moves": ["reject"]},
                  {"turn_index": 1, "speaker": "B", "moves": ["reject"]}])
    norm = _resid_norm({"reject": 0.0})
    view = V2.build_view2(item, CM.comment(item, norm, None, k=3))
    assert view["settlement"] == "A 잔고 50 → 21 · B 잔고 50 → 14"
    assert view["square_note"] == "둘 다 평균(50) 아래, A가 덜 잃음"
    html = V2.render_html(view)
    assert ("A 잔고 50 → 21 · B 잔고 50 → 14 — 둘 다 평균(50) 아래, "
            "A가 덜 잃음") in html
    assert "둘 다 잘됨" not in html
    assert "둘 다 잃음" not in html
    assert "한쪽만" not in html
    assert "square_note" in V2._JS  # replay keeps the sentence in sync


def test_empty_record_has_no_caption_or_note():
    item = _item("empty", [], [])
    view = V2.build_view2(item, CM.comment(
        item, {"mu": 0.0, "sd": 1.0}, None, k=3))
    assert view["caption"] == ""
    assert view["square_note"] == ""
    assert view["tide_start"] == {"x": -0.5, "m": 0.0}
    html = V2.render_html(view)
    assert 'id="caption"' not in html
