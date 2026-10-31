"""Contract for SPEC_task57_misread_frame (synthetic data only, no API)."""
import ast
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from forecaster import board as B
from forecaster import board2 as B2
from forecaster import commentary as CM
from forecaster import game as G
from forecaster import labels as L
from forecaster import misread as MR
from forecaster import viewer2 as V2


def _reading(action, rel=0.0, face=0):
    return {"next_action": action, "relationship_shift": rel,
            "face_threat": face}


def _peff(idx, speaker, readings, listener, text=""):
    counts = Counter(r["next_action"] for r in readings)
    n = len(readings)
    dist = {a: c / n for a, c in counts.items()} if n else {}
    rel_m = sum(r["relationship_shift"] for r in readings) / n if n else 0.0
    face_m = -sum(r["face_threat"] for r in readings) / n if n else 0.0
    return {"index": idx, "speaker": speaker, "text": text or f"panel t{idx}",
            "listeners": [listener],
            "relationship": [rel_m, rel_m, rel_m],
            "face": [face_m, face_m, face_m],
            "action_dist": dist, "disagreement": 0.0,
            "n_readings": n, "readings": list(readings)}


def _panel_fixture():
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
        {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
        {"turn_index": 2, "speaker": "A", "moves": ["ask"]},
        {"turn_index": 3, "speaker": "B", "moves": ["reject"]},
    ]
    effects = [
        _peff(0, "A", [_reading("negotiate", 0.5, 0),
                       _reading("negotiate", 0.5, 0),
                       _reading("comply", 0.0, 1)], "B"),
        _peff(1, "B", [_reading("comply", -1.0, 2),
                       _reading("comply", -1.2, 2),
                       _reading("refuse", -0.5, 1)], "A"),
        _peff(2, "A", [_reading("ask_clarification", 0.1, 0),
                       _reading("ask_clarification", 0.0, 0),
                       _reading("ask_clarification", 0.2, 0)], "B"),
        _peff(3, "B", [_reading("refuse", -0.2, 1),
                       _reading("refuse", -0.3, 1),
                       _reading("refuse", -0.1, 1)], "A"),
    ]
    return {"id": "dlg-57", "genre": "negotiation", "scale": "fine100",
            "moves": moves, "effects": effects, "hits": []}


def _pairwise_story():
    effects = [
        {"index": 0, "speaker": "A", "text": "A proposes",
         "value": 0.2, "n_readings": 3, "listeners": ["X"]},
        {"index": 1, "speaker": "B", "text": "B rejects coldly",
         "value": -2.2, "n_readings": 3, "listeners": ["X"]},
        {"index": 2, "speaker": "A", "text": "A asks",
         "value": 0.1, "n_readings": 3, "listeners": ["X"]},
        {"index": 3, "speaker": "B", "text": "B rejects again",
         "value": -0.2, "n_readings": 3, "listeners": ["X"]},
    ]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
        {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
        {"turn_index": 2, "speaker": "A", "moves": ["ask"]},
        {"turn_index": 3, "speaker": "B", "moves": ["reject"]},
    ]
    item = {"id": "dlg-57", "genre": "negotiation", "scale": "pairwise",
            "moves": moves, "effects": effects, "hits": []}
    norm = {"mu": 0.0, "sd": 1.0, "scale": "pairwise",
            "table": {"propose": 0.0, "reject": 0.0, "ask": 0.0},
            "sd_resid": 1.0}
    return item, CM.comment(item, norm, None, k=3)


def _data_of(html):
    m = re.search(r'<script id="data" type="application/json">(.*?)</script>',
                  html, re.S)
    assert m, "per-turn data table missing"
    return json.loads(m.group(1))


# --- misread unit: majority, invited set, severity ---------------------------

def test_majority_invited_and_severity():
    panel = _panel_fixture()
    got = {a["index"]: a for a in MR.analyze_panel(panel)}
    assert [a["index"] for a in MR.analyze_panel(panel)] == [0, 1, 2, 3]
    t0, t1, t2, t3 = got[0], got[1], got[2], got[3]
    assert (t0["majority"], t0["majority_ko"]) == ("negotiate", "흥정한다")
    assert t0["share"] == pytest.approx(2 / 3)
    assert set(t0["invited"]) == {"negotiate", "comply", "refuse", "proceed"}
    assert t0["misread"] is False
    assert t0["severity"] == pytest.approx(0.0)
    assert (t1["majority"], t1["majority_ko"]) == ("comply", "따른다")
    assert t1["share"] == pytest.approx(2 / 3)
    assert t1["invited"] == ["refuse", "negotiate", "ask_clarification",
                             "withdraw"]
    assert t1["misread"] is True
    assert t1["severity"] == pytest.approx(2 / 3)
    assert t1["face"] == "큼"
    assert (t2["majority"], t2["share"]) == ("ask_clarification", 1.0)
    assert t2["misread"] is False and t2["face"] == "없음"
    assert (t3["majority"], t3["share"]) == ("refuse", 1.0)
    assert t3["misread"] is False and t3["face"] == "약간"


def test_invited_actions_reuses_board_mapping():
    assert MR.invited_actions(["reject"]) == {
        "refuse", "negotiate", "ask_clarification", "withdraw"}
    assert MR.invited_actions(["propose"]) == {
        "negotiate", "comply", "refuse", "proceed"}
    assert MR.invited_actions(["reject", "appeal_need"]) == {
        "refuse", "negotiate", "ask_clarification", "withdraw"}
    assert MR.invited_actions([]) == set()
    # Inverse is faithful to board.ACTION_TO_MOVES in both directions.
    for action, moves in B.ACTION_TO_MOVES.items():
        for m in moves:
            assert action in MR.invited_actions([m])


def test_majority_tie_break_is_canonical():
    action, share = MR.majority_of_counts(Counter({"refuse": 1, "comply": 1}))
    assert action == "comply"  # earlier in labels.NEXT_ACTIONS
    assert share == pytest.approx(0.5)
    assert MR.majority_of_counts(Counter()) == (None, 0.0)


def test_face_word_thresholds():
    assert MR.face_word(0.0) == "없음"
    assert MR.face_word(0.49) == "없음"
    assert MR.face_word(0.5) == "약간"
    assert MR.face_word(1.0) == "약간"
    assert MR.face_word(1.49) == "약간"
    assert MR.face_word(1.5) == "큼"
    assert MR.face_word(2.0) == "큼"
    assert MR.face_word(None) == "없음"


def test_next_action_ko_covers_labels():
    assert set(MR.NEXT_ACTION_KO) == set(L.NEXT_ACTIONS)
    assert MR.NEXT_ACTION_KO["comply"] == "따른다"
    assert MR.NEXT_ACTION_KO["refuse"] == "거절한다"
    assert MR.NEXT_ACTION_KO["negotiate"] == "흥정한다"
    assert MR.NEXT_ACTION_KO["ask_clarification"] == "되묻는다"
    assert MR.NEXT_ACTION_KO["escalate"] == "맞선다"
    assert MR.NEXT_ACTION_KO["withdraw"] == "물러난다"


def test_counts_as_hit_lenient_only_when_key_invited():
    inv = ["negotiate", "proceed"]
    assert MR.counts_as_hit("negotiate", "negotiate", inv) is True
    assert MR.counts_as_hit("proceed", "negotiate", inv) is True
    assert MR.counts_as_hit("comply", "negotiate", inv) is False
    # Key outside the invited set: exact only.
    assert MR.counts_as_hit("escalate", "escalate", ["refuse"]) is True
    assert MR.counts_as_hit("refuse", "escalate", ["refuse"]) is False
    assert MR.counts_as_hit("refuse", "escalate", None) is False


def test_options_include_key_and_stay_five_to_six():
    got = {a["index"]: a for a in MR.analyze_panel(_panel_fixture())}
    for a in got.values():
        opts = MR.options_for(a["majority"], a["invited"], a["dist"])
        assert 5 <= len(opts) <= 6
        assert a["majority"] in [v for v, _ in opts]
        assert all(lb == MR.NEXT_ACTION_KO[v] for v, lb in opts)
        # Canonical order, so the key position gives nothing away.
        assert [v for v, _ in opts] == sorted(
            [v for v, _ in opts],
            key=lambda v: (L.NEXT_ACTIONS.index(v), v))
    labels = [lb for _, lb in MR.options_for(
        got[1]["majority"], got[1]["invited"], got[1]["dist"])]
    assert "따른다" in labels


def test_strip_flag_and_card_line_format():
    assert (MR.strip_flag(1, "B", "거절", "A", "압박")
            == "#1 B 거절 → A는 '압박'으로 읽음")
    reading = {"moves": ["reject", "appeal_need"],
               "intent": ["거절", "사정 호소"], "majority": "escalate",
               "majority_ko": "맞선다", "share": 2 / 3, "face": "큼",
               "rel_mean": 0.5}
    assert (MR.card_line(reading, before=50, after=8)
            == "의도: 거절·사정 호소 → 상대가 읽은 것: 맞선다(67%) · "
               "체면 위협 큼 · 관계 50 → 8")
    assert "관계 +0.50" in MR.card_line(reading)
    assert (MR.reveal_line({**reading, "majority_ko": "거절한다",
                            "share": 2 / 3, "face": "약간"},
                           before=50, after=8)
            == "상대 판정: 거절한다(67%) · 체면 위협 약간 · 관계 50 → 8")


def test_top_misread_caps_at_five():
    analyses = [{"index": i, "misread": True, "severity": 0.9 - i * 0.1}
                for i in range(7)]
    top = MR.top_misread(analyses, k=5)
    assert len(top) == 5
    assert [a["index"] for a in top] == [0, 1, 2, 3, 4]


def test_load_panel_sources(tmp_path):
    assert MR.load_panel(None) is None
    panel = _panel_fixture()
    assert MR.load_panel(panel) is panel
    assert MR.load_panel(tmp_path / "missing.json") is None
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(panel, ensure_ascii=False), encoding="utf-8")
    assert MR.load_panel(path)["id"] == "dlg-57"
    assert MR.analyze_panel(None) == []


# --- board2 -----------------------------------------------------------------

def test_board2_with_panel_shows_strip_and_three_dials():
    item, com = _pairwise_story()
    board = B2.build_board2(item, com, panel=_panel_fixture())
    assert board["panel"] is True
    assert board["misread_count"] == 1
    assert board["misread_total"] == 4
    assert board["misread_strip"] == ["#1 B 거절 → A는 '수락'으로 읽음"]
    assert 1 in [d["index"] for d in board["decisive"]]
    page = B2.render_html(board)
    assert "오해 지점" in page
    assert "#1 B 거절 → A는 '수락'으로 읽음" in page
    assert ("의도: 거절 → 상대가 읽은 것: 따른다(67%) · "
            "체면 위협 큼 · 관계 50 → ") in page
    assert "어긋난 턴 1 / 4" in page
    assert "관계 축만" not in page
    # Verdict line stays.
    assert board["verdict"] and board["verdict"] in page


def test_board2_without_panel_shows_badge_only():
    item, com = _pairwise_story()
    board = B2.build_board2(item, com)
    assert board["panel"] is False
    page = B2.render_html(board)
    assert "관계 축만" in page
    assert "오해 지점" not in page
    assert "상대가 읽은 것" not in page
    assert "어긋난 턴" not in page


def test_board2_panel_cli(tmp_path):
    item, com = _pairwise_story()
    item_p = tmp_path / "record.json"
    com_p = tmp_path / "commentary.json"
    panel_p = tmp_path / "panel.json"
    out_p = tmp_path / "board.html"
    item_p.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
    com_p.write_text(json.dumps(com, ensure_ascii=False), encoding="utf-8")
    panel_p.write_text(json.dumps(_panel_fixture(), ensure_ascii=False),
                       encoding="utf-8")
    B2.main(["--item", str(item_p), "--commentary", str(com_p),
             "--out", str(out_p), "--panel", str(panel_p)])
    page = out_p.read_text(encoding="utf-8")
    assert "오해 지점" in page
    assert "#1 B 거절 → A는 '수락'으로 읽음" in page
    assert "어긋난 턴 1 / 4" in page


# --- game -------------------------------------------------------------------

def test_game_with_panel_asks_next_actions():
    item, com = _pairwise_story()
    game = G.build_game(item, com, panel=_panel_fixture())
    assert game["panel"] is True
    turns = game["turns"]
    assert [t["verdict"] for t in turns] == [
        "negotiate", "comply", "ask_clarification", "refuse"]
    assert [t["asked"] for t in turns] == [True, True, True, True]
    assert all(t["question"] == "상대는 이 말을 어떻게 받을까?"
               for t in turns)
    t1 = turns[1]
    assert t1["invited"] == ["refuse", "negotiate", "ask_clarification",
                             "withdraw"]
    assert t1["key_in_invited"] is False
    assert 5 <= len(t1["options"]) <= 6
    assert "comply" in [o["value"] for o in t1["options"]]
    assert "따른다" in [o["label"] for o in t1["options"]]
    assert t1["reveal"].startswith(
        "상대 판정: 따른다(67%) · 체면 위협 큼 · 관계 50 → ")
    assert t1["reason"]  # 49b reason line stays
    assert t1["misread"] is True
    assert turns[0]["key_in_invited"] is True
    assert [t["misread"] for t in turns] == [False, True, False, False]
    page = G.render_html(game)
    data = _data_of(page)
    assert data["panel"] is True
    assert data["asked"] == 4
    assert "오해 지점" in page and "미리 봤다" in page
    assert "상대가 어떻게 받을까" in page
    assert "관계 축만" not in page


def test_game_panel_asks_only_majority_half_and_up():
    item, com = _pairwise_story()
    panel = _panel_fixture()
    panel["effects"][0] = _peff(
        0, "A", [_reading("negotiate", 0.5, 0),
                 _reading("comply", 0.0, 1),
                 _reading("refuse", -0.2, 1)], "B")
    game = G.build_game(item, com, panel=panel)
    turns = game["turns"]
    assert turns[0]["verdict"] == "comply"  # 1/3 tie, canonical first
    assert turns[0]["asked"] is False
    assert [t["asked"] for t in turns[1:]] == [True, True, True]
    assert game["asked"] == 3


def test_game_without_panel_keeps_direction_and_badge(tmp_path):
    item, com = _pairwise_story()
    game = G.build_game(item, com)
    assert game["panel"] is False
    page = G.render_html(game)
    assert "관계 축만" in page
    assert "오를까, 그대로일까, 내릴까?" in page
    data = _data_of(page)
    assert data["panel"] is False
    assert all(t["options"] == [] for t in data["turns"])


def test_game_panel_cli(tmp_path):
    item, com = _pairwise_story()
    item_p = tmp_path / "record.json"
    com_p = tmp_path / "commentary.json"
    panel_p = tmp_path / "panel.json"
    out_p = tmp_path / "game.html"
    item_p.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
    com_p.write_text(json.dumps(com, ensure_ascii=False), encoding="utf-8")
    panel_p.write_text(json.dumps(_panel_fixture(), ensure_ascii=False),
                       encoding="utf-8")
    G.main(["--item", str(item_p), "--commentary", str(com_p),
            "--out", str(out_p), "--panel", str(panel_p)])
    page = out_p.read_text(encoding="utf-8")
    data = _data_of(page)
    assert [t["verdict"] for t in data["turns"]] == [
        "negotiate", "comply", "ask_clarification", "refuse"]
    assert "오해 지점" in page and "미리 봤다" in page


# --- reuse + stdlib ----------------------------------------------------------

def test_reuses_board_labels_and_misread():
    assert MR.ACTION_TO_MOVES is B.ACTION_TO_MOVES
    assert MR.MOVE_PRIORITY is B.MOVE_PRIORITY
    assert MR.NEXT_ACTIONS is L.NEXT_ACTIONS
    assert MR.MOVE_KO is V2.MOVE_KO
    assert B2.misread is MR
    assert G.misread is MR


def test_misread_stdlib_only():
    src = Path(MR.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    stdlib = set(sys.stdlib_module_names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in stdlib, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue
            modname = (node.module or "").split(".")[0]
            assert modname in stdlib, node.module
