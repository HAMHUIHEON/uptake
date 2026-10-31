"""Contract for SPEC_task58_alert_timeline (synthetic data only, no API)."""
import ast
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from forecaster import alerts as AL
from forecaster import board2 as B2
from forecaster import commentary as CM
from forecaster import misread as MR


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
    return {"id": "dlg-58", "genre": "negotiation", "scale": "fine100",
            "moves": moves, "effects": effects, "hits": []}


def _pairwise_story(genre="medical", cockpit=True, times=False):
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
    if cockpit:
        effects[1]["cockpit_before"] = "농담으로 풀려 했다"
    if times:
        effects[0]["time"] = "0:00"
        effects[1]["time"] = "2:05"
        effects[2]["time"] = "5:40"
        effects[3]["time"] = "8:31"
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
        {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
        {"turn_index": 2, "speaker": "A", "moves": ["ask"]},
        {"turn_index": 3, "speaker": "B", "moves": ["reject"]},
    ]
    item = {"id": "dlg-58", "genre": genre, "scale": "pairwise",
            "moves": moves, "effects": effects, "hits": []}
    norm = {"mu": 0.0, "sd": 1.0, "scale": "pairwise",
            "table": {"propose": 0.0, "reject": 0.0, "ask": 0.0},
            "sd_resid": 1.0}
    return item, CM.comment(item, norm, None, k=3)


def _case_fixture():
    return {"title": "Tenerife 참사", "outcome": "583명 사망",
            "source_url": "https://example.com/report", "note": "1977년"}


# --- levels and thresholds -------------------------------------------------

def test_stakes_weight_values_and_default():
    assert AL.STAKES_WEIGHT == {"aviation_atc": 1.0, "medical": 1.0,
                                "meeting": 0.6, "negotiation": 0.4}
    assert AL.stakes_weight("aviation_atc") == 1.0
    assert AL.stakes_weight("medical") == 1.0
    assert AL.stakes_weight("meeting") == 0.6
    assert AL.stakes_weight("negotiation") == 0.4
    assert AL.stakes_weight("casual") == 0.5
    assert AL.stakes_weight(None) == 0.5
    assert AL.stakes_weight("") == 0.5


def test_base_from_z_thresholds():
    assert AL.base_from_z(-2.5) == 1.0
    assert AL.base_from_z(-2.0) == 1.0
    assert AL.base_from_z(-1.5) == 0.5
    assert AL.base_from_z(-1.0) == 0.5
    assert AL.base_from_z(-0.99) == 0.0
    assert AL.base_from_z(0.0) == 0.0
    assert AL.base_from_z(1.5) == 0.0
    assert AL.base_from_z(None) == 0.0
    assert AL.base_from_z("bad") == 0.0


def test_level_thresholds():
    assert AL.level_of(0.0) == "관심"
    assert AL.level_of(0.24) == "관심"
    assert AL.level_of(0.25) == "주의"
    assert AL.level_of(0.49) == "주의"
    assert AL.level_of(0.5) == "경계"
    assert AL.level_of(0.74) == "경계"
    assert AL.level_of(0.75) == "심각"
    assert AL.level_of(1.0) == "심각"


def test_compute_alerts_panel_mode_uses_severity_times_weight():
    panel = _panel_fixture()
    analyses = MR.analyze_panel(panel)
    # z values must be ignored when a panel is given.
    turns = [{"index": i, "z": -3.0} for i in range(4)]
    got = AL.compute_alerts(turns, genre="medical", analyses=analyses)
    assert [a["level"] for a in got] == ["관심", "경계", "관심", "관심"]
    assert got[1]["base"] == pytest.approx(2 / 3)
    assert got[1]["score"] == pytest.approx(2 / 3)
    assert got[0]["score"] == pytest.approx(0.0)
    assert all(a["weight"] == 1.0 for a in got)
    low = AL.compute_alerts(turns, genre="negotiation", analyses=analyses)
    assert low[1]["score"] == pytest.approx(2 / 3 * 0.4)
    assert low[1]["level"] == "주의"
    assert AL.first_alert(low) is None
    assert AL.first_alert(got)["index"] == 1


def test_compute_alerts_zdrop_mode_prefers_z_resid():
    turns = [
        {"index": 0, "z": -0.5},
        {"index": 1, "z": -1.0},
        {"index": 2, "z": -2.0},
        {"index": 3, "z": -1.5, "z_resid": 0.0},
    ]
    got = AL.compute_alerts(turns, genre="aviation_atc")
    assert [a["base"] for a in got] == [0.0, 0.5, 1.0, 0.0]
    assert [a["level"] for a in got] == ["관심", "경계", "심각", "관심"]
    assert AL.first_alert(got)["index"] == 1
    # Unknown genre falls back to weight 0.5: base 1.0 -> 주의.
    unk = AL.compute_alerts(turns, genre="nope")
    assert unk[2]["score"] == pytest.approx(0.5)
    assert unk[2]["level"] == "경계"


# --- first alert and lead time ---------------------------------------------

def test_lead_time_turns_only_without_clock():
    turns = [{"index": 0, "z": 0.0}, {"index": 1, "z": -1.0},
             {"index": 2, "z": 0.0}, {"index": 3, "z": 0.0}]
    alerts = AL.compute_alerts(turns, genre="medical")
    lead = AL.lead_time(alerts)
    assert lead["first_index"] == 1
    assert lead["first_level"] == "경계"
    assert lead["outcome_index"] == 3
    assert lead["turns"] == 2
    assert lead["seconds"] is None
    assert lead["text"] == "결과까지 2턴"
    assert (AL.flag_text(AL.first_alert(alerts), lead)
            == "#1 경계 — 결과까지 2턴")


def test_lead_time_with_clock_and_exact_flag_format():
    turns = [{"index": i, "z": 0.0} for i in range(35)]
    turns[19] = {"index": 19, "z": -1.0, "time": "0:00"}
    turns[34] = {"index": 34, "z": 0.0, "time": "8:31"}
    alerts = AL.compute_alerts(turns, genre="medical")
    assert AL.first_alert(alerts)["index"] == 19
    lead = AL.lead_time(alerts)
    assert lead["turns"] == 15
    assert lead["seconds"] == pytest.approx(511.0)
    assert lead["text"] == "결과까지 15턴 / 8분 31초"
    assert (AL.flag_text(AL.first_alert(alerts), lead)
            == "#19 경계 — 결과까지 15턴 / 8분 31초")


def test_parse_time_and_format_duration():
    assert AL.parse_time("8:31") == 511.0
    assert AL.parse_time("1:02:03") == 3723.0
    assert AL.parse_time("0:00") == 0.0
    assert AL.parse_time(45) == 45.0
    assert AL.parse_time("45") == 45.0
    assert AL.parse_time(None) is None
    assert AL.parse_time("nope") is None
    assert AL.parse_time(-5) is None
    assert AL.parse_time("8:75") is None
    assert AL.format_duration(511) == "8분 31초"
    assert AL.format_duration(45) == "45초"
    assert AL.format_duration(3723) == "1시간 2분 3초"


def test_outcome_turn_override_and_fallback():
    turns = [{"index": i, "z": (-1.0 if i == 1 else 0.0)}
             for i in range(5)]
    alerts = AL.compute_alerts(turns, genre="medical")
    assert AL.lead_time(alerts, outcome_turn=3)["turns"] == 2
    assert AL.lead_time(alerts, outcome_turn=3)["outcome_index"] == 3
    assert AL.lead_time(alerts, outcome_turn=99)["outcome_index"] == 4
    assert AL.lead_time(alerts, outcome_turn=99)["turns"] == 3


def test_no_first_alert_when_levels_stay_low():
    alerts = AL.compute_alerts([{"index": 0, "z": 0.0}], genre="medical")
    assert AL.first_alert(alerts) is None
    assert AL.lead_time(alerts) is None
    assert AL.lead_time([]) is None
    strip = AL.alert_strip_html(alerts, genre="medical")
    assert "경보 타임라인" in strip
    assert "경계 이상 없음" in strip
    assert AL.alert_strip_html([]) == ""


def test_alert_strip_ticks_and_flag():
    turns = [{"index": i, "z": (-1.0 if i == 1 else 0.0)}
             for i in range(4)]
    alerts = AL.compute_alerts(turns, genre="medical")
    strip = AL.alert_strip_html(alerts, genre="medical")
    assert strip.count('class="tick"') == 4
    assert "#1 경계 — 결과까지 2턴" in strip
    assert 'data-level="경계"' in strip
    assert 'title="#1 경계"' in strip


# --- intent table ----------------------------------------------------------

def test_intent_rows_from_cockpit_before_with_panel():
    item, com = _pairwise_story()
    board = B2.build_board2(item, com, panel=_panel_fixture())
    rows = board["intent_rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["index"] == 1
    assert row["intent"] == "농담으로 풀려 했다"
    assert row["said"] == "B rejects coldly"
    assert row["read"] == "따른다(67%)"
    page = B2.render_html(board)
    assert "의도 vs 전달 vs 해석" in page
    assert "뜻한 것" in page and "말한 것" in page and "읽힌 것" in page
    assert "농담으로 풀려 했다" in page
    assert "B rejects coldly" in page
    assert "따른다(67%)" in page


def test_intent_rows_fallback_to_pairwise_verdict_without_panel():
    item, com = _pairwise_story()
    board = B2.build_board2(item, com)
    rows = board["intent_rows"]
    assert len(rows) == 1
    assert rows[0]["read"].startswith("나쁘게 ")
    assert " → " in rows[0]["read"]
    page = B2.render_html(board)
    assert "의도 vs 전달 vs 해석" in page
    assert rows[0]["read"] in page


def test_intent_map_overrides_and_extends_cockpit(tmp_path):
    item, com = _pairwise_story()
    board = B2.build_board2(item, com, intent={1: "명시적 의도", 2: "추가 메모"})
    rows = {r["index"]: r for r in board["intent_rows"]}
    assert set(rows) == {1, 2}
    assert rows[1]["intent"] == "명시적 의도"
    assert rows[2]["said"] == "A asks"
    assert AL.load_intent({"1": "x", "bad": "y", 2: ""}) == {1: "x"}
    assert AL.load_intent('{"1": "x"}') == {1: "x"}
    assert AL.load_intent("not json") == {}
    assert AL.load_intent(None) == {}
    path = tmp_path / "intent.json"
    path.write_text('{"3": "file note"}', encoding="utf-8")
    assert AL.load_intent(path) == {3: "file note"}
    assert AL.load_intent(tmp_path / "missing.json") == {}


def test_no_intent_section_without_notes():
    item, com = _pairwise_story(cockpit=False)
    page = B2.render_html(B2.build_board2(item, com))
    assert "의도 vs 전달 vs 해석" not in page
    assert AL.intent_table_html([]) == ""


# --- case header -----------------------------------------------------------

def test_case_header_and_footer_source_link():
    item, com = _pairwise_story()
    board = B2.build_board2(item, com, case=_case_fixture())
    assert board["case"]["title"] == "Tenerife 참사"
    page = B2.render_html(board)
    assert "사건과 결과" in page
    assert "Tenerife 참사" in page
    assert "583명 사망" in page
    assert "1977년" in page
    assert 'href="https://example.com/report"' in page
    assert "모델이 발언을 읽은 것이지 사람을 판정한 것이 아닙니다" in page
    # Header block sits above the verdict; source link also in the footer.
    assert page.index("사건과 결과") < page.index("한 문장")
    footer = page.split("<footer>")[1]
    assert "출처" in footer
    assert "https://example.com/report" in footer


def test_no_case_block_by_default():
    item, com = _pairwise_story()
    page = B2.render_html(B2.build_board2(item, com))
    assert "사건과 결과" not in page
    assert "출처" not in page
    assert AL.load_case(None) is None
    assert AL.load_case("not json") is None
    assert AL.case_block_html(None) == ""


# --- board2 wiring ---------------------------------------------------------

def test_board_alerts_use_panel_severity_and_genre_weight():
    item, com = _pairwise_story()
    board = B2.build_board2(item, com, panel=_panel_fixture())
    assert board["alert_genre"] == "medical"
    assert board["alert_weight"] == 1.0
    assert [a["level"] for a in board["alerts"]] == ["관심", "경계",
                                                    "관심", "관심"]
    assert board["alert_first"]["index"] == 1
    assert board["alert_lead"]["turns"] == 2
    page = B2.render_html(board)
    assert "경보 타임라인" in page
    assert "#1 경계 — 결과까지 2턴" in page
    # Strip sits above the tide.
    assert page.index("경보 타임라인") < page.index("<h2>흐름")


def test_board_alerts_zdrop_without_panel_and_clock_lead():
    item, com = _pairwise_story(times=True)
    board = B2.build_board2(item, com)
    assert board["alert_first"]["index"] == 1
    assert board["alert_first"]["level"] == "심각"
    assert board["alert_lead"]["text"] == "결과까지 2턴 / 6분 26초"
    page = B2.render_html(board)
    assert "#1 심각 — 결과까지 2턴 / 6분 26초" in page


def test_board_outcome_turn_option():
    item, com = _pairwise_story()
    board = B2.build_board2(item, com, outcome_turn=2)
    assert board["alert_lead"]["outcome_index"] == 2
    assert board["alert_lead"]["turns"] == 1


def test_no_game_link_anywhere_in_board():
    item, com = _pairwise_story()
    board = B2.build_board2(item, com)
    assert board.get("game") is None
    page = B2.render_html(board)
    assert "game_" not in page
    assert "게임으로 풀어 보기" not in page
    assert "uptake-game" not in page
    assert "dashboard_dlg-58.html" in page


def test_board2_cli_with_intent_case_outcome_turn(tmp_path):
    item, com = _pairwise_story(cockpit=False)
    item_p = tmp_path / "record.json"
    com_p = tmp_path / "commentary.json"
    panel_p = tmp_path / "panel.json"
    case_p = tmp_path / "case.json"
    out_p = tmp_path / "board.html"
    item_p.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
    com_p.write_text(json.dumps(com, ensure_ascii=False), encoding="utf-8")
    panel_p.write_text(json.dumps(_panel_fixture(), ensure_ascii=False),
                       encoding="utf-8")
    case_p.write_text(json.dumps(_case_fixture(), ensure_ascii=False),
                      encoding="utf-8")
    B2.main(["--item", str(item_p), "--commentary", str(com_p),
             "--out", str(out_p), "--panel", str(panel_p),
             "--intent", '{"2": "CLI 메모"}', "--case", str(case_p),
             "--outcome-turn", "3"])
    page = out_p.read_text(encoding="utf-8")
    assert "경보 타임라인" in page
    assert "#1 경계 — 결과까지 2턴" in page
    assert "의도 vs 전달 vs 해석" in page
    assert "CLI 메모" in page
    assert "사건과 결과" in page
    assert "Tenerife 참사" in page
    assert "game_" not in page
    assert "게임으로 풀어 보기" not in page


# --- reuse + stdlib --------------------------------------------------------

def test_board2_reuses_alerts():
    assert B2.alerts is AL


def test_alerts_stdlib_only():
    src = Path(AL.__file__).read_text(encoding="utf-8")
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
