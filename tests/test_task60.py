"""Contract for SPEC_task60_board_polish (synthetic data only, no API)."""
import html as htmlmod
import json
from collections import Counter

import pytest

from forecaster import board2 as B2
from forecaster import commentary as CM
from forecaster import labels as L
from forecaster import misread as MR
from forecaster import pairwise as P


def _reading(action, rel=0.0, face=0):
    return {"next_action": action, "relationship_shift": rel,
            "face_threat": face}


def _effect_for(actions):
    return {"readings": [_reading(a) for a in actions]}


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


def _item(cid, effects, moves, scale="pairwise", genre="medical"):
    return {"id": cid, "genre": genre, "scale": scale,
            "moves": moves, "effects": effects, "hits": []}


def _pw_effects(speakers, values):
    return [{"index": i, "speaker": s, "text": f"turn {i} by {s}",
             "value": v, "n_readings": 3, "listeners": ["X"]}
            for i, (s, v) in enumerate(zip(speakers, values))]


def _pw_moves(speakers, move="inform"):
    return [{"turn_index": i, "speaker": s, "moves": [move]}
            for i, s in enumerate(speakers)]


def _pw_norm(moves=("inform",)):
    return {"mu": 0.0, "sd": 1.0, "scale": "pairwise",
            "table": {m: 0.0 for m in moves}, "sd_resid": 1.0}


def _story_3speaker():
    # Scrambled arrival (C first) with counts A3/B2/C1: activity order
    # must win over appearance order for cards, tide pair and legend.
    speakers = ["C", "A", "B", "A", "B", "A"]
    values = [0.30, 0.40, 0.20, 0.35, 0.10, 0.30]
    item = _item("dlg-60-3sp", _pw_effects(speakers, values),
                 _pw_moves(speakers))
    return item, CM.comment(item, _pw_norm(), None, k=3)


def _story_7speaker():
    speakers = [f"S{i}" for i in range(7)]
    values = [0.05] * 7
    item = _item("dlg-60-7sp", _pw_effects(speakers, values),
                 _pw_moves(speakers))
    return item, CM.comment(item, _pw_norm(), None, k=3)


def _quiet_panel(alert_idx):
    # Six quiet turns; only alert_idx carries conflict severity.
    speakers = ["A", "B", "A", "B", "A", "B"]
    moves = _pw_moves(speakers)
    effects = []
    for i, s in enumerate(speakers):
        if i == alert_idx:
            readings = [_reading("refuse", -1.0, 2),
                        _reading("refuse", -1.0, 2),
                        _reading("refuse", -1.0, 2)]
        else:
            readings = [_reading("comply", 0.1, 0),
                        _reading("comply", 0.1, 0),
                        _reading("comply", 0.1, 0)]
        effects.append(_peff(i, s, readings, "X"))
    return {"id": "dlg-60-quiet", "genre": "medical", "scale": "fine100",
            "moves": moves, "effects": effects, "hits": []}


def _quiet_story():
    speakers = ["A", "B", "A", "B", "A", "B"]
    values = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
    item = _item("dlg-60-quiet", _pw_effects(speakers, values),
                 _pw_moves(speakers))
    return item, CM.comment(item, _pw_norm(), None, k=3)


# --- misread = conflict ------------------------------------------------------

def test_conflict_table_matches_spec():
    assert MR.CONFLICTS["accept"] == {"refuse", "escalate", "retaliate",
                                      "ignore"}
    assert MR.CONFLICTS["concede"] == {"refuse", "escalate", "retaliate",
                                       "ignore"}
    assert MR.CONFLICTS["propose"] == {"ignore", "retaliate", "escalate"}
    assert MR.CONFLICTS["counter"] == {"ignore", "retaliate", "escalate"}
    assert MR.CONFLICTS["appeal_need"] == {"ignore", "retaliate", "escalate"}
    assert MR.CONFLICTS["appeal_fairness"] == {"ignore", "retaliate",
                                               "escalate"}
    assert MR.CONFLICTS["ask"] == {"ignore", "refuse", "escalate"}
    assert MR.CONFLICTS["probe"] == {"ignore", "refuse", "escalate"}
    assert MR.CONFLICTS["inform"] == {"refuse", "ignore", "retaliate",
                                      "escalate"}
    assert MR.CONFLICTS["pressure"] == {"ignore", "withdraw"}
    assert MR.CONFLICTS["challenge"] == {"ignore", "withdraw"}
    assert MR.CONFLICTS["repair"] == {"retaliate", "escalate", "refuse",
                                      "ignore"}
    assert MR.CONFLICTS["empathize"] == {"retaliate", "escalate", "refuse",
                                         "ignore"}
    assert MR.CONFLICTS["praise"] == {"retaliate", "escalate", "refuse",
                                      "ignore"}
    assert MR.CONFLICTS["reject"] == {"comply", "ignore"}
    # Invited and conflict never overlap: uptake can never be misread.
    for move in MR.CONFLICTS:
        assert MR.invited_actions([move]).isdisjoint(
            MR.conflict_actions([move]))


def test_invited_sets_widened_for_uptake():
    assert MR.invited_actions(["inform"]) == {
        "proceed", "comply", "ask_clarification"}
    assert MR.invited_actions(["ask"]) == {
        "ask_clarification", "comply", "proceed"}
    assert MR.invited_actions(["probe"]) == {
        "ask_clarification", "comply", "proceed"}
    assert MR.invited_actions(["other"]) == set(L.NEXT_ACTIONS)
    assert MR.invited_actions(["smalltalk"]) == set(L.NEXT_ACTIONS)
    assert MR.invited_actions(["propose"]) == {
        "negotiate", "comply", "refuse", "proceed"}
    assert MR.invited_actions(["reject"]) == {
        "refuse", "negotiate", "ask_clarification", "withdraw"}
    # The rest keep the pure board inverse.
    assert MR.invited_actions(["accept"]) == {"comply", "proceed"}
    assert MR.invited_actions(["counter"]) == {"negotiate"}
    assert MR.invited_actions(["deflect"]) == {"withdraw", "ignore"}
    assert MR.invited_actions(["repair"]) == {"repair"}


def test_severity_zero_when_move_has_no_conflict_set():
    for moves in (["deflect"], ["other"], ["smalltalk"], ["joke"],
                  ["cooperate"], ["reveal_preference"], ["unknown_move"],
                  []):
        row = MR.reading_for_turn(moves, _effect_for(["refuse"] * 3))
        assert row["conflict"] == []
        assert row["severity"] == pytest.approx(0.0)
        assert row["misread"] is False


def test_1549_style_comply_and_answer_not_misread():
    # An instruction complied with is uptake.
    row = MR.reading_for_turn(["inform"], _effect_for(["comply"] * 3))
    assert row["misread"] is False
    assert row["severity"] == pytest.approx(0.0)
    # A question answered (comply/proceed) is uptake, not misread.
    for moves in (["ask"], ["probe"]):
        for majority in ("comply", "proceed"):
            row = MR.reading_for_turn(
                moves, _effect_for([majority] * 2 + ["ask_clarification"]))
            assert row["majority"] == majority
            assert row["misread"] is False
            assert row["severity"] == pytest.approx(0.0)
    # Asking back on an instruction is uptake too.
    row = MR.reading_for_turn(
        ["inform"], _effect_for(["ask_clarification"] * 2 + ["comply"]))
    assert row["misread"] is False
    # Instruction chatter ("other") can never misread.
    row = MR.reading_for_turn(["other"], _effect_for(["escalate"] * 3))
    assert row["misread"] is False
    assert row["severity"] == pytest.approx(0.0)


def test_lear_nothing_style_refusal_still_misread():
    # A bare statement ("Nothing.") read as a refusal stays flagged.
    row = MR.reading_for_turn(
        ["inform"], _effect_for(["refuse"] * 2 + ["ignore"]))
    assert row["misread"] is True
    assert row["severity"] == pytest.approx(1.0)
    assert row["conflict"] == ["refuse", "escalate", "retaliate", "ignore"]
    # So does a question read as a refusal.
    row = MR.reading_for_turn(
        ["ask"], _effect_for(["refuse"] * 2 + ["comply"]))
    assert row["misread"] is True
    assert row["severity"] == pytest.approx(2 / 3)
    # Rejection conflicts: taken as a go-ahead, or blanked.
    row = MR.reading_for_turn(
        ["reject"], _effect_for(["comply"] * 2 + ["refuse"]))
    assert row["misread"] is True
    assert row["severity"] == pytest.approx(2 / 3)
    row = MR.reading_for_turn(["reject"], _effect_for(["ignore"] * 3))
    assert row["misread"] is True
    assert row["severity"] == pytest.approx(1.0)


def test_severity_is_conflict_share_from_action_dist():
    eff = {"action_dist": {"refuse": 0.5, "comply": 0.5}, "n_readings": 4}
    row = MR.reading_for_turn(["inform"], eff)
    assert row["majority"] == "comply"  # tie breaks canonically
    assert row["misread"] is False
    assert row["severity"] == pytest.approx(0.5)
    assert row["conflict"] == ["refuse", "escalate", "retaliate", "ignore"]


# --- multi-speaker boards ----------------------------------------------------

def test_three_speaker_cards_legend_and_others_line():
    item, com = _story_3speaker()
    board = B2.build_board2(item, com)
    assert [c["speaker"] for c in board["cards"]] == ["A", "B", "C"]
    assert len(board["other_speakers"]) == 1
    assert board["other_speakers"][0]["speaker"] == "C"
    assert isinstance(board["other_speakers"][0]["standing"], int)
    # The tide stays two-colour, led by the two most active (A on top:
    # turn 0 is C with z > 0, so M0 goes negative).
    assert board["momentum"][0] < 0
    page = B2.render_html(board)
    assert page.count('class="score"') == 3
    assert "빨강 A · 파랑 B · 선이 내려오면 A 우세" in page
    assert 'class="tide-a"' in page and 'class="tide-b"' in page
    assert "band" not in page.lower()
    standing = board["other_speakers"][0]["standing"]
    assert f"다른 화자: C {standing}" in page


def test_seven_speakers_cap_cards_at_six():
    item, com = _story_7speaker()
    board = B2.build_board2(item, com)
    assert [c["speaker"] for c in board["cards"]] == [f"S{i}" for i in range(6)]
    assert [o["speaker"] for o in board["other_speakers"]] == [
        f"S{i}" for i in range(2, 7)]
    page = B2.render_html(board)
    assert page.count('class="score"') == 6
    assert "빨강 S0 · 파랑 S1 · 선이 내려오면 S0 우세" in page
    assert "다른 화자:" in page
    for i in range(2, 7):
        assert f"S{i} " in page


def test_two_speaker_board_has_no_others_line():
    item, com = _quiet_story()
    board = B2.build_board2(item, com)
    assert len(board["cards"]) == 2
    assert board["other_speakers"] == []
    assert "다른 화자" not in B2.render_html(board)


# --- first-alert card --------------------------------------------------------

def test_first_alert_card_added_when_outside_top3():
    item, com = _quiet_story()
    assert com  # commentary builds; decisive ranking comes from the tide
    board = B2.build_board2(item, com, panel=_quiet_panel(5))
    assert board["alert_first"]["index"] == 5
    assert [d["index"] for d in board["decisive"]] == [5, 0, 1, 2]
    first = board["decisive"][0]
    assert first["first_alert"] is True
    assert "상대가 읽은 것" in (first["misread_line"] or "")
    assert all(not d.get("first_alert") for d in board["decisive"][1:])
    page = B2.render_html(board)
    assert page.count("첫 경보") == 1
    assert page.count('class="moment"') == 4
    assert "#5 " in page


def test_first_alert_card_moves_to_front_when_inside_top3():
    item, com = _quiet_story()
    board = B2.build_board2(item, com, panel=_quiet_panel(2))
    assert board["alert_first"]["index"] == 2
    assert [d["index"] for d in board["decisive"]] == [2, 0, 1]
    assert board["decisive"][0]["first_alert"] is True
    page = B2.render_html(board)
    assert page.count("첫 경보") == 1
    assert page.count('class="moment"') == 3


def test_no_first_alert_badge_without_alert():
    item, com = _quiet_story()
    panel = _quiet_panel(5)
    for e in panel["effects"]:
        e["readings"] = [_reading("comply", 0.1, 0)] * 3
        e["action_dist"] = {"comply": 1.0}
    board = B2.build_board2(item, com, panel=panel)
    assert board["alert_first"] is None
    assert all(not d.get("first_alert") for d in board["decisive"])
    assert "첫 경보" not in B2.render_html(board)


# --- outcome lines and case verdict ------------------------------------------

def test_no_result_line_without_outcomes():
    item, com = _quiet_story()
    page = B2.render_html(B2.build_board2(item, com))
    assert "결과 없음" not in page
    assert 'class="outcome"' not in page


def test_outcome_line_shown_with_outcomes():
    item, com = _quiet_story()
    outcomes = {"A": {"points": 17, "likeness": 1},
                "B": {"points": 12, "likeness": 2}}
    page = B2.render_html(B2.build_board2(item, com, outcomes=outcomes))
    assert "점수 17, 상대를 약간 좋아함" in page
    assert "점수 12, 상대를 매우 좋아함" in page


def test_case_verdict_uses_outcome_first_sentence():
    item, com = _quiet_story()
    case = {"title": "T", "outcome": "전원 생존했다. 기적의 불시착이었다.",
            "source_url": "", "note": ""}
    board = B2.build_board2(item, com, case=case)
    assert board["verdict"] == "전원 생존했다."
    assert board["caption"]
    assert board["verdict_sub"] == board["caption"]
    page = B2.render_html(board)
    assert '<p class="verdict">전원 생존했다.</p>' in page
    vpos = page.index('<p class="verdict">')
    cpos = page.index('<p class="sub">' + htmlmod.escape(board["caption"])
                      + "</p>")
    assert vpos < cpos


def test_case_verdict_without_outcome_falls_back_to_caption():
    item, com = _quiet_story()
    board = B2.build_board2(item, com,
                            case={"title": "T", "outcome": "",
                                  "source_url": "", "note": ""})
    assert board["verdict"] == board["caption"]
    assert board["verdict_sub"] == ""
    plain = B2.build_board2(item, com)
    assert plain["verdict"] == plain["caption"]
    assert plain["verdict_sub"] == ""
    assert plain["verdict"] in B2.render_html(plain)


# --- defaults: pairwise k, no game docs --------------------------------------

def test_board_has_no_game_link_or_section():
    item, com = _quiet_story()
    page = B2.render_html(B2.build_board2(item, com, panel=_quiet_panel(5)))
    assert "game_" not in page
    assert "게임으로 풀어 보기" not in page
    assert "uptake-game" not in page
    assert "dashboard_dlg-60-quiet.html" in page


def test_pairwise_run_default_k_is_8(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    for did in ("casino-1", "casino-2"):
        rec = {"id": did, "genre": "g", "scale": "fine100", "moves": [],
               "effects": [{"index": 0, "speaker": "A", "text": "a0"},
                           {"index": 1, "speaker": "B", "text": "b1"}]}
        (src / f"{did}.json").write_text(json.dumps(rec), encoding="utf-8")

    class _FakeMuse:
        def __init__(self, *a, **k):
            pass

        def complete(self, messages, model=None, temperature=1.0,
                     reasoning_effort=None, response_format=None):
            return '{"better": "tie", "why": "x"}'

    monkeypatch.setattr(P, "MuseClient", _FakeMuse)
    out = tmp_path / "out"
    P.main(["run", "--src", str(src), "--out", str(out),
            "--seed", "0", "--budget", "10", "--workers", "2"])
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["settings"]["k"] == 8
    assert run["n_pairs"] > 0
    out4 = tmp_path / "out4"
    P.main(["run", "--src", str(src), "--out", str(out4), "--k", "4",
            "--seed", "0", "--budget", "10", "--workers", "2"])
    run4 = json.loads((out4 / "run.json").read_text(encoding="utf-8"))
    assert run4["settings"]["k"] == 4
