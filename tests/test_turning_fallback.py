"""G2: turning points exist even without an urgent mismatch.

Fallback rules are general (steering moves / generic negotiation cues),
never case names or quoted phrases.
"""
import json

from forecaster import report as R


def _write_panel_case(root, case_id, turns, moves, effects, outcome_turn=0):
    hs = root / "highstakes"
    hs.mkdir(parents=True, exist_ok=True)
    with open(hs / f"casino-{case_id}_turns.jsonl", "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    with open(hs / f"casino-{case_id}_case.json", "w", encoding="utf-8") as f:
        json.dump({"title": "Synthetic", "outcome": "Synthetic.",
                   "source_url": "https://example.com/r",
                   "outcome_turn": outcome_turn}, f)
    panel_dir = hs / "cases" / f"casino-{case_id}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    with open(panel_dir / "panel_record.json", "w", encoding="utf-8") as f:
        json.dump({"id": f"casino-{case_id}", "genre": "synthetic",
                   "moves": moves, "effects": effects}, f)


def _mv(idx, speaker, moves, quote):
    return {"turn_index": idx, "speaker": speaker, "moves": moves,
            "quote": quote}


def _eff(idx, speaker, text, dist, actions):
    return {"index": idx, "speaker": speaker, "text": text,
            "action_dist": dist, "disagreement": 0.5,
            "n_readings": len(actions),
            "readings": [{"turn": idx, "listener": "X", "stance": "neutral",
                          "next_action": a, "relationship_shift": 0,
                          "face_threat": 0} for a in actions]}


def _check_tp_shape(tp):
    for key in ("turn", "speaker", "opened", "received_by", "received_as",
                "response_turn", "effect", "wanted_vs_happened",
                "intentional"):
        assert key in tp, key
    assert tp["opened"].strip()
    assert tp["received_as"].strip()
    assert tp["wanted_vs_happened"].strip()
    assert "wanted:" in tp["wanted_vs_happened"]
    assert "happened:" in tp["wanted_vs_happened"]
    assert tp["intentional"] in ("yes", "no", "unclear")
    assert tp["effect"] in ("uptake", "no uptake")


def test_fallback_with_moves_no_urgent(tmp_path):
    # Cooperative panel: no urgent intent anywhere, but a proposal
    # steers and the other side receives it.
    turns = [
        {"speaker": "A", "text": "How are you feeling today?"},
        {"speaker": "B", "text": "A bit tired lately."},
        {"speaker": "A", "text": "Let us order another test to check."},
        {"speaker": "B", "text": "Sure, that sounds fine."},
    ]
    moves = [_mv(0, "A", ["ask"], "feeling"),
             _mv(1, "B", ["inform"], "tired"),
             _mv(2, "A", ["inform", "propose"], "order another test"),
             _mv(3, "B", ["accept"], "sounds fine")]
    effects = [_eff(i, t["speaker"], t["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]) for i, t in enumerate(turns)]
    _write_panel_case(tmp_path, "coop", turns, moves, effects,
                       outcome_turn=3)
    r = R.build_report("coop", data_root=str(tmp_path))
    assert r["turning_points"], "expected fallback turning points"
    for tp in r["turning_points"]:
        _check_tp_shape(tp)
    assert any("propose" in tp["opened"] for tp in r["turning_points"])


def test_fallback_without_panel_negotiation(tmp_path):
    # Control-style dialogue: no panel record at all. Generic
    # negotiation cues (offer / deal markers) must still yield
    # turning points with steered/received/wanted/intent.
    hs = tmp_path / "highstakes"
    hs.mkdir(parents=True, exist_ok=True)
    turns = [
        {"speaker": "X", "text": "Hello, I would like to split things fairly."},
        {"speaker": "Y", "text": "What would you give me in return?"},
        {"speaker": "X", "text": "My final offer is two for one, take it or leave it."},
        {"speaker": "Y", "text": "No, I cannot accept that, I will walk away."},
        {"speaker": "X", "text": "Submit-Deal"},
        {"speaker": "Y", "text": "Accept-Deal"},
    ]
    with open(hs / "casino-noneg_turns.jsonl", "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    with open(hs / "casino-noneg_case.json", "w", encoding="utf-8") as f:
        json.dump({"title": "Synthetic", "outcome": "Synthetic.",
                   "source_url": "https://example.com/r",
                   "outcome_turn": 5}, f)
    # No panel_record.json on purpose.
    r = R.build_report("noneg", data_root=str(tmp_path))
    assert r["turning_points"], "expected text-inferred turning points"
    for tp in r["turning_points"]:
        _check_tp_shape(tp)


def _write_nopanel_case(root, case_id, turns, outcome_turn):
    hs = root / "highstakes"
    hs.mkdir(parents=True, exist_ok=True)
    with open(hs / f"casino-{case_id}_turns.jsonl", "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    with open(hs / f"casino-{case_id}_case.json", "w", encoding="utf-8") as f:
        json.dump({"title": "Synthetic", "outcome": "Synthetic.",
                   "source_url": "https://example.com/r",
                   "outcome_turn": outcome_turn}, f)
    # No panel_record.json on purpose.


def test_terminal_accept_is_recorded_as_reception(tmp_path):
    # G2: a terminal closing accept has no later reply, so it is not
    # its own turning point: no observed reply means no received-as
    # on record (standalone terminal entries were flagged as missing
    # their reception items at 227:#21 / 243:#38). The close is
    # recorded instead as the reception half of the proposal it
    # answers: the Submit entry links response_turn to the accept,
    # names the accept in received_as, and counts uptake.
    turns = [
        {"speaker": "X", "text": "I would like to split things fairly."},
        {"speaker": "Y", "text": "Submit-Deal"},
        {"speaker": "X", "text": "Accept-Deal"},
    ]
    _write_nopanel_case(tmp_path, "termacc", turns, outcome_turn=2)
    r = R.build_report("termacc", data_root=str(tmp_path))
    assert r["turning_points"]
    for tp in r["turning_points"]:
        _check_tp_shape(tp)
    _check_response_linkage(turns, r["turning_points"])
    by_turn = {tp["turn"]: tp for tp in r["turning_points"]}
    assert 2 not in by_turn, "terminal accept must not be standalone"
    tp = by_turn[1]
    assert tp["response_turn"] == 2
    assert tp["received_by"] == "X"
    assert "accept" in tp["received_as"]
    assert "accept" in tp["wanted_vs_happened"]
    assert tp["effect"] == "uptake"


def test_terminal_reject_is_recorded_as_reception(tmp_path):
    # Same reception rule for a refused close: the terminal reject
    # is not its own entry; the proposal it answers links to it.
    turns = [
        {"speaker": "X", "text": "My final offer is two for one."},
        {"speaker": "Y", "text": "Submit-Deal"},
        {"speaker": "X", "text": "Reject-Deal"},
    ]
    _write_nopanel_case(tmp_path, "termrej", turns, outcome_turn=2)
    r = R.build_report("termrej", data_root=str(tmp_path))
    assert r["turning_points"]
    for tp in r["turning_points"]:
        _check_tp_shape(tp)
    _check_response_linkage(turns, r["turning_points"])
    by_turn = {tp["turn"]: tp for tp in r["turning_points"]}
    assert 2 not in by_turn, "terminal reject must not be standalone"
    tp = by_turn[1]
    assert tp["response_turn"] == 2
    assert tp["received_by"] == "X"
    assert "reject" in tp["received_as"]


def test_terminal_mismatch_falls_back_to_answered_steering(tmp_path):
    # Same reception rule on the mismatch path: a mismatch with no
    # reply is skipped, and answered steering stands in, so the
    # closing challenge is shown as the reception of turn #0 (with
    # the panel read kept) rather than as its own unreceived entry.
    turns = [
        {"speaker": "A", "text": "Standing by for your call."},
        {"speaker": "B", "text": "That gauge looks wrong, check it."},
    ]
    moves = [_mv(0, "A", ["accept"], "standing by"),
             _mv(1, "B", ["challenge"], "looks wrong")]
    effects = [_eff(0, "A", turns[0]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]),
               _eff(1, "B", turns[1]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"])]
    _write_panel_case(tmp_path, "termmis", turns, moves, effects,
                       outcome_turn=1)
    r = R.build_report("termmis", data_root=str(tmp_path))
    assert r["turning_points"]
    for tp in r["turning_points"]:
        _check_tp_shape(tp)
    _check_response_linkage(turns, r["turning_points"])
    by_turn = {tp["turn"]: tp for tp in r["turning_points"]}
    assert 1 not in by_turn, "terminal mismatch must not be standalone"
    tp = by_turn[0]
    assert tp["response_turn"] == 1
    assert tp["received_by"] == "B"
    assert "proceed" in tp["received_as"]
    assert "challenge" in tp["received_as"]


def test_single_speaker_reports_no_turning_points(tmp_path):
    # Edge: one speaker only, so no steering turn is ever answered
    # by another speaker. With no reception on record anywhere,
    # the honest report is zero turning points.
    turns = [
        {"speaker": "X", "text": "I would like to split things fairly."},
        {"speaker": "X", "text": "Submit-Deal"},
    ]
    _write_nopanel_case(tmp_path, "termsolo", turns, outcome_turn=1)
    r = R.build_report("termsolo", data_root=str(tmp_path))
    assert r["turning_points"] == []


def _check_response_linkage(turns, turning_points):
    # G2 linkage invariant (general rule, both turning-point paths):
    # response_turn is always the observed forward reply -- a
    # strictly later turn spoken by received_by. Never null, never
    # backward: entries without a reply are not built.
    by_speaker = [t["speaker"] for t in turns]
    for tp in turning_points:
        t = tp["turn"]
        assert 0 <= t < len(turns)
        assert tp["speaker"] == by_speaker[t]
        rt = tp["response_turn"]
        assert isinstance(rt, int) and not isinstance(rt, bool), tp
        assert t < rt < len(turns), tp
        assert by_speaker[rt] == tp["received_by"], tp
        assert tp["received_by"] != tp["speaker"], tp


def test_response_turn_never_points_backward_nopanel(tmp_path):
    # Regression for 227:#21 / 243:#38: the terminal Accept-Deal
    # is the reception of the Submit-Deal entry, never a backward
    # pointer and never its own entry.
    turns = [
        {"speaker": "X", "text": "I would like to split things fairly."},
        {"speaker": "Y", "text": "What would you give me in return?"},
        {"speaker": "X", "text": "Submit-Deal"},
        {"speaker": "Y", "text": "Accept-Deal"},
    ]
    _write_nopanel_case(tmp_path, "linkfb", turns, outcome_turn=3)
    r = R.build_report("linkfb", data_root=str(tmp_path))
    assert r["turning_points"]
    _check_response_linkage(turns, r["turning_points"])
    by_turn = {tp["turn"]: tp for tp in r["turning_points"]}
    assert 3 not in by_turn
    assert by_turn[2]["response_turn"] == 3


def test_response_turn_never_points_backward_panel(tmp_path):
    # Same invariant on the mismatch path, with a mid-dialogue
    # mismatch (forward pointer) and a terminal mismatch (null).
    turns = [
        {"speaker": "A", "text": "We need priority, we are low on fuel."},
        {"speaker": "B", "text": "Roger, continue as cleared."},
        {"speaker": "A", "text": "We are declaring an emergency now."},
    ]
    moves = [_mv(0, "A", ["appeal_need"], "low on fuel"),
             _mv(1, "B", ["accept", "inform"], "continue as cleared"),
             _mv(2, "A", ["appeal_need"], "declaring an emergency")]
    effects = [_eff(i, t["speaker"], t["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]) for i, t in enumerate(turns)]
    _write_panel_case(tmp_path, "linkmis", turns, moves, effects,
                       outcome_turn=2)
    r = R.build_report("linkmis", data_root=str(tmp_path))
    assert r["turning_points"]
    _check_response_linkage(turns, r["turning_points"])


def test_closing_accept_is_reception_not_standalone_entry(tmp_path):
    # G2 reception rule (general, both turning-point paths): a turning
    # point pairs steering with an observed reply, so the final turn --
    # which no reply answers -- is recorded as the reception half of
    # the proposal it closes, never as its own entry. Harness evidence
    # 227:#21 / 243:#38 (terminal Accept-Deal turns) flags standalone
    # terminal entries as missing their reception items.
    turns = [
        {"speaker": "X", "text": "I would like to split things fairly."},
        {"speaker": "Y", "text": "What would you give me in return?"},
        {"speaker": "X", "text": "Submit-Deal"},
        {"speaker": "Y", "text": "Accept-Deal"},
    ]
    _write_nopanel_case(tmp_path, "closerec", turns, outcome_turn=3)
    r = R.build_report("closerec", data_root=str(tmp_path))
    assert r["turning_points"]
    for tp in r["turning_points"]:
        _check_tp_shape(tp)
    _check_response_linkage(turns, r["turning_points"])
    by_turn = {tp["turn"]: tp for tp in r["turning_points"]}
    assert 3 not in by_turn, "terminal accept must not be standalone"
    assert by_turn[2]["response_turn"] == 3
    assert by_turn[2]["received_by"] == "Y"
    assert "accept" in by_turn[2]["received_as"]
    assert by_turn[2]["effect"] == "uptake"


def test_every_turning_point_has_observed_forward_reply(tmp_path):
    # G2 linkage invariant, strict form: response_turn is always the
    # observed forward reply -- a strictly later turn spoken by
    # received_by. No nulls, no backward pointers.
    turns = [
        {"speaker": "X", "text": "I would like to split things fairly."},
        {"speaker": "Y", "text": "Submit-Deal"},
        {"speaker": "X", "text": "Accept-Deal"},
    ]
    _write_nopanel_case(tmp_path, "strictlink", turns, outcome_turn=2)
    r = R.build_report("strictlink", data_root=str(tmp_path))
    assert r["turning_points"]
    by_speaker = [t["speaker"] for t in turns]
    for tp in r["turning_points"]:
        rt = tp["response_turn"]
        assert isinstance(rt, int) and not isinstance(rt, bool), tp
        assert tp["turn"] < rt < len(turns), tp
        assert by_speaker[rt] == tp["received_by"], tp


def test_infer_text_moves_uses_general_cues_only():
    # General English negotiation vocabulary, no case content.
    assert "propose" in R.infer_text_moves("I would like to split fairly")
    assert "propose" in R.infer_text_moves("Submit-Deal")
    assert "accept" in R.infer_text_moves("Accept-Deal")
    assert "reject" in R.infer_text_moves("Reject-Deal")
    assert "reject" in R.infer_text_moves("No, I will walk away.")
    # 'know' must not trigger the 'no' reject cue.
    assert "reject" not in R.infer_text_moves("I know how you feel")
