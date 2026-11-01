"""Contract for uptake-report: one-case misreading report from the record.

Synthetic fixtures only (no private data, no network). Covers the CLI
contract (--case/--out/--budget, SPEND last line, caching), the report
schema, the general urgent-read-as-routine alert rule, and the HTML
report order/content.
"""
import json

import pytest

from forecaster import report as R


def _write_case(root, case_id="t1", n_extra=1):
    hs = root / "highstakes"
    hs.mkdir(parents=True, exist_ok=True)
    turns = [
        {"speaker": "A", "text": "We need priority, we are low on fuel."},
        {"speaker": "B", "text": "Roger, continue as cleared."},
        {"speaker": "A", "text": "Confirming we continue."},
    ]
    for i in range(n_extra):
        turns.append({"speaker": "B" if i % 2 else "A",
                      "text": f"Routine follow-up {i}."})
    with open(hs / f"casino-{case_id}_turns.jsonl", "w",
              encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    with open(hs / f"casino-{case_id}_case.json", "w",
              encoding="utf-8") as f:
        json.dump({"title": "Synthetic fuel case",
                   "outcome": "Synthetic outcome.",
                   "source_url": "https://example.com/record",
                   "outcome_turn": 2}, f)

    def eff(idx, speaker, dist, readings):
        return {"index": idx, "speaker": speaker,
                "text": turns[idx]["text"], "action_dist": dist,
                "disagreement": 0.5, "n_readings": len(readings),
                "readings": readings}

    def reading(turn, listener, action, rel=0, face=0, stance="neutral"):
        return {"turn": turn, "listener": listener, "stance": stance,
                "next_action": action, "relationship_shift": rel,
                "face_threat": face}

    moves = [
        {"turn_index": 0, "speaker": "A",
         "moves": ["inform", "appeal_need"], "quote": "need priority"},
        {"turn_index": 1, "speaker": "B", "moves": ["accept", "inform"],
         "quote": "continue as cleared"},
        {"turn_index": 2, "speaker": "A", "moves": ["accept"],
         "quote": "confirming"},
    ]
    effects = [
        eff(0, "A", {"proceed": 0.5, "comply": 0.5},
            [reading(0, "B", "proceed"), reading(0, "B", "comply")]),
        eff(1, "B", {"proceed": 1.0},
            [reading(1, "A", "proceed"), reading(1, "A", "proceed")]),
        eff(2, "A", {"proceed": 1.0},
            [reading(2, "B", "proceed"), reading(2, "B", "proceed")]),
    ]
    for i in range(n_extra):
        idx = 3 + i
        speaker = turns[idx]["speaker"]
        moves.append({"turn_index": idx, "speaker": speaker,
                      "moves": ["inform"], "quote": f"follow-up {i}"})
        effects.append(eff(idx, speaker, {"proceed": 1.0},
                           [reading(idx, "A", "proceed"),
                            reading(idx, "B", "proceed")]))
    panel_dir = hs / "cases" / f"casino-{case_id}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    with open(panel_dir / "panel_record.json", "w", encoding="utf-8") as f:
        json.dump({"id": f"casino-{case_id}", "genre": "aviation_atc",
                   "moves": moves, "effects": effects}, f)
    return root


def _run(case_id, root, out, budget=5.0):
    R.main(["--case", case_id, "--out", str(out),
            "--budget", str(budget), "--data-root", str(root)])


# --- CLI contract ---------------------------------------------------------------

def test_cli_writes_json_html_and_spend_last(tmp_path, capsys):
    _write_case(tmp_path)
    out = tmp_path / "out"
    _run("t1", tmp_path, out)
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith("SPEND=")
    assert float(lines[-1].split("=", 1)[1]) == 0.0
    assert (out / "report.json").exists()
    assert (out / "report.html").exists()


def test_cli_cache_hit_reuses_report(tmp_path, capsys):
    _write_case(tmp_path)
    out = tmp_path / "out"
    _run("t1", tmp_path, out)
    first = (out / "report.json").read_text(encoding="utf-8")
    capsys.readouterr()
    _run("t1", tmp_path, out)
    lines = capsys.readouterr().out.strip().splitlines()
    assert "cached" in lines[0]
    assert lines[-1] == "SPEND=0.00"
    assert (out / "report.json").read_text(encoding="utf-8") == first


def test_cli_unknown_case_exits_with_spend(tmp_path, capsys):
    (tmp_path / "highstakes").mkdir()
    (tmp_path / "datasets").mkdir()
    with pytest.raises(SystemExit) as exc:
        _run("nope", tmp_path, tmp_path / "out")
    assert exc.value.code == 1
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith("SPEND=")


# --- schema -----------------------------------------------------------------------

def _report(tmp_path):
    _write_case(tmp_path)
    out = tmp_path / "out"
    _run("t1", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        return json.load(f)


def test_report_schema(tmp_path):
    r = _report(tmp_path)
    for key in ("id", "title", "domain", "result_type", "outcome",
                "source_url", "n_turns", "alerts", "decisive",
                "alternatives", "perspective", "reliability",
                "speakers", "turning_points"):
        assert key in r, key
    assert r["n_turns"] == 4
    for a in r["alerts"]:
        assert set(("turn", "level", "axis", "reason", "resolved",
                    "basis", "lead_turns")) <= set(a)
        assert a["level"] in ("advisory", "caution", "warning", "critical")
        assert a["basis"] in ("before_reply", "after_reply")
    d = r["decisive"]
    for key in ("turn", "quote", "intended", "intended_source", "said",
                "read", "read_prob", "scales"):
        assert key in d, key
    assert set(("next_action", "relationship", "face")) <= set(d["scales"])
    assert len(r["alternatives"]) >= 2
    for alt in r["alternatives"]:
        assert set(("text", "p_better", "n_compared", "ci_low",
                    "ci_high")) <= set(alt)
    p = r["perspective"]
    assert p["line"] and p["work"] and p["source"]
    rel = r["reliability"]
    for key in ("split_half", "k", "outcome_r1", "perturb", "limits",
                "measured_on"):
        assert key in rel, key
    assert rel["limits"]
    assert rel["measured_on"]
    assert r["speakers"], "speakers must not be empty"
    for sp in r["speakers"]:
        for key in ("speaker", "initial_goal", "goal_source", "shifts",
                    "end_balance", "end_basis"):
            assert key in sp, key
        assert -1.0 <= sp["end_balance"] <= 1.0
        for sh in sp["shifts"]:
            for key in ("turn", "from", "to", "evidence_quote"):
                assert key in sh, key
    for tp in r["turning_points"]:
        for key in ("turn", "speaker", "opened", "received_by",
                    "received_as", "response_turn", "effect",
                    "wanted_vs_happened", "intentional"):
            assert key in tp, key
        assert tp["intentional"] in ("yes", "no", "unclear")


# --- alert rule ---------------------------------------------------------------------

def test_urgent_routine_unchanged_is_warning(tmp_path):
    r = _report(tmp_path)
    by_turn = {a["turn"]: a for a in r["panel_alerts"]}
    assert by_turn[0]["level"] == "warning"
    assert by_turn[0]["axis"] == "urgent-read-as-routine"
    assert by_turn[0]["resolved"] is False


def test_decisive_is_strongest_mismatch(tmp_path):
    r = _report(tmp_path)
    assert r["decisive"]["turn"] == 0
    assert "appeal_need" in r["decisive"]["intended"]
    assert r["decisive"]["intended_source"].startswith("model estimate:")


def test_decisive_intended_source_is_honest_about_model_labels(tmp_path):
    # C4: move labels are model judgments, never record facts. The
    # decisive intended_source must say so up front, and the three
    # scales (next action, relationship, face) must all be present.
    r = _report(tmp_path)
    d = r["decisive"]
    assert d["intended"] and d["said"] and d["read"]
    assert d["intended_source"].startswith("model estimate:")
    assert "model judgment" in d["intended_source"]
    assert set(("next_action", "relationship", "face")) <= set(d["scales"])


def test_decisive_intended_source_without_panel_is_model_estimate():
    turns = [{"speaker": "A", "text": "Hello."}]
    d = R.build_decisive("t9", turns, {}, {}, {}, [])
    assert d["intended_source"].startswith("model estimate:")


# --- decisive turn from the per-item readings -----------------------------------

_GRADES = {"status": "measured", "cuts": [11.3, 21.8, 30.8, 35.1], "causes": {},
           "items": {"deliverable": "measured", "facts": "measured", "warmth": "reference"}}


def _item(cls, wanted, actual, spread=15.0, reply=None, quote=""):
    return {"class": cls,
            "wanted": {"score": wanted, "spread": spread} if wanted is not None else None,
            "actual": {"score": actual, "spread": spread} if actual is not None else None,
            "reply_turn": reply, "quote": quote}


def _reading(turn, speaker="A", **items):
    from forecaster.multilayer import ITEM_CLASS
    return {"turn": turn, "speaker": speaker, "broken": [],
            "items": {k: _item(ITEM_CLASS[k], *v) for k, v in items.items()}}


def _layered(readings, weight=100.0):
    from forecaster.multilayer import build_alerts_items
    build_alerts_items(readings, {"stakes_weight": weight}, _GRADES)
    return readings


_USE = {"deliverable", "facts"}
_TURNS = [{"speaker": s, "text": f"Plain words of turn {i}."}
          for i, s in enumerate("ABABA")]


def test_layered_pick_takes_largest_severity():
    readings = _layered([_reading(0, facts=(50.0, 50.0)),
                         _reading(1, deliverable=(80.0, 20.0)),
                         _reading(2, facts=(60.0, 40.0))])
    severities = [r["layers"]["severity"] for r in readings]
    assert severities[1] == max(severities) and severities.count(severities[1]) == 1
    assert R.pick_decisive_layered(readings, _USE) == (1, "deliverable")


def test_layered_pick_tie_goes_to_earlier_turn():
    readings = _layered([_reading(0, facts=(10.0, 90.0)),
                         _reading(1, facts=(70.0, 30.0)),
                         _reading(2, facts=(70.0, 30.0))])
    assert readings[1]["layers"]["severity"] == readings[2]["layers"]["severity"]
    assert R.pick_decisive_layered(readings, _USE)[0] == 1


def test_layered_item_is_the_gap_behind_severity_and_measured_only():
    # A reference item with a bigger gap never drives the pick or the text.
    readings = _layered([_reading(0, warmth=(95.0, 5.0), facts=(60.0, 50.0),
                                  deliverable=(70.0, 30.0))])
    assert R.top_measured_item(readings[0], _USE) == "deliverable"
    assert readings[0]["layers"]["miss"] == 49.0  # 70 x (100 - 30) / 100


def test_layered_pick_ignores_turns_with_only_reference_or_half_values():
    readings = _layered([_reading(0, warmth=(95.0, 5.0)),
                         _reading(1, facts=(80.0, None)),
                         _reading(2, facts=(50.0, 50.0))])
    assert R.pick_decisive_layered(readings, _USE)[0] == 2


def test_layered_pick_is_none_without_values():
    from forecaster.multilayer import blank_reading
    turns = _TURNS[:2]
    readings = _layered([blank_reading(turns, 0)])
    assert R.pick_decisive_layered(readings, _USE) is None
    assert R.pick_decisive_layered([], _USE) is None


def test_build_decisive_layered_fields_come_from_the_turn_and_item():
    readings = _layered([_reading(0, facts=(50.0, 50.0)),
                         _reading(1, deliverable=(82.6, 19.2, 15.0, 2, "say again"))])
    moves = {1: {"moves": ["pressure"], "quote": "not in this turn at all"}}
    effects = {1: {"action_dist": {"comply": 0.75, "proceed": 0.25},
                   "readings": [{"next_action": "comply", "relationship_shift": 1,
                                 "face_threat": 2}]}}
    d = R.build_decisive("t1", _TURNS, {}, moves, effects, [], uptake=readings, use=_USE)
    assert d["turn"] == 1
    assert d["said"] == _TURNS[1]["text"]
    assert d["quote"] in d["said"]
    assert d["intended"].startswith("deliverable: reached for 83/100")
    assert d["read"] == ("deliverable: the wanted effect landed 19/100 "
                         "(reply #2: say again)")
    assert d["intended_source"].startswith("model estimate:")
    assert "model judgment" in d["intended_source"]
    assert d["read_prob"] is None
    assert d["scales"] == {"next_action": "comply", "relationship": 1.0, "face": -2.0}


def test_build_decisive_keeps_a_verbatim_panel_quote():
    readings = _layered([_reading(1, deliverable=(80.0, 20.0))])
    moves = {1: {"moves": ["inform"], "quote": "words of turn"}}
    d = R.build_decisive("t1", _TURNS, {}, moves, {}, [], uptake=readings, use=_USE)
    assert d["quote"] == "words of turn"


def test_report_uses_layered_turn_for_alternatives_and_html(tmp_path, monkeypatch):
    import forecaster.multilayer as ML
    _write_case(tmp_path)
    real = ML.build_multilayer

    def fake(turns, **kwargs):
        got = real(turns, grades=_GRADES)
        got["uptake"][2]["items"]["facts"] = _item("understanding", 80.0, 20.0)
        return {**got, "alerts": ML.build_alerts_items(got["uptake"], {"stakes_weight": 90.0}, _GRADES)}

    monkeypatch.setattr(ML, "build_multilayer", fake)
    r = R.build_report("t1", data_root=str(tmp_path))
    d = r["decisive"]
    # The panel rule alone picks turn 0 on this case.
    assert d["turn"] == 2
    top = max(r["uptake"], key=lambda u: u["layers"]["severity"])
    assert top["turn"] == d["turn"]
    assert d["said"] == "Confirming we continue."
    assert d["quote"] == "Confirming we continue."
    assert d["intended"].startswith("facts: reached for 80/100")
    assert d["read"].startswith("facts: the wanted effect landed 20/100")
    assert d["scales"]["next_action"] == "proceed"
    for alt in r["alternatives"]:
        assert d["quote"] in alt["text"]
    assert d["quote"] in r["alternatives_control"]["worse"]["text"]
    page = R.render_html(r)
    assert "Decisive turn #2." in page
    assert "None" not in page.split('id="intent"')[1].split("</section>")[0]


def test_report_without_readings_keeps_the_panel_rule(tmp_path):
    r = _report(tmp_path)
    assert r["decisive"]["turn"] == 0
    assert "appeal_need" in r["decisive"]["intended"]
    assert r["decisive"]["read"] in ("proceed", "comply")
    assert r["decisive"]["read_prob"] == 0.5
    assert all(u["layers"]["severity"] == 0.0 for u in r["uptake"])


def test_engaged_response_downgrades_to_caution():
    eff = {"action_dist": {"proceed": 0.6, "comply": 0.4}}
    level, _, mismatch, unchanged = R.score_turn(
        ["challenge"], eff, ["propose", "accept"])
    assert (level, mismatch, unchanged) == ("caution", True, False)
    level, _, _, _ = R.score_turn(["challenge"], eff, ["accept", "inform"])
    assert level == "warning"


def test_routine_intent_never_alerts():
    eff = {"action_dist": {"proceed": 1.0}}
    level, _, mismatch, _ = R.score_turn(["inform"], eff, ["accept"])
    assert level is None and mismatch is False


def test_urgent_read_as_escalation_is_quiet():
    eff = {"action_dist": {"escalate": 0.8, "proceed": 0.2}}
    level, _, mismatch, _ = R.score_turn(["appeal_need"], eff, ["accept"])
    assert level is None and mismatch is False


def test_relationship_scores_do_not_trigger():
    eff = {"action_dist": {"proceed": 1.0},
           "readings": [{"next_action": "proceed", "relationship_shift": -2,
                         "face_threat": 2}]}
    level, _, mismatch, _ = R.score_turn(["inform"], eff, ["accept"])
    assert level is None and mismatch is False


def test_repeated_press_escalates_to_warning():
    eff = {"action_dist": {"proceed": 0.6, "comply": 0.4}}
    level, _, mismatch, unchanged = R.score_turn(
        ["challenge"], eff, ["propose", "accept"], repeated=True)
    assert (level, mismatch, unchanged) == ("warning", True, False)
    thin = {"action_dist": {"proceed": 0.4, "escalate": 0.6}}
    level, _, mismatch, _ = R.score_turn(
        ["challenge"], thin, ["propose"], repeated=True)
    assert (level, mismatch) == ("advisory", False)


def _write_custom_case(root, case_id, turns, moves, effects, outcome_turn=0,
                       genre="synthetic"):
    hs = root / "highstakes"
    hs.mkdir(parents=True, exist_ok=True)
    with open(hs / f"casino-{case_id}_turns.jsonl", "w",
              encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    with open(hs / f"casino-{case_id}_case.json", "w",
              encoding="utf-8") as f:
        json.dump({"title": "Synthetic case", "outcome": "Synthetic outcome.",
                   "source_url": "https://example.com/record",
                   "outcome_turn": outcome_turn}, f)
    panel_dir = hs / "cases" / f"casino-{case_id}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    with open(panel_dir / "panel_record.json", "w", encoding="utf-8") as f:
        json.dump({"id": f"casino-{case_id}", "genre": genre,
                   "moves": moves, "effects": effects}, f)
    return root


def _mv(idx, speaker, moves, quote, addressed_to=None):
    m = {"turn_index": idx, "speaker": speaker, "moves": moves,
         "quote": quote}
    if addressed_to:
        m["addressed_to"] = addressed_to
    return m


def _eff(idx, speaker, text, dist, actions):
    stances = ("neutral", "trusting", "suspicious")
    return {"index": idx, "speaker": speaker, "text": text,
            "action_dist": dist, "disagreement": 0.5,
            "n_readings": len(actions),
            "readings": [{"turn": idx, "listener": "X",
                          "stance": stances[i % 3], "next_action": a,
                          "relationship_shift": 0, "face_threat": 0}
                         for i, a in enumerate(actions)]}


def test_repeat_press_warns_in_report(tmp_path):
    turns = [
        {"speaker": "A", "text": "We need priority now."},
        {"speaker": "B", "text": "What is your alternate?"},
        {"speaker": "A", "text": "Confirm priority, we are urgent."},
        {"speaker": "B", "text": "Roger."},
    ]
    moves = [_mv(0, "A", ["appeal_need"], "need priority"),
             _mv(1, "B", ["probe"], "alternate"),
             _mv(2, "A", ["appeal_need"], "confirm priority"),
             _mv(3, "B", ["accept"], "roger")]
    effects = [_eff(0, "A", turns[0]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]),
               _eff(1, "B", turns[1]["text"], {"comply": 1.0},
                    ["comply", "comply"]),
               _eff(2, "A", turns[2]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]),
               _eff(3, "B", turns[3]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"])]
    _write_custom_case(tmp_path, "rep", turns, moves, effects,
                       outcome_turn=3)
    out = tmp_path / "out"
    _run("rep", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_turn = {a["turn"]: a for a in r["panel_alerts"]}
    assert by_turn[0]["level"] == "warning"
    assert "press again at #2" in by_turn[0]["reason"]
    assert by_turn[0]["resolved"] is False


def test_delayed_uptake_resolves_original_press(tmp_path):
    # 1549-style persistence: the first press needs a repeat, but the
    # repeat is taken up (options offered). The issue is resolved, with
    # the delay disclosed in the reason.
    turns = [
        {"speaker": "A", "text": "We need priority, fuel low."},
        {"speaker": "B", "text": "Stand by."},
        {"speaker": "A", "text": "Confirm priority, urgent."},
        {"speaker": "B", "text": "Cleared direct, what do you need?"},
        {"speaker": "A", "text": "Direct, thank you."},
    ]
    moves = [_mv(0, "A", ["appeal_need"], "need priority"),
             _mv(1, "B", ["inform"], "stand by"),
             _mv(2, "A", ["appeal_need"], "confirm priority"),
             _mv(3, "B", ["propose"], "cleared direct"),
             _mv(4, "A", ["accept"], "thank you")]
    effects = [_eff(i, t["speaker"], t["text"], {"proceed": 1.0},
                    ["proceed", "proceed"])
               for i, t in enumerate(turns)]
    _write_custom_case(tmp_path, "delayed", turns, moves, effects,
                       outcome_turn=4)
    out = tmp_path / "out"
    _run("delayed", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_turn = {a["turn"]: a for a in r["panel_alerts"]}
    assert by_turn[0]["level"] == "warning"
    assert "press again at #2" in by_turn[0]["reason"]
    assert by_turn[0]["resolved"] is True
    assert by_turn[2]["resolved"] is True


def test_concede_alone_does_not_resolve(tmp_path):
    # Dismissal-style yielding ("maybe it is") without a new course is
    # not uptake: the warning stays unresolved.
    turns = [
        {"speaker": "A", "text": "That reading looks wrong."},
        {"speaker": "B", "text": "Checked, continue."},
        {"speaker": "B", "text": "Maybe it is fine."},
        {"speaker": "A", "text": "Roger."},
    ]
    moves = [_mv(0, "A", ["challenge"], "looks wrong"),
             _mv(1, "B", ["inform"], "continue"),
             _mv(2, "B", ["concede"], "maybe fine"),
             _mv(3, "A", ["accept"], "roger")]
    effects = [_eff(i, t["speaker"], t["text"], {"proceed": 1.0},
                    ["proceed", "proceed"])
               for i, t in enumerate(turns)]
    _write_custom_case(tmp_path, "dismiss", turns, moves, effects,
                       outcome_turn=3)
    out = tmp_path / "out"
    _run("dismiss", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_turn = {a["turn"]: a for a in r["panel_alerts"]}
    assert by_turn[0]["level"] == "warning"
    assert by_turn[0]["resolved"] is False


def test_same_speaker_propose_does_not_resolve(tmp_path):
    # The speaker's own continued proposing is not uptake: only the
    # other side can take the warning up.
    turns = [
        {"speaker": "A", "text": "That reading looks wrong."},
        {"speaker": "A", "text": "I propose we recheck it now."},
        {"speaker": "B", "text": "Checked, continue."},
        {"speaker": "B", "text": "Continue."},
    ]
    moves = [_mv(0, "A", ["challenge"], "looks wrong"),
             _mv(1, "A", ["propose"], "recheck it"),
             _mv(2, "B", ["inform"], "continue"),
             _mv(3, "B", ["inform"], "continue")]
    effects = [_eff(0, "A", turns[0]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]),
               _eff(1, "A", turns[1]["text"], {"escalate": 1.0},
                    ["escalate", "escalate"]),
               _eff(2, "B", turns[2]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]),
               _eff(3, "B", turns[3]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"])]
    _write_custom_case(tmp_path, "selfprop", turns, moves, effects,
                       outcome_turn=3)
    out = tmp_path / "out"
    _run("selfprop", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_turn = {a["turn"]: a for a in r["panel_alerts"]}
    assert by_turn[0]["level"] == "warning"
    assert by_turn[0]["resolved"] is False


def test_hostile_repair_read_does_not_resolve(tmp_path):
    # A panel repair guess against a hostile turn (reject + pressure)
    # is contradicted by the record and does not resolve the alert.
    turns = [
        {"speaker": "A", "text": "We need priority now."},
        {"speaker": "B", "text": "Denied, hold your course."},
        {"speaker": "B", "text": "Continue as cleared."},
    ]
    moves = [_mv(0, "A", ["appeal_need"], "need priority"),
             _mv(1, "B", ["reject", "pressure"], "denied"),
             _mv(2, "B", ["inform"], "continue")]
    effects = [_eff(0, "A", turns[0]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]),
               _eff(1, "B", turns[1]["text"],
                    {"repair": 2.0 / 3.0, "withdraw": 1.0 / 3.0},
                    ["repair", "repair", "withdraw"]),
               _eff(2, "B", turns[2]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"])]
    _write_custom_case(tmp_path, "hostile", turns, moves, effects,
                       outcome_turn=2)
    out = tmp_path / "out"
    _run("hostile", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_turn = {a["turn"]: a for a in r["panel_alerts"]}
    assert by_turn[0]["resolved"] is False


def test_continued_appeal_inherits_urgency(tmp_path):
    turns = [
        {"speaker": "A", "text": "That gauge looks wrong."},
        {"speaker": "A", "text": "Really not right."},
        {"speaker": "B", "text": "It is fine, continue."},
        {"speaker": "A", "text": "Check it again."},
        {"speaker": "B", "text": "Checked."},
    ]
    moves = [_mv(0, "A", ["challenge"], "looks wrong"),
             _mv(1, "A", ["other"], "not right"),
             _mv(2, "B", ["challenge"], "it is fine"),
             _mv(3, "A", ["challenge"], "check again"),
             _mv(4, "B", ["accept"], "checked")]
    effects = [_eff(i, t["speaker"], t["text"], {"proceed": 1.0},
                    ["proceed", "proceed"])
               for i, t in enumerate(turns)]
    _write_custom_case(tmp_path, "cont", turns, moves, effects,
                       outcome_turn=4)
    out = tmp_path / "out"
    _run("cont", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_turn = {a["turn"]: a for a in r["panel_alerts"]}
    assert by_turn[1]["level"] == "warning"
    assert "continuation of #0" in by_turn[1]["reason"]
    assert "press again at #3" in by_turn[1]["reason"]


def test_concession_does_not_inherit_urgency():
    turns = [{"speaker": "A", "text": "x"},
             {"speaker": "A", "text": "maybe it is fine"}]
    moves = {0: {"moves": ["challenge"]}, 1: {"moves": ["concede"]}}
    effects = {0: {"action_dist": {"proceed": 1.0}},
               1: {"action_dist": {"proceed": 1.0}}}
    assert R.inherited_urgent(moves, effects, turns, 1) is None
    moves[1] = {"moves": ["other"]}
    assert R.inherited_urgent(moves, effects, turns, 1) == (0, ["challenge"])
    moves[1] = {"moves": ["accept"]}
    assert R.inherited_urgent(moves, effects, turns, 1) is None


def test_routine_read_then_countered_warns(tmp_path):
    turns = [
        {"speaker": "A", "text": "We are set, going ahead."},
        {"speaker": "B", "text": "Stand by, wait for my call."},
        {"speaker": "A", "text": "Standing by."},
    ]
    moves = [_mv(0, "A", ["accept", "inform"], "going ahead",
                 addressed_to="B"),
             _mv(1, "B", ["deflect", "inform"], "stand by"),
             _mv(2, "A", ["accept"], "standing by")]
    effects = [_eff(0, "A", turns[0]["text"],
                    {"proceed": 2.0 / 3.0, "refuse": 1.0 / 3.0},
                    ["proceed", "proceed", "refuse"]),
               _eff(1, "B", turns[1]["text"], {"comply": 1.0},
                    ["comply", "comply"]),
               _eff(2, "A", turns[2]["text"], {"comply": 1.0},
                    ["comply", "comply"])]
    _write_custom_case(tmp_path, "r2", turns, moves, effects,
                       outcome_turn=2)
    out = tmp_path / "out"
    _run("r2", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_turn = {a["turn"]: a for a in r["panel_alerts"]}
    assert by_turn[0]["level"] == "warning"
    assert by_turn[0]["axis"] == "routine-read-then-countered"
    assert "#1 (deflect)" in by_turn[0]["reason"]


def test_countered_routine_gates():
    turns = [{"speaker": "A", "text": "x"},
             {"speaker": "B", "text": "y"}]
    row = {"share": 1.0}
    moves = {0: {"moves": ["inform"], "addressed_to": "B"},
             1: {"moves": ["deflect"]}}
    effects = {0: {"action_dist": {"proceed": 1.0}}}
    assert R.countered_routine(moves, effects, turns, 0, row) is None
    effects = {0: {"action_dist": {"proceed": 0.6, "refuse": 0.4}}}
    hit = R.countered_routine(moves, effects, turns, 0, row)
    assert hit is not None and hit[0] == 1
    moves[1] = {"moves": ["inform"]}
    assert R.countered_routine(moves, effects, turns, 0, row) is None
    moves[1] = {"moves": ["deflect"]}
    moves[0] = {"moves": ["challenge"], "addressed_to": "B"}
    assert R.countered_routine(moves, effects, turns, 0, row) is None


def _designator_case(turns_text):
    turns = [{"speaker": "B" if i % 2 == 0 else "A", "text": text}
             for i, text in enumerate(turns_text)]
    moves = [_mv(i, t["speaker"], ["inform"], t["text"][:24])
             for i, t in enumerate(turns)]
    effects = [_eff(i, t["speaker"], t["text"], {"proceed": 1.0},
                    ["proceed", "proceed"])
               for i, t in enumerate(turns)]
    return turns, moves, effects


def test_designator_variant_warns_and_resolves(tmp_path):
    turns, moves, effects = _designator_case([
        "Cactus 1549, turn left heading 270.",
        "Ah, Cactus 1539, we lost thrust.",
        "Cactus 1529, say your intentions.",
        "All right, Cactus 1549 returning.",
    ])
    _write_custom_case(tmp_path, "design", turns, moves, effects,
                       outcome_turn=3, genre="aviation_atc")
    out = tmp_path / "out"
    _run("design", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_turn = {a["turn"]: a for a in r["panel_alerts"]}
    assert by_turn[1]["level"] == "warning"
    assert by_turn[1]["axis"] == "designator-inconsistency"
    assert by_turn[1]["resolved"] is True
    assert "#3" in by_turn[1]["reason"]
    assert by_turn[2]["resolved"] is True


def test_designator_variant_unresolved_when_never_corrected(tmp_path):
    turns, moves, effects = _designator_case([
        "KLM 4805 is on the runway.",
        "KLM 8705, you are cleared.",
        "Roger, rolling.",
    ])
    _write_custom_case(tmp_path, "sic", turns, moves, effects,
                       outcome_turn=2, genre="aviation_atc")
    out = tmp_path / "out"
    _run("sic", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_turn = {a["turn"]: a for a in r["panel_alerts"]}
    assert by_turn[1]["axis"] == "designator-inconsistency"
    assert by_turn[1]["resolved"] is False
    assert "never returns" in by_turn[1]["reason"]


def test_designator_rule_quiet_when_consistent(tmp_path):
    turns, moves, effects = _designator_case([
        "Avianca 052, confirm your alternate.",
        "Confirming Avianca 052, no alternate.",
        "Avianca 052, cleared to land.",
    ])
    _write_custom_case(tmp_path, "steady", turns, moves, effects,
                       outcome_turn=2, genre="aviation_atc")
    out = tmp_path / "out"
    _run("steady", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    assert [a for a in r["alerts"]
            if a["axis"] == "designator-inconsistency"] == []


def test_designator_rule_inert_outside_atc(tmp_path):
    turns, moves, effects = _designator_case([
        "Cactus 1549, turn left heading 270.",
        "Ah, Cactus 1539, we lost thrust.",
        "All right, Cactus 1549 returning.",
    ])
    _write_custom_case(tmp_path, "nogenre", turns, moves, effects,
                       outcome_turn=2, genre="diplomacy")
    out = tmp_path / "out"
    _run("nogenre", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    assert [a for a in r["alerts"]
            if a["axis"] == "designator-inconsistency"] == []


def test_routine_instruction_numbers_do_not_confuse(tmp_path):
    turns, moves, effects = _designator_case([
        "Heading 220, maintain.",
        "Heading 270 now.",
        "Runway 13 in sight.",
        "The 325 beacon passed.",
    ])
    _write_custom_case(tmp_path, "headings", turns, moves, effects,
                       outcome_turn=3, genre="aviation_atc")
    out = tmp_path / "out"
    _run("headings", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    assert [a for a in r["alerts"]
            if a["axis"] == "designator-inconsistency"] == []


# --- alternatives ---------------------------------------------------------------------

def test_alternatives_uncertain_with_interval(tmp_path):
    r = _report(tmp_path)
    for alt in r["alternatives"]:
        assert 0.0 < alt["p_better"] < 1.0
        assert alt["n_compared"] >= 2
        assert alt["ci_low"] <= alt["p_better"] <= alt["ci_high"]


# --- no hardcoded case phrases --------------------------------------------------------

def test_report_module_has_no_case_phrases():
    from pathlib import Path
    src = Path(R.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "avianca052", "airflorida90", "tenerife1977", "lear_1_1",
        "usair1549", "sejong_1444", "reykjavik1986", "aci_D2N068",
        '"227"', '"243"',
        "Avianca", "Tenerife", "Cordelia", "Lear", "KLM", "Cove Neck",
        "Hudson", "Nothing, my lord", "takeoff",
        "CaSiNo 227", "CaSiNo 243",
    ):
        assert forbidden not in src, forbidden


def test_case_meta_loads_from_data_file():
    meta = R.load_case_meta()
    assert meta["avianca052"]["domain"] == "atc"
    assert meta["avianca052"]["result_type"] == "tragedy"
    assert meta["usair1549"]["result_type"] == "success"
    assert meta["227"]["result_type"] == "control"


# --- HTML -----------------------------------------------------------------------------

def test_html_order_notice_and_source(tmp_path):
    _write_case(tmp_path)
    out = tmp_path / "out"
    _run("t1", tmp_path, out)
    page = (out / "report.html").read_text(encoding="utf-8")
    assert '<html lang="en">' in page
    heads = ["What happened", "goals and strategy shifts",
             "turning points", "Alert timeline", "Intended, said, read",
             "said differently", "perspective", "Plus and minus",
             "trust this"]
    idxs = [page.find(h) for h in heads]
    assert all(i >= 0 for i in idxs), heads
    assert idxs == sorted(idxs)
    assert "reads the utterance, not the person" in page
    assert "https://example.com/record" in page
    assert "관계 보드" not in page
    for banned in ("체스", "바둑", "형세", "관계보드", "게임 보드"):
        assert banned not in page, banned
    assert "chess" not in page.lower()
    assert "relationship board" not in page.lower()


def test_html_goals_and_balance_render_speakers(tmp_path):
    _write_case(tmp_path)
    out = tmp_path / "out"
    _run("t1", tmp_path, out)
    page = (out / "report.html").read_text(encoding="utf-8")
    assert "Initial goal" in page
    assert "Plus and minus" in page
    assert "not a score of the person" in page


def test_html_english_first_no_korean_ui(tmp_path):
    import re
    _write_case(tmp_path)
    out = tmp_path / "out"
    _run("t1", tmp_path, out)
    page = (out / "report.html").read_text(encoding="utf-8")
    body = re.sub(r"<blockquote>.*?</blockquote>", "", page, flags=re.S)
    assert not re.search(r"[\uac00-\ud7a3]", body)


def test_perspective_work_is_english(tmp_path):
    r = _report(tmp_path)
    import re
    assert not re.search(r"[\uac00-\ud7a3]", r["perspective"]["work"])


def test_speakers_goals_and_balance(tmp_path):
    r = _report(tmp_path)
    by_sp = {sp["speaker"]: sp for sp in r["speakers"]}
    assert set(by_sp) == {"A", "B"}
    # The goal itself is a sentence (a move label is not a goal); the
    # first labeled move stays cited in the source, not the sentence.
    assert "appeal_need" not in by_sp["A"]["initial_goal"]
    assert "appeal_need" in by_sp["A"]["goal_source"]
    assert by_sp["A"]["goal_source"].startswith("model estimate:")
    assert by_sp["A"]["evidence_quote"] in (
        "We need priority, we are low on fuel.")
    assert by_sp["A"]["end_balance"] == -1.0
    assert by_sp["B"]["end_balance"] == 0.0


def test_speaker_quotes_are_verbatim(tmp_path):
    import json
    _write_case(tmp_path)
    texts = []
    with open(tmp_path / "highstakes" / "casino-t1_turns.jsonl",
              encoding="utf-8") as f:
        for line in f:
            texts.append(json.loads(line)["text"])
    out = tmp_path / "out"
    _run("t1", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    for sp in r["speakers"]:
        assert sp["evidence_quote"] in texts
        for sh in sp["shifts"]:
            assert sh["evidence_quote"] in texts
    for tp in r["turning_points"]:
        assert tp["quote"] in texts


def test_turning_points_contrast_wanted_and_happened(tmp_path):
    r = _report(tmp_path)
    assert r["turning_points"], "expected a mismatch turning point"
    tp = r["turning_points"][0]
    assert tp["turn"] == 0
    assert tp["speaker"] == "A"
    assert "appeal_need" in tp["opened"]
    assert tp["received_by"] == "B"
    assert "proceed" in tp["received_as"] or "comply" in tp["received_as"]
    assert tp["response_turn"] == 1
    assert tp["effect"] in ("uptake", "no uptake")
    assert "wanted:" in tp["wanted_vs_happened"]
    assert "happened:" in tp["wanted_vs_happened"]


# --- turning points from the per-item readings ----------------------------------

_TP_TURNS = [{"speaker": "A", "text": "Please send the signed form today."},
             {"speaker": "B", "text": "The form is with legal, I cannot send it."},
             {"speaker": "A", "text": "Then tell me who has it."},
             {"speaker": "B", "text": "Someone over there, I think."},
             {"speaker": "A", "text": "I still need the form."},
             {"speaker": "B", "text": "Noted."}]


def _tp_points(readings, weight=100.0, turns=_TP_TURNS, use=_USE):
    from forecaster.multilayer import build_alerts_items, build_standing_items, broken_items
    for r in readings:
        r["broken"] = broken_items(r["items"])
    alerts = build_alerts_items(readings, {"stakes_weight": weight}, _GRADES)
    standing = build_standing_items(readings, use=use)
    return R.build_reading_turning_points(turns, readings, alerts, standing, use), alerts, standing


def test_alert_turn_becomes_a_turning_point_from_its_item_values():
    points, alerts, _ = _tp_points([
        _reading(0, deliverable=(82.6, 19.2, 15.0, 1, "I cannot send it")),
        _reading(1, speaker="B", facts=(70.0, 90.0))])
    assert [a["turn"] for a in alerts] == [0]
    assert [p["turn"] for p in points] == [0]
    tp = points[0]
    assert tp["speaker"] == "A" and tp["received_by"] == "B" and tp["response_turn"] == 1
    assert tp["opened"].startswith("deliverable: reached for 83/100")
    assert tp["quote"] == _TP_TURNS[0]["text"]
    assert tp["received_as"] == "deliverable landed 19/100; reply #1: I cannot send it"
    assert tp["wanted_vs_happened"] == ("wanted: deliverable at 83/100; happened: 19/100; "
                                        "reply #1: I cannot send it")
    assert tp["effect"] == "64 points short of what was wanted"
    assert (tp["item"], tp["wanted"], tp["actual"]) == ("deliverable", 82.6, 19.2)
    assert tp["triggers"][0].startswith("alert:")
    assert tp["source"].startswith("model estimate:")


def test_split_turn_under_the_alert_cuts_becomes_a_turning_point():
    # stakes_weight 10 keeps severity under every cut: only the split speaks.
    points, alerts, standing = _tp_points([
        _reading(0, facts=(70.0, 90.0)),
        _reading(1, speaker="B", deliverable=(80.0, 20.0, 15.0, 2, "tell me who has it"))],
        weight=10.0)
    assert alerts == []
    assert standing["splits"]["outcome"]["turn"] == 1
    assert [p["turn"] for p in points] == [1]
    assert points[0]["triggers"] == ["split:outcome"]
    assert points[0]["item"] == "deliverable"


def test_turn_that_landed_stays_out_even_where_standing_split():
    points, alerts, standing = _tp_points([
        _reading(0, facts=(80.0, 90.0)),
        _reading(1, speaker="B", facts=(80.0, 90.0))], weight=10.0)
    assert alerts == []
    assert standing["splits"]["understanding"]["turn"] == 1
    assert points == []


def test_split_turn_where_actual_passes_wanted_stays_out():
    # Inside the broken cuts, yet the wanted effect landed past what was wanted.
    points, _, standing = _tp_points([
        _reading(0, facts=(50.0, 90.0)),
        _reading(1, speaker="B", facts=(45.0, 55.0))], weight=10.0)
    assert standing["splits"]["understanding"]["turn"] == 1
    assert points == []


def test_opened_is_the_most_wanted_item_that_fell_short_and_measured():
    # warmth is reference and facts landed: neither may name the entry.
    points, _, _ = _tp_points([
        _reading(0, warmth=(95.0, 5.0), facts=(90.0, 85.0), deliverable=(60.0, 30.0))])
    assert [p["item"] for p in points] == ["deliverable"]
    assert "warmth" not in points[0]["opened"]


def test_most_wanted_item_is_picked_over_the_largest_gap():
    grades_use = {"deliverable", "facts"}
    points, alerts, _ = _tp_points([
        _reading(0, facts=(90.0, 55.0), deliverable=(60.0, 10.0))], use=grades_use)
    assert sorted(alerts[0]["items"]) == ["deliverable", "facts"]
    assert points[0]["item"] == "facts"


def test_reading_without_a_cited_reply_says_so_and_uses_the_next_speaker():
    points, _, _ = _tp_points([_reading(0, deliverable=(80.0, 20.0))])
    tp = points[0]
    assert tp["response_turn"] == 1 and tp["received_by"] == "B"
    assert ("the reading cites no reply; next reply #1: "
            "The form is with legal, I cannot send it.") in tp["received_as"]


def test_reply_by_the_speaker_is_not_shown_as_the_other_side_receiving():
    points, _, _ = _tp_points([
        _reading(0, deliverable=(80.0, 20.0, 15.0, 2, "tell me who has it"))])
    tp = points[0]
    assert tp["response_turn"] == 1 and tp["received_by"] == "B"
    assert "the speaker's own #2: tell me who has it" in tp["received_as"]


def test_turn_nobody_else_answers_is_not_built():
    solo = [{"speaker": "A", "text": "First."}, {"speaker": "A", "text": "Second."}]
    points, alerts, _ = _tp_points([_reading(0, deliverable=(80.0, 20.0))], turns=solo)
    assert alerts and points == []


def test_intentional_is_unclear_without_a_later_reach():
    points, _, _ = _tp_points([_reading(0, deliverable=(80.0, 20.0)),
                               _reading(1, speaker="B", deliverable=(80.0, 90.0)),
                               _reading(2, facts=(80.0, 90.0))])
    assert points[0]["intentional"] == "unclear"


def test_intentional_is_yes_when_the_same_speaker_reaches_again():
    points, _, _ = _tp_points([_reading(0, deliverable=(80.0, 20.0)),
                               _reading(1, speaker="B", facts=(80.0, 90.0)),
                               _reading(2, deliverable=(70.0, 30.0))])
    # 승미님 지시: 의도 여부를 재는 측정값이 없으므로 yes를 내지 않는다.
    assert points[0]["intentional"] == "unclear"


def test_no_measured_reading_returns_none_for_the_panel_rule():
    from forecaster.multilayer import blank_reading
    assert R.build_reading_turning_points(_TP_TURNS, [blank_reading(_TP_TURNS, 0)],
                                          [], {"splits": {}}, _USE) is None
    reference_only = [_reading(0, warmth=(95.0, 5.0))]
    assert R.build_reading_turning_points(_TP_TURNS, reference_only, [], {"splits": {}}, _USE) is None


def test_report_with_readings_takes_turning_points_from_them_not_the_panel(tmp_path, monkeypatch):
    import forecaster.multilayer as ML
    _write_case(tmp_path, n_extra=2)  # turn 2 needs a later turn by the other speaker
    real = ML.build_multilayer

    def fake(turns, **kwargs):
        got = real(turns, grades=_GRADES)
        got["uptake"][2]["items"]["facts"] = _item("understanding", 80.0, 20.0, 15.0, 4, "Routine follow-up")
        got["uptake"][2]["broken"] = ["facts"]
        return {**got, "alerts": ML.build_alerts_items(got["uptake"], {"stakes_weight": 90.0}, _GRADES),
                "standing": ML.build_standing_items(got["uptake"], use=ML.measured_items(_GRADES))}

    monkeypatch.setattr(ML, "build_multilayer", fake)
    r = R.build_report("t1", data_root=str(tmp_path))
    assert r["turning_points_source"] == "per-item readings"
    # The panel alone puts its mismatch on turn 0.
    assert [tp["turn"] for tp in r["turning_points"]] == [2]
    tp = r["turning_points"][0]
    reading = next(u for u in r["uptake"] if u["turn"] == 2)
    assert tp["wanted"] == reading["items"][tp["item"]]["wanted"]["score"] == 80.0
    assert tp["actual"] == reading["items"][tp["item"]]["actual"]["score"] == 20.0
    for key in ("turn", "speaker", "opened", "received_by", "received_as",
                "response_turn", "effect", "wanted_vs_happened"):
        assert tp[key] not in (None, ""), key
    assert tp["intentional"] in ("yes", "no", "unclear")
    assert isinstance(tp["response_turn"], int) and tp["response_turn"] > tp["turn"]
    assert "read as" not in tp["received_as"]
    assert "appeal_need" not in tp["opened"]
    page = R.render_html(r).split('id="turning-points"')[1].split("</section>")[0]
    assert "Basis: model estimate:" in page
    assert "urgent intent" not in page


def test_report_without_readings_keeps_the_panel_turning_points(tmp_path):
    r = _report(tmp_path)
    assert r["turning_points_source"] == "panel or text"
    assert "source" not in r["turning_points"][0]
    page = R.render_html(r).split('id="turning-points"')[1].split("</section>")[0]
    assert "urgent intent" in page
    assert "Basis:" not in page


def test_html_section_ids_in_order(tmp_path):
    _write_case(tmp_path)
    out = tmp_path / "out"
    _run("t1", tmp_path, out)
    page = (out / "report.html").read_text(encoding="utf-8")
    ids = ["event", "goals", "turning-points", "alerts", "intent",
           "alternatives", "perspective", "balance", "reliability"]
    idxs = [page.find(f'<section id="{i}"') for i in ids]
    assert all(i >= 0 for i in idxs), ids
    assert idxs == sorted(idxs)


def test_split_verbatim_finds_embedded_quote():
    quote = "We need priority now."
    texts = R.alternative_texts(quote)
    for text in texts:
        pre, span, post = R.split_verbatim(text, quote)
        assert span == quote
        assert pre + span + post == text
    long_quote = "word " * 100
    short = R.alternative_texts(long_quote)[0]
    _, span, _ = R.split_verbatim(short, long_quote)
    assert span and len(span) <= 220 and span in short
    pre, span, post = R.split_verbatim("A fresh paraphrase.", quote)
    assert (pre, span, post) == ("A fresh paraphrase.", "", "")


def test_html_verbatim_quote_stays_in_blockquote(tmp_path):
    import re
    korean = "연료가 부족하니 우선권을 주십시오"
    turns = [
        {"speaker": "A", "text": korean},
        {"speaker": "B", "text": "Roger, continue as cleared."},
        {"speaker": "A", "text": "Roger."},
    ]
    moves = [_mv(0, "A", ["appeal_need"], korean),
             _mv(1, "B", ["accept", "inform"], "continue as cleared"),
             _mv(2, "A", ["accept"], "roger")]
    effects = [_eff(0, "A", turns[0]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]),
               _eff(1, "B", turns[1]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]),
               _eff(2, "A", turns[2]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"])]
    _write_custom_case(tmp_path, "ko", turns, moves, effects,
                       outcome_turn=2)
    out = tmp_path / "out"
    _run("ko", tmp_path, out)
    page = (out / "report.html").read_text(encoding="utf-8")
    assert korean in page  # original utterance preserved
    body = re.sub(r"<blockquote>.*?</blockquote>", "", page, flags=re.S)
    assert not re.search(r"[\uac00-\ud7a3]", body)
    alt_section = page[page.find('id="alternatives"'):]
    assert f"<blockquote>{korean}</blockquote>" in alt_section


# --- preregistered reliability (C7) ---------------------------------------------

def test_reliability_is_preregistered(tmp_path):
    # C7: reliability is never computed per case. Every report carries
    # the same pre-registered measurements with their target population.
    r = _report(tmp_path)
    rel = r["reliability"]
    assert rel["split_half"] == pytest.approx(0.863)
    assert rel["k"] == 8
    assert 0.27 <= rel["outcome_r1"] <= 0.35
    assert rel["perturb"] == pytest.approx(0.8)
    assert "casino" in str(rel["measured_on"]).lower()
    blob = json.dumps(rel)
    assert "0.277" in blob and "0.344" in blob  # both holdout R1s on record
    assert "16/20" in blob or "16 / 20" in blob
    limits = " ".join(rel["limits"]).lower()
    assert "not" in limits and "this case" in limits


def test_reliability_identical_across_cases(tmp_path):
    # C7: different conversations must not change the reliability block.
    _write_case(tmp_path, case_id="c1", n_extra=1)
    _write_case(tmp_path, case_id="c2", n_extra=4)
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    _run("c1", tmp_path, out1)
    _run("c2", tmp_path, out2)
    with open(out1 / "report.json", encoding="utf-8") as f:
        rel1 = json.load(f)["reliability"]
    with open(out2 / "report.json", encoding="utf-8") as f:
        rel2 = json.load(f)["reliability"]
    for key in ("split_half", "k", "outcome_r1", "perturb", "measured_on"):
        assert rel1[key] == rel2[key], key


# --- G1: per-speaker balances must differ when records differ -------------------

def test_g1_caution_vs_warning_balances_differ(tmp_path):
    # Same count of unresolved urgent turns, but different severity:
    # A gets an engaged reply (caution, partial uptake), B gets none
    # (warning, missed). Balances must differ (partial vs missed).
    turns = [
        {"speaker": "A", "text": "We need priority now."},
        {"speaker": "B", "text": "Here is another course instead."},
        {"speaker": "B", "text": "That reading looks wrong."},
        {"speaker": "A", "text": "Noted, continuing."},
    ]
    moves = [_mv(0, "A", ["appeal_need"], "need priority"),
             _mv(1, "B", ["counter"], "another course"),
             _mv(2, "B", ["challenge"], "looks wrong"),
             _mv(3, "A", ["inform"], "continuing")]
    effects = [_eff(i, t["speaker"], t["text"], {"proceed": 1.0},
                    ["proceed", "proceed"])
               for i, t in enumerate(turns)]
    _write_custom_case(tmp_path, "sev", turns, moves, effects,
                       outcome_turn=3)
    out = tmp_path / "out"
    _run("sev", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_sp = {sp["speaker"]: sp for sp in r["speakers"]}
    assert by_sp["A"]["end_balance"] == pytest.approx(-0.5)
    assert by_sp["B"]["end_balance"] == pytest.approx(-1.0)
    assert by_sp["A"]["end_balance"] != by_sp["B"]["end_balance"]


def test_g1_no_urgent_with_panel_balances_differ(tmp_path):
    # No urgent presses anywhere: fall back to intent-vs-read matching
    # over all turns. A has one non-routine read (aside ignored), B is
    # all routine. Balances must differ.
    turns = [
        {"speaker": "A", "text": "Hello."},
        {"speaker": "B", "text": "Hi."},
        {"speaker": "A", "text": "Aside to someone else."},
        {"speaker": "B", "text": "Okay."},
    ]
    moves = [_mv(0, "A", ["inform"], "hello"),
             _mv(1, "B", ["inform"], "hi"),
             _mv(2, "A", ["other"], "aside"),
             _mv(3, "B", ["inform"], "okay")]
    effects = [_eff(0, "A", turns[0]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]),
               _eff(1, "B", turns[1]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"]),
               _eff(2, "A", turns[2]["text"], {"ignore": 1.0},
                    ["ignore", "ignore"]),
               _eff(3, "B", turns[3]["text"], {"proceed": 1.0},
                    ["proceed", "proceed"])]
    _write_custom_case(tmp_path, "nou", turns, moves, effects,
                       outcome_turn=3)
    out = tmp_path / "out"
    _run("nou", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_sp = {sp["speaker"]: sp for sp in r["speakers"]}
    assert by_sp["A"]["end_balance"] == pytest.approx(0.0)
    assert by_sp["B"]["end_balance"] == pytest.approx(1.0)
    assert by_sp["A"]["end_balance"] != by_sp["B"]["end_balance"]


def test_g1_control_points_balances_differ(tmp_path):
    # No panel record (control dialogue): fall back to the participant
    # record (points scored against own values). Different points must
    # give different balances, and goals come from the value record.
    ds = tmp_path / "datasets"
    ds.mkdir(parents=True, exist_ok=True)
    dlg = {
        "dialogue_id": 999,
        "chat_logs": [
            {"id": "mturk_agent_1", "text": "I need food most."},
            {"id": "mturk_agent_2", "text": "I need firewood most."},
            {"id": "mturk_agent_1", "text": "How about two food for one firewood?"},
            {"id": "mturk_agent_2", "text": "Deal, submit."},
        ],
        "participant_info": {
            "mturk_agent_1": {
                "value2issue": {"High": "Food", "Medium": "Water",
                                "Low": "Firewood"},
                "outcomes": {"points_scored": 20,
                             "satisfaction": "Slightly satisfied"},
            },
            "mturk_agent_2": {
                "value2issue": {"High": "Firewood", "Medium": "Food",
                                "Low": "Water"},
                "outcomes": {"points_scored": 10,
                             "satisfaction": "Slightly dissatisfied"},
            },
        },
    }
    with open(ds / "casino.json", "w", encoding="utf-8") as f:
        json.dump([dlg], f)
    out = tmp_path / "out"
    _run("999", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    by_sp = {sp["speaker"]: sp for sp in r["speakers"]}
    assert by_sp["mturk_agent_1"]["end_balance"] == pytest.approx(1 / 3,
                                                                  abs=0.01)
    assert by_sp["mturk_agent_2"]["end_balance"] == pytest.approx(-1 / 3,
                                                                  abs=0.01)
    assert ("Food" in by_sp["mturk_agent_1"]["initial_goal"]
            and "Food" in by_sp["mturk_agent_2"]["initial_goal"])


def test_g1_initial_goals_are_sentences_not_labels(tmp_path):
    # G1: initial_goal is one sentence saying what the speaker wanted
    # in THIS conversation. A move-classification label ("ask: ask for
    # ...") is not a goal. Sources stay honest, quotes stay verbatim,
    # and shift endpoints carry no "label:" prefix either.
    import re
    r = _report(tmp_path)
    label_prefix = re.compile(r"^[a-z_]+:\s")
    move_token = re.compile(
        r"\b(appeal_need|appeal_fairness|reveal_preference)\b")
    for sp in r["speakers"]:
        goal = sp["initial_goal"]
        assert " " in goal.strip(), goal
        assert goal.strip().endswith("."), goal
        assert not label_prefix.match(goal), goal
        assert not move_token.search(goal), goal
        assert sp["goal_source"].startswith(
            ("model estimate:", "record:")), sp["goal_source"]
        assert -1.0 <= sp["end_balance"] <= 1.0
        assert sp["end_basis"]
        for sh in sp["shifts"]:
            assert not label_prefix.match(sh["from"]), sh
            assert not label_prefix.match(sh["to"]), sh


def test_g1_control_value_goals_are_sentences(tmp_path):
    # G1 record path: value-ranking goals keep their issue nouns but
    # read as sentences, not bare "High: ...; ..." fragments.
    ds = tmp_path / "datasets"
    ds.mkdir(parents=True, exist_ok=True)
    dlg = {
        "dialogue_id": 998,
        "chat_logs": [
            {"id": "mturk_agent_1", "text": "I need food most."},
            {"id": "mturk_agent_2", "text": "I need firewood most."},
        ],
        "participant_info": {
            "mturk_agent_1": {
                "value2issue": {"High": "Food", "Medium": "Water",
                                "Low": "Firewood"},
                "outcomes": {"points_scored": 20,
                             "satisfaction": "Slightly satisfied"},
            },
            "mturk_agent_2": {
                "value2issue": {"High": "Firewood", "Medium": "Food",
                                "Low": "Water"},
                "outcomes": {"points_scored": 10,
                             "satisfaction": "Slightly dissatisfied"},
            },
        },
    }
    with open(ds / "casino.json", "w", encoding="utf-8") as f:
        json.dump([dlg], f)
    out = tmp_path / "out"
    _run("998", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    for sp in r["speakers"]:
        goal = sp["initial_goal"]
        assert "Food" in goal
        assert " " in goal.strip() and goal.strip().endswith(".")
        assert sp["goal_source"].startswith("record:")


# --- o3: end plus and minus by stake ---------------------------------------------

def _balance_section(page):
    return page.split('<section id="balance">')[1].split("</section>")[0]


def _end_row(score, judged, mean, item):
    return {"score": score, "judged": judged, "mean": mean,
            "items": {item: score} if judged else {}}


def _with_end_values(r):
    """Same shape build_report writes: four stakes, one left unjudged."""
    a, b = r["speakers"][:2]
    a["end_by_stake"] = {
        "outcome": _end_row(-1.375, 12, -0.115, "deliverable"),
        "relationship": _end_row(-0.824, 7, -0.118, "warmth"),
        "face": _end_row(-0.157, 1, -0.157, "competence"),
        "understanding": _end_row(0.321, 12, 0.027, "facts")}
    a["end_balance"] = -0.091
    b["end_by_stake"] = {
        "outcome": _end_row(1.068, 18, 0.059, "deliverable"),
        "relationship": _end_row(-1.011, 10, -0.101, "warmth"),
        "face": _end_row(0.0, 0, None, "competence"),
        "understanding": _end_row(0.267, 7, 0.038, "facts")}
    b["end_balance"] = -0.001
    return r


def test_balance_shows_end_values_by_speaker_and_stake_first(tmp_path):
    r = _with_end_values(_report(tmp_path))
    section = _balance_section(R.render_html(r))
    rows = [row.split("<td>")[1:] for row in section.split("<tr>")
            if row.startswith("<td>")]
    by_sp = {row[0].split("</td>")[0]: row for row in rows if len(row) == 5}
    assert set(by_sp) == {"A", "B"}
    assert by_sp["A"][1].startswith("-1.375 (mean -0.115, 12 turns)")
    assert by_sp["A"][3].startswith("-0.157 (mean -0.157, 1 turn)")
    assert by_sp["B"][1].startswith("+1.068 (mean +0.059, 18 turns)")
    # An unjudged stake is a dash, not a zero.
    assert by_sp["B"][3].startswith("–")
    # The heading opens the section, the stake table comes before the average.
    assert section.strip().startswith("<h2>8. Plus and minus at the end</h2>")
    assert section.index("-1.375") < section.index("Average (sorting only)")
    assert section.index("-1.375") < section.index("<td>-0.091</td>")


def test_balance_scope_says_conversation_only_not_event_outcome(tmp_path):
    r = _with_end_values(_report(tmp_path))
    section = _balance_section(R.render_html(r))
    assert ("how far each speaker&#x27;s words landed as read through up to "
            "3 later turns; they do not include the outcome of the event."
            ) in section


def test_end_values_page_has_no_empty_stake_notice_anywhere(tmp_path):
    r = _with_end_values(_report(tmp_path))
    page = R.render_html(r)
    assert "no judged stake gaps" not in page.lower()
    assert R.NO_STAKE_VALUES not in page
    position = page.split('<section id="turning-points">')[1].split(
        "</section>")[0]
    assert "never merged into one number" in position


def test_render_keeps_end_values_and_average_fields_as_given(tmp_path):
    r = _with_end_values(_report(tmp_path))
    before = json.dumps(r["speakers"], sort_keys=True)
    R.render_html(r)
    assert json.dumps(r["speakers"], sort_keys=True) == before
    assert r["speakers"][0]["end_balance"] == -0.091


def test_end_values_from_the_reading_pipeline_reach_the_page(tmp_path, monkeypatch):
    import forecaster.multilayer as ML
    _write_case(tmp_path, n_extra=2)
    real = ML.build_multilayer

    def fake(turns, **kwargs):
        got = real(turns, grades=_GRADES)
        got["uptake"][2]["items"]["facts"] = _item(
            "understanding", 80.0, 20.0, 15.0, 4, "Routine follow-up")
        return {**got, "standing": ML.build_standing_items(
            got["uptake"], use=ML.measured_items(_GRADES))}

    monkeypatch.setattr(ML, "build_multilayer", fake)
    r = R.build_report("t1", data_root=str(tmp_path))
    speaker = str(r["uptake"][2]["speaker"])
    sp = next(s for s in r["speakers"] if s["speaker"] == speaker)
    want = sp["end_by_stake"]["understanding"]
    assert want["judged"] == 1
    page = R.render_html(r)
    section = _balance_section(page)
    assert "{:+.3f} (mean {:+.3f}, 1 turn)".format(
        want["score"], want["mean"]) in section
    assert "no judged stake gaps" not in page.lower()
    assert "do not include the outcome of the event" in section


# --- turn effect mark -----------------------------------------------------------------

_EFFECT = {"value": -0.72932, "mark": "?!", "items": {"deliverable": -0.2964, "facts": -0.43292}}


def _with_effect_marks(r, effect=_EFFECT):
    """A report with one reading on turn 1, shown in every place a turn number goes."""
    r["uptake"][1]["effect"] = effect
    r["alerts"] = [{"turn": 1, "level": "warning", "axis": "understanding",
                    "reason": "Fell short.", "resolved": False}]
    r["turning_points"] = [{"turn": 1, "speaker": "B", "opened": "o", "quote": "q"}]
    r["judgment"] = {"turns": [{"turn": 1, "speaker": "B", "opened": "o", "response_turn": 2,
                                "received_by": "A", "received": "r"}]}
    return r


def _sections(page):
    return (page.split('<section id="turning-points">')[1].split("</section>")[0],
            page.split('<section id="alerts">')[1].split("</section>")[0])


def test_effect_mark_stands_next_to_the_turn_in_all_three_places(tmp_path):
    third, fourth = _sections(R.render_html(_with_effect_marks(_report(tmp_path))))
    assert "<h3>Turn #1 ?! (B)</h3>" in third
    assert "<tr><td>#1 ?!</td><td>B</td><td>o</td>" in third
    assert "<tr><td>#1 ?!</td><td>warning</td>" in fourth


def test_effect_items_are_visible_turn_by_turn(tmp_path):
    _, fourth = _sections(R.render_html(_with_effect_marks(_report(tmp_path))))
    assert "<td>?! -0.729 (deliverable -0.296, facts -0.433)</td>" in fourth


def test_effect_note_says_measured_blank_and_not_the_person(tmp_path):
    _, fourth = _sections(R.render_html(_report(tmp_path)))
    assert f"<p>{R.TURN_EFFECT_NOTE}</p>" in fourth
    for words in ("measured items only", "stays blank", "does not rate the person"):
        assert words in R.TURN_EFFECT_NOTE


def test_blank_effect_shows_no_mark_and_panel_mark_is_not_borrowed(tmp_path):
    r = _with_effect_marks(_report(tmp_path), {"value": None, "mark": None, "items": {}})
    r["uptake"][1]["mark"] = "!!"
    page = R.render_html(r)
    third, fourth = _sections(page)
    assert "<h3>Turn #1 (B)</h3>" in third
    assert "<tr><td>#1</td><td>B</td><td>o</td>" in third
    assert "<tr><td>#1</td><td>warning</td>" in fourth
    assert "!!" not in third + fourth.replace(R.TURN_EFFECT_NOTE, "")


def test_equal_mark_is_shown_for_a_real_value_near_zero(tmp_path):
    r = _with_effect_marks(_report(tmp_path), {"value": 0.18384, "mark": "=", "items": {"warmth": 0.18384}})
    third, fourth = _sections(R.render_html(r))
    assert "<h3>Turn #1 = (B)</h3>" in third
    assert "<td>= +0.184 (warmth +0.184)</td>" in fourth


def test_effect_from_the_reading_pipeline_keeps_other_values(tmp_path, monkeypatch):
    import copy
    import forecaster.multilayer as ML
    _write_case(tmp_path, n_extra=2)
    real = ML.build_multilayer
    seen = {}

    def fake(turns, **kwargs):
        got = real(turns, grades=_GRADES)
        got["uptake"][2]["items"]["facts"] = _item("understanding", 80.0, 20.0, 15.0, 4, "Routine follow-up")
        got["uptake"][2]["items"]["warmth"] = _item("relationship", 100.0, 0.0)
        got["uptake"][2]["broken"] = ["facts"]
        out = {**got, "alerts": ML.build_alerts_items(got["uptake"], {"stakes_weight": 90.0}, _GRADES),
               "standing": ML.build_standing_items(got["uptake"], use=ML.measured_items(_GRADES))}
        seen["uptake"], seen["alerts"] = copy.deepcopy(out["uptake"]), copy.deepcopy(out["alerts"])
        return out

    monkeypatch.setattr(ML, "build_multilayer", fake)
    r = R.build_report("t1", data_root=str(tmp_path))
    # warmth is reference in these grades: only facts counts.
    assert r["uptake"][2]["effect"] == {"value": -0.48, "mark": "?!", "items": {"facts": -0.48}}
    assert r["uptake"][0]["effect"] == {"value": None, "mark": None, "items": {}}
    assert r["alerts"] == seen["alerts"]
    for before, after in zip(seen["uptake"], r["uptake"]):
        assert {k: v for k, v in after.items() if k not in ("effect", "scales", "mark")} == before
    # This panel record carries no annotation, so the panel mark stays blank beside the effect.
    assert r["uptake"][2]["mark"] is None
    assert "<td>?! -0.480 (facts -0.480)</td>" in R.render_html(r)


def test_balance_without_end_values_says_so(tmp_path):
    r = _report(tmp_path)
    assert all("end_by_stake" not in sp or not any(
        v["judged"] for v in sp["end_by_stake"].values())
        for sp in r["speakers"])
    section = _balance_section(R.render_html(r))
    assert "No per-stake values on record" in section
    assert "do not include the outcome of the event" not in section
    assert "Average (sorting only)" not in section
    # The notice comes before any number the record still carries.
    assert section.index("No per-stake values") < section.index("<table>")


_STAKES_BASIS = "The board report lists a total loss of the aircraft."


def _stakes_report(tmp_path, monkeypatch, stakes_record=None):
    """A report through the real reading path, on a stub model, with the record in the case file."""
    import forecaster.multilayer as ML
    _write_case(tmp_path)
    case_path = tmp_path / "highstakes" / "casino-t1_case.json"
    case = json.loads(case_path.read_text(encoding="utf-8"))
    if stakes_record is not None:
        case["stakes_record"] = stakes_record
    case_path.write_text(json.dumps(case), encoding="utf-8")
    seen = []

    class Stub:
        def complete(self, messages, **kwargs):
            seen.append(json.dumps(messages, ensure_ascii=False))
            system = messages[0]["content"]
            if "describe its conditions" in system:
                return json.dumps({"relationship": "crew and tower", "stakes_weight": 20,
                                   "evidence": "Roger, continue as cleared."})
            hi, lo = [0, 0, 0, .2, .8], [.8, .2, 0, 0, 0]
            return json.dumps({"items": {i: {"wanted": hi, "actual": lo} for i in ML.ITEMS}})

    real = ML.build_multilayer

    def fake(turns, **kwargs):
        return real(turns, client=Stub(), model="m", cache_dir=tmp_path / "cache", max_turns=2,
                    record=kwargs["record"], stakes_record=kwargs["stakes_record"], grades=_GRADES)

    monkeypatch.setattr(ML, "build_multilayer", fake)
    return R.build_report("t1", data_root=str(tmp_path)), seen


def test_report_shows_recorded_stakes_weight_and_source(tmp_path, monkeypatch):
    r, seen = _stakes_report(tmp_path, monkeypatch, {"weight": 100, "basis": _STAKES_BASIS})
    assert r["conditions"]["stakes_weight"] == 100.0
    assert r["conditions"]["stakes_weight_model"] == 20.0
    assert r["conditions"]["stakes_source"] == f"record: {_STAKES_BASIS}"
    assert r["alerts"] and all(a["severity"] == r["uptake"][a["turn"]]["layers"]["severity"] for a in r["alerts"])
    page = R.render_html(r)
    assert f"stakes_weight: 100.0 (record: {_STAKES_BASIS}; model read 20.0)" in page
    assert _STAKES_BASIS not in "".join(seen)


def test_report_without_or_with_bad_record_shows_model_estimate(tmp_path, monkeypatch):
    plain, _ = _stakes_report(tmp_path / "a", monkeypatch)
    bad, _ = _stakes_report(tmp_path / "b", monkeypatch, {"weight": 140, "basis": _STAKES_BASIS})
    assert plain["conditions"]["stakes_source"] == bad["conditions"]["stakes_source"] == "model estimate"
    assert plain["conditions"]["stakes_weight"] == bad["conditions"]["stakes_weight"] == 20.0
    assert plain["alerts"] == bad["alerts"]
    assert "stakes_weight: 20.0 (model estimate)" in R.render_html(bad)


def test_report_limits_state_how_far_the_stakes_weight_is_tested(tmp_path, monkeypatch):
    r, _ = _stakes_report(tmp_path, monkeypatch)
    assert R.STAKES_LIMIT in r["reliability"]["limits"]
    assert "development turns" in R.STAKES_LIMIT and "fails" in R.STAKES_LIMIT and "locked evaluation" in R.STAKES_LIMIT
    assert R.STAKES_LIMIT in R.render_html(r)


def test_judgment_limit_names_where_the_weight_it_saw_came_from():
    assert R.judgment_limit({"stakes_source": "model estimate"}) == R.JUDGMENT_LIMIT
    recorded = R.judgment_limit({"stakes_source": f"record: {_STAKES_BASIS}"})
    assert "stakes weight it saw is the case record's" in recorded
    assert "relationship and channel it saw were read earlier" in recorded
