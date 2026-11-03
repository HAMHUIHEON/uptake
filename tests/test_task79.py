"""Task79: measured end values on the report (stub client, no network)."""
import json
from pathlib import Path

import pytest

from forecaster import multilayer as M
from forecaster import report as R
from forecaster.board import annotate_turn

HI = [0, 0, 0, .2, .8]  # mean 86.0
LO = [.8, .2, 0, 0, 0]  # mean 14.0
MID = [0, 0, 1, 0, 0]  # mean 50.0


class Stub:
    def __init__(self, replies):
        self.replies = list(replies)
        self.seen = []
        self.n = 0

    def complete(self, messages, **kwargs):
        self.seen.append(messages)
        out = self.replies[min(self.n, len(self.replies) - 1)]
        self.n += 1
        return out if isinstance(out, str) else json.dumps(out)


def _cell(wanted, actual):
    return {"wanted": {"score": wanted, "spread": 5.0} if wanted is not None else None,
            "actual": {"score": actual, "spread": 5.0} if actual is not None else None}


def _reading(speaker, turn, **items):
    full = {name: {"class": M.ITEM_CLASS[name], "wanted": None, "actual": None,
                   "reply_turn": None, "quote": ""}
            for name in M.ITEMS}
    for name, (wanted, actual) in items.items():
        full[name] = {"class": M.ITEM_CLASS[name], **_cell(wanted, actual),
                      "reply_turn": None, "quote": ""}
    return {"turn": turn, "speaker": speaker, "items": full, "broken": []}


def test_calm_reference_values():
    assert M.CALM_REFERENCE == {"deliverable": -30.481, "warmth": -22.482,
                                "competence": -25.66, "facts": -28.932}


def test_build_end_gaps_arithmetic_and_absent():
    use = {"deliverable", "warmth", "facts"}
    readings = [_reading("A", 0, deliverable=(80.0, 20.0), warmth=(60.0, 50.0)),
                _reading("A", 1, deliverable=(70.0, 30.0)),
                _reading("B", 2, deliverable=(50.0, 50.0))]
    gaps = M.build_end_gaps(readings, use)
    # A deliverable: diffs -60, -40, mean -50, minus (-30.481), over 100.
    assert gaps["A"]["deliverable"] == pytest.approx((-50.0 + 30.481) / 100.0)
    # A warmth: one diff -10, minus (-22.482), over 100.
    assert gaps["A"]["warmth"] == pytest.approx((-10.0 + 22.482) / 100.0)
    # No values: absent, never zero.
    assert "facts" not in gaps["A"]
    assert gaps["B"]["deliverable"] == pytest.approx((0.0 + 30.481) / 100.0)
    assert "warmth" not in gaps["B"]
    # No merged number per stake.
    for speaker_gaps in gaps.values():
        assert set(speaker_gaps) <= use
        assert "outcome" not in speaker_gaps


def _end_reply(outcome_bins, settled=None):
    return {"settled": settled if settled is not None else [],
            "stakes": {"outcome": {"score": outcome_bins, "quote": ""},
                       "relationship": {"score": None, "quote": ""},
                       "face": {"score": None, "quote": ""},
                       "understanding": {"score": None, "quote": ""}}}


def test_three_reads_averaged_and_null_skipped():
    hi = M.bins_to_score(HI)["score"]
    lo = M.bins_to_score(LO)["score"]
    stub = Stub([_end_reply(HI), _end_reply(LO), _end_reply(None)])
    turns = [{"speaker": "A", "text": "Hello."}, {"speaker": "B", "text": "Hi."}]
    out = R.build_end_read(turns, "A", "Want food most.", stub, "m", None, None)
    assert out["goal_source"] == "record"
    assert out["goal"] == "Want food most."
    assert out["stakes"]["outcome"]["scores"] == [hi, lo, None]
    assert out["stakes"]["outcome"]["mean"] == pytest.approx((hi + lo) / 2.0)
    assert out["stakes"]["relationship"]["mean"] is None
    assert len(stub.seen) == 3


def test_settled_from_first_read_with_any():
    first = [{"item": "Food", "speaker_gets": 2, "others_get": 1}]
    second = [{"item": "Water", "speaker_gets": 1, "others_get": 0}]
    stub = Stub([_end_reply(HI, []), _end_reply(HI, second), _end_reply(HI, first)])
    turns = [{"speaker": "A", "text": "Hello."}]
    out = R.build_end_read(turns, "A", "Goal.", stub, "m", None, None)
    assert out["settled"] == [{"item": "Water", "speaker_gets": 1.0, "others_get": 0.0}]
    stub2 = Stub([_end_reply(HI, first), _end_reply(HI, second), _end_reply(HI, [])])
    out2 = R.build_end_read(turns, "A", "Goal.", stub2, "m", None, None)
    assert out2["settled"] == [{"item": "Food", "speaker_gets": 2.0, "others_get": 1.0}]


def test_goal_source_record_and_none():
    case = {"participant_info": {"A": {"value2issue": {"High": "Food", "Medium": "Water",
                                                        "Low": "Firewood"}}}}
    goal = R.recorded_goal_for(case, "A")
    assert goal and "Food" in goal
    turns = [{"speaker": "A", "text": "Hello."}]
    stub = Stub([_end_reply(HI)] * 3)
    out = R.build_end_read(turns, "A", goal, stub, "m", None, None)
    assert out["goal_source"] == "record"
    assert "Goal at the start:" in stub.seen[0][1]["content"]
    # No recorded goal: goal None, source none, no goal line sent.
    assert R.recorded_goal_for({}, "A") is None
    assert R.recorded_goal_for({"participant_info": {"A": {}}}, "A") is None
    stub2 = Stub([_end_reply(HI)] * 3)
    out2 = R.build_end_read(turns, "A", None, stub2, "m", None, None)
    assert out2["goal"] is None and out2["goal_source"] == "none"
    for call in stub2.seen:
        assert "Goal at the start:" not in call[1]["content"]


def _write_min_case(root, case_id="t79"):
    hs = Path(root) / "highstakes"
    hs.mkdir(parents=True, exist_ok=True)
    turns = [{"speaker": "A", "text": "We need priority, we are low on fuel."},
             {"speaker": "B", "text": "Roger, continue as cleared."},
             {"speaker": "A", "text": "Confirming we continue."}]
    with open(hs / f"casino-{case_id}_turns.jsonl", "w", encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    with open(hs / f"casino-{case_id}_case.json", "w", encoding="utf-8") as f:
        json.dump({"title": "Synthetic fuel case", "outcome": "Synthetic outcome.",
                   "source_url": "https://example.com/record", "outcome_turn": 2}, f)
    moves = [{"turn_index": 0, "speaker": "A", "moves": ["inform", "appeal_need"],
              "quote": "need priority"},
             {"turn_index": 1, "speaker": "B", "moves": ["accept", "inform"],
              "quote": "continue as cleared"},
             {"turn_index": 2, "speaker": "A", "moves": ["accept"], "quote": "confirming"}]
    effects = []
    for i, t in enumerate(turns):
        effects.append({"index": i, "speaker": t["speaker"], "text": t["text"],
                        "action_dist": {"proceed": 1.0}, "disagreement": 0.0,
                        "n_readings": 2,
                        "readings": [{"turn": i, "listener": "X", "stance": "neutral",
                                      "next_action": "proceed", "relationship_shift": 0,
                                      "face_threat": 0}] * 2})
    panel_dir = hs / "cases" / f"casino-{case_id}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    with open(panel_dir / "panel_record.json", "w", encoding="utf-8") as f:
        json.dump({"id": f"casino-{case_id}", "genre": "aviation_atc",
                   "moves": moves, "effects": effects}, f)


def test_evidence_in_meta_json_and_report(tmp_path):
    meta_path = Path(R.__file__).resolve().parent / "report_meta.json"
    blob = meta_path.read_text(encoding="utf-8")
    for needle in ("+0.41", "+0.28", "+0.53", "+0.31", "+0.18", "+0.43",
                   "+0.08", "+0.04", "+0.39", "+0.25", "+0.52", "0.05",
                   "22.5%", "0.725", "13.8%", "0.525", "82%",
                   "100 negotiation conversations", "aviation"):
        assert needle in blob, needle
    _write_min_case(tmp_path)
    report = R.build_report("t79", data_root=str(tmp_path))
    body = json.dumps(report["reliability"], ensure_ascii=False)
    for needle in ("+0.41", "+0.31", "+0.39", "22.5%", "0.725", "13.8%",
                   "0.525", "82%", "well-understood", "aviation",
                   "court", "diplomacy", "100 negotiation conversations",
                   "second attempt", "+0.18"):
        assert needle in body, needle
    page = R.render_html(report)
    for needle in ("+0.41", "+0.31", "+0.39", "22.5%", "82%",
                   "reference only", "reliability section"):
        assert needle in page, needle
    # Alerts section carries one reference-only sentence with a link.
    alerts = page.split('<section id="alerts">')[1].split("</section>")[0]
    assert "reference only" in alerts.lower()
    assert '<a href="#reliability">' in alerts
    # Balance keeps section id and shows the end read goal source.
    balance = page.split('<section id="balance">')[1].split("</section>")[0]
    assert "Goal source" in balance and "none" in balance


def test_end_balance_from_gaps_and_not_headline(tmp_path, monkeypatch):
    import forecaster.multilayer as ML
    _write_min_case(tmp_path, case_id="g79")
    grades = {"status": "measured", "cuts": [11.3, 21.8, 30.8, 35.1], "causes": {},
              "items": {"deliverable": "measured", "facts": "measured",
                        "warmth": "measured", "competence": "measured"}}
    real = ML.build_multilayer

    def fake(turns, **kwargs):
        got = real(turns, grades=grades)
        got["uptake"][0]["items"]["deliverable"] = {
            "class": "outcome", "wanted": {"score": 80.0, "spread": 5.0},
            "actual": {"score": 20.0, "spread": 5.0}, "reply_turn": None, "quote": ""}
        got["uptake"][0]["items"]["facts"] = {
            "class": "understanding", "wanted": {"score": 70.0, "spread": 5.0},
            "actual": {"score": 30.0, "spread": 5.0}, "reply_turn": None, "quote": ""}
        return {**got, "standing": ML.build_standing_items(
            got["uptake"], use=ML.measured_items(grades))}

    monkeypatch.setattr(ML, "build_multilayer", fake)
    report = R.build_report("g79", data_root=str(tmp_path))
    speaker = str(report["uptake"][0]["speaker"])
    sp = next(s for s in report["speakers"] if s["speaker"] == speaker)
    # (20-80+30.481)/100 and (30-70+28.932)/100, mean clipped, 3 places.
    d = (-60.0 + 30.481) / 100.0
    f = (-40.0 + 28.932) / 100.0
    assert sp["end_gaps"]["deliverable"] == pytest.approx(d)
    assert sp["end_gaps"]["facts"] == pytest.approx(f)
    assert "warmth" not in sp["end_gaps"]
    assert sp["end_balance"] == pytest.approx(round((d + f) / 2.0, 3))
    assert "sorting only" in sp["end_basis"]
    page = R.render_html(report)
    balance = page.split('<section id="balance">')[1].split("</section>")[0]
    assert f"{d:+.3f}" in balance and f"{f:+.3f}" in balance
    assert "<td>-0.091</td>" not in balance


def test_marks_and_bands_unchanged(tmp_path):
    _write_min_case(tmp_path, case_id="m79")
    report = R.build_report("m79", data_root=str(tmp_path))
    assert M.BROKEN_WANTED == 40.0 and M.BROKEN_ACTUAL == 60.0
    assert M.load_grades()["cuts"] == [11.3, 21.8, 30.8, 35.1]
    assert annotate_turn(-0.25) == "?!" and annotate_turn(0.0) == "="
    items = {"deliverable": {"wanted": {"score": 80.0, "spread": 5.0},
                             "actual": {"score": 20.0, "spread": 5.0}}}
    assert M.broken_items(items) == ["deliverable"]
    items["deliverable"]["actual"]["score"] = 61.0
    assert M.broken_items(items) == []
    # Per-turn values come before the marks in the readings table.
    page = R.render_html(report)
    alerts = page.split('<section id="alerts">')[1].split("</section>")[0]
    assert "<th></th><th>Fell short</th>" in alerts
    import html as _html
    valued = {"turn": 0, "speaker": "A", "layers": {"classes": {}},
              "items": {"deliverable": {"class": "outcome",
                                        "wanted": {"score": 80.0, "spread": 5.0},
                                        "actual": {"score": 20.0, "spread": 5.0},
                                        "reply_turn": None, "quote": ""}},
              "expected": {}, "broken": ["deliverable"], "cause": None,
              "effect": {"value": -0.48, "mark": "?!", "items": {"deliverable": -0.48}}}
    row_html = R.render_readings([valued], {"source": "model estimate"}, _html.escape)
    assert row_html.index("wanted 80") < row_html.index("deliverable</td>")
    assert row_html.index("wanted 80") < row_html.index("?! -0.480")
