"""Full frame on a turn reading: gap kind, why, sub-stakes, tolerant quotes."""
import json
from forecaster.axes import MISREAD_TYPES, build_uptake, clean_turn_fields, read_turn, validate_axes
from forecaster.report import turn_scales

TURNS = [{"speaker": "A", "text": "We need priority, please."},
         {"speaker": "B", "text": "Turn left, heading 250."},
         {"speaker": "A", "text": "OK."}]

def reply(gap="missed", **kw):
    e = {"wanted": "priority", "sub_stakes": ["commitment", "warmth"], "happened": "routine", "gap": gap, "reply_turn": 1, "quote": "turn left heading 250"}
    d = {"axes": {a: dict(e) for a in ("outcome", "relationship", "face", "understanding")}, "gap_kind": "accident", "why": {"layer": "say", "factor": "no emergency word", "type": "B4"}}
    d.update(kw)
    return d

class Stub:
    def __init__(self, out): self.out, self.calls = out, 0
    def complete(self, *a, **k):
        self.calls += 1
        return self.out if isinstance(self.out, str) else json.dumps(self.out)

def test_quote_ignores_case_and_punctuation():
    assert validate_axes(reply(), 0, TURNS)["outcome"]["gap"] == "missed"

def test_quote_not_in_reply_is_unclear():
    d = reply(); d["axes"]["outcome"]["quote"] = "clear to land"
    assert validate_axes(d, 0, TURNS)["outcome"]["gap"] == "unclear"

def test_sub_stakes_limited_to_their_stake():
    assert validate_axes(reply(), 0, TURNS)["outcome"]["sub_stakes"] == ["commitment"]

def test_read_turn_has_full_frame_and_caches(tmp_path):
    c = Stub(reply())
    a = read_turn(0, TURNS, client=c, model="m", cache_dir=tmp_path)
    b = read_turn(0, TURNS, client=c, model="m", cache_dir=tmp_path)
    assert a == b and c.calls == 1
    assert a["gap_kind"] == "accident" and a["why"] == {"layer": "say", "factor": "no emergency word", "type": "B4"}

def test_invalid_values_become_honest():
    axes = validate_axes(reply(), 0, TURNS)
    out = clean_turn_fields({"gap_kind": "oops", "why": {"layer": "x", "factor": "f", "type": "Z9"}}, axes)
    assert out["gap_kind"] == "unclear" and out["why"]["layer"] == "unclear" and out["why"]["type"] == "none"

def test_no_why_when_nothing_went_wrong():
    assert clean_turn_fields(reply(), validate_axes(reply(gap="met"), 0, TURNS))["why"] is None

def test_bad_json_is_unclear_not_invented(tmp_path):
    r = read_turn(0, TURNS, client=Stub("not json"), model="m", cache_dir=tmp_path)
    assert r["gap_kind"] == "unclear" and r["axes"]["outcome"]["gap"] == "unclear"

def test_last_turn_never_read(tmp_path):
    u = build_uptake(TURNS, client=Stub(reply()), model="m", cache_dir=tmp_path)
    assert [x["turn"] for x in u] == [0, 1]

def test_types_are_23_plus_none():
    assert len(MISREAD_TYPES) == 24

def test_scales_from_panel():
    s = turn_scales({"action_dist": {"comply": .67, "proceed": .33}, "relationship": [0, -1], "face": [0, 0]})
    assert s == {"next_action": "comply", "relationship": -0.5, "face": 0.0}
    assert turn_scales(None)["next_action"] is None


def _u(turn, sp, **gaps):
    axes = {a: {"gap": gaps.get(a, "not_relevant")} for a in ("outcome", "relationship", "face", "understanding")}
    return {"turn": turn, "speaker": sp, "axes": axes}


def test_standing_keeps_stakes_apart_and_shows_win_with_relationship_loss():
    from forecaster.report import build_standing
    s = build_standing([_u(0, "A", outcome="met", relationship="cost"), _u(1, "B", outcome="missed"), _u(2, "A", outcome="met", relationship="missed")])
    assert s["end"]["A"]["outcome"]["score"] == 2.0
    assert s["end"]["A"]["relationship"]["score"] == -2.0
    assert s["end"]["B"]["outcome"]["score"] == -1.0
    assert s["series"]["outcome"]["A"][-1]["value"] == 2.0


def test_split_is_per_stake_and_empty_stake_has_none():
    from forecaster.report import build_standing
    s = build_standing([_u(0, "A", outcome="met"), _u(1, "B", outcome="met"), _u(2, "A", outcome="missed"), _u(3, "B", outcome="met"), _u(4, "A", outcome="missed")])
    assert s["splits"]["outcome"]["turn"] == 1 and s["splits"]["outcome"]["shift"] == -3.0
    assert s["splits"]["face"] is None


def test_unclear_never_scores():
    from forecaster.report import build_standing
    s = build_standing([_u(0, "A", outcome="unclear")])
    assert s["end"]["A"]["outcome"] == {"score": 0.0, "judged": 0, "mean": None}
