import json
import pytest
from forecaster.v2 import (Goal, ReadingV2, STAKES, parse_reading_v2, gaps, any_gap,
                           judge_same_meaning, score_v2, false_alarm_rate, build_messages_v2)

MSG = "Can you get the report to me by Friday? No pressure."


def R(rid, s=0, act="comply", rel=0, face=0, und="due friday"):
    return ReadingV2(receiver_id=rid, sample=s, next_action=act, relationship_shift=rel,
                     face_threat=face, understood=und)


def test_goal_validation():
    with pytest.raises(ValueError):
        Goal(acceptable_actions=[])
    with pytest.raises(ValueError):
        Goal(acceptable_actions=["fly"])
    assert STAKES == ["outcome", "relationship", "face", "understanding"]


def test_parse_v2():
    raw = ('```json\n{"next_action":"Comply","relationship_shift":-3,"face_threat":"2.4",'
           '"understood":"I must deliver by Friday","speech_act":"command","trigger":"No pressure"}\n```')
    r = parse_reading_v2(raw, "r-0", 1, MSG)
    assert (r.next_action, r.relationship_shift, r.face_threat, r.trigger) == ("comply", -2, 2, "No pressure")
    assert parse_reading_v2('{"next_action":"comply","relationship_shift":0,"face_threat":0,"understood":""}', "r", 0, MSG) is None
    assert parse_reading_v2('{"next_action":"dance","relationship_shift":0,"face_threat":0,"understood":"x"}', "r", 0, MSG) is None


def test_gaps_no_pressure_case():
    goal = Goal(acceptable_actions=["comply", "proceed"], relationship_floor=0, face_ceiling=1,
                core_meaning="Friday would be nice but flexible")
    g = gaps(R("a", act="comply", rel=-1, face=1), goal, same_meaning=False)
    assert g == {"outcome": False, "relationship": True, "face": False, "understanding": True}
    assert any_gap(g)
    ok = gaps(R("a", act="proceed", rel=0, face=0), goal, same_meaning=None)
    assert not any_gap(ok)


def test_judge(fake_client_cls):
    c = fake_client_cls(['{"same": false, "why": "flexibility lost"}', "garbage"])
    assert judge_same_meaning(c, "flexible Friday", "hard deadline", model="j") is False
    assert judge_same_meaning(c, "flexible Friday", "hard deadline", model="j") is None
    assert judge_same_meaning(c, "", "anything", model="j") is None
    assert len(c.calls) == 2


def test_score_and_false_alarm():
    goal = Goal(acceptable_actions=["comply"], relationship_floor=0, face_ceiling=1)
    rs = [R("a", 0), R("a", 1, rel=-1), R("b", 0), R("b", 1, act="refuse")]
    same = {("a", 0): True, ("a", 1): True, ("b", 0): False, ("b", 1): None}
    out = score_v2(rs, goal, same, b=500, seed=0)
    assert out["n_readings"] == 4
    assert out["outcome"][0] == 0.25 and out["relationship"][0] == 0.25
    assert out["understanding"][0] == 0.25 and out["face"][0] == 0.0
    assert out["any"][0] == 0.75
    for k in STAKES + ["any"]:
        p, lo, hi = out[k]
        assert lo <= p <= hi
    controls = [{"any": (0.2, 0, 0.4)}, {"any": (0.6, 0.4, 0.8)}, {"any": (0.0, 0, 0)}, {"any": (0.5, 0.3, 0.7)}]
    rate, lo, hi = false_alarm_rate(controls)
    assert rate == 0.25 and lo <= rate <= hi


def test_prompt_v2_hides_intent(base_receiver):
    text = json.dumps(build_messages_v2(MSG, base_receiver), ensure_ascii=False)
    for key in ["relationship_shift", "face_threat", "understood", "next_action"]:
        assert key in text
    assert MSG in text and "intended" not in text.lower()
