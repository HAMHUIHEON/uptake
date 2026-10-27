import json
import pytest
from forecaster.panel import allocate, build_panel, DEFAULT_MIX, STANCE_ORDER
from forecaster.v2 import (Goal, ReadingV2, STAKE_TREE, SUB_STAKES, parse_reading_v2, sub_gaps,
                           score_v2, build_messages_v2)
from forecaster.v2_run import run_forecast_v2, load_case_v2


def test_allocate_largest_remainder():
    assert allocate(6, DEFAULT_MIX) == {"neutral": 3, "trusting": 2, "suspicious": 1}
    assert allocate(4, DEFAULT_MIX) == {"neutral": 2, "trusting": 1, "suspicious": 1}
    assert allocate(3, {"neutral": 1, "trusting": 1, "suspicious": 1}) == {"neutral": 1, "trusting": 1, "suspicious": 1}
    assert sum(allocate(7, {"neutral": 2, "suspicious": 1}).values()) == 7
    with pytest.raises(ValueError):
        allocate(3, {"angry": 1})


def test_build_panel_mix(base_receiver):
    p = build_panel(base_receiver, n=6, stance_mix=DEFAULT_MIX)
    assert [r.stance for r in p] == ["neutral"] * 3 + ["trusting"] * 2 + ["suspicious"]
    assert [r.id for r in p] == [f"r-{i}" for i in range(6)]
    assert [r.stance for r in build_panel(base_receiver, n=3)] == ["neutral", "suspicious", "trusting"]
    assert STANCE_ORDER == ["neutral", "trusting", "suspicious"]


def test_stake_tree():
    assert list(STAKE_TREE) == ["outcome", "relationship", "face", "understanding"]
    assert "autonomy" in SUB_STAKES and "money" in SUB_STAKES and len(SUB_STAKES) == 13
    with pytest.raises(ValueError):
        Goal(acceptable_actions=["comply"], sub_stakes=["vibes"])


def test_sub_effects_parse_and_gaps():
    raw = ('{"next_action":"comply","relationship_shift":0,"face_threat":1,"understood":"x",'
           '"sub_effects":{"autonomy":-2,"time":3,"bogus":1}}')
    r = parse_reading_v2(raw, "r-0", 0, "msg")
    assert r.sub_effects == {"autonomy": -2, "time": 2}
    goal = Goal(acceptable_actions=["comply"], sub_stakes=["autonomy", "time", "trust"], sub_floor=-1)
    assert sub_gaps(r, goal) == {"autonomy": True, "time": False, "trust": False}


def test_prompt_asks_for_subs(base_receiver):
    text = json.dumps(build_messages_v2("hi", base_receiver, subs=["autonomy", "time"]))
    assert "sub_effects" in text and "autonomy" in text and "time" in text
    assert "sub_effects" not in json.dumps(build_messages_v2("hi", base_receiver))


def test_score_subs():
    goal = Goal(acceptable_actions=["comply"], sub_stakes=["autonomy"])
    rs = [ReadingV2("a", 0, "comply", 0, 0, "x", sub_effects={"autonomy": -2}),
          ReadingV2("b", 0, "comply", 0, 0, "x", sub_effects={"autonomy": 0})]
    out = score_v2(rs, goal, {("a", 0): None, ("b", 0): None}, b=300, seed=0)
    assert out["subs"]["autonomy"][0] == 0.5 and out["any"][0] == 0.0


class Client:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, model, temperature):
        self.calls += 1
        text = json.dumps(messages)
        if "same" in text and "relationship_shift" not in text:
            return '{"same": true, "why": ""}'
        assert "sub_effects" in text
        return '{"next_action":"comply","relationship_shift":0,"face_threat":0,"understood":"ok","sub_effects":{"time":-2}}'


def test_pipeline_uses_mix_and_subs(base_receiver, tmp_path):
    goal = Goal(acceptable_actions=["comply"], core_meaning="ok", sub_stakes=["time"])
    out = run_forecast_v2(Client(), "m", goal, base_receiver, n=4, k=1, b=200, stance_mix=DEFAULT_MIX)
    assert [r.stance for r in out["panel"]] == ["neutral", "neutral", "trusting", "suspicious"]
    assert out["score"]["subs"]["time"][0] == 1.0
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"message": "m", "goal": {"acceptable_actions": ["comply"], "sub_stakes": ["time"]},
                             "receiver": {"id": "x", "relationship": "r", "power": "equal", "channel": "c",
                                          "state": "s", "stance_mix": {"neutral": 1}}}))
    msg, g, rcv, mix = load_case_v2(p)
    assert g.sub_stakes == ["time"] and mix == {"neutral": 1}
