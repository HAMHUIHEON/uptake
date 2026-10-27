import math
import json
import pytest
from forecaster.board import actual_class, forecast_skill, evaluate_turns
from forecaster.moves import MoveTag
from forecaster.transcribe import Turn


def test_actual_class_priority():
    assert actual_class(["inform", "reject"]) == "refuse"
    assert actual_class(["smalltalk", "propose"]) == "negotiate"
    assert actual_class(["challenge", "pressure"]) == "escalate"
    assert actual_class(["praise"]) == "repair"
    assert actual_class(["joke"]) == "proceed" and actual_class([]) == "proceed"


def test_skill_positive_and_zero():
    good = [{"actual_class": c, "action_dist": {c: 1.0}, "n_readings": 4, "predicted": c, "hit": True}
            for c in ["refuse", "comply", "negotiate", "repair", "refuse", "comply"]]
    s = forecast_skill(good, b=300, seed=0)
    assert s["skill_mean"] > 0 and s["skill_ci"][0] > 0 and s["n"] == 6
    p_model = (4 + 0.5) / (4 + 5)
    seen_twice = math.log(p_model) - math.log((1 + 0.5) / (5 + 5))   # refuse, comply turns (4)
    seen_once = math.log(p_model) - math.log((0 + 0.5) / (5 + 5))    # negotiate, repair turns (2)
    assert s["skill_mean"] == pytest.approx((4 * seen_twice + 2 * seen_once) / 6)
    flat = [{"actual_class": "proceed", "action_dist": {"proceed": 1.0}, "n_readings": 4, "predicted": "proceed", "hit": True}
            for _ in range(5)]
    f = forecast_skill(flat, b=300, seed=0)
    assert f["majority_class"] == "proceed" and f["majority_hit_rate"] == 1.0
    assert abs(f["skill_mean"]) < 0.25


class Spy:
    def __init__(self):
        self.listeners = []

    def complete(self, messages, model, temperature):
        text = json.dumps(messages)
        for l in ["A", "B", "C"]:
            if f"You are {l}" in text or f"you are {l}" in text.lower():
                pass
        self.listeners.append(text)
        return '{"next_action":"proceed","relationship_shift":0,"face_threat":0,"understood":"ok"}'


def test_addressed_listener_narrows_panel():
    turns = [Turn("A", "B, what do you think?"), Turn("B", "I agree"), Turn("C", "me too")]
    tags = [MoveTag(0, "A", ["ask"], addressed_to="B"), MoveTag(1, "B", ["inform"], addressed_to="all"),
            MoveTag(2, "C", ["inform"], addressed_to="")]
    eff = evaluate_turns(Spy(), turns, n=2, k=1, b=100, max_workers=1, move_tags=tags)
    assert eff[0]["listeners"] == ["B"] and eff[0]["n_readings"] == 2
    assert eff[1]["listeners"] == ["A", "C"] and eff[2]["listeners"] == ["A", "B"]
