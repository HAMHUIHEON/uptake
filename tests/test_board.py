import json
import pytest
from forecaster.board import (build_listener_messages, evaluate_turns, position_curve, annotate_turn,
                              turning_point, disagreement_spike, forecast_hits, forecast_score, ACTION_TO_MOVES)
from forecaster.moves import MoveTag, MOVES
from forecaster.transcribe import Turn

TURNS = [Turn("A", "Hey, thanks for meeting."), Turn("B", "Sure. What do you want?"),
         Turn("A", "Honestly your last report was sloppy."), Turn("B", "Excuse me?"),
         Turn("A", "Sorry, that came out wrong.")]


def test_listener_prompt_causal():
    from forecaster.types import Receiver
    r = Receiver(id="B@2", relationship="participant in this conversation", power="equal", channel="conversation", state="")
    text = json.dumps(build_listener_messages(TURNS, 2, "B", r, window=1))
    assert "TARGET" in text and "sloppy" in text and "What do you want" in text
    assert "thanks for meeting" not in text and "Excuse me" not in text and "came out wrong" not in text
    assert "relationship_shift" in text and "face_threat" in text


class Client:
    def complete(self, messages, model, temperature):
        target = json.dumps(messages).split("TARGET")[-1]
        if "sloppy" in target:
            return '{"next_action":"retaliate","relationship_shift":-2,"face_threat":2,"understood":"insult"}'
        if "Sorry" in target:
            return '{"next_action":"repair","relationship_shift":1,"face_threat":0,"understood":"apology"}'
        return '{"next_action":"proceed","relationship_shift":0,"face_threat":0,"understood":"ok"}'


def test_evaluate_and_curve():
    eff = evaluate_turns(Client(), TURNS, n=3, k=1, b=200, max_workers=4)
    assert [e["index"] for e in eff] == [0, 1, 2, 3, 4]
    assert eff[2]["listeners"] == ["B"] and eff[2]["n_readings"] == 3
    assert eff[2]["relationship"][0] == -2 and eff[2]["face"][0] == -2
    assert eff[2]["action_dist"] == {"retaliate": 1.0} and eff[2]["disagreement"] == 0.0
    vals = [e["relationship"][0] + e["face"][0] for e in eff]
    assert vals == [0, 0, -4, 0, 1]
    assert position_curve(eff) == [0, 0, -4, -4, -3]
    assert turning_point(eff) == 2


def test_annotate_turn():
    assert [annotate_turn(v) for v in (-2, -1, -0.5, 0, 0.8, 1.6)] == ["??", "?", "?!", "=", "!", "!!"]


def test_disagreement_spike():
    eff = [{"disagreement": d, "relationship": (0, 0, 0), "face": (0, 0, 0)} for d in (0.1, 0.12, 0.1, 0.11, 0.9, 0.2)]
    assert disagreement_spike(eff, z=2.0, min_turn=3) == 4
    flat = [{"disagreement": 0.1, "relationship": (0, 0, 0), "face": (0, 0, 0)} for _ in range(6)]
    assert disagreement_spike(flat) is None


def test_forecast_hits_and_score():
    assert set().union(*ACTION_TO_MOVES.values()) <= set(MOVES)
    eff = [{"index": 0, "speaker": "A", "listeners": ["B"], "action_dist": {"retaliate": 0.6, "repair": 0.4}},
           {"index": 1, "speaker": "B", "listeners": ["A"], "action_dist": {"proceed": 1.0}},
           {"index": 2, "speaker": "A", "listeners": ["B"], "action_dist": {"ask_clarification": 0.5, "comply": 0.5}}]
    tags = [MoveTag(0, "A", ["challenge"]), MoveTag(1, "B", ["challenge"]), MoveTag(2, "A", ["propose"]),
            MoveTag(3, "B", ["accept"])]
    hits = forecast_hits(eff, tags)
    assert [(h["index"], h["predicted"], h["hit"]) for h in hits] == [(0, "retaliate", True), (1, "proceed", False),
                                                                      (2, "ask_clarification", False)]
    assert [h["actual_class"] for h in hits] == ["retaliate", "negotiate", "comply"]
    assert hits[0]["p_actual"] == pytest.approx(0.6) and hits[2]["p_actual"] == pytest.approx(0.5)
    s = forecast_score(hits, b=200, seed=0)
    assert s["hit_rate"] == pytest.approx(1 / 3)
    lo, hi = s["hit_rate_ci"]
    assert lo <= 1 / 3 <= hi and "baseline" in s
