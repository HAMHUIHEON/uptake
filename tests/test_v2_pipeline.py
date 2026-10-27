import json
from pathlib import Path

from forecaster.v2 import Goal
from forecaster.v2_run import run_forecast_v2, annotate, load_case_v2

READ_OK = '{"next_action":"comply","relationship_shift":0,"face_threat":0,"understood":"Friday if possible","speech_act":"request","trigger":""}'
READ_BAD = '{"next_action":"comply","relationship_shift":-1,"face_threat":2,"understood":"Hard Friday deadline","speech_act":"command","trigger":"No pressure"}'


class Client:
    def __init__(self, bad_word):
        self.bad_word = bad_word
        self.calls = 0

    def complete(self, messages, model, temperature):
        self.calls += 1
        text = json.dumps(messages, ensure_ascii=False)
        if "same" in text and "relationship_shift" not in text:  # judge prompt
            return '{"same": false, "why": "x"}' if "Hard Friday" in text else '{"same": true, "why": "y"}'
        return READ_BAD if self.bad_word in text else READ_OK


def test_run_v2_end_to_end(base_receiver):
    goal = Goal(acceptable_actions=["comply", "proceed"], core_meaning="Friday is preferred but flexible")
    c = Client(bad_word="No pressure")
    out = run_forecast_v2(c, "Report by Friday? No pressure.", goal, base_receiver, n=3, k=2, b=300, max_workers=4)
    assert len(out["readings"]) == 6 and c.calls == 12
    s = out["score"]
    assert s["outcome"][0] == 0.0 and s["relationship"][0] == 1.0 and s["face"][0] == 1.0
    assert s["understanding"][0] == 1.0 and s["any"][0] == 1.0
    assert out["annotation"] == "??"
    good = run_forecast_v2(Client(bad_word="NEVER"), "Could you aim for Friday?", goal, base_receiver, n=3, k=2, b=300)
    assert good["score"]["any"][0] == 0.0 and good["annotation"] == "!"
    assert [(r.receiver_id, r.sample) for r in out["readings"]] == [(f"r-{i}", s) for i in range(3) for s in range(2)]


def test_annotate_thresholds():
    mk = lambda p: {"any": (p, p, p)}
    assert [annotate(mk(p)) for p in (0.0, 0.15, 0.35, 0.6, 0.85)] == ["!", "=", "?!", "?", "??"]


def test_load_case_v2(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"message": "hi", "goal": {"acceptable_actions": ["proceed"], "core_meaning": "hello"},
                             "receiver": {"id": "x", "relationship": "friends", "power": "equal",
                                          "channel": "text", "state": "relaxed"}}), encoding="utf-8")
    msg, goal, rcv, _mix = load_case_v2(p)
    assert msg == "hi" and goal.acceptable_actions == ["proceed"] and rcv.id == "x"


def test_controls_exist_and_load():
    d = Path(__file__).resolve().parents[1] / "examples" / "controls"
    files = sorted(d.glob("*.json"))
    assert len(files) >= 6
    for f in files:
        load_case_v2(f)
