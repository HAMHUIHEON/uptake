import json
from forecaster.moves import MOVES, CASINO_MAP, MOVE_EXAMPLES, build_move_messages, tag_conversation
from forecaster.eval_moves import project, CASINO_SCHEME
from forecaster.casino import load_casino
from forecaster.transcribe import Turn


def test_reveal_preference():
    assert MOVES[-2:] == ["reveal_preference", "other"]
    assert CASINO_MAP["no-need"] == "reveal_preference"
    for k in ["cooperate", "appeal_fairness", "ask", "concede", "reveal_preference", "propose"]:
        assert MOVE_EXAMPLES[k]


def test_prompt_has_examples_and_guidance():
    text = json.dumps(build_move_messages([Turn("A", "hi"), Turn("B", "We don't need water")], 1))
    assert "We don't really need water" in text and "rhetorical" in text


def test_project():
    pred = [{"propose", "appeal_need"}, {"accept"}, {"ask"}]
    assert project(pred, CASINO_SCHEME) == [{"appeal_need"}, {"other"}, {"ask"}]
    assert "other" in CASINO_SCHEME and "propose" not in CASINO_SCHEME


def test_exclude_ids(tmp_path):
    d = [{"dialogue_id": i, "chat_logs": [{"id": "mturk_agent_1", "text": f"t{i}"}],
          "annotations": [[f"t{i}", "small-talk"]]} for i in range(10)]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d))
    out = load_casino(p, limit=5, seed=1, exclude_ids={0, 1, 2, 3, 4})
    assert len(out) == 5 and all(x["dialogue_id"] >= 5 for x in out)


class EffortClient:
    def __init__(self):
        self.efforts = []

    def complete(self, messages, model, temperature, reasoning_effort=None):
        self.efforts.append(reasoning_effort)
        return '{"moves":["smalltalk"]}'


class PlainClient:
    def complete(self, messages, model, temperature):
        return '{"moves":["smalltalk"]}'


def test_effort_passthrough():
    turns = [Turn("A", "hi"), Turn("B", "hello")]
    c = EffortClient()
    tag_conversation(c, turns, model="m", reasoning_effort="low", max_workers=1)
    assert c.efforts == ["low", "low"]
    assert len(tag_conversation(PlainClient(), turns, model="m", max_workers=1)) == 2
