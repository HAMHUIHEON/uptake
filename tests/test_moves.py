import json
import pytest
from forecaster.moves import MOVES, CASINO_MAP, MoveTag, build_move_messages, parse_moves, tag_conversation
from forecaster.casino import load_casino
from forecaster.eval_moves import prf, eval_with_ci, majority_baseline
from forecaster.transcribe import Turn

TURNS = [Turn("A", "Hi! How's the trip prep going?"), Turn("B", "Good. I really need firewood, my kids get cold."),
         Turn("A", "I could give you 2 firewood if I get 2 food."), Turn("B", "Hmm, would you do 1 food?"),
         Turn("A", "Deal, that's fair.")]


def test_move_sets():
    assert len(MOVES) == 20 and MOVES[-1] == "other"
    assert set(CASINO_MAP.values()) <= set(MOVES)
    assert set(CASINO_MAP) == {"small-talk", "self-need", "other-need", "no-need", "elicit-pref", "uv-part",
                               "vouch-fair", "showing-empathy", "promote-coordination", "non-strategic"}


def test_prompt_is_causal():
    text = json.dumps(build_move_messages(TURNS, 2, window=1))
    assert "I could give you 2 firewood" in text and "I really need firewood" in text
    assert "How's the trip" not in text            # outside window
    assert "would you do 1 food" not in text        # future turn
    for m in ["probe", "pressure", "deflect", "cooperate"]:
        assert m in text


def test_parse_moves():
    raw = 'ok ```json\n{"moves":["Propose","propose","counter","dance","ask","inform"],"addressed_to":"B","quote":"2 firewood"}\n```'
    out = parse_moves(raw, "I could give you 2 firewood if I get 2 food.")
    assert out["moves"] == ["propose", "counter", "ask"] and out["quote"] == "2 firewood" and out["addressed_to"] == "B"
    assert parse_moves('{"moves":["dance"]}', "x") is None
    assert parse_moves('{"moves":["ask"],"quote":"nope"}', "hello")["quote"] == ""


class Fake:
    def __init__(self):
        self.calls = 0

    def complete(self, messages, model, temperature):
        self.calls += 1
        target = messages[-1]["content"]
        if "Deal" in target.split("TARGET")[-1]:
            return '{"moves":["accept","appeal_fairness"],"addressed_to":"B","quote":"Deal"}'
        if self.calls % 2 == 0:
            return "garbage"
        return '{"moves":["inform"],"addressed_to":"all","quote":""}'


def test_tag_conversation_order_and_failures():
    tags = tag_conversation(Fake(), TURNS, model="m", window=2, max_workers=1)
    assert [t.turn_index for t in tags] == [0, 1, 2, 3, 4]
    assert [t.speaker for t in tags] == ["A", "B", "A", "B", "A"]
    assert tags[-1].moves == ["accept", "appeal_fairness"]
    assert all(isinstance(t, MoveTag) for t in tags)


def _casino_fixture(tmp_path):
    d = [{"dialogue_id": 7,
          "chat_logs": [{"id": "mturk_agent_1", "text": "Hello friend!"},
                        {"id": "mturk_agent_2", "text": "Hi! I need water badly."},
                        {"id": "mturk_agent_1", "text": "Submit-Deal"},
                        {"id": "mturk_agent_2", "text": "Accept-Deal"}],
          "annotations": [["Hello friend!", "small-talk"], ["Hi! I need water badly.", "small-talk,self-need"]]},
         {"dialogue_id": 3, "chat_logs": [{"id": "mturk_agent_1", "text": "x"}], "annotations": []}]
    p = tmp_path / "casino.json"
    p.write_text(json.dumps(d))
    return p


def test_load_casino(tmp_path):
    out = load_casino(_casino_fixture(tmp_path))
    assert len(out) == 1 and out[0]["dialogue_id"] == 7
    assert [(t.speaker, t.text) for t in out[0]["turns"]] == [("A", "Hello friend!"), ("B", "Hi! I need water badly.")]
    assert out[0]["gold"] == [{"smalltalk"}, {"smalltalk", "appeal_need"}]


def test_prf_and_ci():
    gold = [{"a"}, {"b"}, {"a", "b"}]
    pred = [{"a"}, {"a"}, {"b"}]
    m = prf(pred, gold)
    assert m["a"]["f1"] == pytest.approx(0.5) and m["b"]["f1"] == pytest.approx(2 / 3)
    assert m["micro_f1"] == pytest.approx(4 / 7) and m["macro_f1"] == pytest.approx((0.5 + 2 / 3) / 2)
    res = eval_with_ci([(pred, gold), (gold, gold)], b=300, seed=0)
    lo, hi = res["micro_f1_ci"]
    assert lo <= res["micro_f1"] <= hi
    assert majority_baseline([{"a"}, {"a", "b"}, {"c"}]) == {"a"}
