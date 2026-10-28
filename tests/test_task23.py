"""Contract for SPEC_task23_stakes (synthetic data only, fake clients)."""
import json

import pytest

from forecaster import stats as S
from forecaster import stakes as ST
from forecaster.transcribe import Turn
from forecaster.usage import BudgetExceeded


def _entry():
    return {
        "dialogue_id": 1,
        "participant_info": {
            "mturk_agent_1": {"value2issue": {
                "High": "Firewood", "Medium": "Food", "Low": "Water"}},
            "mturk_agent_2": {"value2issue": {
                "High": "Firewood", "Medium": "Water", "Low": "Food"}},
        },
    }


def _prio():
    return ST.priorities(_entry())


def test_priorities_and_offer_points_example():
    prio = _prio()
    assert prio["A"] == {"Firewood": 5, "Food": 4, "Water": 3}
    assert prio["B"] == {"Firewood": 5, "Water": 4, "Food": 3}
    offer = {"A": {"Food": 1, "Water": 0, "Firewood": 3},
             "B": {"Food": 2, "Water": 3, "Firewood": 0}}
    assert ST.offer_points(offer, prio) == {"A": 19, "B": 18}


def test_load_priorities(tmp_path):
    p = tmp_path / "casino.json"
    p.write_text(json.dumps([_entry(), {"dialogue_id": "bad"}]))
    got = ST.load_priorities(p)
    assert list(got) == [1]
    assert got[1]["A"]["Firewood"] == 5
    assert got[1]["B"]["Water"] == 4


def _full(sg, og):
    return {"is_offer": True, "speaker_gets": sg, "other_gets": og}


def test_normalize_completes_missing_sides():
    raw = _full({"Food": 2, "Water": 1, "Firewood": 0}, {})
    got = ST.normalize_offer(raw, "A")
    assert got == {"A": {"Food": 2, "Water": 1, "Firewood": 0},
                   "B": {"Food": 1, "Water": 2, "Firewood": 3}}
    # completion works from the other side too
    raw2 = _full({}, {"Food": 2, "Water": 2, "Firewood": 2})
    got2 = ST.normalize_offer(raw2, "A")
    assert got2["A"] == {"Food": 1, "Water": 1, "Firewood": 1}
    assert got2["B"] == {"Food": 2, "Water": 2, "Firewood": 2}


def test_normalize_rejects_bad_counts():
    base_sg = {"Food": 1, "Water": 1, "Firewood": 1}
    base_og = {"Food": 2, "Water": 2, "Firewood": 2}
    # count > 3
    bad = dict(base_sg, Food=4)
    assert ST.normalize_offer(_full(bad, {}), "A") is None
    # sums != 3
    assert ST.normalize_offer(_full({"Food": 2}, {"Food": 2}), "A") is None
    assert ST.normalize_offer(
        _full(dict(base_sg, Food=2), dict(base_og, Food=2)), "A") is None
    # not an offer / bad speaker / unmentioned item
    assert ST.normalize_offer({"is_offer": False}, "A") is None
    assert ST.normalize_offer({}, "A") is None
    assert ST.normalize_offer(_full(base_sg, base_og), "C") is None
    assert ST.normalize_offer(_full({"Food": 1}, {"Food": 2}), "A") is None


def test_normalize_maps_speaker_b():
    raw = _full({"Food": 1, "Water": 1, "Firewood": 1},
                {"Food": 2, "Water": 2, "Firewood": 2})
    got = ST.normalize_offer(raw, "B")
    assert got == {"A": {"Food": 2, "Water": 2, "Firewood": 2},
                   "B": {"Food": 1, "Water": 1, "Firewood": 1}}


class _FakeOfferClient:
    accepts_meta = True

    def __init__(self, by_turn):
        self.by_turn = dict(by_turn)
        self.turns = []

    def complete(self, messages, model, temperature, meta=None):
        self.turns.append(meta["turn"])
        return self.by_turn.get(meta["turn"], "no json here")


def _offer_raw(sg, og):
    return json.dumps({"is_offer": True, "speaker_gets": sg, "other_gets": og})


def _turns():
    return [Turn("A", "hello", 0), Turn("B", "hi back", 0),
            Turn("A", "take it", 0), Turn("B", "deal", 0)]


def test_extract_offers_only_offer_turns_and_skips_bad_json():
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
        {"turn_index": 1, "speaker": "B", "moves": ["inform"]},
        {"turn_index": 2, "speaker": "A", "moves": ["smalltalk"]},
        {"turn_index": 3, "speaker": "B", "moves": ["accept"]},
    ]
    sg = {"Food": 1, "Water": 1, "Firewood": 1}
    og = {"Food": 2, "Water": 2, "Firewood": 2}
    client = _FakeOfferClient({0: _offer_raw(sg, og),
                               3: "```json\n" + _offer_raw(og, sg) + "\n```"})
    got = ST.extract_offers(client, _turns(), moves, "m", item_id="casino-1")
    assert sorted(client.turns) == [0, 3]
    assert set(got) == {0, 3}
    assert got[0]["A"] == sg
    assert got[3]["B"] == og  # speaker B's share lands on B
    assert got[3]["A"] == sg


def test_extract_offers_skips_bad_json():
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["counter"]},
        {"turn_index": 1, "speaker": "B", "moves": ["accept"]},
    ]
    sg = {"Food": 1, "Water": 1, "Firewood": 1}
    og = {"Food": 2, "Water": 2, "Firewood": 2}
    client = _FakeOfferClient({0: "not json at all",
                               1: _offer_raw(sg, og)})
    got = ST.extract_offers(client, _turns()[:2], moves, "m")
    assert sorted(client.turns) == [0, 1]
    assert set(got) == {1}


def test_extract_offers_budget_propagates():
    class _Boom:
        def complete(self, messages, model, temperature):
            raise BudgetExceeded("over")

    moves = [{"turn_index": 0, "speaker": "A", "moves": ["propose"]}]
    with pytest.raises(BudgetExceeded):
        ST.extract_offers(_Boom(), _turns()[:1], moves, "m")


def test_outcome_track_carries_latest_forward():
    prio = _prio()
    o1 = {"A": {"Food": 1, "Water": 0, "Firewood": 3},
          "B": {"Food": 2, "Water": 3, "Firewood": 0}}
    o2 = {"A": {"Food": 3, "Water": 3, "Firewood": 3},
          "B": {"Food": 0, "Water": 0, "Firewood": 0}}
    track = ST.outcome_track(4, {1: o1, 3: o2}, prio)
    assert track[0] == {"A": None, "B": None, "offer_turn": None}
    assert track[1] == {"A": 19, "B": 18, "offer_turn": 1}
    assert track[2] == {"A": 19, "B": 18, "offer_turn": 1}
    assert track[3] == {"A": 36, "B": 0, "offer_turn": 3}


def _track_row(a, b, t):
    return {"A": a, "B": b, "offer_turn": t}


def test_evaluate_e1_share_and_coverage():
    tracks = {
        1: [_track_row(None, None, None), _track_row(19, 18, 1)],
        2: [_track_row(None, None, None), _track_row(10, 10, 1)],
        3: [_track_row(None, None, None), _track_row(None, None, None)],
    }
    outcomes = {
        1: {"A": {"points": 19}, "B": {"points": 18}},
        2: {"A": {"points": 10}, "B": {"points": 12}},
        3: {"A": {"points": 5}, "B": {"points": 5}},
    }
    res = ST.evaluate(tracks, outcomes, b=50, seed=0)
    assert res["E1"]["k"] == 1
    assert res["E1"]["n"] == 2
    assert res["E1"]["share"] == pytest.approx(0.5)
    assert res["E1"]["ci"] == S.wilson_ci(1, 2)
    assert res["coverage"] == pytest.approx(2 / 3)
    assert len(res["E2"]) == 3
    assert len(res["E3"]) == 3


def test_build_offer_messages_shape():
    msgs = ST.build_offer_messages(_turns(), 2, window=1)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "TARGET turn A: take it" in msgs[1]["content"]
    assert "B: hi back" in msgs[1]["content"]
    assert "A: hello" not in msgs[1]["content"]
    assert "is_offer" in msgs[1]["content"]
