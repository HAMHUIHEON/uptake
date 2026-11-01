"""Task70 step 1: per-turn four-stake reader (stub client, no network)."""
import json

import pytest

from forecaster import axes as A
from forecaster import v2 as V2
from forecaster.usage import BudgetExceeded, Meter


TURNS = [
    {"speaker": "A", "text": "We need priority, we are low on fuel."},
    {"speaker": "B", "text": "Roger, continue as cleared for now."},
    {"speaker": "A", "text": "Confirming we continue, fuel critical."},
    {"speaker": "B", "text": "Cleared direct to the field, report short final."},
    {"speaker": "A", "text": "Direct, thank you."},
]


def _reply(quote, gap="missed", reply_turn=1, wanted="priority now",
           happened="kept on the same course", sub_stakes=None):
    axes = {}
    for axis in A.AXES:
        if sub_stakes is None:
            subs = list(V2.STAKE_TREE[axis][:1])
        elif isinstance(sub_stakes, dict):
            subs = list(sub_stakes.get(axis, []))
        else:
            subs = list(sub_stakes)
        axes[axis] = {"wanted": wanted, "sub_stakes": subs,
                      "happened": happened, "gap": gap,
                      "reply_turn": reply_turn, "quote": quote}
    return json.dumps({"axes": axes})


class _Stub:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0
        self.messages = None

    def complete(self, messages, model=None, temperature=0.0, **kwargs):
        self.calls += 1
        self.messages = messages
        if not self.replies:
            raise RuntimeError("no more replies")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_axis_names_equal_v2_stakes():
    assert tuple(A.AXES) == tuple(V2.STAKES)
    assert set(A.AXES) == {"outcome", "relationship", "face",
                           "understanding"}
    assert "result" not in set(A.AXES)


def test_one_call_returns_four_axes(tmp_path):
    quote = "continue as cleared"
    client = _Stub([_reply(quote)])
    got = A.read_axes(0, TURNS, client=client, model="stub",
                      meter=Meter(budget_usd=5.0), cache_dir=tmp_path)
    assert client.calls == 1
    assert set(got) == set(V2.STAKES)
    for axis in V2.STAKES:
        entry = got[axis]
        assert set(("wanted", "sub_stakes", "happened", "gap",
                    "reply_turn", "quote")) <= set(entry)
        assert entry["gap"] == "missed"
        assert entry["reply_turn"] == 1
        assert entry["quote"] == quote
        assert entry["sub_stakes"] == list(V2.STAKE_TREE[axis][:1])
    # Prior 3 + target + next 3 are shown; one call only.
    user = client.messages[1]["content"]
    assert "TARGET #0" in user
    assert "Cleared direct" in user


def test_sub_stakes_limited_to_stake_tree(tmp_path):
    quote = "continue as cleared"
    subs = {"outcome": ["money", "bogus", "money", "trust"],
            "relationship": "trust",
            "face": ["status"],
            "understanding": []}
    client = _Stub([_reply(quote, sub_stakes=subs)])
    got = A.read_axes(0, TURNS, client=client, model="stub",
                      meter=Meter(budget_usd=5.0), cache_dir=tmp_path)
    assert got["outcome"]["sub_stakes"] == ["money"]
    assert got["relationship"]["sub_stakes"] == []
    assert got["face"]["sub_stakes"] == ["status"]
    assert got["understanding"]["sub_stakes"] == []


def test_invalid_quote_becomes_unclear(tmp_path):
    client = _Stub([_reply("words never said anywhere")])
    got = A.read_axes(0, TURNS, client=client, model="stub",
                      meter=Meter(budget_usd=5.0), cache_dir=tmp_path)
    for axis in A.AXES:
        assert got[axis]["gap"] == "unclear"
        assert got[axis]["quote"] == "words never said anywhere"


def test_reply_out_of_window_becomes_unclear(tmp_path):
    quote = "Direct, thank you."
    client = _Stub([_reply(quote, reply_turn=4)])
    got = A.read_axes(0, TURNS, client=client, model="stub",
                      meter=Meter(budget_usd=5.0), cache_dir=tmp_path)
    for axis in A.AXES:
        assert got[axis]["gap"] == "unclear"


def test_unparseable_json_reads_unclear_without_cache(tmp_path):
    client = _Stub(["not json at all"])
    got = A.read_axes(0, TURNS, client=client, model="stub",
                      meter=Meter(budget_usd=5.0), cache_dir=tmp_path)
    for axis in A.AXES:
        assert got[axis]["gap"] == "unclear"
    assert list(tmp_path.glob("axes-*.json")) == []
    # A second call retries the model (failures are not cached).
    client.replies.append(_reply("continue as cleared"))
    got2 = A.read_axes(0, TURNS, client=client, model="stub",
                       meter=Meter(budget_usd=5.0), cache_dir=tmp_path)
    assert client.calls == 2
    assert got2["outcome"]["gap"] == "missed"


def test_cache_reuses_identical_input(tmp_path):
    client = _Stub([_reply("continue as cleared")])
    meter = Meter(budget_usd=5.0)
    first = A.read_axes(0, TURNS, client=client, model="stub",
                        meter=meter, cache_dir=tmp_path)
    assert client.calls == 1
    client2 = _Stub([RuntimeError("must not be called")])
    second = A.read_axes(0, TURNS, client=client2, model="stub",
                         meter=Meter(budget_usd=5.0), cache_dir=tmp_path)
    assert client2.calls == 0
    assert second == first


def test_budget_stop_propagates_without_call(tmp_path):
    client = _Stub([_reply("continue as cleared")])
    with pytest.raises(BudgetExceeded):
        A.read_axes(0, TURNS, client=client, model="stub",
                    meter=Meter(budget_usd=0.0), cache_dir=tmp_path)
    assert client.calls == 0


def test_last_turn_and_offline_are_unclear(tmp_path):
    client = _Stub([RuntimeError("must not be called")])
    last = A.read_axes(len(TURNS) - 1, TURNS, client=client,
                       model="stub", meter=Meter(budget_usd=5.0),
                       cache_dir=tmp_path)
    assert client.calls == 0
    for axis in A.AXES:
        assert last[axis]["gap"] == "unclear"
    off = A.read_axes(0, TURNS)
    for axis in A.AXES:
        assert off[axis]["gap"] == "unclear"


def test_build_uptake_skips_last_turn(tmp_path):
    quote = "continue as cleared"
    client = _Stub([_reply(quote)] * 4)
    uptake = A.build_uptake(TURNS, client=client, model="stub",
                            meter=Meter(budget_usd=5.0),
                            cache_dir=tmp_path)
    assert [u["turn"] for u in uptake] == [0, 1, 2, 3]
    assert uptake[0]["speaker"] == "A"
    assert set(uptake[0]["axes"]) == set(A.AXES)


def test_build_uptake_max_turns_caps_entries(tmp_path):
    client = _Stub([_reply("x")] * 4)
    uptake = A.build_uptake(TURNS, client=client, model="stub",
                            meter=Meter(budget_usd=5.0),
                            cache_dir=tmp_path, max_turns=2)
    assert [u["turn"] for u in uptake] == [0, 1]
    assert client.calls == 2
