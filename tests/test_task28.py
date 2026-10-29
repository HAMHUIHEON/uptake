"""Contract for SPEC_task28_rescore (synthetic only, fake clients)."""
import json
import random

import pytest

from forecaster import board as B
from forecaster import outcome
from forecaster import rescore as R
from forecaster.panel import DEFAULT_MIX
from forecaster.transcribe import Turn
from forecaster.types import Receiver


def _receiver(rid="B@1"):
    return Receiver(id=rid, relationship="participant in this conversation",
                    power="equal", channel="conversation", state="", stance="neutral")


def _turns():
    return [Turn("A", "hey there"), Turn("B", "hello back"), Turn("A", "take this deal")]


# prompts

def test_prompt_v2_equals_board():
    turns = _turns()
    r = _receiver()
    assert R.build_scaled_messages(turns, 1, "B", r, "v2") == B.build_listener_messages(turns, 1, "B", r, 6)
    assert R.build_scaled_messages(turns, 1, "B", r, "v2", window=1) == B.build_listener_messages(turns, 1, "B", r, 1)


def test_prompt_fine100_keys_and_ranges():
    turns = _turns()
    msgs = R.build_scaled_messages(turns, 1, "A", _receiver("A@1"), "fine100")
    assert [m["role"] for m in msgs] == ["system", "user"]
    sys = msgs[0]["content"]
    assert "relationship_after" in sys
    assert "face_effect" in sys
    assert "face_threat" not in sys
    assert "0 to 100" in sys
    assert "50 = unchanged" in sys
    assert "next_action, relationship_after, face_effect, understood" in sys
    assert "TARGET turn B: hello back" in msgs[1]["content"]


def test_prompt_seven_keys_and_ranges():
    turns = _turns()
    msgs = R.build_scaled_messages(turns, 2, "B", _receiver("B@2"), "seven")
    sys = msgs[0]["content"]
    assert "relationship_shift" in sys
    assert "face_effect" in sys
    assert "face_threat" not in sys
    assert "-3" in sys and "3" in sys
    assert "next_action, relationship_shift, face_effect, understood" in sys


def test_prompt_unknown_scale_raises():
    with pytest.raises(ValueError):
        R.build_scaled_messages(_turns(), 0, "B", _receiver(), "bad")


# parse

def test_parse_v2():
    got = R.parse_scaled('{"next_action":"proceed","relationship_shift":-1,"face_threat":2,"understood":"ok"}', "v2")
    assert got == {"next_action": "proceed", "rel_raw": -1, "face_raw": 2, "rel_norm": -1.0, "face_norm": -2.0}


def test_parse_fine100():
    got = R.parse_scaled('{"next_action":"proceed","relationship_after":75,"face_effect":50,"understood":"ok"}', "fine100")
    assert got["rel_raw"] == 75 and got["face_raw"] == 50
    assert got["rel_norm"] == pytest.approx(1.0)
    assert got["face_norm"] == pytest.approx(0.0)


def test_parse_seven():
    got = R.parse_scaled('{"next_action":"repair","relationship_shift":3,"face_effect":-3,"understood":"ok"}', "seven")
    assert got["rel_raw"] == 3 and got["face_raw"] == -3
    assert got["rel_norm"] == pytest.approx(2.0)
    assert got["face_norm"] == pytest.approx(-2.0)


def test_parse_clamping():
    got = R.parse_scaled('{"next_action":"proceed","relationship_after":150,"face_effect":-20,"understood":"ok"}', "fine100")
    assert got["rel_raw"] == 100 and got["face_raw"] == 0
    assert got["rel_norm"] == pytest.approx(2.0)
    assert got["face_norm"] == pytest.approx(-2.0)
    got2 = R.parse_scaled('{"next_action":"proceed","relationship_shift":9,"face_effect":-9,"understood":"ok"}', "seven")
    assert got2["rel_raw"] == 3 and got2["face_raw"] == -3
    got3 = R.parse_scaled('{"next_action":"proceed","relationship_shift":-9,"face_threat":9,"understood":"ok"}', "v2")
    assert got3["rel_raw"] == -2 and got3["face_raw"] == 2


def test_parse_string_numbers_and_case():
    got = R.parse_scaled('{"next_action":"Proceed","relationship_after":"75","face_effect":"50.0","understood":"ok"}', "fine100")
    assert got["next_action"] == "proceed"
    assert got["rel_raw"] == 75 and got["face_raw"] == 50


def test_parse_missing_key_bool_nonnumeric():
    assert R.parse_scaled('{"next_action":"proceed","relationship_after":60,"understood":"ok"}', "fine100") is None
    assert R.parse_scaled('{"next_action":"proceed","face_effect":60,"understood":"ok"}', "fine100") is None
    assert R.parse_scaled('{"next_action":"proceed","relationship_after":true,"face_effect":50,"understood":"ok"}', "fine100") is None
    assert R.parse_scaled('{"next_action":"proceed","relationship_after":"warm","face_effect":50,"understood":"ok"}', "fine100") is None
    assert R.parse_scaled('{"relationship_after":60,"face_effect":50,"understood":"ok"}', "fine100") is None


def test_parse_bad_next_action():
    assert R.parse_scaled('{"next_action":"dance","relationship_after":60,"face_effect":50,"understood":"ok"}', "fine100") is None
    assert R.parse_scaled('{"next_action":"","relationship_shift":0,"face_effect":0,"understood":"ok"}', "seven") is None


def test_parse_prose_before_json():
    raw = 'Sure, here you go: {"next_action":"proceed","relationship_after":60,"face_effect":40,"understood":"ok"} trailing'
    got = R.parse_scaled(raw, "fine100")
    assert got["rel_raw"] == 60 and got["face_raw"] == 40
    assert R.parse_scaled("no json here", "v2") is None
    assert R.parse_scaled("[1,2,3]", "v2") is None


# evaluate_turns_scaled

class _Fixed:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def complete(self, messages, model, temperature):
        self.calls += 1
        return self.raw


class _MetaCap:
    accepts_meta = True

    def __init__(self, raw):
        self.raw = raw
        self.metas = []

    def complete(self, messages, model, temperature, meta=None):
        self.metas.append(dict(meta or {}))
        return self.raw


class _GarbageOne:
    accepts_meta = True

    def __init__(self, good):
        self.good = good

    def complete(self, messages, model, temperature, meta=None):
        if meta is not None and str(meta.get("member", "")).endswith("-0"):
            return "garbage no json"
        return self.good


def test_evaluate_value_is_rel_plus_face():
    raw = json.dumps({"next_action": "proceed", "relationship_after": 75, "face_effect": 50, "understood": "ok"})
    eff = R.evaluate_turns_scaled(_Fixed(raw), _turns(), scale="fine100", n=2, k=1, b=50, max_workers=2)
    assert len(eff) == 3
    for e in eff:
        assert e["scale"] == "fine100"
        assert e["n_expected"] == 2 * 1 * len(e["listeners"])
        assert e["relationship"][0] == pytest.approx(1.0)
        assert e["face"][0] == pytest.approx(0.0)
        assert B._value_of(e) == pytest.approx(1.0)
        assert e["n_readings"] == e["n_expected"]
        for row in e["readings"]:
            assert set(("turn", "listener", "member", "stance", "sample", "next_action", "rel_raw", "face_raw", "rel_norm", "face_norm")) <= set(row)


def test_evaluate_seven_value():
    raw = json.dumps({"next_action": "repair", "relationship_shift": 3, "face_effect": 3, "understood": "ok"})
    eff = R.evaluate_turns_scaled(_Fixed(raw), _turns(), scale="seven", n=1, k=2, b=50, max_workers=2)
    for e in eff:
        assert B._value_of(e) == pytest.approx(4.0)


def test_evaluate_v2_matches_board_listeners():
    raw = json.dumps({"next_action": "proceed", "relationship_shift": 1, "face_threat": 1, "understood": "ok"})
    turns = _turns()
    got = R.evaluate_turns_scaled(_Fixed(raw), turns, scale="v2", n=2, k=1, b=50, max_workers=2)
    want = B.evaluate_turns(_Fixed(raw), turns, n=2, k=1, b=50, max_workers=2)
    assert [e["listeners"] for e in got] == [e["listeners"] for e in want]
    assert [e["n_readings"] for e in got] == [e["n_readings"] for e in want]
    for e in got:
        assert e["n_expected"] == 2 * 1 * len(e["listeners"])
        for row in e["readings"]:
            assert row["relationship_shift"] == row["rel_raw"]
            assert row["face_threat"] == row["face_raw"]


def test_evaluate_garbage_one_member():
    raw = json.dumps({"next_action": "proceed", "relationship_after": 60, "face_effect": 60, "understood": "ok"})
    eff = R.evaluate_turns_scaled(_GarbageOne(raw), _turns(), scale="fine100", n=3, k=1, b=50, max_workers=2)
    for e in eff:
        assert e["n_expected"] == 3
        assert e["n_readings"] < e["n_expected"]
        assert e["n_readings"] == 2


def test_evaluate_meta_includes_scale():
    raw = json.dumps({"next_action": "proceed", "relationship_shift": 0, "face_effect": 0, "understood": "ok"})
    client = _MetaCap(raw)
    R.evaluate_turns_scaled(client, _turns(), scale="seven", n=1, k=1, b=10, max_workers=1)
    assert client.metas
    for m in client.metas:
        assert m["scale"] == "seven"
        assert m["layer"] == "board"


# select_ids

def _write(tmp_path, name):
    (tmp_path / name).write_text("{}")


def test_select_ids_sort_and_sample(tmp_path):
    _write(tmp_path, "casino-b.json")
    _write(tmp_path, "casino-a.json")
    _write(tmp_path, "casino-c.json")
    _write(tmp_path, "other.json")
    got = R.select_ids(tmp_path)
    assert got == ["casino-a", "casino-b", "casino-c"]
    want = random.Random(6).sample(got, 2)
    assert R.select_ids(tmp_path, sample=2, sample_seed=6) == want
    assert R.select_ids(tmp_path, sample=99) == R.select_ids(tmp_path, sample=99, sample_seed=6)
    assert len(R.select_ids(tmp_path, sample=99)) == 3


def test_select_ids_file_order(tmp_path):
    _write(tmp_path, "casino-a.json")
    _write(tmp_path, "casino-b.json")
    _write(tmp_path, "casino-c.json")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("casino-c\nmissing-id\ncasino-a\n")
    assert R.select_ids(tmp_path, ids_file=str(ids_file)) == ["casino-c", "casino-a"]


# rescore_item + outcome compatibility

def _record():
    return {
        "id": "casino-1",
        "genre": "negotiation",
        "model": "old-model",
        "moves": [
            {"turn_index": 0, "speaker": "A", "moves": ["propose"], "addressed_to": "", "quote": "", "failed": False},
            {"turn_index": 1, "speaker": "B", "moves": ["accept"], "addressed_to": "", "quote": "", "failed": False},
            {"turn_index": 2, "speaker": "A", "moves": ["inform"], "addressed_to": "", "quote": "", "failed": False},
        ],
        "effects": [
            {"index": 2, "speaker": "A", "text": "third"},
            {"index": 0, "speaker": "A", "text": "first"},
            {"index": 1, "speaker": "B", "text": "second"},
        ],
        "board": [],
        "hits": [],
        "gold": [["propose"], ["accept"], ["inform"]],
    }


def test_turns_from_record_sorted():
    turns = R.turns_from_record(_record())
    assert [(t.speaker, t.text) for t in turns] == [("A", "first"), ("B", "second"), ("A", "third")]


def test_rescore_item_writes_record(tmp_path):
    rec = _record()
    raw = json.dumps({"next_action": "proceed", "relationship_after": 75, "face_effect": 50, "understood": "ok"})
    out = tmp_path / "out"
    summary = R.rescore_item(_Fixed(raw), rec, out, "fine100", 2, 1, "my-model", 2)
    assert summary["id"] == "casino-1"
    assert summary["n_turns"] == 3
    assert summary["n_readings"] == summary["n_expected"] == 3 * 2
    data = json.loads((out / "casino-1.json").read_text())
    assert data["id"] == "casino-1"
    assert data["genre"] == "negotiation"
    assert data["model"] == "my-model"
    assert data["moves"] == rec["moves"]
    assert data["gold"] == rec["gold"]
    assert data["board"] == data["effects"]
    assert data["scale"] == "fine100"
    assert data["n"] == 2
    assert len(data["effects"]) == 3


def test_outcome_reception_compatible(tmp_path):
    rec = _record()
    raw = json.dumps({"next_action": "proceed", "relationship_after": 75, "face_effect": 75, "understood": "ok"})
    out = tmp_path / "out"
    R.rescore_item(_Fixed(raw), rec, out, "fine100", 1, 1, "m", 1)
    data = json.loads((out / "casino-1.json").read_text())
    mu = 0.5
    got = outcome.reception(data, mu)
    for side in ("A", "B"):
        vals = [B._value_of(e) - mu for e in data["effects"] if e.get("speaker") == side]
        assert got[side] == pytest.approx(sum(vals) / len(vals))
