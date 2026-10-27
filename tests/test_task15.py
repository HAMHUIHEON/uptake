import json
import pytest
from forecaster.trace import CallLog, LoggingClient
from forecaster.analyze import verdict, percentile, confusion, variance_split


class Inner:
    def __init__(self):
        self.last_stats = {"retries": 1, "status_codes": [429], "prompt_tokens": 10, "completion_tokens": 5}

    def complete(self, messages, model, temperature, reasoning_effort=None):
        return "RAW"


class Boom:
    def complete(self, messages, model, temperature):
        raise ValueError("x")


def test_logging_client(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl")
    c = LoggingClient(Inner(), log, run_id="r1")
    assert c.accepts_meta
    out = c.complete([{"role": "system", "content": "SYS"}, {"role": "user", "content": "u"}], "m", 1.0,
                     reasoning_effort="low", meta={"layer": "tag", "item": "c1", "turn": 3})
    assert out == "RAW"
    rec = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[0])
    assert rec["raw"] == "RAW" and rec["meta"]["turn"] == 3 and rec["retries"] == 1 and rec["status_codes"] == [429]
    assert len(rec["prompt_hash"]) == 12 and rec["effort"] == "low" and rec["error"] is None
    b = LoggingClient(Boom(), log, run_id="r1")
    with pytest.raises(ValueError):
        b.complete([{"role": "system", "content": "S"}], "m", 1.0)
    last = json.loads((tmp_path / "calls.jsonl").read_text().splitlines()[-1])
    assert last["error"] == "ValueError" and last["raw"] == ""


def test_helpers():
    assert verdict(0.7, 0.6, ">=") == "OK" and verdict(0.3, 0.2, "<=") == "PROBLEM"
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5) and percentile([5], 95) == 5
    assert confusion(["a", "b", "a"], ["a", "a", "b"]) == {("a", "a"): 1, ("a", "b"): 1, ("b", "a"): 1}


def test_variance_split():
    rs = [{"item": "c", "turn": 0, "member": "m1", "relationship_shift": 1, "face_threat": 0},
          {"item": "c", "turn": 0, "member": "m1", "relationship_shift": 1, "face_threat": 0},
          {"item": "c", "turn": 0, "member": "m2", "relationship_shift": -1, "face_threat": 0},
          {"item": "c", "turn": 0, "member": "m2", "relationship_shift": -1, "face_threat": 0}]
    v = variance_split(rs)
    assert v["within_member"] == 0 and v["between_members"] > 0
