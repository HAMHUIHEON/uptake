"""Contract for SPEC_task35_dist5_structured (synthetic only, fake clients)."""
import json
import math

import pytest

from forecaster import board as B
from forecaster import pairwise as P
from forecaster import rescore as R
from forecaster import scalecheck as SC
from forecaster import schemas as SCH
from forecaster.labels import NEXT_ACTIONS
from forecaster.reader import MuseClient, StructuredOutputError
from forecaster.schemas import panel_schema, response_format
from forecaster.trace import CallLog, LoggingClient
from forecaster.transcribe import Turn
from forecaster.types import Receiver


def _receiver(rid="B@1"):
    return Receiver(id=rid, relationship="participant in this conversation",
                    power="equal", channel="conversation", state="", stance="neutral")


def _turns():
    return [Turn("A", "hey there"), Turn("B", "hello back"), Turn("A", "take this deal")]


# --- schemas ---

def _assert_strict(schema):
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    for banned in ("allOf", "oneOf", "anyOf"):
        assert banned not in schema
    blob = json.dumps(schema)
    for banned in ("minimum", "maximum", "minItems", "maxItems",
                   "minLength", "maxLength"):
        assert banned not in blob
    for name, prop in schema["properties"].items():
        desc = prop.get("description", "")
        assert isinstance(desc, str) and desc.strip(), name
        assert "\n" not in desc, name


@pytest.mark.parametrize("scale", ["v2", "fine100", "seven", "dist5"])
def test_schema_strict_and_action_enum(scale):
    schema = panel_schema(scale)
    _assert_strict(schema)
    assert schema["properties"]["next_action"]["enum"] == list(NEXT_ACTIONS)
    assert schema["properties"]["next_action"]["type"] == "string"
    assert "understood" in schema["properties"]


def test_schema_keys_per_scale():
    assert set(panel_schema("v2")["properties"]) == {
        "next_action", "relationship_shift", "face_threat", "understood"}
    assert panel_schema("v2")["properties"]["relationship_shift"]["type"] == "integer"
    assert set(panel_schema("fine100")["properties"]) == {
        "next_action", "relationship_after", "face_effect", "understood"}
    assert set(panel_schema("seven")["properties"]) == {
        "next_action", "relationship_shift", "face_effect", "understood"}
    dist5 = panel_schema("dist5")["properties"]
    assert set(dist5) == {"next_action", "p_rel", "p_face", "understood"}
    assert dist5["p_rel"]["type"] == "array"
    assert dist5["p_rel"]["items"]["type"] == "number"
    assert dist5["p_face"]["type"] == "array"


def test_schema_bad_scale_raises():
    with pytest.raises(ValueError):
        panel_schema("bad")


def test_response_format_shape():
    rf = response_format("fine100")
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["strict"] is True
    assert rf["json_schema"]["name"] == "panel_fine100"
    assert rf["json_schema"]["schema"] == panel_schema("fine100")
    assert response_format("dist5")["json_schema"]["name"] == "panel_dist5"


# --- dist5 prompt + parse ---

def test_prompt_dist5_keys():
    msgs = R.build_scaled_messages(_turns(), 1, "A", _receiver("A@1"), "dist5")
    assert [m["role"] for m in msgs] == ["system", "user"]
    sys = msgs[0]["content"]
    assert "next_action, p_rel, p_face, understood" in sys
    assert "p_rel" in sys and "p_face" in sys
    assert "sum to 1" in sys
    assert "TARGET turn B: hello back" in msgs[1]["content"]
    assert R.SCALES == ("v2", "fine100", "seven", "dist5")


def _dist5_raw(p_rel, p_face, nxt="proceed"):
    return json.dumps({"next_action": nxt, "p_rel": list(p_rel),
                       "face_probs_ignored": None, "p_face": list(p_face),
                       "understood": "ok"})


def test_parse_dist5_math():
    p_rel = [0.1, 0.2, 0.4, 0.2, 0.1]
    p_face = [0.5, 0.5, 0.0, 0.0, 0.0]
    got = R.parse_scaled(_dist5_raw(p_rel, p_face), "dist5")
    mids = [10, 30, 50, 70, 90]
    ev_rel = sum(p * m for p, m in zip(p_rel, mids))
    ev_face = sum(p * m for p, m in zip(p_face, mids))
    assert ev_rel == pytest.approx(50.0)
    assert got["rel_raw"] == pytest.approx(ev_rel)
    assert got["face_raw"] == pytest.approx(ev_face)
    assert got["rel_norm"] == pytest.approx((ev_rel - 50) / 25)
    assert got["face_norm"] == pytest.approx((ev_face - 50) / 25)
    var_rel = sum(p * (m - ev_rel) ** 2 for p, m in zip(p_rel, mids))
    var_face = sum(p * (m - ev_face) ** 2 for p, m in zip(p_face, mids))
    assert got["rel_spread"] == pytest.approx(math.sqrt(var_rel))
    assert got["face_spread"] == pytest.approx(math.sqrt(var_face))
    assert got["face_spread"] == pytest.approx(10.0)
    assert got["p_rel"] == pytest.approx(p_rel)
    assert got["p_face"] == pytest.approx(p_face)
    assert sum(got["p_rel"]) == pytest.approx(1.0)


def test_parse_dist5_normalizes():
    got = R.parse_scaled(_dist5_raw([2, 2, 2, 2, 2], [0, 0, 3, 0, 0]), "dist5")
    assert got["p_rel"] == pytest.approx([0.2] * 5)
    assert got["rel_raw"] == pytest.approx(50.0)
    assert got["rel_norm"] == pytest.approx(0.0)
    assert got["p_face"] == pytest.approx([0, 0, 1, 0, 0])
    assert got["face_raw"] == pytest.approx(50.0)


def test_parse_dist5_point_mass_spread_zero():
    got = R.parse_scaled(_dist5_raw([1, 0, 0, 0, 0], [0, 0, 0, 0, 1]), "dist5")
    assert got["rel_raw"] == pytest.approx(10.0)
    assert got["rel_spread"] == pytest.approx(0.0)
    assert got["rel_norm"] == pytest.approx((10 - 50) / 25)
    assert got["face_raw"] == pytest.approx(90.0)
    assert got["face_spread"] == pytest.approx(0.0)
    assert got["face_norm"] == pytest.approx((90 - 50) / 25)


def test_parse_dist5_failures():
    good_face = [0.2] * 5
    assert R.parse_scaled(_dist5_raw([0, 0, 0, 0, 0], good_face), "dist5") is None
    assert R.parse_scaled(_dist5_raw([0.5, -0.1, 0.2, 0.2, 0.2], good_face),
                          "dist5") is None
    assert R.parse_scaled(_dist5_raw([0.5, 0.5], good_face), "dist5") is None
    assert R.parse_scaled(_dist5_raw("nope", good_face), "dist5") is None
    assert R.parse_scaled(_dist5_raw([0.2] * 5, [0, 0, 0, 0, 0]), "dist5") is None
    assert R.parse_scaled(json.dumps({"next_action": "proceed",
                                      "p_rel": [0.2] * 5,
                                      "understood": "ok"}), "dist5") is None
    assert R.parse_scaled(_dist5_raw([0.2] * 5, good_face, nxt="dance"),
                          "dist5") is None
    assert R.parse_scaled("no json here", "dist5") is None
    with pytest.raises(ValueError):
        R.parse_scaled("{}", "bad")


# --- structured plumbing ---

class _Cap:
    accepts_meta = True

    def __init__(self, raw):
        self.raw = raw
        self.calls = []

    def complete(self, messages, model, temperature, meta=None,
                 response_format=None):
        self.calls.append({"meta": dict(meta or {}),
                           "response_format": response_format})
        return self.raw


class _Fixed3:
    def __init__(self, raw):
        self.raw = raw

    def complete(self, messages, model, temperature):
        return self.raw


def _fine100_raw():
    return json.dumps({"next_action": "proceed", "relationship_after": 75,
                       "face_effect": 50, "understood": "ok"})


def test_structured_flag_sends_response_format():
    client = _Cap(_fine100_raw())
    R.evaluate_turns_scaled(client, _turns(), scale="fine100", n=1, k=1,
                            b=10, max_workers=1, structured=True)
    assert client.calls
    for c in client.calls:
        rf = c["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"] == panel_schema("fine100")
        assert c["meta"]["scale"] == "fine100"


def test_default_sends_no_response_format():
    client = _Cap(_fine100_raw())
    R.evaluate_turns_scaled(client, _turns(), scale="fine100", n=1, k=1,
                            b=10, max_workers=1)
    assert client.calls
    assert all(c["response_format"] is None for c in client.calls)


def test_dist5_always_structured():
    raw = _dist5_raw([0.2] * 5, [0, 0, 1, 0, 0])
    client = _Cap(raw)
    eff = R.evaluate_turns_scaled(client, _turns(), scale="dist5", n=2, k=1,
                                  b=50, max_workers=2)
    assert client.calls
    assert all(c["response_format"] is not None for c in client.calls)
    assert all(c["response_format"]["json_schema"]["schema"]
               == panel_schema("dist5") for c in client.calls)
    for e in eff:
        assert e["scale"] == "dist5"
        assert "spread" in e
        assert e["spread"] == pytest.approx(
            sum(r["rel_spread"] for r in e["readings"]) / len(e["readings"]))
        assert B._value_of(e) == pytest.approx(0.0)
        for row in e["readings"]:
            assert row["p_rel"] == pytest.approx([0.2] * 5)
            assert row["p_face"] == pytest.approx([0, 0, 1, 0, 0])
            assert row["rel_spread"] == pytest.approx(math.sqrt(800))
            assert row["face_spread"] == pytest.approx(0.0)


def test_dist5_old_fake_client_still_works():
    raw = _dist5_raw([1, 0, 0, 0, 0], [0, 0, 0, 0, 1])
    eff = R.evaluate_turns_scaled(_Fixed3(raw), _turns(), scale="dist5", n=1,
                                  k=1, b=10, max_workers=1)
    assert [e["n_readings"] for e in eff] == [1, 1, 1]
    assert eff[0]["spread"] == pytest.approx(0.0)


def test_rescore_item_structured_flag(tmp_path):
    rec = {"id": "casino-1", "genre": "negotiation", "model": "m", "moves": [],
           "effects": [{"index": 0, "speaker": "A", "text": "hi"},
                       {"index": 1, "speaker": "B", "text": "yo"}],
           "gold": None}
    out = tmp_path / "plain"
    R.rescore_item(_Fixed3(_fine100_raw()), rec, out, "fine100", 1, 1, "m", 1)
    assert json.loads((out / "casino-1.json").read_text())["structured"] is False
    out2 = tmp_path / "struct"
    R.rescore_item(_Fixed3(_fine100_raw()), rec, out2, "fine100", 1, 1, "m", 1,
                   structured=True)
    assert json.loads((out2 / "casino-1.json").read_text())["structured"] is True
    out3 = tmp_path / "dist"
    R.rescore_item(_Fixed3(_dist5_raw([0.2] * 5, [0.2] * 5)), rec, out3,
                   "dist5", 1, 1, "m", 1)
    assert json.loads((out3 / "casino-1.json").read_text())["structured"] is True


def test_logging_client_marks_structured(tmp_path):
    log = CallLog(tmp_path / "calls.jsonl")
    logged = LoggingClient(_Cap(_fine100_raw()), log, "t")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    logged.complete(msgs, "m", 1.0, meta={"layer": "board"},
                    response_format=response_format("seven"))
    logged.complete(msgs, "m", 1.0, meta={"layer": "board"})
    lines = (tmp_path / "calls.jsonl").read_text().splitlines()
    assert json.loads(lines[0])["structured"] is True
    assert json.loads(lines[1])["structured"] is False


class _Err400(Exception):
    status_code = 400

    def __init__(self):
        super().__init__("invalid schema: p_rel must sum to 1")


def test_400_raises_structured_output_error(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key-not-real")
    seen = []

    def _boom(**kwargs):
        seen.append(kwargs)
        raise _Err400()

    client = MuseClient(create=_boom, sleep=lambda s: None)
    with pytest.raises(StructuredOutputError, match="must sum to 1"):
        client.complete([{"role": "user", "content": "hi"}], "m", 1.0,
                        response_format=response_format("dist5"))
    assert len(seen) == 1  # surfaced, not retried
    assert seen[0]["response_format"]["json_schema"]["strict"] is True
    assert client.last_stats["status_codes"] == [400]


def test_response_format_passthrough_and_default(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "test-key-not-real")
    seen = []

    class _Msg:
        content = '{"ok": true}'

    class _Choice:
        message = _Msg()

    class _Usage:
        prompt_tokens = 3
        completion_tokens = 5

    class _Resp:
        choices = [_Choice()]
        usage = _Usage()

    def _ok(**kwargs):
        seen.append(kwargs)
        return _Resp()

    client = MuseClient(create=_ok, sleep=lambda s: None)
    rf = response_format("v2")
    assert client.complete([{"role": "user", "content": "hi"}], "m", 1.0,
                           response_format=rf) == '{"ok": true}'
    assert seen[-1]["response_format"] == rf
    client.complete([{"role": "user", "content": "hi"}], "m", 1.0)
    assert "response_format" not in seen[-1]


# --- resolution on continuous values ---

def _dist5_effect(idx, value, spread, n=3):
    readings = [{"stance": s, "rel_norm": value / 2, "face_norm": value / 2,
                 "rel_raw": 50.0, "face_raw": 50.0,
                 "rel_spread": spread, "face_spread": 0.0,
                 "p_rel": [0.2] * 5, "p_face": [0.2] * 5}
                for s in ("neutral", "trusting", "suspicious")[:n]]
    return {"index": idx, "speaker": "A", "text": f"t{idx}",
            "listeners": ["B"], "n_readings": n, "n_expected": n,
            "value": value, "spread": spread, "readings": readings,
            "scale": "dist5"}


def test_resolution_dist5_rounded_counts_and_spread():
    rec = {"id": "casino-1", "genre": "negotiation", "scale": "dist5",
           "effects": [_dist5_effect(0, 1.041, 10.0),
                       _dist5_effect(1, 1.049, 20.0),
                       _dist5_effect(2, 2.0, 30.0)]}
    assert SC.record_scale(rec) == "dist5"
    out = SC.resolution({"casino-1": rec})
    assert out["n_turns"] == 3
    assert out["distinct_turn_values"] == 3
    assert out["distinct_turn_values_01"] == 3
    assert out["distinct_turn_values_0.01"] == 3
    assert out["distinct_turn_values_1"] == 2
    assert out["distinct_turn_values_0.1"] == 2
    assert out["mean_spread"] == pytest.approx(20.0)


def test_resolution_mean_spread_zero_when_absent():
    rec = {"id": "casino-1", "genre": "negotiation", "scale": "v2",
           "effects": [{"index": 0, "speaker": "A", "text": "t",
                        "listeners": ["B"], "n_readings": 1, "n_expected": 1,
                        "value": 1.0,
                        "readings": [{"relationship_shift": 1,
                                      "face_threat": 0}]}]}
    out = SC.resolution({"casino-1": rec})
    assert out["mean_spread"] == pytest.approx(0.0)
    assert out["distinct_turn_values_01"] == 1
    assert out["distinct_turn_values_1"] == 1


# --- reliability-split CLI (any scale) ---

def _rel_record(cid, values):
    effects = []
    for i, v in enumerate(values):
        readings = [{"stance": s, "rel_norm": float(v), "face_norm": 0.0,
                     "rel_raw": 50.0 + v, "face_raw": 50.0,
                     "rel_spread": 5.0, "face_spread": 0.0}
                    for s in ("neutral", "trusting", "suspicious")]
        effects.append({"index": i, "speaker": "A", "text": f"t{i}",
                        "n_readings": 3, "readings": readings,
                        "spread": 5.0, "scale": "dist5"})
    return {"id": cid, "genre": "negotiation", "scale": "dist5",
            "structured": True, "moves": [], "effects": effects}


def test_reliability_split_any_scale(tmp_path):
    records = {"casino-1": _rel_record("casino-1", [0, 1, 2, 3, 4])}
    got = P.reliability_split(records)
    assert got["mean_rho"] == pytest.approx(1.0)
    assert got == P.fine100_split_half(records)
    src = tmp_path / "src"
    src.mkdir()
    (src / "casino-1.json").write_text(json.dumps(records["casino-1"]))
    out_new = tmp_path / "new.json"
    P.main(["reliability-split", "--src", str(src), "--out", str(out_new)])
    assert json.loads(out_new.read_text())["mean_rho"] == pytest.approx(1.0)
    out_old = tmp_path / "old.json"
    P.main(["reliability-fine100", "--src", str(src), "--out", str(out_old)])
    assert json.loads(out_old.read_text()) == json.loads(out_new.read_text())


def test_rescore_main_structured_end_to_end(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    for cid in ("casino-1", "casino-2"):
        (src / f"{cid}.json").write_text(json.dumps(
            {"id": cid, "genre": "negotiation", "model": "m", "moves": [],
             "effects": [{"index": 0, "speaker": "A", "text": "hi"},
                         {"index": 1, "speaker": "B", "text": "yo"}],
             "gold": None}))
    seen = []

    class _Inner:
        def __init__(self, meter=None, default_effort=None):
            self.last_stats = {"retries": 0, "status_codes": [],
                               "prompt_tokens": 1, "completion_tokens": 1}

        def complete(self, messages, model, temperature,
                     reasoning_effort=None, response_format=None):
            seen.append(response_format)
            return _fine100_raw()

    monkeypatch.setattr(R, "MuseClient", _Inner)
    out = tmp_path / "out"
    R.main(["--src", str(src), "--out", str(out), "--scale", "fine100",
            "--budget", "5", "--workers", "1", "--inner-workers", "1",
            "--structured"])
    for cid in ("casino-1", "casino-2"):
        data = json.loads((out / f"{cid}.json").read_text())
        assert data["structured"] is True
        assert data["scale"] == "fine100"
    assert seen and all(s is not None for s in seen)
    assert all(s["json_schema"]["strict"] is True for s in seen)
    calls = (out / "calls.jsonl").read_text().splitlines()
    assert calls and all(json.loads(line)["structured"] is True
                         for line in calls)
