"""Contract for SPEC_task36_audio_panel (synthetic only, fake clients)."""
import base64
import json
import subprocess
from pathlib import Path

import pytest

from forecaster import audiopanel as AP
from forecaster import commentary as CM
from forecaster import rescore as R
from forecaster import viewer as V
from forecaster.batch import spearman
from forecaster.schemas import response_format
from forecaster.transcribe import Turn, load_turns, parse_diarized_json, save_turns
from forecaster.types import Receiver
from forecaster.usage import AUDIO_USD_PER_HOUR, BudgetExceeded, Meter


def _receiver(rid="B@1", stance="neutral"):
    return Receiver(id=rid, relationship="participant in this conversation",
                    power="equal", channel="conversation", state="",
                    stance=stance)


def _turns():
    return [Turn("A", "hey there", 0, 0.0, 5.0),
            Turn("B", "hello back", 0, 5.0, 12.0),
            Turn("A", "take this deal", 0, 30.0, 41.5)]


def _fine100_raw(rel=75, face=50, nxt="proceed"):
    return json.dumps({"next_action": nxt, "relationship_after": rel,
                       "face_effect": face, "understood": "ok"})


def _dist5_raw(p_rel=None, p_face=None, nxt="proceed"):
    return json.dumps({"next_action": nxt,
                       "p_rel": list(p_rel or [0.2] * 5),
                       "p_face": list(p_face or [0, 0, 1, 0, 0]),
                       "understood": "ok"})


class _Cap:
    accepts_meta = True

    def __init__(self, raw):
        self.raw = raw
        self.calls = []

    def complete(self, messages, model, temperature, meta=None,
                 response_format=None):
        self.calls.append({"messages": messages, "model": model,
                           "meta": dict(meta or {}),
                           "response_format": response_format})
        return self.raw


# --- clip boundaries ---

def test_clip_plan_bounds_and_preroll_clamp():
    assert AP.clip_plan(_turns(), pre=20.0) == [(0.0, 5.0), (0.0, 12.0),
                                                (10.0, 31.5)]
    assert AP.clip_plan(_turns(), pre=0.0) == [(0.0, 5.0), (5.0, 7.0),
                                               (30.0, 11.5)]


def test_clip_plan_rejects_bad_spans():
    with pytest.raises(ValueError):
        AP.clip_plan(_turns(), pre=-1.0)
    with pytest.raises(ValueError):
        AP.clip_plan([Turn("A", "x", 0, 10.0, 5.0)])
    with pytest.raises(ValueError):
        AP.clip_plan([Turn("A", "x", 0, 0.0, 0.0)])


def test_ffmpeg_clip_cmd_shape():
    assert AP.ffmpeg_clip_cmd("in.mp3", "out.wav", 10.0, 31.5) == [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", "10.0", "-t", "31.5", "-i", "in.mp3",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "out.wav"]


# --- missing timestamps -> clear exit ---

def test_clip_plan_missing_timestamps():
    with pytest.raises(AP.MissingTimestamps, match="--timestamps"):
        AP.clip_plan([Turn("A", "hi", 0, 0.0, 2.0), Turn("B", "yo")])


def test_main_missing_timestamps_exit(tmp_path, capsys):
    turns_p = tmp_path / "turns.jsonl"
    save_turns(turns_p, [Turn("A", "hi"), Turn("B", "yo")])
    audio_p = tmp_path / "a.wav"
    audio_p.write_bytes(b"RIFF....")
    with pytest.raises(SystemExit) as ei:
        AP.main(["--audio", str(audio_p), "--turns", str(turns_p),
                 "--out", str(tmp_path / "r.json")])
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert "uptake-transcribe" in err
    assert "--timestamps" in err


# --- base64 payload shape ---

def test_audio_part_shape():
    wav = b"RIFF....WAVEfake"
    assert AP.audio_part(wav) == {
        "type": "input_audio",
        "input_audio": {"data": base64.b64encode(wav).decode("ascii"),
                        "format": "wav"}}


@pytest.mark.parametrize("scale", ["fine100", "dist5"])
def test_audio_messages_text_plus_audio_parts(scale):
    wav = b"RIFF....WAVEfake"
    b64 = base64.b64encode(wav).decode("ascii")
    receiver = _receiver("A@1", "suspicious")
    msgs = AP.build_audio_messages(_turns(), 1, "A", receiver, scale, b64)
    text_msgs = R.build_scaled_messages(_turns(), 1, "A", receiver, scale)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0] == text_msgs[0]
    assert "suspicious" in msgs[0]["content"]
    parts = msgs[1]["content"]
    assert parts[0] == {"type": "text", "text": text_msgs[1]["content"]}
    assert "TARGET turn B: hello back" in parts[0]["text"]
    assert parts[1] == {"type": "input_audio",
                        "input_audio": {"data": b64, "format": "wav"}}
    assert base64.b64decode(parts[1]["input_audio"]["data"]) == wav


def test_audio_messages_bad_scale():
    with pytest.raises(ValueError):
        AP.build_audio_messages(_turns(), 0, "B", _receiver(), "v2", "eA==")


# --- record format = board record ---

def _clips(seconds=(5.0, 12.0, 31.5)):
    return [{"b64": "eHh4", "seconds": s} for s in seconds]


def test_effect_keys_match_scaled_board_record():
    client = _Cap(_fine100_raw())
    eff = AP.evaluate_turns_audio(client, _turns(), _clips(), scale="fine100",
                                  n=2, b=50, max_workers=1)
    rclient = _Cap(_fine100_raw())
    reff = R.evaluate_turns_scaled(rclient, _turns(), scale="fine100", n=2,
                                   k=1, b=50, max_workers=1)
    assert [set(e) for e in eff] == [set(e) for e in reff]
    base = {"index", "speaker", "text", "listeners", "relationship", "face",
            "action_dist", "disagreement", "n_readings", "readings"}
    assert all(base <= set(e) for e in eff)
    assert [e["n_readings"] for e in eff] == [2, 2, 2]
    assert eff[0]["relationship"][0] == pytest.approx(1.0)
    assert eff[0]["face"][0] == pytest.approx(0.0)


def test_calls_logged_as_audio_layer_with_clip_seconds():
    client = _Cap(_fine100_raw())
    AP.evaluate_turns_audio(client, _turns(), _clips(), scale="fine100", n=1,
                            b=10, max_workers=1, item_id="demo")
    assert len(client.calls) == 3
    for c in client.calls:
        assert c["meta"]["layer"] == "audio"
        assert c["meta"]["item"] == "demo"
        assert c["response_format"] == response_format("fine100")
    by_turn = {c["meta"]["turn"]: c["meta"]["clip_seconds"]
               for c in client.calls}
    assert by_turn == {0: 5.0, 1: 12.0, 2: 31.5}
    assert client.calls[0]["model"] == AP.DEFAULT_MODEL == "muse-spark-1.2"


def test_dist5_effect_keys_match_and_spread():
    client = _Cap(_dist5_raw())
    eff = AP.evaluate_turns_audio(client, _turns(), _clips(), scale="dist5",
                                  n=1, b=10, max_workers=1)
    rclient = _Cap(_dist5_raw())
    reff = R.evaluate_turns_scaled(rclient, _turns(), scale="dist5", n=1,
                                   k=1, b=10, max_workers=1)
    assert [set(e) for e in eff] == [set(e) for e in reff]
    assert all("spread" in e for e in eff)
    assert eff[0]["readings"][0]["p_rel"] == pytest.approx([0.2] * 5)
    assert all(c["response_format"] == response_format("dist5")
               for c in client.calls)


def test_multiparty_listener_matches_board():
    turns4 = [Turn("A", "t0", 0, 0.0, 4.0), Turn("B", "t1", 0, 4.0, 9.0),
              Turn("C", "t2", 0, 9.0, 15.0), Turn("D", "t3", 0, 15.0, 22.0)]
    clips4 = [{"b64": "eA==", "seconds": 4.0}] * 4
    tags = [{"turn_index": 2, "speaker": "C", "moves": ["inform"],
             "addressed_to": "A", "quote": "", "failed": False}]
    eff = AP.evaluate_turns_audio(_Cap(_fine100_raw()), turns4, clips4,
                                  scale="fine100", n=1, b=10, max_workers=1,
                                  move_tags=tags)
    assert eff[2]["listeners"] == ["A"]
    ref = R.evaluate_turns_scaled(_Cap(_fine100_raw()), turns4,
                                  scale="fine100", n=1, k=1, b=10,
                                  max_workers=1, move_tags=tags)
    assert [e["listeners"] for e in eff] == [e["listeners"] for e in ref]
    plain = AP.evaluate_turns_audio(_Cap(_fine100_raw()), turns4, clips4,
                                    scale="fine100", n=1, b=10, max_workers=1)
    assert plain[2]["listeners"] == ["A", "B", "D"]


# --- compare.json math ---

def _toy(values):
    return {"effects": [{"index": i, "speaker": "A", "text": f"t{i}",
                         "value": v} for i, v in enumerate(values)]}


def test_compare_math():
    cmp = AP.compare_records(_toy([1.0, 2.0, 3.0, 4.0]),
                             _toy([1.5, 1.0, 4.0, 3.0]))
    assert cmp["n_turns"] == 4
    assert cmp["per_turn"][0] == {"index": 0, "text": 1.0, "audio": 1.5,
                                  "delta": 0.5}
    assert [r["delta"] for r in cmp["per_turn"]] == pytest.approx(
        [0.5, -1.0, 1.0, -1.0])
    assert cmp["spearman"] == pytest.approx(0.6)
    assert cmp["spearman"] == pytest.approx(
        spearman([1.0, 2.0, 3.0, 4.0], [1.5, 1.0, 4.0, 3.0]))
    assert cmp["mean_abs_delta"] == pytest.approx(0.875)
    assert [r["index"] for r in cmp["largest_delta"]] == [1, 2, 3]


def test_compare_uses_board_values_and_common_indices():
    text = {"effects": [
        {"index": 0, "speaker": "A", "text": "t",
         "relationship": [1.0, 0.0, 0.0], "face": [0.5, 0.0, 0.0]},
        {"index": 1, "speaker": "B", "text": "u", "value": 9.0}]}
    audio = {"effects": [
        {"index": 0, "speaker": "A", "text": "t",
         "relationship": [0.0, 0.0, 0.0], "face": [0.0, 0.0, 0.0]},
        {"index": 2, "speaker": "A", "text": "w", "value": 9.0}]}
    cmp = AP.compare_records(text, audio)
    assert cmp["n_turns"] == 1
    assert cmp["per_turn"] == [{"index": 0, "text": 1.5, "audio": 0.0,
                                "delta": -1.5}]
    assert cmp["largest_delta"] == cmp["per_turn"]


def test_compare_empty():
    cmp = AP.compare_records({"effects": []}, {"effects": []})
    assert cmp == {"n_turns": 0, "per_turn": [], "spearman": 0.0,
                   "mean_abs_delta": 0.0, "largest_delta": []}


# --- viewer overlay ---

def _item():
    return {
        "id": "demo-1", "genre": "roleplay",
        "moves": [
            {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
            {"turn_index": 1, "speaker": "B", "moves": ["inform"]},
            {"turn_index": 2, "speaker": "A", "moves": ["reject"]},
        ],
        "effects": [
            {"index": 0, "speaker": "A", "text": "t0", "value": 1.0,
             "disagreement": 0.0},
            {"index": 1, "speaker": "B", "text": "t1", "value": 0.0,
             "disagreement": 0.0},
            {"index": 2, "speaker": "A", "text": "t2", "value": -1.0,
             "disagreement": 0.0},
        ],
        "hits": [],
    }


def _commentary(item):
    return CM.comment(item, {"mu": 0.25, "sd": 1.0}, None, k=1, client=None)


def _blob_data(html):
    tag = '<script type="application/json" id="data">'
    assert html.count(tag) == 1
    start = html.index(tag) + len(tag)
    return json.loads(html[start:html.index("</script>", start)])


def test_viewer_audio_overlay_curve_and_legend():
    audio = _toy([2.0, 0.5, -2.0])
    view = V.build_view(_item(), _commentary(_item()), audio_record=audio)
    assert view["audio"]["curve"] == pytest.approx([1.75, 2.0, -0.25])
    assert view["audio"]["label"] == V.AUDIO_PANEL_LABEL
    assert [t["audio_value"] for t in view["turns"]] == pytest.approx(
        [2.0, 0.5, -2.0])
    assert [t["audio_delta"] for t in view["turns"]] == pytest.approx(
        [1.0, 0.5, -1.0])
    assert [t["text_raw"] for t in view["turns"]] == pytest.approx(
        [1.0, 0.0, -1.0])
    html = V.render_html(view)
    assert "text panel" in html
    assert "audio panel (Muse Spark 1.2, hears the clip)" in html
    assert "line-audio" in html
    assert "audio-dot" in html
    data = _blob_data(html)
    assert data["audio"]["curve"] == pytest.approx([1.75, 2.0, -0.25])


def test_viewer_without_audio_unchanged():
    view = V.build_view(_item(), _commentary(_item()))
    assert view["audio"] is None
    assert all("audio_value" not in t for t in view["turns"])
    html = V.render_html(view)
    assert "audio panel" not in html
    assert "text panel" not in html


def test_viewer_cli_audio_record(tmp_path):
    item_p = tmp_path / "item.json"
    com_p = tmp_path / "commentary.json"
    item_p.write_text(json.dumps(_item()))
    com_p.write_text(json.dumps(_commentary(_item())))
    audio_p = tmp_path / "audio.json"
    audio_p.write_text(json.dumps(_toy([2.0, 0.5, -2.0])))
    out_p = tmp_path / "board.html"
    V.main(["--item", str(item_p), "--commentary", str(com_p),
            "--out", str(out_p), "--audio-record", str(audio_p)])
    html = out_p.read_text()
    assert V.AUDIO_PANEL_LABEL in html
    assert V.TEXT_PANEL_LABEL in html
    assert _blob_data(html)["audio"]["curve"] == pytest.approx(
        [1.75, 2.0, -0.25])


# --- transcribe timestamps (task07 extension) ---

def test_parse_diarized_json_merge_and_offset():
    obj = {"segments": [
        {"speaker": "A", "text": "hello", "start": 1.0, "end": 2.5},
        {"speaker": "A", "text": "there", "start": 2.5, "end": 4.0},
        {"speaker": "B", "text": "yo", "start": 5.0, "end": 6.0}]}
    assert parse_diarized_json(obj, chunk=1, offset_s=540.0) == [
        Turn("A", "hello there", 1, 541.0, 544.0),
        Turn("B", "yo", 1, 545.0, 546.0)]
    assert parse_diarized_json(json.dumps(obj)) == [
        Turn("A", "hello there", 0, 1.0, 4.0),
        Turn("B", "yo", 0, 5.0, 6.0)]


def test_parse_diarized_json_rejects_plain_text():
    with pytest.raises(ValueError):
        parse_diarized_json("A:hi\nB:yo")
    with pytest.raises(ValueError):
        parse_diarized_json({"nope": []})
    with pytest.raises(ValueError):
        parse_diarized_json({"segments": [
            {"speaker": "A", "text": "x", "start": 3.0, "end": 1.0}]})


def test_turns_roundtrip_with_timestamps(tmp_path):
    p = tmp_path / "t.jsonl"
    turns = [Turn("A", "hi", 0, 0.0, 2.5),
             Turn("B", "yo", 1, 540.0, 543.0)]
    save_turns(p, turns)
    assert load_turns(p) == turns
    old = tmp_path / "old.jsonl"
    old.write_text('{"speaker": "A", "text": "hi", "chunk": 0}\n')
    assert load_turns(old) == [Turn("A", "hi", 0)]


def test_transcribe_wav_timestamps_payload(tmp_path, monkeypatch):
    from forecaster.transcribe import transcribe_wav

    monkeypatch.setenv("META_API_KEY", "fake-key")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....WAVE")
    seen = []

    def post(url, headers, files):
        seen.append(files["request"][1])
        return (200, "A:hi")

    transcribe_wav(wav, post=post)
    payload = json.loads(seen[-1])
    assert payload["timestamps"] is True
    assert payload["model"] == "muse-voice-transcribe-1.0"
    transcribe_wav(wav, post=post, timestamps=False)
    assert json.loads(seen[-1])["timestamps"] is False


# --- audio metering ---

def test_meter_audio_cost_and_budget():
    meter = Meter(budget_usd=10.0)
    meter.add("muse-spark-1.2", 1000, 500)
    meter.add_audio("muse-spark-1.2", 3600.0)
    assert meter.audio_seconds() == pytest.approx(3600.0)
    assert meter.audio_cost() == pytest.approx(
        AUDIO_USD_PER_HOUR["muse-spark-1.2"])
    token = 1000 / 1_000_000 * 1.25 + 500 / 1_000_000 * 4.25
    assert meter.cost() == pytest.approx(token + meter.audio_cost())
    tight = Meter(budget_usd=0.001)
    tight.add_audio("muse-spark-1.2", 60.0)
    with pytest.raises(BudgetExceeded):
        tight.check()


# --- end to end (stubbed ffmpeg + fake client) ---

class _Inner:
    def __init__(self, meter=None, default_effort=None):
        self.meter = meter
        self.last_stats = {"retries": 0, "status_codes": [],
                           "prompt_tokens": 0, "completion_tokens": 0}
        self.seen = []

    def complete(self, messages, model, temperature,
                 reasoning_effort=None, response_format=None):
        self.seen.append(messages)
        assert response_format is not None
        return _fine100_raw()


def _stub_ffmpeg(monkeypatch, seen_cmds):
    def fake_run(cmd, **kw):
        seen_cmds.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"RIFF-fake-wav-bytes")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(AP.subprocess, "run", fake_run)


def test_main_end_to_end(tmp_path, monkeypatch, capsys):
    import forecaster.audiopanel as APmod

    turns_p = tmp_path / "turns.jsonl"
    save_turns(turns_p, _turns())
    audio_p = tmp_path / "meet.mp3"
    audio_p.write_bytes(b"fake-audio-bytes")
    text_p = tmp_path / "text.json"
    text_p.write_text(json.dumps(_toy([0.5, 0.5, 0.5])))
    seen_cmds: list = []
    _stub_ffmpeg(monkeypatch, seen_cmds)
    inner = _Inner()
    monkeypatch.setattr(APmod, "MuseClient", lambda **kw: inner)
    out_p = tmp_path / "audio_record.json"
    AP.main(["--audio", str(audio_p), "--turns", str(turns_p),
             "--out", str(out_p), "--text-record", str(text_p),
             "--n", "1", "--b", "20", "--workers", "1",
             "--budget", "5", "--id", "demo-1"])
    record = json.loads(out_p.read_text())
    assert record["id"] == "demo-1"
    assert record["modality"] == "audio"
    assert record["scale"] == "fine100"
    assert record["model"] == "muse-spark-1.2"
    assert record["structured"] is True
    assert record["audio_seconds"] == pytest.approx(48.5)
    assert len(record["effects"]) == 3
    assert record["board"] == record["effects"]
    cmp = json.loads((tmp_path / "compare.json").read_text())
    assert cmp["n_turns"] == 3
    assert [r["audio"] for r in cmp["per_turn"]] == pytest.approx([1.0] * 3)
    calls = [json.loads(line) for line in
             (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 3
    assert all(c["meta"]["layer"] == "audio" for c in calls)
    assert {c["meta"]["clip_seconds"] for c in calls} == {5.0, 12.0, 31.5}
    assert len(seen_cmds) == 3
    assert seen_cmds[0][:4] == ["ffmpeg", "-y", "-loglevel", "error"]
    assert "-ar" in seen_cmds[0] and "16000" in seen_cmds[0]
    payload = inner.seen[0][1]["content"]
    assert payload[0]["type"] == "text"
    assert payload[1]["type"] == "input_audio"
    assert base64.b64decode(payload[1]["input_audio"]["data"]) == \
        b"RIFF-fake-wav-bytes"
    out = capsys.readouterr().out
    assert "audio 48.5s" in out
    assert "spend" in out


def test_main_budget_exceeded(tmp_path, monkeypatch):
    import forecaster.audiopanel as APmod

    turns_p = tmp_path / "turns.jsonl"
    save_turns(turns_p, _turns())
    monkeypatch.setattr(APmod, "MuseClient", lambda **kw: _Inner())
    with pytest.raises(SystemExit) as ei:
        AP.main(["--audio", str(tmp_path / "a.wav"),
                 "--turns", str(turns_p),
                 "--out", str(tmp_path / "r.json"), "--budget", "0.000001"])
    assert "budget" in str(ei.value.code)
