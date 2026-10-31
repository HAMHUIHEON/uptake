"""Contract for SPEC_task59_audio_align (synthetic only, mocked transcriber)."""
import ast
import json
import sys
from pathlib import Path

import pytest

from forecaster import align as AL
from forecaster import audiopanel as AP
from forecaster.transcribe import Turn


def _segments():
    return [
        {"start": 10.0, "end": 15.0,
         "text": "Cactus fifteen forty nine runway heading"},
        {"start": 20.0, "end": 26.0,
         "text": "tower cactus fifteen forty nine clear for takeoff"},
        {"start": 40.0, "end": 48.0,
         "text": "mayday mayday we lost both engines"},
    ]


def _record(texts):
    return {"id": "toy", "genre": "demo", "effects": [
        {"index": i, "speaker": "S", "text": t} for i, t in enumerate(texts)]}


# --- tokenize / similarity ---

def test_tokenize_lower_digits_callsign():
    assert AL.tokenize("Cactus 1549, Climb!") == ["cactus", "1549", "climb"]
    assert AL.tokenize("A320: lost BOTH engines...") == [
        "a320", "lost", "both", "engines"]
    assert AL.tokenize("") == []
    assert AL.tokenize(None) == []


def test_similarity_identical_disjoint_partial():
    assert AL.similarity("Cactus 1549 climb", "climb 1549 cactus") == 1.0
    assert AL.similarity("alpha beta", "gamma delta") == 0.0
    assert AL.similarity("", "alpha") == 0.0
    # |cap|=2, |cup|=4 -> 0.5
    assert AL.similarity("a b c", "a b d") == pytest.approx(0.5)


def test_threshold_accepts_above_rejects_below():
    # 2/6 = 0.333... < 0.35 -> unaligned
    low = AL.align_texts(["a b c d e"], [{"start": 0.0, "end": 1.0,
                                          "text": "a b f g h"}])
    assert low[0]["aligned"] is False
    assert low[0]["segment"] is None
    # 3/5 = 0.6 >= 0.35 -> aligned
    high = AL.align_texts(["a b c"], [{"start": 0.0, "end": 1.0,
                                       "text": "a b c d e"}])
    assert high[0]["aligned"] is True
    assert (high[0]["start"], high[0]["end"]) == (0.0, 1.0)
    # threshold knob respected: 0.5 match rejected at 0.9
    strict = AL.align_texts(["a b c"], [{"start": 0.0, "end": 1.0,
                                         "text": "a b d"}],
                            threshold=0.9)
    assert strict[0]["aligned"] is False
    assert AL.SIM_THRESHOLD == pytest.approx(0.35)


# --- greedy monotone alignment ---

def test_align_monotone_in_order_spans():
    texts = ["cactus fifteen forty nine runway heading",
             "tower cactus fifteen forty nine clear takeoff",
             "mayday lost both engines"]
    matches = AL.align_texts(texts, _segments())
    assert [m["segment"] for m in matches] == [0, 1, 2]
    assert all(m["aligned"] for m in matches)
    assert [(m["start"], m["end"]) for m in matches] == [
        (10.0, 15.0), (20.0, 26.0), (40.0, 48.0)]
    assert matches[0]["similarity"] == pytest.approx(1.0)


def test_align_monotone_never_moves_back():
    segments = [{"start": 0.0, "end": 1.0, "text": "apple banana"},
                {"start": 2.0, "end": 3.0, "text": "cherry date"}]
    # T0 takes S1; T1's global best (S0) is behind the cursor -> unaligned.
    matches = AL.align_texts(["cherry date", "apple banana"], segments)
    assert matches[0]["segment"] == 1
    assert matches[1]["aligned"] is False
    assert matches[1]["segment"] is None
    # reversed record order aligns both
    fwd = AL.align_texts(["apple banana", "cherry date"], segments)
    assert [m["segment"] for m in fwd] == [0, 1]


def test_align_tie_breaks_earliest():
    segments = [{"start": 0.0, "end": 1.0, "text": "alpha beta"},
                {"start": 2.0, "end": 3.0, "text": "alpha beta"}]
    matches = AL.align_texts(["alpha beta"], segments)
    assert matches[0]["segment"] == 0


# --- record output + report ---

def test_align_record_marks_unaligned_and_report():
    record = _record(["cactus fifteen forty nine runway heading",
                      "something completely unrelated zebra",
                      "mayday lost both engines"])
    out, report = AL.align_record(record, _segments())
    assert [e["aligned"] for e in out["effects"]] == [True, False, True]
    assert (out["effects"][0]["start"], out["effects"][0]["end"]) == (10.0, 15.0)
    assert "start" not in out["effects"][1]
    assert "end" not in out["effects"][1]
    assert (out["effects"][2]["start"], out["effects"][2]["end"]) == (40.0, 48.0)
    assert report["aligned"] == 2
    assert report["unaligned"] == 1
    assert report["aligned_indices"] == [0, 2]
    assert report["unaligned_indices"] == [1]
    assert report["mean_similarity"] == pytest.approx(
        (out["effects"][0]["similarity"] + out["effects"][2]["similarity"]) / 2)
    assert report["offset_hint"] == pytest.approx(10.0)
    assert report["threshold"] == pytest.approx(0.35)
    assert out["alignment"] == report
    # input untouched
    assert "aligned" not in record["effects"][0]


def test_report_shape_keys_and_types():
    _, report = AL.align_record(
        _record(["cactus fifteen forty nine runway heading"]), _segments())
    assert {"aligned", "unaligned", "mean_similarity", "offset_hint"} <= set(report)
    assert isinstance(report["aligned"], int)
    assert isinstance(report["unaligned"], int)
    assert isinstance(report["mean_similarity"], float)
    assert isinstance(report["offset_hint"], float)


def test_report_empty_record_and_empty_transcript():
    out, report = AL.align_record({"id": "e", "effects": []}, _segments())
    assert (report["aligned"], report["unaligned"],
            report["mean_similarity"], report["offset_hint"]) == (0, 0, 0.0, 0.0)
    assert report["threshold"] == pytest.approx(0.35)
    assert report["aligned_indices"] == []
    assert report["unaligned_indices"] == []
    assert out["effects"] == []
    out2, report2 = AL.align_record(_record(["hello world"]), [])
    assert [e["aligned"] for e in out2["effects"]] == [False]
    assert (report2["aligned"], report2["unaligned"],
            report2["mean_similarity"], report2["offset_hint"]) == (0, 1, 0.0, 0.0)


def test_align_record_sorts_by_index_but_keeps_order():
    record = {"id": "o", "effects": [
        {"index": 1, "speaker": "S", "text": "cherry date"},
        {"index": 0, "speaker": "S", "text": "apple banana"}]}
    segments = [{"start": 0.0, "end": 1.0, "text": "apple banana"},
                {"start": 2.0, "end": 3.0, "text": "cherry date"}]
    out, report = AL.align_record(record, segments)
    # original order kept, spans filled per index
    assert [e["index"] for e in out["effects"]] == [1, 0]
    assert (out["effects"][0]["start"], out["effects"][1]["start"]) == (2.0, 0.0)
    assert report["aligned"] == 2


def test_segments_from_turns_skips_missing_spans():
    turns = [Turn("A", "hi there", 0, 1.0, 2.0),
             Turn("B", "no times here"),
             {"speaker": "C", "text": "bad span", "start": 5.0, "end": 5.0}]
    assert AL.segments_from_turns(turns) == [
        {"start": 1.0, "end": 2.0, "text": "hi there"}]


# --- CLI with mocked transcriber ---

def _mock_transcribe(monkeypatch, turns):
    seen = {}

    def fake(src, work_dir, post=None, timestamps=True, chunk_s=540,
             model=None):
        seen["src"] = str(src)
        seen["work"] = str(work_dir)
        seen["timestamps"] = timestamps
        seen["model"] = model
        return list(turns)

    monkeypatch.setattr("forecaster.transcribe.transcribe_file", fake)
    return seen


def _toy_turns():
    return [Turn("A", "cactus fifteen forty nine runway heading", 0, 10.0, 15.0),
            Turn("B", "tower cactus fifteen forty nine clear for takeoff",
                 0, 20.0, 26.0)]


def test_main_end_to_end_mocked(tmp_path, monkeypatch, capsys):
    record_p = tmp_path / "record.json"
    record_p.write_text(json.dumps(_record(
        ["cactus fifteen forty nine runway heading",
         "tower cactus fifteen forty nine clear takeoff"])))
    audio_p = tmp_path / "case.wav"
    audio_p.write_bytes(b"RIFF....")
    seen = _mock_transcribe(monkeypatch, _toy_turns())
    out_p = tmp_path / "aligned.json"
    AL.main(["--audio", str(audio_p), "--record", str(record_p),
             "--out", str(out_p)])
    assert seen["src"] == str(audio_p)
    assert seen["timestamps"] is True
    assert seen["model"] == "muse-voice-transcribe-1.0"
    out = json.loads(out_p.read_text())
    assert [e["aligned"] for e in out["effects"]] == [True, True]
    assert (out["effects"][0]["start"], out["effects"][1]["start"]) == (10.0, 20.0)
    assert out["alignment"]["aligned"] == 2
    assert out["alignment"]["offset_hint"] == pytest.approx(10.0)
    assert "aligned 2/2" in capsys.readouterr().out


def test_main_model_passthrough_and_unaligned_notice(tmp_path, monkeypatch,
                                                     capsys):
    record_p = tmp_path / "record.json"
    record_p.write_text(json.dumps(_record(["totally unrelated zebra"])))
    audio_p = tmp_path / "case.wav"
    audio_p.write_bytes(b"RIFF....")
    seen = _mock_transcribe(monkeypatch, _toy_turns())
    out_p = tmp_path / "aligned.json"
    AL.main(["--audio", str(audio_p), "--record", str(record_p),
             "--out", str(out_p), "--model", "custom-model"])
    assert seen["model"] == "custom-model"
    out = json.loads(out_p.read_text())
    assert out["alignment"]["unaligned"] == 1
    assert "unaligned turns: [0]" in capsys.readouterr().out


def test_main_rejects_record_without_effects(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"id": "x"}))
    with pytest.raises(SystemExit) as ei:
        AL.main(["--audio", "a.wav", "--record", str(bad),
                 "--out", str(tmp_path / "o.json")])
    assert ei.value.code == 2


def test_main_uses_given_work_dir(tmp_path, monkeypatch):
    record_p = tmp_path / "record.json"
    record_p.write_text(json.dumps(_record(["hi"])))
    audio_p = tmp_path / "case.wav"
    audio_p.write_bytes(b"RIFF....")
    seen = _mock_transcribe(monkeypatch, [])
    work = tmp_path / "work"
    AL.main(["--audio", str(audio_p), "--record", str(record_p),
             "--out", str(tmp_path / "o.json"), "--work", str(work)])
    assert seen["work"] == str(work)
    assert work.is_dir()


# --- audio panel acceptance ---

def test_aligned_record_accepted_by_audio_panel():
    record = _record(["cactus fifteen forty nine runway heading",
                      "tower cactus fifteen forty nine clear takeoff",
                      "mayday lost both engines"])
    out, _ = AL.align_record(record, _segments())
    turns = [Turn(e["speaker"], e["text"], 0, e["start"], e["end"])
             for e in out["effects"]]
    plan = AP.clip_plan(turns, pre=20.0)
    assert plan == [(0.0, 15.0), (0.0, 26.0), (20.0, 28.0)]
    plan0 = AP.clip_plan(turns, pre=0.0)
    assert plan0 == [(10.0, 5.0), (20.0, 6.0), (40.0, 8.0)]


# --- stdlib only + entry point ---

def test_align_is_stdlib_only():
    src = Path(AL.__file__).read_text()
    stdlib = set(sys.stdlib_module_names)
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [node.module.split(".")[0]] if node.module else []
        else:
            continue
        for mod in mods:
            assert mod in stdlib or mod == "transcribe", mod


def test_entry_point_registered():
    pyproject = Path(AL.__file__).resolve().parent.parent / "pyproject.toml"
    assert 'uptake-align = "forecaster.align:main"' in pyproject.read_text()
