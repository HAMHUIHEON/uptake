from pathlib import Path
from forecaster.transcribe import (Turn, parse_diarized_text, ffmpeg_cmd, transcribe_wav, save_turns, load_turns)


def test_parse_merges_and_skips():
    text = "A: Hello there\nA:How are you?\n\nB: Fine: thanks\nunknown: (laughs)\nnot a turn\nB:ok"
    turns = parse_diarized_text(text, chunk=2)
    assert turns == [Turn("A", "Hello there How are you?", 2), Turn("B", "Fine: thanks", 2),
                     Turn("unknown", "(laughs)", 2), Turn("B", "ok", 2)]


def test_ffmpeg_cmd():
    assert ffmpeg_cmd("in.mp3", "out.wav", 540, 540) == [
        "ffmpeg", "-y", "-loglevel", "error", "-ss", "540", "-t", "540", "-i", "in.mp3",
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "out.wav"]


def test_transcribe_wav_request_and_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "fake-key")
    monkeypatch.setattr("time.sleep", lambda s: None)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....WAVE")
    seen = []

    def post(url, headers, files):
        seen.append((url, headers, files))
        return (429, "busy") if len(seen) == 1 else (200, "A:hi\nB:yo")

    out = transcribe_wav(wav, post=post)
    assert out == "A:hi\nB:yo" and len(seen) == 2
    url, headers, files = seen[0]
    assert url == "https://api.meta.ai/v1/asr/transcribe"
    assert headers["Accept"] == "text/plain" and headers["Authorization"] == "Bearer fake-key"
    req = files["request"]
    assert '"DIARIZATION"' in req[1] and '"muse-voice-transcribe-1.0"' in req[1] and req[2] == "application/json"
    assert files["audio"][2] == "audio/wav"


def test_transcribe_wav_gives_up(tmp_path, monkeypatch):
    monkeypatch.setenv("META_API_KEY", "fake-key")
    monkeypatch.setattr("time.sleep", lambda s: None)
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    import pytest
    with pytest.raises(RuntimeError):
        transcribe_wav(wav, post=lambda u, h, f: (500, "err"), max_retries=2)


def test_turns_roundtrip(tmp_path):
    t = [Turn("A", "hi", 0), Turn("B", "yo", 1)]
    p = tmp_path / "t.jsonl"
    save_turns(p, t)
    assert load_turns(p) == t
