"""Audio transcription with speaker diarization (Muse Voice Transcribe)."""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

TRANSCRIBE_URL = "https://api.meta.ai/v1/asr/transcribe"
REQUEST_PAYLOAD = {
    "model": "muse-voice-transcribe-1.0",
    "audioEncoding": "WAV",
    "mode": "DIARIZATION",
}


@dataclass
class Turn:
    speaker: str
    text: str
    chunk: int = 0
    start: float | None = None
    end: float | None = None


def parse_diarized_text(text: str, chunk: int = 0) -> list[Turn]:
    turns: list[Turn] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        if ":" not in line:
            continue
        speaker, _, rest = line.partition(":")
        speaker = speaker.strip()
        rest = rest.strip()
        if not speaker or not rest:
            continue
        if turns and turns[-1].speaker == speaker:
            turns[-1].text = turns[-1].text + " " + rest
        else:
            turns.append(Turn(speaker=speaker, text=rest, chunk=chunk))
    return turns


def parse_diarized_json(obj, chunk: int = 0, offset_s: float = 0.0) -> list[Turn]:
    """Parse a timestamped transcribe response into turns.

    `obj` is the decoded JSON body (or its text): a dict with a
    `"segments"` list of `{"speaker", "text", "start", "end"}` dicts, where
    start/end are seconds relative to the chunk. Consecutive segments from
    the same speaker merge into one turn (text joined with a space, span
    extended). `offset_s` shifts chunk-relative times into full-audio
    seconds. Raises ValueError/TypeError/KeyError on a non-conforming body.
    """
    if isinstance(obj, str):
        obj = json.loads(obj)
    if not isinstance(obj, dict) or not isinstance(obj.get("segments"), list):
        raise ValueError("timestamped transcribe body has no segments list")
    turns: list[Turn] = []
    for seg in obj["segments"]:
        if not isinstance(seg, dict):
            raise ValueError("bad segment")
        speaker = str(seg["speaker"]).strip()
        text = str(seg["text"]).strip()
        start = float(seg["start"]) + float(offset_s)
        end = float(seg["end"]) + float(offset_s)
        if not speaker or not text:
            continue
        if end < start:
            raise ValueError("segment end before start")
        if turns and turns[-1].speaker == speaker:
            turns[-1].text = turns[-1].text + " " + text
            prev_start = turns[-1].start
            turns[-1].start = min(s for s in (prev_start, start) if s is not None)
            prev_end = turns[-1].end
            turns[-1].end = max(e for e in (prev_end, end) if e is not None)
        else:
            turns.append(Turn(speaker=speaker, text=text, chunk=chunk,
                              start=start, end=end))
    return turns


def ffmpeg_cmd(src, dst, start_s: int, dur_s: int) -> list[str]:
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_s), "-t", str(dur_s),
        "-i", str(src),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dst),
    ]


def audio_duration(src) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(src),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def to_wav_chunks(src, out_dir, chunk_s=540) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dur = audio_duration(src)
    if dur <= 0:
        return []
    n = int(math.ceil(dur / chunk_s))
    paths: list[Path] = []
    for i in range(n):
        start = i * chunk_s
        remaining = dur - start
        if remaining > chunk_s:
            d: int = chunk_s
        else:
            d = int(math.ceil(remaining))
        dst = out / f"chunk_{i:03d}.wav"
        subprocess.run(ffmpeg_cmd(src, dst, start, d), check=True)
        paths.append(dst)
    return paths


def _api_key() -> str:
    return os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY") or ""


def transcribe_wav(path, post=None, max_retries=4, timestamps=True) -> str:
    payload = dict(REQUEST_PAYLOAD)
    payload["timestamps"] = bool(timestamps)
    req_json = json.dumps(payload)
    p = Path(path)
    data = p.read_bytes()
    filename = p.name
    files = {
        "request": (None, req_json, "application/json"),
        "audio": (filename, data, "audio/wav"),
    }
    key = _api_key()
    headers = {"Accept": "text/plain", "Authorization": f"Bearer {key}"}
    if post is None:
        import ssl

        import truststore

        try:
            import httpx
        except ModuleNotFoundError:  # pragma: no cover
            import httpx2 as httpx  # type: ignore

        ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

        def _default_post(url, headers, files):
            with httpx.Client(verify=ssl_context) as client:
                r = client.post(url, headers=headers, files=files, timeout=120)
                return (r.status_code, r.text)

        post_fn = _default_post
    else:
        post_fn = post

    last_status = None
    last_text = ""
    for attempt in range(max_retries + 1):
        try:
            status, text = post_fn(TRANSCRIBE_URL, headers, files)
        except Exception as exc:
            last_status = None
            last_text = str(exc)
            if attempt >= max_retries:
                raise RuntimeError(f"transcribe failed: {exc}") from exc
            time.sleep(2 ** attempt)
            continue
        if 200 <= status < 300:
            return text
        last_status = status
        last_text = text
        if status == 429 or 500 <= status < 600:
            if attempt >= max_retries:
                break
            time.sleep(2 ** attempt)
            continue
        raise RuntimeError(f"transcribe failed: {status} {text}")
    raise RuntimeError(f"transcribe failed after retries: {last_status} {last_text}")


def transcribe_file(src, work_dir, post=None, timestamps=True,
                    chunk_s=540) -> list[Turn]:
    chunks = to_wav_chunks(src, work_dir, chunk_s=chunk_s)
    turns: list[Turn] = []
    for idx, chunk_path in enumerate(chunks):
        text = transcribe_wav(chunk_path, post=post, timestamps=timestamps)
        parsed = None
        if timestamps:
            try:
                parsed = parse_diarized_json(text, chunk=idx,
                                             offset_s=idx * chunk_s)
            except (ValueError, TypeError, KeyError):
                parsed = None
        if parsed is None:
            parsed = parse_diarized_text(text, chunk=idx)
        turns.extend(parsed)
    return turns


def save_turns(path, turns) -> None:
    p = Path(path)
    if str(p.parent) and str(p.parent) != ".":
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for t in turns:
            obj = {"speaker": t.speaker, "text": t.text, "chunk": t.chunk}
            start = getattr(t, "start", None)
            end = getattr(t, "end", None)
            if start is not None:
                obj["start"] = start
            if end is not None:
                obj["end"] = end
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_turns(path) -> list[Turn]:
    turns: list[Turn] = []
    with open(Path(path), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            start = obj.get("start")
            end = obj.get("end")
            turns.append(Turn(
                speaker=obj["speaker"],
                text=obj["text"],
                chunk=obj.get("chunk", 0),
                start=None if start is None else float(start),
                end=None if end is None else float(end),
            ))
    return turns


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="uptake-transcribe")
    ap.add_argument("audio")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--timestamps", dest="timestamps",
                    action=argparse.BooleanOptionalAction, default=True,
                    help="keep per-turn start/end timestamps (needed by uptake-audio-panel)")
    args = ap.parse_args(argv)

    work = Path(args.work) if args.work else Path("data/private/transcribe_work")
    work.mkdir(parents=True, exist_ok=True)
    turns = transcribe_file(args.audio, work, timestamps=args.timestamps)
    save_turns(args.out, turns)
    print(f"{len(turns)} turns")
    by_chunk: dict[int, set[str]] = defaultdict(set)
    for t in turns:
        by_chunk[t.chunk].add(t.speaker)
    for c in sorted(by_chunk):
        print(f"chunk {c}: {len(by_chunk[c])} speakers")
