"""Audio panel: listeners read the turn from the recording (task36, P8-C demo).

Same listener panel as the board (same text window, same three stances), but
each judge also hears the turn's audio clip cut from the recording with a
pre-roll. Model default is muse-spark-1.2 (docs: 1.3 audio quality may drop).
Demo only: agreement with the text panel is reported descriptively.
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
import tempfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from . import board
from . import schemas as SCH
from .batch import spearman
from .moves import MoveTag
from .panel import DEFAULT_MIX, build_panel
from .reader import MuseClient
from .rescore import parse_scaled
from .rescore import build_scaled_messages
from .stats import cluster_bootstrap_ci
from .trace import CallLog, LoggingClient
from .transcribe import load_turns
from .types import Receiver
from .usage import BudgetExceeded, Meter

DEFAULT_MODEL = "muse-spark-1.2"
DEFAULT_PRE = 20.0
AUDIO_SCALES = ("fine100", "dist5")


class MissingTimestamps(ValueError):
    """Turns lack the per-turn start/end spans the audio panel needs."""


def clip_plan(turns, pre=DEFAULT_PRE) -> list[tuple[float, float]]:
    """Per-turn ``(clip_start, clip_seconds)`` for ``[max(0, start-pre), end]``.

    Raises MissingTimestamps when any turn lacks start/end, ValueError on a
    non-positive clip span.
    """
    pre = float(pre)
    if pre < 0:
        raise ValueError(f"pre must be >= 0, got {pre!r}")
    plan: list[tuple[float, float]] = []
    for pos, t in enumerate(turns):
        start = getattr(t, "start", None)
        end = getattr(t, "end", None)
        if start is None or end is None:
            speaker = getattr(t, "speaker", "?")
            raise MissingTimestamps(
                f"turn {pos} ({speaker}) has no start/end timestamps; "
                "re-run uptake-transcribe with --timestamps (on by default) "
                "so each turn keeps its span")
        start_f = float(start)
        end_f = float(end)
        if end_f <= start_f:
            raise ValueError(
                f"turn {pos} has a non-positive span "
                f"(start={start!r}, end={end!r})")
        clip_start = max(0.0, start_f - pre)
        seconds = end_f - clip_start
        if seconds <= 0:
            raise ValueError(
                f"turn {pos} has a non-positive clip span "
                f"(start={start!r}, end={end!r}, pre={pre})")
        plan.append((clip_start, seconds))
    return plan


def ffmpeg_clip_cmd(src, dst, start_s: float, dur_s: float) -> list[str]:
    return [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", str(start_s), "-t", str(dur_s),
        "-i", str(src),
        "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
        str(dst),
    ]


def cut_turn_clips(audio_path, plan, work_dir) -> tuple[list[dict], float]:
    """Cut one 16 kHz mono wav per plan entry; return clips + total seconds.

    Each clip is ``{"b64": base64 wav bytes, "seconds": clip length}``.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    clips: list[dict] = []
    total = 0.0
    for i, (start, seconds) in enumerate(plan):
        dst = work / f"clip_{i:03d}.wav"
        subprocess.run(ffmpeg_clip_cmd(audio_path, dst, start, seconds),
                       check=True)
        clips.append({"b64": base64.b64encode(dst.read_bytes()).decode("ascii"),
                      "seconds": float(seconds)})
        total += float(seconds)
    return clips, total


def audio_part(wav_bytes: bytes) -> dict:
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(wav_bytes).decode("ascii"),
            "format": "wav",
        },
    }


def build_audio_messages(turns, i, listener: str, receiver: Receiver, scale,
                         audio_b64: str, window=6) -> list[dict]:
    """Board text window (judged turn marked) plus the inline audio clip."""
    if scale not in AUDIO_SCALES:
        raise ValueError(f"bad scale: {scale!r}")
    msgs = build_scaled_messages(turns, i, listener, receiver, scale, window)
    return [
        msgs[0],
        {"role": "user", "content": [
            {"type": "text", "text": msgs[1]["content"]},
            {"type": "input_audio",
             "input_audio": {"data": audio_b64, "format": "wav"}},
        ]},
    ]


def _read_one_audio(client, turns, i, listener, receiver, s, model, window,
                    scale, clip, item_id=""):
    messages = build_audio_messages(turns, i, listener, receiver, scale,
                                    clip["b64"], window)
    meta = {"layer": "audio", "scale": scale, "item": item_id, "turn": i,
            "listener": listener, "member": receiver.id,
            "stance": receiver.stance, "sample": s,
            "clip_seconds": float(clip["seconds"])}
    rf = SCH.response_format(scale)
    for _ in range(2):
        try:
            raw = board._call_board(client, messages, model, meta,
                                    response_format=rf)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        parsed = parse_scaled(raw, scale)
        if parsed is not None:
            return parsed
    return None


def evaluate_turns_audio(client, turns, clips, scale="fine100", n=3, k=1,
                         model=DEFAULT_MODEL, stance_mix=DEFAULT_MIX,
                         window=6, b=1000, seed=0, max_workers=8,
                         move_tags=None, item_id="") -> list[dict]:
    """Read every turn from its clip; effects carry the board record keys."""
    if scale not in AUDIO_SCALES:
        raise ValueError(f"bad scale: {scale!r}")
    effects: list[dict] = []
    if not turns:
        return effects
    if max_workers is None or max_workers < 1:
        max_workers = 1
    seen: list[str] = []
    for t in turns:
        if t.speaker not in seen:
            seen.append(t.speaker)
    addr_by_index: dict[int, str] = {}
    if move_tags:
        for t in move_tags:
            if isinstance(t, dict):
                idx = t.get("turn_index")
            else:
                idx = getattr(t, "turn_index", None)
            if isinstance(idx, int) and idx not in addr_by_index:
                addr_by_index[idx] = board._addressed_to(t)
    for i, turn in enumerate(turns):
        listeners = [s for s in seen if s != turn.speaker and s != "unknown"]
        if move_tags:
            addr = addr_by_index.get(i, "")
            if addr and addr != turn.speaker and addr in seen and addr != "unknown":
                listeners = [addr]
        if not listeners:
            lonely = {
                "index": i, "speaker": turn.speaker, "text": turn.text,
                "listeners": [], "relationship": (0.0, 0.0, 0.0),
                "face": (0.0, 0.0, 0.0), "action_dist": {},
                "disagreement": 0.0, "n_readings": 0, "readings": [],
                "scale": scale, "n_expected": 0,
            }
            if scale == "dist5":
                lonely["spread"] = 0.0
            effects.append(lonely)
            continue
        tasks: list[tuple] = []
        for listener in listeners:
            base = Receiver(
                id=f"{listener}@{i}",
                relationship="participant in this conversation",
                power="equal", channel="conversation", state="",
            )
            panel = build_panel(base, n, seed, stance_mix)
            for receiver in panel:
                for s in range(k):
                    tasks.append((listener, receiver, s))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_read_one_audio, client, turns, i,
                                   listener, receiver, s, model, window,
                                   scale, clips[i], item_id)
                       for listener, receiver, s in tasks]
            results = [f.result() for f in futures]
        readings = [r for r in results if r is not None]
        n_readings = len(readings)
        readings_out = []
        for (listener, receiver, s), r in zip(tasks, results):
            if r is None:
                continue
            row = {"turn": i, "listener": listener, "member": receiver.id,
                   "stance": receiver.stance, "sample": s,
                   "next_action": r["next_action"],
                   "rel_raw": r["rel_raw"], "face_raw": r["face_raw"],
                   "rel_norm": r["rel_norm"], "face_norm": r["face_norm"]}
            if scale == "dist5":
                row["p_rel"] = list(r["p_rel"])
                row["p_face"] = list(r["p_face"])
                row["rel_spread"] = r["rel_spread"]
                row["face_spread"] = r["face_spread"]
            readings_out.append(row)
        if not readings:
            empty = {
                "index": i, "speaker": turn.speaker, "text": turn.text,
                "listeners": listeners, "relationship": (0.0, 0.0, 0.0),
                "face": (0.0, 0.0, 0.0), "action_dist": {},
                "disagreement": 0.0, "n_readings": 0, "readings": [],
                "scale": scale, "n_expected": len(tasks),
            }
            if scale == "dist5":
                empty["spread"] = 0.0
            effects.append(empty)
            continue
        rel_clusters: dict[str, list[float]] = {}
        face_clusters: dict[str, list[float]] = {}
        counts: Counter = Counter()
        for (listener, receiver, s), r in zip(tasks, results):
            if r is None:
                continue
            rel_clusters.setdefault(receiver.id, []).append(float(r["rel_norm"]))
            face_clusters.setdefault(receiver.id, []).append(float(r["face_norm"]))
            counts[r["next_action"]] += 1
        rel_ci = cluster_bootstrap_ci(rel_clusters, b=b, seed=seed)
        face_ci = cluster_bootstrap_ci(face_clusters, b=b, seed=seed)
        action_dist = {a: counts[a] / n_readings for a in counts}
        effect = {
            "index": i, "speaker": turn.speaker, "text": turn.text,
            "listeners": listeners, "relationship": rel_ci, "face": face_ci,
            "action_dist": action_dist,
            "disagreement": board._disagreement(action_dist, n_readings),
            "n_readings": n_readings, "readings": readings_out,
            "scale": scale, "n_expected": len(tasks),
        }
        if scale == "dist5":
            effect["spread"] = float(
                sum(float(r["rel_spread"]) for r in readings) / len(readings))
        effects.append(effect)
    return effects


def _idx_of(effect: dict, pos: int) -> int:
    idx = effect.get("index", pos)
    return idx if isinstance(idx, int) else pos


def compare_records(text_record, audio_record) -> dict:
    """Per-turn text/audio values and Δ, Spearman, mean |Δ|, top-3 |Δ| turns."""
    text_by_idx: dict[int, dict] = {}
    for pos, e in enumerate((text_record or {}).get("effects", []) or []):
        if isinstance(e, dict):
            text_by_idx.setdefault(_idx_of(e, pos), e)
    audio_by_idx: dict[int, dict] = {}
    for pos, e in enumerate((audio_record or {}).get("effects", []) or []):
        if isinstance(e, dict):
            audio_by_idx.setdefault(_idx_of(e, pos), e)
    per_turn: list[dict] = []
    for idx in sorted(set(text_by_idx) & set(audio_by_idx)):
        tv = float(board._value_of(text_by_idx[idx]))
        av = float(board._value_of(audio_by_idx[idx]))
        per_turn.append({"index": idx, "text": tv, "audio": av,
                         "delta": av - tv})
    if per_turn:
        rho = float(spearman([r["text"] for r in per_turn],
                             [r["audio"] for r in per_turn]))
        mean_abs = float(sum(abs(r["delta"]) for r in per_turn)
                         / len(per_turn))
    else:
        rho = 0.0
        mean_abs = 0.0
    largest = sorted(per_turn, key=lambda r: (-abs(r["delta"]), r["index"]))[:3]
    return {
        "n_turns": len(per_turn),
        "per_turn": per_turn,
        "spearman": rho,
        "mean_abs_delta": mean_abs,
        "largest_delta": [dict(r) for r in largest],
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-audio-panel")
    ap.add_argument("--audio", required=True)
    ap.add_argument("--turns", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--moves", default=None)
    ap.add_argument("--text-record", default=None)
    ap.add_argument("--scale", default="fine100", choices=list(AUDIO_SCALES))
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--pre", type=float, default=DEFAULT_PRE)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--window", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--b", type=int, default=1000)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--budget", type=float, default=None)
    ap.add_argument("--id", dest="item_id", default=None)
    ap.add_argument("--genre", default="")
    ap.add_argument("--work", default=None)
    ap.add_argument("--calls", default=None)
    args = ap.parse_args(argv)

    turns = load_turns(args.turns)
    try:
        plan = clip_plan(turns, args.pre)
    except MissingTimestamps as exc:
        ap.error(str(exc))
    except ValueError as exc:
        ap.error(str(exc))
    tags: list = []
    if args.moves:
        with open(args.moves, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                tags.append(MoveTag(
                    turn_index=obj.get("turn_index", 0),
                    speaker=obj.get("speaker", ""),
                    moves=list(obj.get("moves", []) or []),
                    addressed_to=obj.get("addressed_to", ""),
                    quote=obj.get("quote", ""),
                    failed=bool(obj.get("failed", False)),
                ))
    effort = args.effort
    if isinstance(effort, str) and effort.lower() in ("none", "null", ""):
        effort = None
    meter = Meter(budget_usd=args.budget)
    out_path = Path(args.out)
    calls_path = Path(args.calls) if args.calls else out_path.parent / "calls.jsonl"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    client = LoggingClient(MuseClient(meter=meter, default_effort=effort),
                           CallLog(calls_path), run_id)
    meter.add_audio(args.model, sum(s for _, s in plan))
    try:
        meter.check()
    except BudgetExceeded as exc:
        raise SystemExit(f"budget exceeded: {exc}")
    if args.work:
        work = Path(args.work)
        work.mkdir(parents=True, exist_ok=True)
        clips, audio_seconds = cut_turn_clips(args.audio, plan, work)
    else:
        with tempfile.TemporaryDirectory(prefix="uptake-audio-") as tmp:
            clips, audio_seconds = cut_turn_clips(args.audio, plan, tmp)
    try:
        effects = evaluate_turns_audio(
            client, turns, clips, scale=args.scale, n=args.n, model=args.model,
            window=args.window, b=args.b, seed=args.seed,
            max_workers=args.workers, move_tags=tags or None,
            item_id=args.item_id or out_path.stem)
    except BudgetExceeded as exc:
        raise SystemExit(f"budget exceeded: {exc}")
    moves_out = [{
        "turn_index": t.turn_index, "speaker": t.speaker,
        "moves": list(t.moves), "addressed_to": t.addressed_to,
        "quote": t.quote, "failed": bool(t.failed),
    } for t in tags]
    hits = board.forecast_hits(effects, tags)
    item_id = args.item_id or out_path.stem
    record = {
        "id": item_id,
        "genre": args.genre,
        "model": args.model,
        "scale": args.scale,
        "modality": "audio",
        "audio_seconds": audio_seconds,
        "moves": moves_out,
        "effects": effects,
        "board": effects,
        "hits": hits,
        "gold": None,
        "n": args.n,
        "structured": True,
    }
    if str(out_path.parent) and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    if args.text_record:
        with open(args.text_record, encoding="utf-8") as f:
            text_record = json.load(f)
        cmp = compare_records(text_record, record)
        cmp_path = out_path.parent / "compare.json"
        with open(cmp_path, "w", encoding="utf-8") as f:
            json.dump(cmp, f, ensure_ascii=False, indent=2)
    n_readings = sum(int(e.get("n_readings", 0) or 0) for e in effects)
    n_expected = sum(int(e.get("n_expected", 0) or 0) for e in effects)
    total = meter.cost()
    print(f"done {len(turns)} turns readings {n_readings}/{n_expected} "
          f"audio {audio_seconds:.1f}s spend: ${total:.4f} "
          f"(tokens ${total - meter.audio_cost():.4f} + "
          f"audio ${meter.audio_cost():.4f})")
    if args.text_record:
        print(f"compare: spearman {cmp['spearman']:+.3f} "
              f"mean|d| {cmp['mean_abs_delta']:.3f}")
