"""Re-read the board panel with another scale / panel size (task28)."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import random
import sys
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import batch
from . import board
from .labels import NEXT_ACTIONS
from .panel import DEFAULT_MIX, build_panel
from .reader import DEFAULT_MODEL, MuseClient
from .stats import cluster_bootstrap_ci
from .trace import CallLog, LoggingClient
from .transcribe import Turn
from .types import Receiver
from .usage import BudgetExceeded, Meter

SCALES = ("v2", "fine100", "seven")


def build_scaled_messages(turns, i, listener: str, receiver: Receiver, scale, window=6) -> list[dict]:
    if scale == "v2":
        return board.build_listener_messages(turns, i, listener, receiver, window)
    if scale == "fine100":
        keys = "next_action, relationship_after, face_effect, understood"
        rel_line = ("- relationship_after must be an integer from 0 to 100: how you feel "
                    "toward the speaker of the TARGET turn now, where 50 = unchanged "
                    "from before this turn, 0 = much colder, 100 = much warmer")
        face_line = ("- face_effect must be an integer from 0 to 100: what the TARGET turn "
                     "does to your face (dignity, standing), where 50 = no effect, "
                     "0 = severe threat to your face, 100 = strongly affirms your face")
    elif scale == "seven":
        keys = "next_action, relationship_shift, face_effect, understood"
        rel_line = ("- relationship_shift must be an integer from -3 (much colder) to 3 "
                    "(much warmer) toward the speaker of the TARGET turn, 0 = unchanged")
        face_line = ("- face_effect must be an integer from -3 (severe threat to your face) "
                     "to 3 (strongly affirms your face), 0 = no effect")
    else:
        raise ValueError(f"bad scale: {scale!r}")
    system = (
        "You are this listener reacting in a conversation:\n"
        f"- listener: {listener}\n"
        f"- relationship: {receiver.relationship}\n"
        f"- power: {receiver.power}\n"
        f"- channel: {receiver.channel}\n"
        f"- state: {receiver.state}\n"
        f"- stance: {receiver.stance}\n\n"
        "You just heard the TARGET turn below, in the context of the previous turns. "
        "You do not know what the speaker intended. "
        "React the way this listener would, then reply with JSON only "
        f"(no other text) with keys {keys}.\n"
        f"- next_action must be one of: {', '.join(NEXT_ACTIONS)}\n"
        f"{rel_line}\n"
        f"{face_line}\n"
        "- understood must be one sentence saying what you think the speaker meant"
    )
    start = max(0, i - window)
    lines = [f"{turns[j].speaker}: {turns[j].text}" for j in range(start, i)]
    lines.append(f"TARGET turn {turns[i].speaker}: {turns[i].text}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _coerce_num(value, lo: int, hi: int):
    if isinstance(value, bool):
        return None
    try:
        rounded = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return None
    return max(lo, min(hi, rounded))


def parse_scaled(raw, scale) -> dict | None:
    if scale not in SCALES:
        raise ValueError(f"bad scale: {scale!r}")
    if not isinstance(raw, str):
        return None
    try:
        start = raw.find("{")
        if start < 0:
            return None
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
        if not isinstance(obj, dict):
            return None
    except (ValueError, json.JSONDecodeError):
        return None
    nxt = str(obj.get("next_action", "")).strip().lower()
    if nxt not in NEXT_ACTIONS:
        return None
    if scale == "v2":
        rel_key, face_key = "relationship_shift", "face_threat"
        rel_lo, rel_hi = -2, 2
        face_lo, face_hi = 0, 2
    elif scale == "fine100":
        rel_key, face_key = "relationship_after", "face_effect"
        rel_lo, rel_hi = 0, 100
        face_lo, face_hi = 0, 100
    else:  # seven
        rel_key, face_key = "relationship_shift", "face_effect"
        rel_lo, rel_hi = -3, 3
        face_lo, face_hi = -3, 3
    if rel_key not in obj or face_key not in obj:
        return None
    rel = _coerce_num(obj.get(rel_key), rel_lo, rel_hi)
    face = _coerce_num(obj.get(face_key), face_lo, face_hi)
    if rel is None or face is None:
        return None
    if scale == "v2":
        rel_norm = float(rel)
        face_norm = float(-face)
    elif scale == "fine100":
        rel_norm = float((rel - 50) / 25)
        face_norm = float((face - 50) / 25)
    else:
        rel_norm = float(rel * 2 / 3)
        face_norm = float(face * 2 / 3)
    return {
        "next_action": nxt,
        "rel_raw": rel,
        "face_raw": face,
        "rel_norm": rel_norm,
        "face_norm": face_norm,
    }


def _read_one_scaled(client, turns, i, listener, receiver, s, model, window, scale, item_id=""):
    messages = build_scaled_messages(turns, i, listener, receiver, scale, window)
    meta = {"layer": "board", "scale": scale, "item": item_id, "turn": i,
            "listener": listener, "member": receiver.id,
            "stance": receiver.stance, "sample": s}
    for _ in range(2):
        try:
            raw = board._call_board(client, messages, model, meta)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        parsed = parse_scaled(raw, scale)
        if parsed is not None:
            return parsed
    return None


def evaluate_turns_scaled(client, turns, scale="v2", n=3, k=1, model=DEFAULT_MODEL,
                          stance_mix=DEFAULT_MIX, window=6, b=1000, seed=0,
                          max_workers=8, move_tags=None, item_id="") -> list[dict]:
    if scale not in SCALES:
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
            effects.append({
                "index": i, "speaker": turn.speaker, "text": turn.text,
                "listeners": [], "relationship": (0.0, 0.0, 0.0),
                "face": (0.0, 0.0, 0.0), "action_dist": {},
                "disagreement": 0.0, "n_readings": 0, "readings": [],
                "scale": scale, "n_expected": 0,
            })
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
            futures = [pool.submit(_read_one_scaled, client, turns, i, listener,
                                   receiver, s, model, window, scale, item_id)
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
            if scale == "v2":
                row["relationship_shift"] = r["rel_raw"]
                row["face_threat"] = r["face_raw"]
            readings_out.append(row)
        if not readings:
            effects.append({
                "index": i, "speaker": turn.speaker, "text": turn.text,
                "listeners": listeners, "relationship": (0.0, 0.0, 0.0),
                "face": (0.0, 0.0, 0.0), "action_dist": {},
                "disagreement": 0.0, "n_readings": 0, "readings": [],
                "scale": scale, "n_expected": len(tasks),
            })
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
        effects.append({
            "index": i, "speaker": turn.speaker, "text": turn.text,
            "listeners": listeners, "relationship": rel_ci, "face": face_ci,
            "action_dist": action_dist,
            "disagreement": board._disagreement(action_dist, n_readings),
            "n_readings": n_readings, "readings": readings_out,
            "scale": scale, "n_expected": len(tasks),
        })
    return effects


def turns_from_record(record) -> list[Turn]:
    effects = record.get("effects", []) or []

    def _idx(e):
        v = e.get("index", 0) if isinstance(e, dict) else 0
        return v if isinstance(v, int) else 0

    ordered = sorted([e for e in effects if isinstance(e, dict)], key=_idx)
    return [Turn(speaker=e.get("speaker", ""), text=e.get("text", "")) for e in ordered]


def select_ids(src_dir, sample=None, sample_seed=6, ids_file=None) -> list[str]:
    src = Path(src_dir)
    try:
        found = [p.stem for p in src.glob("casino-*.json") if p.is_file()]
    except OSError:
        found = []
    sorted_ids = sorted(found)
    if ids_file is not None:
        want: list[str] = []
        with open(ids_file, encoding="utf-8") as f:
            for line in f:
                cid = line.strip()
                if not cid:
                    continue
                want.append(cid)
        existing = set(sorted_ids)
        return [cid for cid in want if cid in existing]
    if isinstance(sample, int):
        k = min(sample, len(sorted_ids))
        return random.Random(sample_seed).sample(sorted_ids, k)
    return sorted_ids


def rescore_item(client, record, out_dir, scale, n, k, model, inner_workers, window=6) -> dict:
    turns = turns_from_record(record)
    moves = record.get("moves", []) or []
    item_id = record.get("id", "")
    effects = evaluate_turns_scaled(
        client, turns, scale=scale, n=n, k=k, model=model,
        window=window, max_workers=inner_workers, move_tags=moves,
        item_id=item_id)
    hits = board.forecast_hits(effects, moves)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record_out = {
        "id": record.get("id"),
        "genre": record.get("genre", ""),
        "model": model,
        "moves": copy.deepcopy(moves),
        "effects": effects,
        "board": effects,
        "hits": hits,
        "gold": copy.deepcopy(record.get("gold")),
        "scale": scale,
        "n": n,
    }
    with open(out / f"{item_id}.json", "w", encoding="utf-8") as f:
        json.dump(record_out, f, ensure_ascii=False, indent=2)
    return {
        "id": record.get("id"),
        "n_turns": len(turns),
        "n_readings": sum(int(e.get("n_readings", 0) or 0) for e in effects),
        "n_expected": sum(int(e.get("n_expected", 0) or 0) for e in effects),
    }


def main(argv=None) -> None:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser(prog="uptake-rescore")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", required=True, choices=list(SCALES))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--sample-seed", type=int, default=6)
    ap.add_argument("--ids-file", default=None)
    ap.add_argument("--budget", type=float, required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--inner-workers", type=int, default=8)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--window", type=int, default=6)
    args = ap.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    selected = select_ids(src, sample=args.sample, sample_seed=args.sample_seed,
                          ids_file=args.ids_file)
    with open(out / "ids.txt", "w", encoding="utf-8") as f:
        for cid in selected:
            f.write(cid + "\n")
    to_run: list[str] = []
    n_skipped_existing = 0
    for cid in selected:
        if (out / f"{cid}.json").exists():
            n_skipped_existing += 1
        else:
            to_run.append(cid)
    meter = Meter(budget_usd=args.budget)
    inner = MuseClient(meter=meter, default_effort=(args.effort or None))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    client = LoggingClient(inner, CallLog(out / "calls.jsonl"), run_id)

    def _run_one(cid: str) -> dict:
        with open(src / f"{cid}.json", encoding="utf-8") as f:
            record = json.load(f)
        return rescore_item(client, record, out, args.scale, args.n, args.k,
                            args.model, args.inner_workers, window=args.window)

    summaries: list[dict] = []
    errors: list[dict] = []
    budget_stop = False
    max_workers = args.workers if isinstance(args.workers, int) and args.workers >= 1 else 1
    queue: deque[str] = deque(to_run)
    in_flight: dict[cf.Future, str] = {}
    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        while queue or in_flight:
            while queue and len(in_flight) < max_workers and not budget_stop:
                cid = queue.popleft()
                in_flight[pool.submit(_run_one, cid)] = cid
            if not in_flight:
                break
            done, _ = cf.wait(list(in_flight), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                cid = in_flight.pop(fut)
                try:
                    summaries.append(fut.result())
                except BudgetExceeded:
                    errors.append({"id": cid, "error": "budget"})
                    budget_stop = True
                except Exception as exc:  # noqa: BLE001 - keep going
                    errors.append({"id": cid, "error": str(exc)})
    n_selected = len(selected)
    n_done = sum(1 for cid in selected if (out / f"{cid}.json").exists())
    readings = 0
    expected = 0
    for cid in selected:
        p = out / f"{cid}.json"
        if not p.exists():
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        for e in data.get("effects", []) or []:
            if not isinstance(e, dict):
                continue
            try:
                readings += int(e.get("n_readings", 0) or 0)
            except (TypeError, ValueError):
                pass
            try:
                expected += int(e.get("n_expected", 0) or 0)
            except (TypeError, ValueError):
                pass
    parse_fail_rate = (1 - readings / expected) if expected else 0
    spend = meter.by_model()
    cost = meter.cost()
    run_meta = {
        "run_id": run_id,
        "argv": list(sys.argv),
        "settings": {
            "src": args.src, "out": args.out, "scale": args.scale,
            "n": args.n, "k": args.k, "sample": args.sample,
            "sample_seed": args.sample_seed, "ids_file": args.ids_file,
            "budget": args.budget, "workers": args.workers,
            "inner_workers": args.inner_workers, "effort": args.effort,
            "model": args.model, "window": args.window,
        },
        "commit": batch._git_commit(),
    }
    with open(out / "run.json", "w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)
    report = {
        "n_selected": n_selected,
        "n_done": n_done,
        "n_skipped_existing": n_skipped_existing,
        "n_errors": len(errors),
        "errors": errors,
        "readings": readings,
        "expected": expected,
        "parse_fail_rate": parse_fail_rate,
        "spend": spend,
        "cost": cost,
        "budget_stop": bool(budget_stop),
    }
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"done {n_done}/{n_selected} readings {readings}/{expected} "
          f"parse_fail={parse_fail_rate:.3f} errors {len(errors)} spend: ${cost:.4f}")
