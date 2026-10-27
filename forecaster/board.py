"""L5 board: turn-by-turn position curve, turning points, forecast check."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from .labels import NEXT_ACTIONS
from .panel import DEFAULT_MIX, build_panel
from .reader import DEFAULT_MODEL
from .usage import BudgetExceeded
from .stats import cluster_bootstrap_ci, wilson_ci
from .types import Receiver
from .v2 import parse_reading_v2

ACTION_TO_MOVES = {
    "comply": {"accept", "concede"},
    "proceed": {"inform", "propose", "accept", "smalltalk"},
    "refuse": {"reject"},
    "negotiate": {"counter", "propose", "appeal_need", "appeal_fairness"},
    "ask_clarification": {"ask", "probe"},
    "escalate": {"pressure", "challenge"},
    "retaliate": {"challenge", "pressure"},
    "withdraw": {"deflect", "other"},
    "ignore": {"deflect", "other"},
    "repair": {"repair", "empathize", "praise"},
}

MOVE_PRIORITY: list[tuple[str, str]] = [
    ("reject", "refuse"),
    ("accept", "comply"),
    ("concede", "comply"),
    ("counter", "negotiate"),
    ("propose", "negotiate"),
    ("pressure", "escalate"),
    ("challenge", "retaliate"),
    ("repair", "repair"),
    ("ask", "ask_clarification"),
    ("probe", "ask_clarification"),
    ("appeal_need", "negotiate"),
    ("appeal_fairness", "negotiate"),
    ("empathize", "repair"),
    ("praise", "repair"),
    ("deflect", "withdraw"),
]


def actual_class(moves: list[str]) -> str:
    present = set(moves or [])
    for move, cls in MOVE_PRIORITY:
        if move in present:
            return cls
    return "proceed"


_SPARK = "▁▂▃▄▅▆▇█"


def build_listener_messages(turns, i, listener: str, receiver: Receiver, window=6) -> list[dict]:
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
        "(no other text) with keys next_action, relationship_shift, face_threat, understood.\n"
        f"- next_action must be one of: {', '.join(NEXT_ACTIONS)}\n"
        "- relationship_shift must be an integer from -2 (much colder) to 2 (much warmer) "
        "toward the speaker of the TARGET turn\n"
        "- face_threat must be an integer from 0 (none) to 2 (severe) to you as the listener\n"
        "- understood must be one sentence saying what you think the speaker meant"
    )
    start = max(0, i - window)
    lines = [f"{turns[j].speaker}: {turns[j].text}" for j in range(start, i)]
    lines.append(f"TARGET turn {turns[i].speaker}: {turns[i].text}")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _read_one(client, turns, i, listener, receiver, s, model, window):
    messages = build_listener_messages(turns, i, listener, receiver, window)
    text = turns[i].text
    for attempt in range(2):
        try:
            raw = client.complete(messages, model, 1.0)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        reading = parse_reading_v2(raw, receiver.id, s, text)
        if reading is not None:
            return reading
    return None


def _disagreement(action_dist: dict, n_readings: int) -> float:
    if n_readings <= 1 or len(action_dist) <= 1:
        return 0.0
    total = sum(action_dist.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for share in action_dist.values():
        p = share / total if total != 1.0 else share
        if p > 0:
            entropy -= p * math.log(p)
    denom = math.log(len(action_dist))
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, entropy / denom))


def _addressed_to(tag) -> str:
    if isinstance(tag, dict):
        addr = tag.get("addressed_to", "")
    else:
        addr = getattr(tag, "addressed_to", "")
    return addr if isinstance(addr, str) else str(addr)


def evaluate_turns(client, turns, n=3, k=1, model=DEFAULT_MODEL,
                   stance_mix=DEFAULT_MIX, window=6, b=1000, seed=0,
                   max_workers=8, move_tags=None) -> list[dict]:
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
                addr_by_index[idx] = _addressed_to(t)
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
                "disagreement": 0.0, "n_readings": 0,
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
            futures = [pool.submit(_read_one, client, turns, i, listener, receiver, s,
                                   model, window)
                       for listener, receiver, s in tasks]
            results = [f.result() for f in futures]
        readings = [r for r in results if r is not None]
        n_readings = len(readings)
        if not readings:
            effects.append({
                "index": i, "speaker": turn.speaker, "text": turn.text,
                "listeners": listeners, "relationship": (0.0, 0.0, 0.0),
                "face": (0.0, 0.0, 0.0), "action_dist": {},
                "disagreement": 0.0, "n_readings": 0,
            })
            continue
        rel_clusters: dict[str, list[float]] = {}
        face_clusters: dict[str, list[float]] = {}
        counts: Counter = Counter()
        for r in readings:
            rel_clusters.setdefault(r.receiver_id, []).append(float(r.relationship_shift))
            face_clusters.setdefault(r.receiver_id, []).append(float(-r.face_threat))
            counts[r.next_action] += 1
        rel_ci = cluster_bootstrap_ci(rel_clusters, b=b, seed=seed)
        face_ci = cluster_bootstrap_ci(face_clusters, b=b, seed=seed)
        action_dist = {a: counts[a] / n_readings for a in counts}
        effects.append({
            "index": i, "speaker": turn.speaker, "text": turn.text,
            "listeners": listeners, "relationship": rel_ci, "face": face_ci,
            "action_dist": action_dist,
            "disagreement": _disagreement(action_dist, n_readings),
            "n_readings": n_readings,
        })
    return effects


def _value_of(effect: dict) -> float:
    if "value" in effect and isinstance(effect["value"], (int, float)):
        return float(effect["value"])
    rel = effect.get("relationship", (0, 0, 0))[0]
    face = effect.get("face", (0, 0, 0))[0]
    return float(rel) + float(face)


def position_curve(effects) -> list[float]:
    curve: list[float] = []
    total = 0.0
    for e in effects:
        total += _value_of(e)
        curve.append(total)
    return curve


def annotate_turn(v) -> str:
    v = float(v)
    if v <= -1.5:
        return "??"
    if v <= -0.75:
        return "?"
    if v <= -0.25:
        return "?!"
    if v >= 1.5:
        return "!!"
    if v >= 0.75:
        return "!"
    return "="


def _idx_of(effect: dict, pos: int) -> int:
    idx = effect.get("index", pos)
    return idx if isinstance(idx, int) else pos


def turning_point(effects, min_turn=2) -> int | None:
    best = None
    best_abs = 0.0
    for pos, e in enumerate(effects):
        idx = _idx_of(e, pos)
        if idx < min_turn:
            continue
        v = _value_of(e)
        if best is None or abs(v) > best_abs:
            best = idx
            best_abs = abs(v)
    if best is None or best_abs == 0:
        return None
    return best


def disagreement_spike(effects, z=2.0, min_turn=3) -> int | None:
    for pos, e in enumerate(effects):
        idx = _idx_of(e, pos)
        if idx < min_turn:
            continue
        earlier = effects[:pos]
        if not earlier:
            continue
        vals = [float(x.get("disagreement", 0.0)) for x in earlier]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = max(math.sqrt(var), 0.05)
        if float(e.get("disagreement", 0.0)) > mean + z * std:
            return idx
    return None


def _tag_info(tag) -> tuple:
    if isinstance(tag, dict):
        return (tag.get("turn_index"), tag.get("speaker"),
                list(tag.get("moves", []) or []))
    return (getattr(tag, "turn_index", None), getattr(tag, "speaker", None),
            list(getattr(tag, "moves", []) or []))


def forecast_hits(effects, move_tags) -> list[dict]:
    by_index: dict[int, tuple] = {}
    for t in move_tags or []:
        idx, speaker, moves = _tag_info(t)
        if isinstance(idx, int) and idx not in by_index:
            by_index[idx] = (speaker, moves)
    hits: list[dict] = []
    for pos, e in enumerate(effects):
        i = _idx_of(e, pos)
        dist = e.get("action_dist", {}) or {}
        if not dist:
            continue
        nxt = by_index.get(i + 1)
        if nxt is None:
            continue
        next_speaker, actual = nxt
        listeners = e.get("listeners", []) or []
        if next_speaker not in listeners:
            continue
        if not actual:
            continue
        predicted = sorted(dist, key=lambda a: (-dist[a], a))[0]
        cls = actual_class(list(actual))
        hit = (predicted == cls)
        p_actual = float(dist.get(cls, 0.0))
        n_readings = int(e.get("n_readings", 0) or 0)
        hits.append({
            "index": i, "predicted": predicted,
            "actual_moves": list(actual), "actual_class": cls,
            "hit": hit, "p_actual": float(p_actual),
            "n_readings": n_readings,
            "action_dist": dict(dist),
        })
    return hits


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = q * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def forecast_score(hits, b=1000, seed=0) -> dict:
    n = len(hits)
    if n == 0:
        lo, hi = wilson_ci(0, 0)
        return {"hit_rate": 0.0, "hit_rate_ci": (lo, hi),
                "p_actual_mean": 0.0, "p_actual_ci": (0.0, 0.0),
                "baseline": 0.0, "baseline_action": "always_proceed", "n": 0}
    n_hits = sum(1 for h in hits if h["hit"])
    hit_rate = n_hits / n
    lo, hi = wilson_ci(n_hits, n)
    p_vals = [float(h["p_actual"]) for h in hits]
    p_mean = sum(p_vals) / n
    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(max(1, b)):
        sample = [p_vals[rng.randrange(n)] for _ in range(n)]
        reps.append(sum(sample) / n)
    reps.sort()
    plo, phi = _percentile(reps, 0.025), _percentile(reps, 0.975)
    if hits and any("actual_class" in h for h in hits):
        actual_counts: Counter = Counter(h.get("actual_class", "proceed") for h in hits)
        majority = sorted(actual_counts, key=lambda a: (-actual_counts[a], a))[0]
        baseline = actual_counts[majority] / n
        baseline_action = f"always_{majority}"
    else:
        counts: Counter = Counter(h["predicted"] for h in hits)
        common = sorted(counts, key=lambda a: (-counts[a], a))[0]
        common_moves = ACTION_TO_MOVES.get(common, set())
        baseline = sum(1 for h in hits if set(h.get("actual_moves", [])) & common_moves) / n
        baseline_action = f"always_{common}"
    return {"hit_rate": hit_rate, "hit_rate_ci": (lo, hi),
            "p_actual_mean": p_mean, "p_actual_ci": (plo, phi),
            "baseline": baseline, "baseline_action": baseline_action, "n": n}


def forecast_skill(hits, b=1000, seed=0, alpha=0.5) -> dict:
    n = len(hits)
    if n == 0:
        return {"skill_mean": 0.0, "skill_ci": (0.0, 0.0),
                "n": 0, "majority_class": "proceed",
                "majority_hit_rate": 0.0}
    k_actions = len(NEXT_ACTIONS)
    actuals = [h.get("actual_class", "proceed") for h in hits]
    counts_all: Counter = Counter(actuals)
    majority = sorted(counts_all, key=lambda a: (-counts_all[a], a))[0]
    majority_hit_rate = counts_all[majority] / n
    gains: list[float] = []
    for pos, h in enumerate(hits):
        actual = h.get("actual_class", "proceed")
        dist = h.get("action_dist") or {}
        if dist:
            share = float(dist.get(actual, 0.0))
        else:
            share = float(h.get("p_actual", 0.0))
        nn = int(h.get("n_readings", 0) or 0)
        p_model = (share * nn + alpha) / (nn + k_actions * alpha)
        loo_count = sum(1 for j, hh in enumerate(hits)
                        if j != pos and hh.get("actual_class") == actual)
        p_clim = (loo_count + alpha) / ((n - 1) + k_actions * alpha)
        gains.append(math.log(p_model) - math.log(p_clim))
    skill_mean = sum(gains) / n
    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(max(1, b)):
        sample = [gains[rng.randrange(n)] for _ in range(n)]
        reps.append(sum(sample) / n)
    reps.sort()
    slo, shi = _percentile(reps, 0.025), _percentile(reps, 0.975)
    return {"skill_mean": skill_mean, "skill_ci": (slo, shi),
            "n": n, "majority_class": majority,
            "majority_hit_rate": majority_hit_rate}


def _spark_char(pos: float, mn: float, mx: float) -> str:
    if mx <= mn:
        return _SPARK[0]
    frac = (pos - mn) / (mx - mn)
    idx = int(round(frac * (len(_SPARK) - 1)))
    return _SPARK[max(0, min(len(_SPARK) - 1, idx))]


def main(argv=None) -> None:
    from .moves import MoveTag
    from .reader import MuseClient
    from .transcribe import load_turns

    ap = argparse.ArgumentParser(prog="uptake-board")
    ap.add_argument("--turns", required=True)
    ap.add_argument("--moves", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--b", type=int, default=1000)
    ap.add_argument("--window", type=int, default=6)
    args = ap.parse_args(argv)

    turns = load_turns(args.turns)
    tags: list[MoveTag] = []
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
    effects = evaluate_turns(
        MuseClient(), turns, n=args.n, k=args.k, model=args.model,
        window=args.window, b=args.b, seed=args.seed,
        max_workers=args.workers, move_tags=tags or None,
    )
    curve = position_curve(effects)
    mn = min(curve) if curve else 0.0
    mx = max(curve) if curve else 0.0
    out_turns: list[dict] = []
    for e, pos in zip(effects, curve):
        value = _value_of(e)
        ann = annotate_turn(value)
        out_turns.append({**e, "value": value, "annotation": ann,
                          "position": pos})
    tp = turning_point(effects)
    ds = disagreement_spike(effects)
    forecast = None
    skill = None
    if tags:
        hits = forecast_hits(effects, tags)
        if hits:
            forecast = forecast_score(hits)
            skill = forecast_skill(hits)
    board = {"turns": out_turns, "turning_point": tp,
             "disagreement_spike": ds, "forecast": forecast,
             "skill": skill}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(board, f, ensure_ascii=False, indent=2)
    for e in out_turns:
        text = str(e.get("text", "")).replace("\n", " ")[:60]
        print(f"#{e['index']} {e['speaker']} {e['annotation']} "
              f"{e['value']:+.2f} {e['position']:+.2f} "
              f"{_spark_char(e['position'], mn, mx)} {text}")
    if forecast is not None:
        lo, hi = forecast["hit_rate_ci"]
        print(f"Forecast hit rate: {forecast['hit_rate']:.2f} [{lo:.2f}-{hi:.2f}] "
              f"(n={forecast['n']}, baseline {forecast['baseline_action']} {forecast['baseline']:.2f})")
    if skill is not None:
        slo, shi = skill["skill_ci"]
        print(f"Forecast skill vs climatology: {skill['skill_mean']:+.3f} [{slo:+.3f}-{shi:+.3f}] "
              f"(n={skill['n']}, majority {skill['majority_class']} {skill['majority_hit_rate']:.2f})")
        print("skill > 0 with CI above 0 = better than climatology")
