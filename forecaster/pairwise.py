"""Pairwise-comparison reader and Bradley-Terry ratings (task34, Amendment 7 P7-A)."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import copy
import json
import math
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import batch
from . import board
from . import scalecheck as SC
from .batch import spearman
from .panel import DEFAULT_MIX, build_panel
from .reader import DEFAULT_MODEL, MuseClient
from .trace import CallLog, LoggingClient
from .transcribe import Turn
from .types import Receiver
from .usage import BudgetExceeded, Meter

STANCES = ("neutral", "trusting", "suspicious")
WINDOW = 6
VOTES = ("first", "second", "tie")


def pairwise_panel(pair_id) -> list[Receiver]:
    """The three board receiver stances for one pair (same personas as board)."""
    base = Receiver(
        id=f"pair{pair_id}",
        relationship="participant in this conversation",
        power="equal", channel="conversation", state="",
    )
    return build_panel(base, 3, 0, DEFAULT_MIX)


def collect_pool(records) -> tuple[list[dict], dict]:
    """Every effect turn with a resolvable listener.

    Returns (pool, contexts). Pool entries hold dialogue, turn (effect index),
    speaker, listener and genre, sorted by (dialogue, turn). Contexts maps a
    dialogue id to its turns, index->position map and genre for window building.
    Listener resolution mirrors board.evaluate_turns.
    """
    if isinstance(records, dict):
        items = list(records.items())
    else:
        items = [(r.get("id") if isinstance(r, dict) else None, r)
                 for r in (records or [])]
    pool: list[dict] = []
    contexts: dict = {}
    for item_id, record in items:
        if not isinstance(record, dict):
            continue
        did = item_id if isinstance(item_id, str) and item_id else record.get("id")
        if not isinstance(did, str) or not did:
            continue
        genre = record.get("genre", "")
        genre = genre if isinstance(genre, str) else ""
        effects = [e for e in (record.get("effects", []) or [])
                   if isinstance(e, dict)]

        def _idx(e):
            v = e.get("index", 0)
            return v if isinstance(v, int) else 0

        ordered = sorted(effects, key=_idx)
        turns = [Turn(speaker=e.get("speaker", ""), text=e.get("text", ""))
                 for e in ordered]
        seen: list[str] = []
        for t in turns:
            if t.speaker not in seen:
                seen.append(t.speaker)
        moves = record.get("moves", []) or []
        addr_by_index: dict[int, str] = {}
        for m in moves:
            if isinstance(m, dict):
                idx = m.get("turn_index")
            else:
                idx = getattr(m, "turn_index", None)
            if isinstance(idx, int) and idx not in addr_by_index:
                addr_by_index[idx] = board._addressed_to(m)
        pos_of: dict[int, int] = {}
        for pos, e in enumerate(ordered):
            idx = e.get("index", pos)
            idx = idx if isinstance(idx, int) else pos
            pos_of[idx] = pos
            listeners = [s for s in seen
                         if s != turns[pos].speaker and s != "unknown"]
            if moves:
                addr = addr_by_index.get(idx, "")
                if (addr and addr != turns[pos].speaker and addr in seen
                        and addr != "unknown"):
                    listeners = [addr]
            if not listeners:
                continue
            pool.append({"dialogue": did, "turn": idx,
                         "speaker": turns[pos].speaker,
                         "listener": listeners[0], "genre": genre})
        contexts[did] = {"turns": turns, "pos_of": pos_of, "genre": genre}
    pool.sort(key=lambda e: (e["dialogue"], e["turn"]))
    return pool, contexts


def make_pairs(pool, k=4, seed=0) -> list[dict]:
    """Pair each turn with k turns from other dialogues of the same genre.

    Unordered pairs are deduped; each pair gets a random presentation order.
    Pair dicts hold id, turn refs a/b (canonical order) and first ("a"/"b").
    """
    try:
        kk = int(k)
    except (TypeError, ValueError):
        kk = 4
    if kk < 0:
        kk = 0
    rng = random.Random(seed)
    by_genre: dict[str, list[dict]] = {}
    for e in pool or []:
        if not isinstance(e, dict):
            continue
        by_genre.setdefault(e.get("genre", ""), []).append(e)
    found: dict[tuple, tuple[dict, dict]] = {}
    for e in pool or []:
        if not isinstance(e, dict):
            continue
        elig = [p for p in by_genre.get(e.get("genre", ""), [])
                if p.get("dialogue") != e.get("dialogue")]
        if not elig or kk == 0:
            continue
        for q in rng.sample(elig, min(kk, len(elig))):
            ka = (e.get("dialogue"), e.get("turn"))
            kb = (q.get("dialogue"), q.get("turn"))
            key = tuple(sorted([ka, kb]))
            if key not in found:
                first_e, second_e = (e, q) if key[0] == ka else (q, e)
                found[key] = (first_e, second_e)
    pairs = []
    for pid, key in enumerate(sorted(found)):
        ea, eb = found[key]
        pairs.append({
            "id": pid,
            "a": _ref(ea),
            "b": _ref(eb),
            "first": "a" if rng.random() < 0.5 else "b",
        })
    return pairs


def _ref(e: dict) -> dict:
    return {"dialogue": e.get("dialogue"), "turn": e.get("turn"),
            "speaker": e.get("speaker"), "listener": e.get("listener")}


def snippet_lines(turns, pos, window=WINDOW) -> list[str]:
    """Context lines for a judged turn; same window slicing as the board."""
    start = max(0, pos - window)
    return [f"{turns[j].speaker}: {turns[j].text}" for j in range(start, pos)]


def snippet_for(ref, contexts, window=WINDOW) -> str:
    ctx = contexts.get(ref.get("dialogue"), {})
    turns = ctx.get("turns", []) or []
    pos = ctx.get("pos_of", {}).get(ref.get("turn"), 0)
    if not isinstance(pos, int) or pos < 0 or pos >= len(turns):
        pos = 0
    lines = snippet_lines(turns, pos, window)
    lines.append(f"MESSAGE TO JUDGE {turns[pos].speaker}: {turns[pos].text}"
                 if turns else "MESSAGE TO JUDGE (empty)")
    return "\n".join(lines)


def build_pair_messages(pair, contexts, receiver: Receiver,
                        window=WINDOW) -> list[dict]:
    first_key = pair.get("first", "a")
    second_key = "b" if first_key == "a" else "a"
    ref1 = pair.get(first_key, {})
    ref2 = pair.get(second_key, {})
    system = (
        "You are this listener reacting in conversations:\n"
        f"- relationship: {receiver.relationship}\n"
        f"- power: {receiver.power}\n"
        f"- channel: {receiver.channel}\n"
        f"- state: {receiver.state}\n"
        f"- stance: {receiver.stance}\n\n"
        "You will see two conversation snippets. Each snippet ends with a message "
        "to judge (marked MESSAGE TO JUDGE) and states the listener role. "
        "You do not know what the speakers intended. "
        "React the way this listener would, then reply with JSON only "
        "(no other text) with keys better, why.\n"
        "- better must be one of: first, second, tie — after which message is the "
        "listener more favourably disposed toward the speaker? (tie = about the same)\n"
        "- why must be one sentence saying what drove your choice"
    )
    user = (
        f"Snippet 1 (listener: {ref1.get('listener', '')}):\n"
        f"{snippet_for(ref1, contexts, window)}\n\n"
        f"Snippet 2 (listener: {ref2.get('listener', '')}):\n"
        f"{snippet_for(ref2, contexts, window)}\n\n"
        "After which message is the listener more favourably disposed toward the "
        "speaker? Reply with JSON only: "
        '{"better": "first" | "second" | "tie", "why": "<one sentence>"}.'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_pairwise(raw) -> dict | None:
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
    better = str(obj.get("better", "")).strip().lower()
    if better not in VOTES:
        return None
    why = obj.get("why", "")
    why = "" if why is None else str(why)
    return {"better": better, "why": why}


def derandomize(vote, first):
    """Map a first/second/tie vote back to turn-a / turn-b / tie."""
    if vote is None:
        return None
    if vote == "tie":
        return "tie"
    other = "b" if first == "a" else "a"
    if vote == "first":
        return first
    if vote == "second":
        return other
    return None


def _judge_one(client, pair, contexts, receiver, model, window):
    messages = build_pair_messages(pair, contexts, receiver, window)
    meta = {"layer": "pairwise", "pair": pair.get("id"),
            "stance": receiver.stance, "member": receiver.id,
            "dialogues": [pair.get("a", {}).get("dialogue"),
                          pair.get("b", {}).get("dialogue")]}
    attempts = 0
    for _ in range(2):
        try:
            raw = board._call_board(client, messages, model, meta)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        attempts += 1
        parsed = parse_pairwise(raw)
        if parsed is not None:
            return parsed, attempts
    return None, attempts


def judge_all(client, pairs, contexts, model=DEFAULT_MODEL, workers=10,
              meter=None, window=WINDOW):
    """One call per (pair, stance); returns (raw_votes, stats).

    raw_votes[pair_id][stance] is "first"/"second"/"tie"/None. Stops cleanly
    when the meter is exhausted; unjudged votes stay None.
    """
    by_id = {p["id"]: p for p in (pairs or []) if isinstance(p, dict)}
    receivers = {pid: pairwise_panel(pid) for pid in by_id}
    raw = {pid: {s: None for s in STANCES} for pid in by_id}
    stats = {"n_calls": 0, "n_attempted_votes": 0, "budget_stop": False}
    try:
        max_workers = int(workers)
    except (TypeError, ValueError):
        max_workers = 10
    if max_workers < 1:
        max_workers = 1
    queue: deque[tuple] = deque(
        (pid, stance) for pid in sorted(by_id) for stance in STANCES)
    in_flight: dict[cf.Future, tuple] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        while queue or in_flight:
            while queue and len(in_flight) < max_workers \
                    and not stats["budget_stop"]:
                pid, stance = queue[0]
                if meter is not None:
                    try:
                        meter.check()
                    except BudgetExceeded:
                        stats["budget_stop"] = True
                        break
                queue.popleft()
                rec = receivers[pid][STANCES.index(stance)]
                fut = pool.submit(_judge_one, client, by_id[pid], contexts,
                                  rec, model, window)
                in_flight[fut] = (pid, stance)
            if not in_flight:
                break
            done, _ = cf.wait(list(in_flight),
                              return_when=cf.FIRST_COMPLETED)
            for fut in done:
                pid, stance = in_flight.pop(fut)
                try:
                    parsed, n_att = fut.result()
                except BudgetExceeded:
                    stats["budget_stop"] = True
                    continue
                except Exception:  # noqa: BLE001 - keep going
                    continue
                stats["n_attempted_votes"] += 1
                stats["n_calls"] += n_att
                raw[pid][stance] = parsed["better"] if parsed else None
    return raw, stats


def write_run_outputs(out_dir, pairs, raw_votes, stats, settings,
                      spend, cost, run_id="", commit="unknown") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [p for p in (pairs or []) if isinstance(p, dict)]
    n_first = n_second = n_tie = n_null = 0
    lines = []
    for p in sorted(pairs, key=lambda q: q.get("id", 0)):
        pid = p.get("id")
        got = raw_votes.get(pid, {}) if isinstance(raw_votes, dict) else {}
        votes: dict[str, str | None] = {}
        raw_out: dict[str, str | None] = {}
        for stance in STANCES:
            v = got.get(stance)
            v = v if v in VOTES else None
            raw_out[stance] = v
            votes[stance] = derandomize(v, p.get("first", "a"))
            if v == "first":
                n_first += 1
            elif v == "second":
                n_second += 1
            elif v == "tie":
                n_tie += 1
            else:
                n_null += 1
        lines.append({"id": pid, "a": p.get("a", {}), "b": p.get("b", {}),
                      "first": p.get("first", "a"),
                      "votes": votes, "raw": raw_out})
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    attempted = int(stats.get("n_attempted_votes", 0) or 0)
    denom = n_first + n_second
    run = {
        "run_id": run_id,
        "settings": dict(settings or {}),
        "commit": commit,
        "n_pairs": len(lines),
        "n_expected": len(lines) * len(STANCES),
        "n_calls": int(stats.get("n_calls", 0) or 0),
        "n_attempted_votes": attempted,
        "n_votes": n_first + n_second + n_tie,
        "n_null": n_null,
        "n_first": n_first,
        "n_second": n_second,
        "n_tie": n_tie,
        "parse_fail_rate": (n_null / attempted) if attempted else 0.0,
        "position_bias_share": (n_first / denom) if denom else 0.0,
        "spend": spend if isinstance(spend, dict) else {},
        "cost": float(cost),
        "budget_stop": bool(stats.get("budget_stop", False)),
    }
    with open(out / "run.json", "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=2)
    return run


def load_pairs(pairs_dir) -> list[dict]:
    try:
        lines = (Path(pairs_dir) / "pairs.jsonl").read_text(
            encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            out.append(obj)
    out.sort(key=lambda p: p.get("id", 0)
             if isinstance(p.get("id"), int) else 0)
    return out


def pair_votes(pairs) -> list[tuple]:
    """Non-null votes as (key_a, key_b, outcome) with outcome in a/b/tie."""
    rows = []
    for p in pairs or []:
        if not isinstance(p, dict):
            continue
        a = p.get("a", {}) if isinstance(p.get("a"), dict) else {}
        b = p.get("b", {}) if isinstance(p.get("b"), dict) else {}
        ka = (a.get("dialogue"), a.get("turn"))
        kb = (b.get("dialogue"), b.get("turn"))
        votes = p.get("votes", {})
        vals = votes.values() if isinstance(votes, dict) else (
            votes if isinstance(votes, list) else [])
        for v in vals:
            if v in ("a", "b", "tie"):
                rows.append((ka, kb, v))
    return rows


def fit_bradley_terry(rows, max_iter=200, tol=1e-6):
    """MM fit (Hunter 2004); ties = half win each side.

    Uniform prior of 0.5 pseudo-win vs a virtual reference turn (strength 1)
    keeps the comparison graph connected. Returns (ratings, n_votes) keyed by
    (dialogue, turn); ratings are centred to mean 0 on the log scale.
    """
    try:
        iters = int(max_iter)
    except (TypeError, ValueError):
        iters = 200
    if iters < 1:
        iters = 1
    try:
        tol_f = float(tol)
    except (TypeError, ValueError):
        tol_f = 1e-6
    wins: dict = {}
    games: dict = {}
    counts: dict = {}
    players: set = set()
    for ka, kb, outcome in rows or []:
        if ka == kb:
            continue
        if outcome == "a":
            wins[ka] = wins.get(ka, 0.0) + 1.0
        elif outcome == "b":
            wins[kb] = wins.get(kb, 0.0) + 1.0
        elif outcome == "tie":
            wins[ka] = wins.get(ka, 0.0) + 0.5
            wins[kb] = wins.get(kb, 0.0) + 0.5
        else:
            continue
        players.add(ka)
        players.add(kb)
        counts[ka] = counts.get(ka, 0) + 1
        counts[kb] = counts.get(kb, 0) + 1
        gk = tuple(sorted([ka, kb]))
        games[gk] = games.get(gk, 0) + 1
    opp: dict = {p: {} for p in players}
    for (x, y), n in games.items():
        opp[x][y] = opp[x].get(y, 0) + n
        opp[y][x] = opp[y].get(x, 0) + n
    pi = {p: 1.0 for p in players}
    for _ in range(iters):
        new = {}
        for p in players:
            denom = sum(n / (pi[p] + pi[q]) for q, n in opp[p].items())
            denom += 1.0 / (pi[p] + 1.0)
            new[p] = (wins.get(p, 0.0) + 0.5) / denom
        if players:
            delta = max(abs(math.log(new[p]) - math.log(pi[p]))
                        for p in players)
        else:
            delta = 0.0
        pi = new
        if delta < tol_f:
            break
    if not players:
        return {}, {}
    mean = sum(math.log(v) for v in pi.values()) / len(pi)
    ratings = {p: math.log(v) - mean for p, v in pi.items()}
    return ratings, counts


def split_half_reliability(pairs, half_seed=1) -> dict:
    """Spearman between BT ratings fitted on two random halves of the pairs."""
    idx = list(range(len(pairs or [])))
    random.Random(half_seed).shuffle(idx)
    half = len(idx) // 2
    rows1 = pair_votes([pairs[i] for i in idx[:half]])
    rows2 = pair_votes([pairs[i] for i in idx[half:]])
    r1, _ = fit_bradley_terry(rows1)
    r2, _ = fit_bradley_terry(rows2)
    common = sorted(set(r1) & set(r2))
    xs = [r1[k] for k in common]
    ys = [r2[k] for k in common]
    rho = float(spearman(xs, ys)) if len(common) >= 2 else 0.0
    if math.isnan(rho) or math.isinf(rho):
        rho = 0.0
    return {"rho": float(rho), "n_turns": int(len(common)),
            "n_votes_half1": int(len(rows1)),
            "n_votes_half2": int(len(rows2)),
            "half_seed": half_seed}


def speakers_of(pairs) -> dict:
    out: dict = {}
    for p in pairs or []:
        if not isinstance(p, dict):
            continue
        for side in ("a", "b"):
            ref = p.get(side, {})
            if not isinstance(ref, dict):
                continue
            key = (ref.get("dialogue"), ref.get("turn"))
            if key not in out:
                out[key] = ref.get("speaker", "")
    return out


def reception_pw(ratings, speakers) -> dict:
    """Mean rating of each side's turns, keyed by (dialogue, side)."""
    acc: dict = {}
    for key, r in (ratings or {}).items():
        try:
            did, _ = key
        except (TypeError, ValueError):
            continue
        side = (speakers or {}).get(key, "")
        acc.setdefault((did, side), []).append(float(r))
    return {k: (sum(v) / len(v), len(v)) for k, v in acc.items()}


def emit_records(src_records, ratings, counts, out_dir) -> list[str]:
    """Copy each source record with effect values replaced by BT ratings.

    Effects with no rating get value 0.0 (the centred prior mean) and
    n_readings 0; top-level scale is "pairwise".
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if isinstance(src_records, dict):
        items = list(src_records.items())
    else:
        items = [(r.get("id") if isinstance(r, dict) else None, r)
                 for r in (src_records or [])]
    written = []
    for item_id, record in items:
        if not isinstance(record, dict):
            continue
        did = item_id if isinstance(item_id, str) and item_id \
            else record.get("id")
        rec = copy.deepcopy(record)
        effects = [e for e in (rec.get("effects", []) or [])
                   if isinstance(e, dict)]
        for pos, e in enumerate(effects):
            idx = e.get("index", pos)
            idx = idx if isinstance(idx, int) else pos
            key = (did, idx)
            e["value"] = float(ratings[key]) if key in (ratings or {}) else 0.0
            e["n_readings"] = int(counts.get(key, 0)) if isinstance(
                counts, dict) else 0
        rec["effects"] = effects
        rec["board"] = effects
        rec["scale"] = "pairwise"
        if isinstance(did, str) and did:
            with open(out / f"{did}.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, ensure_ascii=False, indent=2)
            written.append(did)
    return sorted(written)


def write_fit_outputs(out_dir, pairs, ratings, counts, speakers,
                      reliability) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rated = sorted(ratings or {}, key=lambda k: (str(k[0]), k[1]
                                          if isinstance(k[1], int) else 0))
    ratings_payload = {
        "ratings": [{"dialogue": k[0], "turn": k[1],
                     "speaker": (speakers or {}).get(k, ""),
                     "rating": float(ratings[k]),
                     "n_votes": int(counts.get(k, 0))}
                    for k in rated],
        "reception_pw": [{"dialogue": did, "side": side,
                          "reception_pw": float(mean), "n_turns": int(n)}
                         for (did, side), (mean, n)
                         in sorted(reception_pw(ratings, speakers).items(),
                                   key=lambda kv: (str(kv[0][0]),
                                                   str(kv[0][1])))],
        "n_votes": int(sum(counts.values())) if counts else 0,
        "n_turns_rated": int(len(rated)),
    }
    with open(out / "ratings.json", "w", encoding="utf-8") as f:
        json.dump(ratings_payload, f, ensure_ascii=False, indent=2)
    with open(out / "reliability.json", "w", encoding="utf-8") as f:
        json.dump(dict(reliability or {}), f, ensure_ascii=False, indent=2)
    return ratings_payload


def fine100_split_half(records) -> dict:
    """Judge-split reliability for absolute readings (same module, for P7-A).

    Per turn: mean of {neutral, trusting} readings vs mean of {suspicious}
    readings, and {neutral, suspicious} vs {trusting}; Spearman over turns
    for each split plus their mean.
    """
    if isinstance(records, dict):
        items = list(records.items())
    else:
        items = [(r.get("id") if isinstance(r, dict) else None, r)
                 for r in (records or [])]
    s1x, s1y, s2x, s2y = [], [], [], []
    for _, record in items:
        if not isinstance(record, dict):
            continue
        for e in (record.get("effects", []) or []):
            if not isinstance(e, dict):
                continue
            groups: dict[str, list[float]] = {}
            for r in (e.get("readings", []) or []):
                if not isinstance(r, dict):
                    continue
                rn, fn = SC.reading_norm(r)
                groups.setdefault(r.get("stance"), []).append(
                    float(rn + fn))
            g_nt = groups.get("neutral", []) + groups.get("trusting", [])
            g_s = groups.get("suspicious", [])
            if g_nt and g_s:
                s1x.append(sum(g_nt) / len(g_nt))
                s1y.append(sum(g_s) / len(g_s))
            g_ns = groups.get("neutral", []) + groups.get("suspicious", [])
            g_t = groups.get("trusting", [])
            if g_ns and g_t:
                s2x.append(sum(g_ns) / len(g_ns))
                s2y.append(sum(g_t) / len(g_t))
    rho1 = float(spearman(s1x, s1y)) if len(s1x) >= 2 else 0.0
    rho2 = float(spearman(s2x, s2y)) if len(s2x) >= 2 else 0.0
    if math.isnan(rho1) or math.isinf(rho1):
        rho1 = 0.0
    if math.isnan(rho2) or math.isinf(rho2):
        rho2 = 0.0
    return {"nt_vs_s": {"rho": float(rho1), "n_turns": int(len(s1x))},
            "ns_vs_t": {"rho": float(rho2), "n_turns": int(len(s2x))},
            "mean_rho": float((rho1 + rho2) / 2)}


# task35: same judge-split reliability, any scale (reads rel_norm).
reliability_split = fine100_split_half


def _normalize_effort(effort):
    if isinstance(effort, str) and effort.lower() in ("none", "null", ""):
        return None
    return effort


def main(argv=None) -> None:
    from datetime import datetime, timezone

    ap = argparse.ArgumentParser(prog="uptake-pairwise")
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("run")
    rp.add_argument("--src", required=True)
    rp.add_argument("--out", required=True)
    rp.add_argument("--k", type=int, default=4)
    rp.add_argument("--seed", type=int, default=0)
    rp.add_argument("--budget", type=float, default=1.5)
    rp.add_argument("--workers", type=int, default=10)
    rp.add_argument("--effort", default="low")
    rp.add_argument("--model", default=DEFAULT_MODEL)
    fp = sub.add_parser("fit")
    fp.add_argument("--pairs", required=True)
    fp.add_argument("--src", required=True)
    fp.add_argument("--out", required=True)
    fp.add_argument("--half-seed", type=int, default=1)
    op = sub.add_parser("reliability-split",
                        aliases=["reliability-fine100"])
    op.add_argument("--src", required=True)
    op.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        src = Path(args.src)
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        records = SC.load_dir(src)
        pool, contexts = collect_pool(records)
        pairs = make_pairs(pool, k=args.k, seed=args.seed)
        meter = Meter(budget_usd=args.budget)
        inner = MuseClient(meter=meter,
                           default_effort=_normalize_effort(args.effort))
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        client = LoggingClient(inner, CallLog(out / "calls.jsonl"), run_id)
        raw, stats = judge_all(client, pairs, contexts, model=args.model,
                               workers=args.workers, meter=meter)
        settings = {"src": args.src, "out": args.out, "k": args.k,
                    "seed": args.seed, "budget": args.budget,
                    "workers": args.workers, "effort": args.effort,
                    "model": args.model, "n_dialogues": len(contexts),
                    "n_turns": len(pool)}
        run = write_run_outputs(out, pairs, raw, stats, settings,
                                meter.by_model(), meter.cost(), run_id,
                                batch._git_commit())
        print(f"done pairs {run['n_pairs']} votes {run['n_votes']}/"
              f"{run['n_expected']} parse_fail={run['parse_fail_rate']:.3f} "
              f"pos_bias={run['position_bias_share']:.3f} "
              f"budget_stop={run['budget_stop']} "
              f"spend: ${run['cost']:.4f}")
        return
    if args.cmd == "fit":
        pairs = load_pairs(args.pairs)
        records = SC.load_dir(args.src)
        rows = pair_votes(pairs)
        ratings, counts = fit_bradley_terry(rows)
        speakers = speakers_of(pairs)
        reliability = split_half_reliability(pairs, half_seed=args.half_seed)
        out = Path(args.out)
        payload = write_fit_outputs(out, pairs, ratings, counts, speakers,
                                    reliability)
        written = emit_records(records, ratings, counts, out / "records")
        print(f"done turns {payload['n_turns_rated']} "
              f"votes {payload['n_votes']} "
              f"split_half_rho={reliability['rho']:+.3f} "
              f"n={reliability['n_turns']} records {len(written)}")
        return
    records = SC.load_dir(args.src)
    result = fine100_split_half(records)
    out_p = Path(args.out)
    if str(out_p.parent):
        out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"done nt_vs_s={result['nt_vs_s']['rho']:+.3f} "
          f"n={result['nt_vs_s']['n_turns']} "
          f"ns_vs_t={result['ns_vs_t']['rho']:+.3f} "
          f"n={result['ns_vs_t']['n_turns']} "
          f"mean={result['mean_rho']:+.3f}")
