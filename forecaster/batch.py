"""Overnight batch runner: manifest routing, per-item runs, genre report."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import glob
import json
import math
import random
from collections import Counter, deque
from pathlib import Path

from .board import evaluate_turns, forecast_hits, forecast_skill, position_curve
from .casino import load_casino
from .eval_moves import CASINO_SCHEME, eval_with_ci, project
from .moves import tag_conversation
from .reader import MuseClient
from .transcribe import load_turns
from .usage import BudgetExceeded, Meter

PUBLIC_MODEL = "muse-spark-1.3-contributor"
PRIVATE_MODEL = "muse-spark-1.3"


def model_for(item, public_model=PUBLIC_MODEL,
              private_model=PRIVATE_MODEL) -> str:
    if isinstance(item, dict) and item.get("public"):
        return public_model
    return private_model


def _normalize_effort(effort):
    if isinstance(effort, str) and effort.lower() in ("none", "null", ""):
        return None
    return effort


def _tag_dict(t) -> dict:
    if isinstance(t, dict):
        return {
            "turn_index": t.get("turn_index", 0),
            "speaker": t.get("speaker", ""),
            "moves": list(t.get("moves", []) or []),
            "addressed_to": t.get("addressed_to", ""),
            "quote": t.get("quote", ""),
            "failed": bool(t.get("failed", False)),
        }
    return {
        "turn_index": getattr(t, "turn_index", 0),
        "speaker": getattr(t, "speaker", ""),
        "moves": list(getattr(t, "moves", []) or []),
        "addressed_to": getattr(t, "addressed_to", ""),
        "quote": getattr(t, "quote", ""),
        "failed": bool(getattr(t, "failed", False)),
    }


def run_item(client, item, casino_index, out_dir, n=3, k=1,
             effort="low", inner_workers=8,
             public_model=PUBLIC_MODEL,
             private_model=PRIVATE_MODEL) -> dict:
    model = model_for(item, public_model=public_model,
                      private_model=private_model)
    effort = _normalize_effort(effort)
    kind = item.get("kind", "turns")
    gold = None
    if kind == "casino":
        dialogue = casino_index[item["dialogue_id"]]
        turns = dialogue["turns"]
        gold = [sorted(set(g)) for g in dialogue["gold"]]
    else:
        turns = load_turns(item["path"])
    tags = tag_conversation(client, turns, model=model,
                            max_workers=inner_workers,
                            reasoning_effort=effort)
    effects = evaluate_turns(client, turns, n=n, k=k, model=model,
                             max_workers=inner_workers, move_tags=tags)
    hits = forecast_hits(effects, tags)
    moves = [_tag_dict(t) for t in tags]
    n_turns = len(moves)
    other_rate = (sum(1 for m in moves if m["moves"] == ["other"]) / n_turns
                  if n_turns else 0.0)
    failed_rate = (sum(1 for m in moves if m["failed"]) / n_turns
                   if n_turns else 0.0)
    pred = [list(m["moves"]) for m in moves]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "id": item["id"],
        "genre": item.get("genre", ""),
        "model": model,
        "moves": moves,
        "effects": effects,
        "board": effects,
        "hits": hits,
        "gold": gold,
    }
    with open(out / f"{item['id']}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    return {
        "id": item["id"],
        "genre": item.get("genre", ""),
        "n_turns": n_turns,
        "other_rate": other_rate,
        "failed_rate": failed_rate,
        "hits": hits,
        "pred": pred,
        "gold": gold,
    }


def _summary_from_record(data: dict) -> dict:
    moves = data.get("moves", [])
    n_turns = len(moves)
    other_rate = (sum(1 for m in moves if list(m.get("moves", [])) == ["other"])
                  / n_turns if n_turns else 0.0)
    failed_rate = (sum(1 for m in moves if m.get("failed"))
                   / n_turns if n_turns else 0.0)
    return {
        "id": data["id"],
        "genre": data.get("genre", ""),
        "n_turns": n_turns,
        "other_rate": other_rate,
        "failed_rate": failed_rate,
        "hits": data.get("hits", []),
        "pred": [list(m.get("moves", [])) for m in moves],
        "gold": data.get("gold"),
    }


def _error(item, message: str) -> dict:
    return {"id": item.get("id"), "error": message}


def run_batch(manifest_items, client, out_dir, casino_path=None,
              workers=4, public_model=PUBLIC_MODEL,
              private_model=PRIVATE_MODEL, **item_kwargs) -> list[dict]:
    item_kwargs.setdefault("public_model", public_model)
    item_kwargs.setdefault("private_model", private_model)
    items = list(manifest_items or [])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    casino_index: dict = {}
    if casino_path is not None:
        for d in load_casino(casino_path):
            casino_index[d["dialogue_id"]] = d
    results: dict[int, dict] = {}
    todo: list[int] = []
    for idx, item in enumerate(items):
        path = out / f"{item.get('id')}.json"
        if path.exists():
            with open(path, encoding="utf-8") as f:
                results[idx] = _summary_from_record(json.load(f))
        else:
            todo.append(idx)

    def _run_one(idx: int) -> dict:
        return run_item(client, items[idx], casino_index, out, **item_kwargs)

    max_workers = workers if isinstance(workers, int) and workers >= 1 else 1
    queue: deque[int] = deque(todo)
    in_flight: dict[cf.Future, int] = {}
    budget_hit = False
    with cf.ThreadPoolExecutor(max_workers=max_workers) as pool:
        while queue or in_flight:
            while queue and len(in_flight) < max_workers and not budget_hit:
                idx = queue.popleft()
                in_flight[pool.submit(_run_one, idx)] = idx
            if not in_flight:
                break
            done, _ = cf.wait(list(in_flight), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                idx = in_flight.pop(fut)
                try:
                    results[idx] = fut.result()
                except BudgetExceeded:
                    results[idx] = _error(items[idx], "budget")
                    budget_hit = True
                except Exception as exc:  # noqa: BLE001 - batch continues
                    results[idx] = _error(items[idx], str(exc))
            if budget_hit:
                while queue:
                    idx = queue.popleft()
                    results[idx] = _error(items[idx], "budget")
    return [results[i] for i in range(len(items))]


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


def _mean_ci(vals: list[float], b: int, rng: random.Random):
    if not vals:
        return (0.0, 0.0, 0.0)
    mean = sum(vals) / len(vals)
    if len(vals) == 1 or b <= 0:
        return (mean, mean, mean)
    n = len(vals)
    reps = sorted(
        sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(b))
    lo = min(_percentile(reps, 0.025), mean)
    hi = max(_percentile(reps, 0.975), mean)
    return (mean, lo, hi)


def genre_report(summaries, b=1000, seed=0) -> dict:
    rng = random.Random(seed)
    by_genre: dict[str, list[dict]] = {}
    n_errors = 0
    for s in summaries or []:
        if isinstance(s, dict) and "error" in s:
            n_errors += 1
            continue
        by_genre.setdefault(s.get("genre", ""), []).append(s)
    report: dict = {}
    for genre in sorted(by_genre):
        convos = by_genre[genre]
        other_vals = [c["other_rate"] for c in convos]
        failed_vals = [c["failed_rate"] for c in convos]
        pooled_hits = [h for c in convos for h in c.get("hits", [])]
        n_hits = len(pooled_hits)
        hit_rate = (sum(1 for h in pooled_hits if h.get("hit")) / n_hits
                    if n_hits else 0.0)
        entry: dict = {
            "n_conv": len(convos),
            "other_rate": _mean_ci(other_vals, b, rng),
            "failed_rate": (sum(failed_vals) / len(failed_vals)
                            if failed_vals else 0.0),
            "hit_rate": hit_rate,
            "skill": forecast_skill(pooled_hits, b=b, seed=seed),
        }
        per_dialogue = []
        for c in convos:
            if c.get("gold") is not None:
                per_dialogue.append(
                    (project(c.get("pred", []), CASINO_SCHEME), c["gold"]))
        if per_dialogue:
            entry["micro_f1"] = eval_with_ci(per_dialogue, b=b, seed=seed)
        report[genre] = entry
    report["n_errors"] = n_errors
    return report


def cohen_kappa(a: list[str], b: list[str]) -> float:
    if len(a) != len(b):
        raise ValueError("sequences must have equal length")
    n = len(a)
    if n == 0:
        return 1.0
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca = Counter(a)
    cb = Counter(b)
    pe = sum((ca[l] / n) * (cb.get(l, 0) / n) for l in ca)
    if abs(1.0 - pe) < 1e-12:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def _ranks(vals) -> list[float]:
    n = len(vals)
    order = sorted(range(n), key=lambda i: vals[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for kk in range(i, j + 1):
            ranks[order[kk]] = avg
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    if len(x) != len(y):
        raise ValueError("sequences must have equal length")
    n = len(x)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0 if x[0] == y[0] else 0.0
    rx, ry = _ranks(list(x)), _ranks(list(y))
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (bb - my) for a, bb in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((bb - my) ** 2 for bb in ry)
    denom = math.sqrt(dx * dy)
    if denom == 0:
        return 1.0 if list(x) == list(y) else 0.0
    return num / denom


def _first_moves(record: dict) -> list[str]:
    out = []
    for m in record.get("moves", []):
        labels = m.get("moves", []) or []
        out.append(labels[0] if labels else "other")
    return out


def repeat_check(items, client, out_dir, sample=5, seed=0, **kw) -> dict:
    out = Path(out_dir)
    repeat_dir = out / "repeat"
    items = list(items or [])
    casino_path = kw.pop("casino_path", None)
    casino_index: dict = {}
    if casino_path is not None:
        for d in load_casino(casino_path):
            casino_index[d["dialogue_id"]] = d
    rng = random.Random(seed)
    k = min(sample, len(items))
    chosen = rng.sample(items, k) if k else []
    per_item: dict = {}
    for item in chosen:
        cid = item.get("id")
        try:
            with open(out / f"{cid}.json", encoding="utf-8") as f:
                orig = json.load(f)
        except OSError:
            continue
        try:
            run_item(client, item, casino_index, repeat_dir, **kw)
            with open(repeat_dir / f"{cid}.json", encoding="utf-8") as f:
                rep = json.load(f)
        except Exception:  # noqa: BLE001 - record failure, keep going
            continue
        kappa = cohen_kappa(_first_moves(orig), _first_moves(rep))
        curve_a = position_curve(orig.get("effects", []))
        curve_b = position_curve(rep.get("effects", []))
        if len(curve_a) != len(curve_b) or not curve_a:
            rho = 0.0
        else:
            rho = spearman(curve_a, curve_b)
        per_item[cid] = {"kappa": kappa, "spearman": rho}
    vals_k = [v["kappa"] for v in per_item.values()]
    vals_s = [v["spearman"] for v in per_item.values()]
    mean_kappa = sum(vals_k) / len(vals_k) if vals_k else 0.0
    mean_spearman = sum(vals_s) / len(vals_s) if vals_s else 0.0
    return {
        "per_item": per_item,
        "items": per_item,
        "mean_kappa": mean_kappa,
        "kappa_mean": mean_kappa,
        "mean_spearman": mean_spearman,
        "spearman_mean": mean_spearman,
    }


def _exclude_ids(paths) -> set:
    ids: set = set()
    for group in paths or []:
        files = group if isinstance(group, list) else [group]
        for path in files:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict) and "dialogue_id" in rec:
                        ids.add(rec["dialogue_id"])
    return ids


def manifest_main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-manifest")
    ap.add_argument("--casino", default=None)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--exclude", nargs="*", action="append", default=[])
    ap.add_argument("--turns-glob", default=None)
    ap.add_argument("--turns-genre", default="youtube")
    args = ap.parse_args(argv)

    items: list[dict] = []
    if args.casino:
        dialogues = load_casino(args.casino, limit=args.n, seed=args.seed,
                                exclude_ids=_exclude_ids(args.exclude))
        for d in dialogues:
            items.append({
                "id": f"casino-{d['dialogue_id']}",
                "genre": "negotiation",
                "kind": "casino",
                "dialogue_id": d["dialogue_id"],
                "public": True,
            })
    if args.turns_glob:
        seen: set[str] = set()
        for path in sorted(glob.glob(args.turns_glob)):
            stem = Path(path).stem
            cid, suffix = stem, 1
            while cid in seen:
                suffix += 1
                cid = f"{stem}-{suffix}"
            seen.add(cid)
            items.append({
                "id": cid,
                "genre": args.turns_genre,
                "kind": "turns",
                "path": path,
                "public": False,
            })
    for item in items:
        print(json.dumps(item, ensure_ascii=False))


def _load_manifest(path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="uptake-batch")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--casino", default=None)
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inner-workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--effort", default="low")
    ap.add_argument("--repeat-sample", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--b", type=int, default=1000)
    ap.add_argument("--public-model", default=PUBLIC_MODEL)
    ap.add_argument("--private-model", default=PRIVATE_MODEL)
    return ap


def batch_main(argv=None) -> None:
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    print(f"public_model={args.public_model} private_model={args.private_model}")
    items = _load_manifest(args.manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meter = Meter(budget_usd=args.budget)
    client = MuseClient(meter=meter, default_effort=args.effort)
    item_kwargs = {"n": args.n, "k": args.k, "effort": args.effort,
                   "inner_workers": args.inner_workers,
                   "public_model": args.public_model,
                   "private_model": args.private_model}
    summaries = run_batch(items, client, out, casino_path=args.casino,
                          workers=args.workers, **item_kwargs)
    rep = genre_report(summaries, b=args.b, seed=args.seed)
    stab = repeat_check(items, client, out, sample=args.repeat_sample,
                        seed=args.seed, casino_path=args.casino,
                        **item_kwargs)
    spend = meter.by_model()
    total = meter.cost()
    report = dict(rep)
    report["genre_report"] = rep
    report["genres"] = rep
    report["stability"] = stab
    report["repeat"] = stab
    report["spend"] = spend
    report["usage"] = spend
    report["by_model"] = spend
    report["cost"] = total
    report["public_model"] = args.public_model
    report["private_model"] = args.private_model
    with open(out / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    for genre in sorted(g for g in rep if g != "n_errors"):
        entry = rep[genre]
        omean = entry["other_rate"][0]
        extra = ""
        if "micro_f1" in entry:
            extra = f" micro_f1={entry['micro_f1']['micro_f1']:.3f}"
        print(f"{genre}: n={entry['n_conv']} other={omean:.3f} "
              f"failed={entry['failed_rate']:.3f} hit={entry['hit_rate']:.3f}"
              f"{extra}")
    print(f"errors: {rep['n_errors']}")
    print(f"stability: mean_kappa={stab['mean_kappa']:.3f} "
          f"mean_spearman={stab['mean_spearman']:.3f}")
    print(f"spend: ${total:.4f}")
    for model in sorted(spend):
        row = spend[model]
        print(f"  {model}: {row['calls']} calls ${row['usd']:.4f}")


main = batch_main
