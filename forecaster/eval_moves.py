"""Move evaluation: PRF, bootstrap CI, baselines, CLIs."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter


def prf(pred: list[set], gold: list[set]) -> dict:
    labels_gold: set[str] = set()
    for g in gold:
        labels_gold |= set(g)
    labels_pred: set[str] = set()
    for p in pred:
        labels_pred |= set(p)
    out: dict = {}
    for lbl in sorted(labels_gold | labels_pred):
        tp = fp = fn = 0
        support = 0
        for p, g in zip(pred, gold):
            in_p = lbl in p
            in_g = lbl in g
            if in_p and in_g:
                tp += 1
            elif in_p and not in_g:
                fp += 1
            elif not in_p and in_g:
                fn += 1
            if in_g:
                support += 1
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        out[lbl] = {"p": prec, "r": rec, "f1": f1, "support": support}
    total_tp = total_fp = total_fn = 0
    for p, g in zip(pred, gold):
        for lbl in set(p) | set(g):
            if lbl in p and lbl in g:
                total_tp += 1
            elif lbl in p and lbl not in g:
                total_fp += 1
            elif lbl not in p and lbl in g:
                total_fn += 1
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
    if labels_gold:
        macro_f1 = sum(out[l]["f1"] for l in labels_gold) / len(labels_gold)
    else:
        macro_f1 = 0.0
    out["micro_f1"] = micro_f1
    out["macro_f1"] = macro_f1
    return out


def _pooled(per_dialogue):
    pred_all: list[set] = []
    gold_all: list[set] = []
    for p, g in per_dialogue:
        pred_all.extend(p)
        gold_all.extend(g)
    return pred_all, gold_all


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(round(q * (len(s) - 1)))
    idx = max(0, min(len(s) - 1, idx))
    return s[idx]


def eval_with_ci(per_dialogue: list[tuple[list[set], list[set]]], b=1000, seed=0) -> dict:
    pred_all, gold_all = _pooled(per_dialogue)
    point = prf(pred_all, gold_all)
    micro = point["micro_f1"]
    macro = point["macro_f1"]
    n = len(per_dialogue)
    if n == 0 or b <= 0:
        return {"micro_f1": micro, "macro_f1": macro,
                "micro_f1_ci": [micro, micro], "macro_f1_ci": [macro, macro]}
    rng = random.Random(seed)
    micro_vals: list[float] = []
    macro_vals: list[float] = []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        rp: list[set] = []
        rg: list[set] = []
        for j in idx:
            rp.extend(per_dialogue[j][0])
            rg.extend(per_dialogue[j][1])
        m = prf(rp, rg)
        micro_vals.append(m["micro_f1"])
        macro_vals.append(m["macro_f1"])
    lo_micro = _percentile(micro_vals, 0.025)
    hi_micro = _percentile(micro_vals, 0.975)
    lo_macro = _percentile(macro_vals, 0.025)
    hi_macro = _percentile(macro_vals, 0.975)
    lo_micro = min(lo_micro, micro)
    hi_micro = max(hi_micro, micro)
    lo_macro = min(lo_macro, macro)
    hi_macro = max(hi_macro, macro)
    return {"micro_f1": micro, "macro_f1": macro,
            "micro_f1_ci": [lo_micro, hi_micro], "macro_f1_ci": [lo_macro, hi_macro]}


def majority_baseline(gold_all) -> set:
    counts: Counter = Counter()
    for g in gold_all:
        for lbl in g:
            counts[lbl] += 1
    if not counts:
        return set()
    top = max(counts.values())
    cands = sorted([lbl for lbl, c in counts.items() if c == top])
    return {cands[0]}


def main(argv=None) -> None:
    from .casino import load_casino
    from .moves import tag_conversation
    from .reader import DEFAULT_MODEL, MuseClient

    ap = argparse.ArgumentParser(prog="uptake-eval-moves")
    ap.add_argument("--casino", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    dialogues = load_casino(args.casino, limit=args.n, seed=args.seed)
    client = MuseClient()
    per_dialogue: list[tuple[list[set], list[set]]] = []
    records: list[dict] = []
    for d in dialogues:
        tags = tag_conversation(client, d["turns"], model=args.model,
                                max_workers=args.workers)
        pred = [set(t.moves) for t in tags]
        gold = d["gold"]
        per_dialogue.append((pred, gold))
        records.append({"dialogue_id": d["dialogue_id"],
                        "pred": [sorted(s) for s in pred],
                        "gold": [sorted(s) for s in gold]})
    pred_all, gold_all = _pooled(per_dialogue)
    table = prf(pred_all, gold_all)
    res = eval_with_ci(per_dialogue, seed=args.seed)
    base = majority_baseline(gold_all)
    base_pred = [set(base) for _ in gold_all]
    base_micro = prf(base_pred, gold_all)["micro_f1"] if gold_all else 0.0
    labels = sorted([k for k in table if k not in ("micro_f1", "macro_f1")])
    print(f"{'label':20s} {'p':>6s} {'r':>6s} {'f1':>6s} {'support':>7s}")
    for lbl in labels:
        row = table[lbl]
        print(f"{lbl:20s} {row['p']:6.3f} {row['r']:6.3f} {row['f1']:6.3f} {row['support']:7d}")
    lo, hi = res["micro_f1_ci"]
    mlo, mhi = res["macro_f1_ci"]
    print(f"micro_f1: {res['micro_f1']:.3f} [{lo:.3f}-{hi:.3f}]")
    print(f"macro_f1: {res['macro_f1']:.3f} [{mlo:.3f}-{mhi:.3f}]")
    print(f"baseline {sorted(base)}: micro_f1={base_micro:.3f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
