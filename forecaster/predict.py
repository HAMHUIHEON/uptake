"""Markov baseline, learned forecast pooling, z-score board (offline)."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from .board import _tag_info, actual_class
from .labels import NEXT_ACTIONS
from . import calibrate as C
from . import stats as S


def with_context(item) -> list[dict]:
    moves = item.get("moves", []) if isinstance(item, dict) else []
    by_index: dict[int, list] = {}
    for t in moves or []:
        idx, _speaker, mv = _tag_info(t)
        if isinstance(idx, int) and idx not in by_index:
            by_index[idx] = list(mv or [])
    item_id = item.get("id", "") if isinstance(item, dict) else ""
    hits = item.get("hits", []) if isinstance(item, dict) else []
    out: list[dict] = []
    for h in hits or []:
        c = dict(h) if isinstance(h, dict) else {}
        if isinstance(h, dict):
            c = dict(h)
        c["item"] = item_id
        idx = h.get("index") if isinstance(h, dict) else None
        if isinstance(idx, int) and idx in by_index:
            c["cur_class"] = actual_class(by_index[idx])
        else:
            c["cur_class"] = "proceed"
        out.append(c)
    return out


def fit_markov(hits, base, beta=2.0) -> dict:
    beta = float(beta)
    pair: Counter = Counter()
    cur_total: Counter = Counter()
    for h in hits or []:
        cur = h.get("cur_class", "proceed") if isinstance(h, dict) else "proceed"
        actual = h.get("actual_class", "proceed") if isinstance(h, dict) else "proceed"
        if cur not in NEXT_ACTIONS:
            cur = "proceed"
        if actual not in NEXT_ACTIONS:
            continue
        pair[(cur, actual)] += 1
        cur_total[cur] += 1
    out: dict = {}
    for cur in NEXT_ACTIONS:
        denom_count = cur_total.get(cur, 0)
        if denom_count == 0:
            out[cur] = {a: float(base.get(a, 0.0)) for a in NEXT_ACTIONS}
            continue
        denom = denom_count + beta
        row = {}
        for nxt in NEXT_ACTIONS:
            row[nxt] = float((pair.get((cur, nxt), 0) + beta * float(base.get(nxt, 0.0))) / denom)
        out[cur] = row
    return out


def markov_prob(hit, markov) -> dict:
    cur = hit.get("cur_class", "proceed") if isinstance(hit, dict) else "proceed"
    if cur not in markov:
        cur = "proceed" if "proceed" in markov else next(iter(markov))
    return markov[cur]


def pool(dists, weights) -> dict:
    k = len(NEXT_ACTIONS)
    if not dists or not weights or len(dists) != len(weights):
        return {a: 1.0 / k for a in NEXT_ACTIONS}
    vals: dict[str, float] = {}
    for a in NEXT_ACTIONS:
        prod = 1.0
        for d, w in zip(dists, weights):
            try:
                p = float(d.get(a, 0.0)) if isinstance(d, dict) else 0.0
            except (TypeError, ValueError):
                p = 0.0
            wf = float(w)
            if wf == 0.0:
                continue
            if p <= 0.0:
                prod = 0.0
                break
            prod *= p ** wf
        vals[a] = float(prod)
    s = sum(vals.values())
    if not math.isfinite(s) or s <= 0:
        return {a: 1.0 / k for a in NEXT_ACTIONS}
    return {a: float(vals[a] / s) for a in NEXT_ACTIONS}


def _mean_log_pool(hits, markov, a, b, alpha=0.5) -> float:
    if not hits:
        return 0.0
    total = 0.0
    for h in hits:
        actual = h.get("actual_class", "proceed") if isinstance(h, dict) else "proceed"
        m = markov_prob(h, markov)
        panel = C.panel_prob(h, alpha=alpha)
        p = float(pool([m, panel], [a, b]).get(actual, 0.0))
        if p <= 0:
            p = 1e-12
        total += math.log(p)
    return total / len(hits)


def fit_pool(hits, markov, grid=None, alpha=0.5) -> tuple[float, float]:
    if grid is None:
        grid = [i * 0.25 for i in range(9)]
    if not hits:
        return (float(grid[0]), float(grid[0]))
    best_a = grid[0]
    best_b = grid[0]
    best = None
    for b in grid:
        for a in grid:
            s = _mean_log_pool(hits, markov, a, b, alpha=alpha)
            if best is None or s > best:
                best = s
                best_a = a
                best_b = b
    return (float(best_a), float(best_b))


def fit_linear(hits, prior_fn, grid=None, alpha=0.5) -> float:
    if grid is None:
        grid = [i / 20 for i in range(21)]
    if not hits:
        return float(grid[0])
    best_lam = grid[0]
    best = None
    for lam in grid:
        lf = float(lam)
        total = 0.0
        for h in hits:
            actual = h.get("actual_class", "proceed") if isinstance(h, dict) else "proceed"
            panel = C.panel_prob(h, alpha=alpha)
            try:
                prior = prior_fn(h)
            except Exception:
                prior = {}
            pp = float(panel.get(actual, 0.0)) if isinstance(panel, dict) else 0.0
            qp = float(prior.get(actual, 0.0)) if isinstance(prior, dict) else 0.0
            p = lf * pp + (1.0 - lf) * qp
            if p <= 0:
                p = 1e-12
            total += math.log(p)
        s = total / len(hits)
        if best is None or s > best:
            best = s
            best_lam = lam
    return float(best_lam)


def compare(hits, models, b=2000, seed=0) -> dict:
    if "base" not in models or "markov" not in models:
        raise ValueError("models must include 'base' and 'markov'")
    n = len(hits) if hits else 0
    if n == 0:
        return {
            name: {
                "log_score": 0.0,
                "skill_vs_base": [0.0, 0.0, 0.0],
                "vs_markov": [0.0, 0.0, 0.0],
                "hit_rate": 0.0,
                "n": 0,
            }
            for name in models
        }
    keys: list[str] = []
    for i, h in enumerate(hits):
        if isinstance(h, dict) and h.get("item") is not None:
            keys.append(str(h.get("item")))
        else:
            keys.append(f"hit-{i}")
    logs: dict[str, list[float]] = {}
    hit_rate: dict[str, float] = {}
    for name, fn in models.items():
        ls: list[float] = []
        n_hit = 0
        for h in hits:
            actual = h.get("actual_class", "proceed") if isinstance(h, dict) else "proceed"
            try:
                dist = fn(h)
            except Exception:
                dist = {}
            if not isinstance(dist, dict):
                dist = {}
            p = float(dist.get(actual, 0.0) or 0.0)
            if p <= 0:
                p = 1e-12
            ls.append(math.log(p))
            pred = sorted(NEXT_ACTIONS, key=lambda a, _d=dist: (-float(_d.get(a, 0.0) or 0.0), a))[0]
            if pred == actual:
                n_hit += 1
        logs[name] = ls
        hit_rate[name] = n_hit / n
    base_logs = logs["base"]
    markov_logs = logs["markov"]
    before: dict[str, list[float]] = {}
    for k, v in zip(keys, markov_logs):
        before.setdefault(k, []).append(v)
    out: dict = {}
    for name in models:
        ls = logs[name]
        mean = sum(ls) / n
        diff_clusters: dict[str, list[float]] = {}
        for k, vm, vb in zip(keys, ls, base_logs):
            diff_clusters.setdefault(k, []).append(vm - vb)
        point, lo, hi = S.cluster_bootstrap_ci(diff_clusters, b=b, seed=seed)
        after: dict[str, list[float]] = {}
        for k, v in zip(keys, ls):
            after.setdefault(k, []).append(v)
        m_point, m_lo, m_hi = S.paired_bootstrap_diff(before, after, b=b, seed=seed)
        out[name] = {
            "log_score": float(mean),
            "skill_vs_base": [float(point), float(lo), float(hi)],
            "vs_markov": [float(m_point), float(m_lo), float(m_hi)],
            "hit_rate": float(hit_rate[name]),
            "n": n,
        }
    return out


def transfer(fit_items, eval_items, genre="negotiation", b=2000, seed=0) -> dict:
    fit_sel = {cid: d for cid, d in (fit_items or {}).items()
               if isinstance(d, dict) and d.get("genre") == genre}
    eval_sel = {cid: d for cid, d in (eval_items or {}).items()
                if isinstance(d, dict) and d.get("genre") == genre}
    fit_hits: list[dict] = []
    for cid in sorted(fit_sel):
        fit_hits.extend(with_context(fit_sel[cid]))
    eval_hits: list[dict] = []
    for cid in sorted(eval_sel):
        eval_hits.extend(with_context(eval_sel[cid]))

    base = C.base_rates(fit_hits)
    markov = fit_markov(fit_hits, base)
    lam_base = C.fit_lambda(fit_hits, base)
    lam_markov = fit_linear(fit_hits, lambda h, _m=markov: markov_prob(h, _m))
    pa, pb = fit_pool(fit_hits, markov)

    models = {
        "raw_panel": (lambda h: C.panel_prob(h)),
        "base": (lambda h, _b=base: _b),
        "markov": (lambda h, _m=markov: markov_prob(h, _m)),
        "panel_base": (lambda h, _b=base, _l=lam_base: C.blend(C.panel_prob(h), _b, _l)),
        "panel_markov": (
            lambda h, _m=markov, _l=lam_markov: C.blend(C.panel_prob(h), markov_prob(h, _m), _l)
        ),
        "pool": (lambda h, _m=markov, _a=pa, _b=pb: pool([markov_prob(h, _m), C.panel_prob(h)], [_a, _b])),
    }
    res = compare(eval_hits, models, b=b, seed=seed)

    def _log_lists(fn) -> dict[str, list[float]]:
        clusters: dict[str, list[float]] = {}
        for i, h in enumerate(eval_hits):
            if isinstance(h, dict) and h.get("item") is not None:
                key = str(h.get("item"))
            else:
                key = f"hit-{i}"
            actual = h.get("actual_class", "proceed") if isinstance(h, dict) else "proceed"
            try:
                dist = fn(h)
            except Exception:
                dist = {}
            if not isinstance(dist, dict):
                dist = {}
            p = float(dist.get(actual, 0.0) or 0.0)
            if p <= 0:
                p = 1e-12
            clusters.setdefault(key, []).append(math.log(p))
        return clusters

    logs = {name: _log_lists(fn) for name, fn in models.items()}

    def _contrast(before_name: str, after_name: str) -> dict:
        pt, lo, hi = S.paired_bootstrap_diff(logs[before_name], logs[after_name], b=b, seed=seed)
        return {"diff": [float(pt), float(lo), float(hi)], "supported": bool(lo > 0)}

    contrasts = {
        "B1_panel_base_vs_raw": _contrast("raw_panel", "panel_base"),
        "B2_markov_vs_base": _contrast("base", "markov"),
        "B3_pool_vs_markov": _contrast("markov", "pool"),
        "B4_panel_base_vs_markov": _contrast("markov", "panel_base"),
    }
    return {
        "models": res,
        "fitted": {"lam_base": float(lam_base), "lam_markov": float(lam_markov),
                   "a": float(pa), "b": float(pb)},
        "n_fit_items": len(fit_sel),
        "n_eval_items": len(eval_sel),
        "n_eval_hits": len(eval_hits),
        "contrasts": contrasts,
    }


def _jsonable(obj):
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-predict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--frac", type=float, default=0.5)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(args.seeds) if args.seeds else [0, 1, 2]
    items = C._load_items(out)

    by_genre: dict[str, list[str]] = {}
    for cid, data in items.items():
        genre = data.get("genre", "")
        by_genre.setdefault(genre, []).append(cid)

    rows: list[dict] = []
    board_rows: list[dict] = []
    per_genre: dict[str, dict[int, dict]] = {}

    for genre in sorted(by_genre):
        ids = sorted(by_genre[genre])
        per_genre[genre] = {}
        for seed in seeds:
            dev_ids, test_ids = C.split_items(ids, frac=args.frac, seed=seed)
            dev_hits: list[dict] = []
            for cid in dev_ids:
                dev_hits.extend(with_context(items[cid]))
            test_hits: list[dict] = []
            for cid in test_ids:
                test_hits.extend(with_context(items[cid]))
            dev_effects = [items[cid].get("effects", []) or [] for cid in dev_ids]
            test_effects = [items[cid].get("effects", []) or [] for cid in test_ids]

            base = C.base_rates(dev_hits)
            markov = fit_markov(dev_hits, base)
            lam_base = C.fit_lambda(dev_hits, base)
            lam_markov = fit_linear(dev_hits, lambda h, _m=markov: markov_prob(h, _m))
            pa, pb = fit_pool(dev_hits, markov)
            norm = C.fit_board_norm(dev_effects, method="z")

            models = {
                "base": (lambda h, _b=base: _b),
                "markov": (lambda h, _m=markov: markov_prob(h, _m)),
                "panel_base": (lambda h, _b=base, _l=lam_base: C.blend(C.panel_prob(h), _b, _l)),
                "panel_markov": (
                    lambda h, _m=markov, _l=lam_markov: C.blend(C.panel_prob(h), markov_prob(h, _m), _l)
                ),
                "pool": (lambda h, _m=markov, _a=pa, _b=pb: pool([markov_prob(h, _m), C.panel_prob(h)], [_a, _b])),
            }
            res = compare(test_hits, models, b=2000, seed=seed)
            per_genre[genre][seed] = res
            fitted = {"lam_base": float(lam_base), "lam_markov": float(lam_markov), "a": float(pa), "b": float(pb)}
            for name in ("base", "markov", "panel_base", "panel_markov", "pool"):
                r = res[name]
                rows.append({
                    "genre": genre,
                    "seed": seed,
                    "model": name,
                    "log_score": float(r["log_score"]),
                    "skill_vs_base": [float(v) for v in r["skill_vs_base"]],
                    "vs_markov": [float(v) for v in r["vs_markov"]],
                    "hit_rate": float(r["hit_rate"]),
                    "n": int(r["n"]),
                    "fitted": dict(fitted),
                    "params": dict(fitted),
                    "lam_base": float(lam_base),
                    "lam_markov": float(lam_markov),
                    "a": float(pa),
                    "b": float(pb),
                })
            try:
                shares = C._annotation_shares(test_effects, norm)
            except Exception:
                shares = {lab: 0.0 for lab in ["??", "?", "?!", "=", "!", "!!"]}
            try:
                mean_before, mean_after = C._mean_end_values(test_effects, norm)
            except Exception:
                mean_before, mean_after = 0.0, 0.0
            board_rows.append({
                "genre": genre,
                "seed": seed,
                "annotation_shares": dict(shares),
                "ann_shares": dict(shares),
                "mean_end_before": float(mean_before),
                "mean_end_after": float(mean_after),
                "norm": _jsonable(norm),
                "n_dev": len(dev_ids),
                "n_test": len(test_ids),
            })

    summary: dict[str, str] = {}
    for genre in sorted(per_genre):
        res_by_seed = per_genre[genre]
        n_seeds = len(seeds)
        x = 0
        y = 0
        for seed in seeds:
            res = res_by_seed.get(seed, {})
            m = res.get("markov", {})
            svb = m.get("skill_vs_base", [0.0, 0.0, 0.0])
            try:
                if float(svb[1]) > 0:
                    x += 1
            except (TypeError, IndexError, ValueError):
                pass
            beaten = False
            for pname in ("panel_base", "panel_markov", "pool"):
                pm = res.get(pname, {})
                vm = pm.get("vs_markov", [0.0, 0.0, 0.0])
                try:
                    if float(vm[1]) > 0:
                        beaten = True
                        break
                except (TypeError, IndexError, ValueError):
                    continue
            if beaten:
                y += 1
        summary[genre] = (
            f"seeds with CI lower > 0: markov vs base {x}/{n_seeds}; "
            f"best panel model vs markov {y}/{n_seeds}"
        )

    payload = {
        "seeds": list(seeds),
        "frac": float(args.frac),
        "rows": _jsonable(rows),
        "board": _jsonable(board_rows),
        "summary": dict(summary),
    }
    with open(out / "predict.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)

    lines: list[str] = []
    lines.append("# Forecast: markov baseline, learned pooling, z-score board")
    lines.append("")
    lines.append(f"seeds={','.join(str(s) for s in seeds)} frac={args.frac}")
    lines.append("")
    lines.append("| genre | seed | model | log_score | skill vs base [CI] | vs markov [CI] | hit_rate | n | fitted |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        try:
            svb = r["skill_vs_base"]
            svb_txt = f"{svb[0]:+.3f} [{svb[1]:+.3f}-{svb[2]:+.3f}]"
        except (KeyError, TypeError, IndexError):
            svb_txt = "n/a"
        try:
            vm = r["vs_markov"]
            vm_txt = f"{vm[0]:+.3f} [{vm[1]:+.3f}-{vm[2]:+.3f}]"
        except (KeyError, TypeError, IndexError):
            vm_txt = "n/a"
        fit = r.get("fitted", {})
        fit_txt = f"lam_base={fit.get('lam_base', 0.0):.2f} lam_markov={fit.get('lam_markov', 0.0):.2f} a={fit.get('a', 0.0):.2f} b={fit.get('b', 0.0):.2f}"
        lines.append(
            f"| {r['genre']} | {r['seed']} | {r['model']} | {r['log_score']:+.3f} |"
            f" {svb_txt} | {vm_txt} | {r['hit_rate']:.3f} | {r['n']} | {fit_txt} |"
        )
    lines.append("")
    lines.append("| genre | seed | end before->after | annotation shares |")
    lines.append("|---|---|---|---|")
    for br in board_rows:
        shares = br.get("annotation_shares", {})
        share_txt = ", ".join(f"{k}={shares.get(k, 0.0):.3f}" for k in ["??", "?", "?!", "=", "!", "!!"])
        lines.append(
            f"| {br['genre']} | {br['seed']} |"
            f" {br['mean_end_before']:+.3f}->{br['mean_end_after']:+.3f} | {share_txt} |"
        )
    lines.append("")
    for genre in sorted(summary):
        lines.append(f"{genre}: {summary[genre]}")
    lines.append("")
    with open(out / "predict.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def transfer_main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-transfer")
    ap.add_argument("--fit", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--genre", default="negotiation")
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    fit_items = C._load_items(Path(args.fit))
    eval_path = Path(args.eval)
    eval_items = C._load_items(eval_path)
    res = transfer(fit_items, eval_items, genre=args.genre, b=args.b, seed=args.seed)

    payload = {"fit": str(args.fit), "eval": str(args.eval), "genre": args.genre,
               "b": int(args.b), "seed": int(args.seed)}
    payload.update(_jsonable(res))
    with open(eval_path / "transfer.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)

    fit = res.get("fitted", {})
    lines: list[str] = []
    lines.append("# Transfer: fit -> held-out eval")
    lines.append("")
    lines.append(f"fit={args.fit} eval={args.eval} genre={args.genre} b={args.b} seed={args.seed}")
    lines.append(
        f"fitted: lam_base={fit.get('lam_base', 0.0):.2f}"
        f" lam_markov={fit.get('lam_markov', 0.0):.2f}"
        f" a={fit.get('a', 0.0):.2f} b={fit.get('b', 0.0):.2f}"
        f" n_fit={res.get('n_fit_items', 0)}"
        f" n_eval={res.get('n_eval_items', 0)}"
        f" n_eval_hits={res.get('n_eval_hits', 0)}"
    )
    lines.append("")
    lines.append("| model | log_score | skill vs base [CI] | vs markov [CI] | hit_rate | n |")
    lines.append("|---|---|---|---|---|---|")
    models = res.get("models", {})
    for name in ("raw_panel", "base", "markov", "panel_base", "panel_markov", "pool"):
        r = models.get(name, {})
        try:
            svb = r["skill_vs_base"]
            svb_txt = f"{svb[0]:+.3f} [{svb[1]:+.3f}-{svb[2]:+.3f}]"
        except (KeyError, TypeError, IndexError):
            svb_txt = "n/a"
        try:
            vm = r["vs_markov"]
            vm_txt = f"{vm[0]:+.3f} [{vm[1]:+.3f}-{vm[2]:+.3f}]"
        except (KeyError, TypeError, IndexError):
            vm_txt = "n/a"
        try:
            ls_txt = f"{float(r['log_score']):+.3f}"
        except (KeyError, TypeError, ValueError):
            ls_txt = "n/a"
        try:
            hr_txt = f"{float(r['hit_rate']):.3f}"
        except (KeyError, TypeError, ValueError):
            hr_txt = "n/a"
        lines.append(
            f"| {name} | {ls_txt} | {svb_txt} | {vm_txt} | {hr_txt} | {r.get('n', 0)} |"
        )
    lines.append("")
    lines.append("| contrast | diff [CI] | verdict |")
    lines.append("|---|---|---|")
    contrasts = res.get("contrasts", {})
    for name in ("B1_panel_base_vs_raw", "B2_markov_vs_base", "B3_pool_vs_markov",
                 "B4_panel_base_vs_markov"):
        c = contrasts.get(name, {})
        diff = c.get("diff", [0.0, 0.0, 0.0])
        try:
            diff_txt = f"{diff[0]:+.3f} [{diff[1]:+.3f}-{diff[2]:+.3f}]"
        except (TypeError, IndexError):
            diff_txt = "n/a"
        verdict = "SUPPORTED" if c.get("supported") else "NOT SUPPORTED"
        lines.append(f"| {name} | {diff_txt} | {verdict} |")
    lines.append("")
    with open(eval_path / "transfer.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
