"""Forecast calibration (base-rate blend) and centered board norm (offline)."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from .board import _percentile, _value_of, annotate_turn, position_curve
from .labels import NEXT_ACTIONS


def base_rates(hits, alpha=0.5) -> dict:
    n = len(hits) if hits else 0
    k = len(NEXT_ACTIONS)
    counts = Counter()
    for h in hits or []:
        if isinstance(h, dict):
            counts[h.get("actual_class", "proceed")] += 1
    denom = n + k * alpha
    return {a: (counts.get(a, 0) + alpha) / denom for a in NEXT_ACTIONS}


def panel_prob(hit, alpha=0.5) -> dict:
    k = len(NEXT_ACTIONS)
    dist = {}
    nn = 0
    if isinstance(hit, dict):
        dist = hit.get("action_dist") or {}
        try:
            nn = int(hit.get("n_readings", 0) or 0)
        except (TypeError, ValueError):
            nn = 0
    denom = nn + k * alpha
    out = {}
    for a in NEXT_ACTIONS:
        try:
            share = float(dist.get(a, 0.0) or 0.0)
        except (TypeError, ValueError):
            share = 0.0
        out[a] = (share * nn + alpha) / denom
    return out


def blend(panel_p, base, lam) -> dict:
    lam = float(lam)
    out = {}
    for a in NEXT_ACTIONS:
        out[a] = lam * float(panel_p[a]) + (1.0 - lam) * float(base[a])
    return out


def log_score(hits, base, lam, alpha=0.5) -> float:
    if not hits:
        return 0.0
    total = 0.0
    for h in hits:
        actual = h.get("actual_class", "proceed") if isinstance(h, dict) else "proceed"
        panel = panel_prob(h, alpha=alpha)
        b = blend(panel, base, lam)
        p = float(b.get(actual, 0.0))
        if p <= 0:
            p = 1e-12
        total += math.log(p)
    return total / len(hits)


def fit_lambda(hits, base, grid=None, alpha=0.5) -> float:
    if grid is None:
        grid = [i / 20 for i in range(21)]
    best_lam = grid[0]
    best_score = None
    for lam in grid:
        s = log_score(hits, base, lam, alpha=alpha)
        if best_score is None or s > best_score:
            best_score = s
            best_lam = lam
    return float(best_lam)


def blend_skill(hits, base, lam, b=1000, seed=0, alpha=0.5) -> dict:
    n = len(hits) if hits else 0
    if n == 0:
        return {
            "skill_mean": 0.0,
            "skill_ci": (0.0, 0.0),
            "n": 0,
            "lam": float(lam),
            "hit_rate": 0.0,
            "majority_hit_rate": 0.0,
        }
    majority = sorted(base, key=lambda a: (-base[a], a))[0]
    gains: list[float] = []
    n_hit = 0
    n_maj = 0
    for h in hits:
        actual = h.get("actual_class", "proceed") if isinstance(h, dict) else "proceed"
        panel = panel_prob(h, alpha=alpha)
        bl = blend(panel, base, lam)
        p_blend = float(bl.get(actual, 0.0))
        if p_blend <= 0:
            p_blend = 1e-12
        p_base = float(base.get(actual, 0.0))
        if p_base <= 0:
            p_base = 1e-12
        gains.append(math.log(p_blend) - math.log(p_base))
        pred = sorted(bl, key=lambda a: (-bl[a], a))[0]
        if pred == actual:
            n_hit += 1
        if actual == majority:
            n_maj += 1
    skill_mean = sum(gains) / n
    rng = random.Random(seed)
    reps: list[float] = []
    for _ in range(max(1, b)):
        sample = [gains[rng.randrange(n)] for _ in range(n)]
        reps.append(sum(sample) / n)
    reps.sort()
    lo = _percentile(reps, 0.025)
    hi = _percentile(reps, 0.975)
    return {
        "skill_mean": skill_mean,
        "skill_ci": (lo, hi),
        "n": n,
        "lam": float(lam),
        "hit_rate": n_hit / n,
        "majority_hit_rate": n_maj / n,
    }


def split_items(ids, frac=0.5, seed=0):
    ordered = sorted(ids)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    k = round(len(ordered) * frac)
    k = max(0, min(len(ordered), k))
    return (ordered[:k], ordered[k:])


def fit_board_norm(effects_lists, method="quantile") -> dict:
    values: list[float] = []
    for lst in effects_lists or []:
        for e in lst or []:
            try:
                values.append(float(_value_of(e)))
            except (TypeError, ValueError):
                continue
    n = len(values)
    if n == 0:
        return {"mu": 0.0, "sd": 0.0, "n": 0, "thresholds": None}
    mu = sum(values) / n
    var = sum((v - mu) ** 2 for v in values) / n
    sd = math.sqrt(var) if var > 0 else 0.0
    if method == "z":
        if sd == 0:
            return {"mu": float(mu), "sd": float(sd), "n": n, "thresholds": None}
        thresholds = {
            "??": float(-2.0 * sd),
            "?": float(-1.0 * sd),
            "?!": float(-0.5 * sd),
            "!": float(1.0 * sd),
            "!!": float(2.0 * sd),
        }
        return {"mu": float(mu), "sd": float(sd), "n": n, "thresholds": thresholds}
    centered = sorted(v - mu for v in values)
    thresholds = {
        "??": float(_percentile(centered, 0.025)),
        "?": float(_percentile(centered, 0.10)),
        "?!": float(_percentile(centered, 0.25)),
        "!": float(_percentile(centered, 0.90)),
        "!!": float(_percentile(centered, 0.975)),
    }
    return {"mu": float(mu), "sd": float(sd), "n": n, "thresholds": thresholds}


_SKIP_NAMES = {"report.json", "run.json", "calibration.json", "calibration.md", "report.md",
               "transfer.json", "outcome.json", "predict.json"}


def _is_skipped(name: str) -> bool:
    if name in _SKIP_NAMES:
        return True
    low = name.lower()
    if low.startswith("commentary") and low.endswith(".json"):
        return True
    if "manifest" in low:
        return True
    if low.startswith("log") or "log" in low and low.endswith((".json", ".jsonl")) and "catalog" not in low:
        # skip logs / run logs but keep normal item files
        if low.startswith("log"):
            return True
    return False


def _load_items(out: Path) -> dict:
    items: dict = {}
    for p in sorted(out.iterdir()):
        if not p.is_file():
            continue
        if p.suffix != ".json":
            continue
        if _is_skipped(p.name):
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        if "genre" not in data or "hits" not in data or "effects" not in data:
            continue
        cid = data.get("id", p.stem)
        items[cid] = data
    return items


def _annotation_shares(test_effects_lists, norm) -> dict:
    labels = ["??", "?", "?!", "=", "!", "!!"]
    counts = Counter()
    total = 0
    for lst in test_effects_lists:
        for e in lst or []:
            try:
                v = float(_value_of(e))
            except (TypeError, ValueError):
                continue
            mu = float(norm.get("mu", 0.0)) if isinstance(norm, dict) else 0.0
            ann = annotate_turn(v - mu, norm=norm)
            counts[ann] += 1
            total += 1
    if total == 0:
        return {lab: 0.0 for lab in labels}
    return {lab: counts.get(lab, 0) / total for lab in labels}


def _mean_end_values(test_effects_lists, norm):
    befores: list[float] = []
    afters: list[float] = []
    for lst in test_effects_lists:
        if not lst:
            continue
        curve_before = position_curve(lst, norm=None)
        curve_after = position_curve(lst, norm=norm)
        if curve_before:
            befores.append(float(curve_before[-1]))
        if curve_after:
            afters.append(float(curve_after[-1]))
    mean_before = sum(befores) / len(befores) if befores else 0.0
    mean_after = sum(afters) / len(afters) if afters else 0.0
    return mean_before, mean_after


def _jsonable(obj):
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    return obj


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-calibrate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.5)
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items = _load_items(out)

    by_genre: dict[str, list[str]] = {}
    for cid, data in items.items():
        genre = data.get("genre", "")
        by_genre.setdefault(genre, []).append(cid)

    genres: dict = {}
    for genre in sorted(by_genre):
        ids = sorted(by_genre[genre])
        dev_ids, test_ids = split_items(ids, frac=args.frac, seed=args.seed)
        dev_hits: list[dict] = []
        dev_effects: list[list] = []
        for cid in dev_ids:
            data = items[cid]
            dev_hits.extend(data.get("hits", []) or [])
            dev_effects.append(data.get("effects", []) or [])
        test_hits: list[dict] = []
        test_effects: list[list] = []
        for cid in test_ids:
            data = items[cid]
            test_hits.extend(data.get("hits", []) or [])
            test_effects.append(data.get("effects", []) or [])

        base = base_rates(dev_hits)
        lam = fit_lambda(dev_hits, base)
        norm = fit_board_norm(dev_effects)
        skill = blend_skill(test_hits, base, lam, seed=args.seed)
        raw = blend_skill(test_hits, base, 1.0, seed=args.seed)
        shares = _annotation_shares(test_effects, norm)
        mean_before, mean_after = _mean_end_values(test_effects, norm)

        entry = {
            "dev_ids": list(dev_ids),
            "test_ids": list(test_ids),
            "n_dev": len(dev_ids),
            "n_test": len(test_ids),
            "n_dev_items": len(dev_ids),
            "n_test_items": len(test_ids),
            "lam": float(lam),
            "base": dict(base),
            "base_rates": dict(base),
            "norm": _jsonable(norm),
            "skill": _jsonable(skill),
            "test_skill": _jsonable(skill),
            "blend_skill": _jsonable(skill),
            "raw_skill": _jsonable(raw),
            "raw_panel_skill": _jsonable(raw),
            "annotation_shares": dict(shares),
            "ann_shares": dict(shares),
            "mean_end_before": float(mean_before),
            "mean_end_after": float(mean_after),
            "mean_end_value_before": float(mean_before),
            "mean_end_value_after": float(mean_after),
            "mean_final_before": float(mean_before),
            "mean_final_after": float(mean_after),
        }
        genres[genre] = entry

    payload: dict = {"seed": args.seed, "frac": args.frac, "genres": _jsonable(genres)}
    for genre, entry in genres.items():
        payload[genre] = _jsonable(entry)
    with open(out / "calibration.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)

    lines: list[str] = []
    lines.append("# Calibration (base-rate blend + centered board)")
    lines.append("")
    lines.append(f"seed={args.seed} frac={args.frac}")
    lines.append("")
    lines.append(
        "| genre | n_dev | n_test | lam | test skill [CI] | raw skill [CI] |"
        " hit vs majority | mu | end before->after | annotation shares |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for genre in sorted(genres):
        e = genres[genre]
        s = e["skill"]
        r = e["raw_skill"]
        try:
            s_ci = s["skill_ci"]
            s_txt = f"{s['skill_mean']:+.3f} [{s_ci[0]:+.3f}-{s_ci[1]:+.3f}]"
        except (KeyError, TypeError, IndexError):
            s_txt = "n/a"
        try:
            r_ci = r["skill_ci"]
            r_txt = f"{r['skill_mean']:+.3f} [{r_ci[0]:+.3f}-{r_ci[1]:+.3f}]"
        except (KeyError, TypeError, IndexError):
            r_txt = "n/a"
        try:
            hit_txt = f"{s['hit_rate']:.3f} vs {s['majority_hit_rate']:.3f}"
        except (KeyError, TypeError):
            hit_txt = "n/a"
        shares = e.get("annotation_shares", {})
        share_txt = ", ".join(
            f"{k}={shares.get(k, 0.0):.3f}" for k in ["??", "?", "?!", "=", "!", "!!"]
        )
        lines.append(
            f"| {genre} | {e['n_dev']} | {e['n_test']} | {e['lam']:.2f} |"
            f" {s_txt} | {r_txt} | {hit_txt} | {e['norm'].get('mu', 0.0):+.3f} |"
            f" {e['mean_end_before']:+.3f}->{e['mean_end_after']:+.3f} |"
            f" {share_txt} |"
        )
    lines.append("")
    with open(out / "calibration.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
