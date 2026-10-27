"""Morning analysis for the pre-registered overnight run.

Pure helpers (verdict, percentile, confusion, variance_split) plus the
report builder behind `uptake-analyze --out DIR`, which writes
DIR/report.md following specs/EVAL_PLAN_overnight_1.md.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

HELDOUT_F1 = 0.624
HELDOUT_F1_LO = 0.569
HELDOUT_F1_HI = 0.677


def verdict(value, ok, direction) -> str:
    """OK when the value meets the threshold in the given direction."""
    if direction == ">=":
        return "OK" if value >= ok else "PROBLEM"
    if direction == "<=":
        return "OK" if value <= ok else "PROBLEM"
    raise ValueError(f"bad direction: {direction!r}")


def percentile(values, q: float) -> float:
    """Linear-interpolation percentile; q in [0, 100]."""
    vals = sorted(values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return float(vals[0])
    q = min(100.0, max(0.0, float(q)))
    rank = q / 100.0 * (len(vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(vals[lo])
    frac = rank - lo
    return float(vals[lo] * (1 - frac) + vals[hi] * frac)


def confusion(pred_labels, gold_labels) -> dict:
    """Count (gold, pred) pairs over single labels."""
    out: dict = {}
    for pred, gold in zip(pred_labels, gold_labels):
        key = (gold, pred)
        out[key] = out.get(key, 0) + 1
    return out


def _pvar(xs: list[float]) -> float:
    if not xs:
        return 0.0
    mean = sum(xs) / len(xs)
    return sum((x - mean) ** 2 for x in xs) / len(xs)


def variance_split(readings) -> dict:
    """Split v = relationship_shift - face_threat variance per (item, turn).

    Between-members = variance of per-member means; within-member = mean
    of per-member variances (0 when one sample). Returns overall means of
    both across groups plus their ratio.
    """
    groups: dict = {}
    for r in readings or []:
        try:
            v = float(r["relationship_shift"]) - float(r["face_threat"])
        except (KeyError, TypeError, ValueError):
            continue
        key = (r.get("item"), r.get("turn"))
        groups.setdefault(key, {}).setdefault(r.get("member"), []).append(v)
    betweens: list[float] = []
    withins: list[float] = []
    for members in groups.values():
        per_member = [vals for vals in members.values() if vals]
        if not per_member:
            continue
        means = [sum(vals) / len(vals) for vals in per_member]
        betweens.append(_pvar(means))
        withins.append(sum(_pvar(vals) for vals in per_member) / len(means))
    between = sum(betweens) / len(betweens) if betweens else 0.0
    within = sum(withins) / len(withins) if withins else 0.0
    if within > 0:
        ratio = between / within
    elif between > 0:
        ratio = float("inf")
    else:
        ratio = 0.0
    return {"between_members": between, "within_member": within,
            "ratio": ratio}


def _load_json(path: Path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def _load_calls(path: Path) -> list[dict]:
    recs: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict):
                    recs.append(rec)
    except OSError:
        pass
    return recs


def _item_records(out: Path) -> dict:
    items: dict = {}
    for path in sorted(out.glob("*.json")):
        if path.name in ("report.json", "run.json"):
            continue
        data = _load_json(path, None)
        if isinstance(data, dict) and "id" in data:
            items[data["id"]] = data
    return items


def _first_label(moves) -> str:
    if moves:
        return moves[0]
    return "other"


def _fmt_ci(ci) -> str:
    try:
        return f"[{ci[0]:.3f}-{ci[1]:.3f}]"
    except (TypeError, IndexError):
        return "[?]"


class Report:
    def __init__(self, out: Path, seed: int = 0):
        self.out = out
        self.rng = random.Random(seed)
        self.seed = seed
        self.batch = _load_json(out / "report.json", {})
        self.run = _load_json(out / "run.json", {})
        self.calls = _load_calls(out / "calls.jsonl")
        self.items = _item_records(out)
        self.genres = _load_json(out / "report.json", {}).get("genre_report", {})
        if not self.genres:
            self.genres = {k: v for k, v in self.batch.items()
                           if isinstance(v, dict) and "n_conv" in v}
        self.problems: list[str] = []
        self.lines: list[str] = []
        self.table: list[tuple] = []

    # -- Q1 ------------------------------------------------------------
    def q1(self):
        entry = self.genres.get("negotiation", {})
        micro = entry.get("micro_f1")
        if not micro:
            self.table.append(("Q1", "move F1 on new negotiation data",
                               "n/a", "CI overlaps held-out "
                               f"{HELDOUT_F1:.3f} [{HELDOUT_F1_LO:.3f}-"
                               f"{HELDOUT_F1_HI:.3f}]", "PROBLEM"))
            self.problems.append("Q1: no negotiation micro-F1 in report.")
            return
        lo, hi = micro["micro_f1_ci"]
        value = f"{micro['micro_f1']:.3f} [{lo:.3f}-{hi:.3f}]"
        ok = lo <= HELDOUT_F1_HI and hi >= HELDOUT_F1_LO
        self.table.append(("Q1", "move F1 on new negotiation data", value,
                           f"CI overlaps held-out {HELDOUT_F1:.3f} "
                           f"[{HELDOUT_F1_LO:.3f}-{HELDOUT_F1_HI:.3f}]",
                           "OK" if ok else "PROBLEM"))
        if not ok:
            self.problems.append("Q1")
            self.lines.append("## Drill-down: Q1 (move confusion + wrong turns)")
            self._q1_drilldown()
            self.lines.append("")

    def _q1_drilldown(self):
        preds, golds, wrong = [], [], []
        raws: dict = {}
        for rec in self.calls:
            meta = rec.get("meta") or {}
            if meta.get("layer") == "tag":
                raws[(meta.get("item"), meta.get("turn"))] = rec.get("raw", "")
        for cid, data in sorted(self.items.items()):
            if data.get("genre") != "negotiation":
                continue
            moves = data.get("moves", []) or []
            gold = data.get("gold") or []
            effects = {e.get("index"): e for e in data.get("effects", []) or []}
            for pos, m in enumerate(moves):
                pred = _first_label(m.get("moves", []))
                g = _first_label(gold[pos]) if pos < len(gold) else "other"
                preds.append(pred)
                golds.append(g)
                if pred != g:
                    eff = effects.get(m.get("turn_index", pos), {})
                    wrong.append({"item": cid, "turn": pos,
                                  "speaker": m.get("speaker", ""),
                                  "pred": pred, "gold": g,
                                  "text": eff.get("text", ""),
                                  "raw": raws.get((cid, pos), "")})
        conf = confusion(preds, golds)
        self.lines.append("per-label confusion (gold -> pred: count):")
        for (g, p) in sorted(conf):
            self.lines.append(f"- {g} -> {p}: {conf[(g, p)]}")
        picks = self.rng.sample(wrong, min(20, len(wrong))) if wrong else []
        self.lines.append(f"20 random wrong turns (n={len(wrong)}):")
        for w in picks:
            self.lines.append(f"- {w['item']} turn {w['turn']} "
                              f"[{w['speaker']}] pred={w['pred']} "
                              f"gold={w['gold']} text={w['text']!r}")
            if w["raw"]:
                self.lines.append(f"  raw: {w['raw'][:1000]}")

    # -- Q2 ------------------------------------------------------------
    def q2(self):
        for genre in sorted(self.genres):
            entry = self.genres[genre]
            other = entry["other_rate"][0]
            failed = entry["failed_rate"]
            v = "OK" if (other <= 0.20 and failed <= 0.05) else "PROBLEM"
            self.table.append((f"Q2/{genre}", "move coverage",
                               f"other={other:.3f} failed={failed:.3f}",
                               "other<=0.20 failed<=0.05", v))
            if v == "PROBLEM":
                self.problems.append(f"Q2/{genre}")
                self.lines.append(f"## Drill-down: Q2/{genre} ('other' turns)")
                self._q2_drilldown(genre)
                self.lines.append("")

    def _q2_drilldown(self, genre: str):
        others = []
        for cid, data in sorted(self.items.items()):
            if data.get("genre") != genre:
                continue
            effects = {e.get("index"): e for e in data.get("effects", []) or []}
            for m in data.get("moves", []) or []:
                if list(m.get("moves", [])) == ["other"]:
                    pos = m.get("turn_index", 0)
                    eff = effects.get(pos, {})
                    others.append({"item": cid, "turn": pos,
                                   "speaker": m.get("speaker", ""),
                                   "text": eff.get("text", ""),
                                   "quote": m.get("quote", "")})
        picks = self.rng.sample(others, min(20, len(others))) if others else []
        self.lines.append(f"20 'other' turns in {genre} (n={len(others)}); "
                          "read for missing move types:")
        for w in picks:
            self.lines.append(f"- {w['item']} turn {w['turn']} "
                              f"[{w['speaker']}] quote={w['quote']!r} "
                              f"text={w['text']!r}")

    # -- Q3 ------------------------------------------------------------
    def q3(self):
        stab = self.batch.get("stability", {}) or {}
        kappa = stab.get("mean_kappa", 0.0)
        rho = stab.get("mean_spearman", 0.0)
        n = len(stab.get("per_item", stab.get("items", {})) or {})
        vk = verdict(kappa, 0.60, ">=")
        self.table.append(("Q3a", "tag stability (kappa, first move)",
                           f"{kappa:.3f} (n={n})", ">=0.60", vk))
        if vk == "PROBLEM":
            self.problems.append("Q3a")
            self.lines.append("## Drill-down: Q3a (per-label disagreement)")
            self._q3a_drilldown()
            self.lines.append("")
        vr = verdict(rho, 0.70, ">=")
        self.table.append(("Q3b", "position curve stability (Spearman)",
                           f"{rho:.3f} (n={n})", ">=0.70", vr))
        if vr == "PROBLEM":
            self.problems.append("Q3b")
            self.lines.append("## Drill-down: Q3b (variance split)")
            self._q3b_drilldown()
            self.lines.append("")

    def _q3a_drilldown(self):
        rep_dir = self.out / "repeat"
        pairs = []
        if rep_dir.is_dir():
            for path in sorted(rep_dir.glob("*.json")):
                rep = _load_json(path, None)
                orig = self.items.get(path.stem)
                if not isinstance(rep, dict) or orig is None:
                    continue
                rm = {m.get("turn_index"): _first_label(m.get("moves", []))
                      for m in rep.get("moves", []) or []}
                for m in orig.get("moves", []) or []:
                    pos = m.get("turn_index")
                    if pos in rm:
                        pairs.append((_first_label(m.get("moves", [])),
                                      rm[pos]))
        conf = confusion([p[1] for p in pairs], [p[0] for p in pairs])
        self.lines.append("repeat-vs-original first-move disagreement "
                          f"(n={len(pairs)}; gold=original, pred=repeat):")
        for (g, p) in sorted(conf):
            if g != p:
                self.lines.append(f"- {g} -> {p}: {conf[(g, p)]}")
        if not pairs:
            self.lines.append("- no repeat pairs found")

    def _all_readings(self) -> list[dict]:
        out = []
        for cid, data in self.items.items():
            for e in data.get("effects", []) or []:
                for r in e.get("readings", []) or []:
                    row = dict(r)
                    row.setdefault("item", cid)
                    out.append(row)
        rep_dir = self.out / "repeat"
        if rep_dir.is_dir():
            for path in sorted(rep_dir.glob("*.json")):
                rep = _load_json(path, None)
                if not isinstance(rep, dict):
                    continue
                for e in rep.get("effects", []) or []:
                    for r in e.get("readings", []) or []:
                        row = dict(r)
                        row.setdefault("item", rep.get("id", path.stem))
                        out.append(row)
        return out

    def _q3b_drilldown(self):
        readings = self._all_readings()
        overall = variance_split(readings)
        self.lines.append(
            f"variance split overall: between_members="
            f"{overall['between_members']:.3f} within_member="
            f"{overall['within_member']:.3f} ratio={overall['ratio']:.3f}")
        by_turn: dict = {}
        for r in readings:
            by_turn.setdefault(r.get("turn"), []).append(r)
        self.lines.append("per turn:")
        for turn in sorted(by_turn, key=lambda t: (t is None, t)):
            v = variance_split(by_turn[turn])
            self.lines.append(f"- turn {turn}: n={len(by_turn[turn])} "
                              f"between={v['between_members']:.3f} "
                              f"within={v['within_member']:.3f} "
                              f"ratio={v['ratio']:.3f}")
        self.lines.append("follow-up: rerun n=6 panel on 3 conversations "
                          "to split between-member vs between-run variance.")

    # -- Q4 ------------------------------------------------------------
    def q4(self):
        for genre in sorted(self.genres):
            entry = self.genres[genre]
            skill = entry.get("skill")
            if not skill:
                self.table.append((f"Q4/{genre}", "forecast skill",
                                   "n/a", "skill CI lower > 0", "PROBLEM"))
                self.problems.append(f"Q4/{genre}")
                continue
            lo, hi = skill["skill_ci"]
            value = f"{skill['skill_mean']:+.3f} [{lo:+.3f}-{hi:+.3f}]"
            v = "OK" if lo > 0 else "PROBLEM"
            self.table.append((f"Q4/{genre}", "forecast skill vs climatology",
                               f"{value} (n={skill.get('n', 0)})",
                               "skill CI lower > 0", v))
            if v == "PROBLEM":
                self.problems.append(f"Q4/{genre}")
                self.lines.append(f"## Drill-down: Q4/{genre} (hits)")
                self._q4_drilldown(genre)
                self.lines.append("")

    def _q4_drilldown(self, genre: str):
        hits = []
        for cid, data in sorted(self.items.items()):
            if data.get("genre") != genre:
                continue
            for h in data.get("hits", []) or []:
                hits.append(h)
        conf = confusion([h.get("predicted") for h in hits],
                         [h.get("actual_class") for h in hits])
        self.lines.append("predicted-vs-actual class confusion "
                          "(gold=actual, pred=predicted):")
        for (g, p) in sorted(conf, key=str):
            self.lines.append(f"- {g} -> {p}: {conf[(g, p)]}")
        n = len(hits)
        share = (sum(1 for h in hits if h.get("predicted") == "proceed") / n
                 if n else 0.0)
        self.lines.append(f"share of 'proceed' predictions: {share:.3f} "
                          f"(n={n})")
        by_pos: dict = {}
        for h in hits:
            by_pos.setdefault(h.get("index"), []).append(h)
        self.lines.append("hits by turn position:")
        for pos in sorted(by_pos, key=lambda t: (t is None, t)):
            hh = by_pos[pos]
            rate = sum(1 for h in hh if h.get("hit")) / len(hh)
            self.lines.append(f"- turn {pos}: hit_rate={rate:.3f} "
                              f"(n={len(hh)})")

    # -- Q5 ------------------------------------------------------------
    def q5(self):
        n_conv = sum(e.get("n_conv", 0) for e in self.genres.values())
        cost = float(self.batch.get("cost", 0.0) or 0.0)
        per_conv = cost / n_conv if n_conv else 0.0
        errors = self.batch.get("errors", []) or []
        budget_stop = any(e.get("error") == "budget" for e in errors)
        health = self.batch.get("health_stop")
        stopped = budget_stop or (health is not None)
        v = ("OK" if (per_conv <= 0.05 and not stopped) else "PROBLEM")
        extra = ""
        if health is not None:
            extra = " HEALTH_STOP"
        if budget_stop:
            extra += " BUDGET_STOP"
        self.table.append(("Q5", "cost/speed",
                           f"${per_conv:.4f}/conv (n={n_conv}){extra}",
                           "<=$0.05/conv, no budget stop", v))
        if v == "PROBLEM":
            self.problems.append("Q5")
        self.lines.append("## Cost and speed (Q5 detail)")
        self._q5_detail(cost, n_conv)
        self.lines.append("")

    def _q5_detail(self, cost: float, n_conv: int):
        lats = [float(r.get("latency_s", 0.0) or 0.0) for r in self.calls]
        self.lines.append(f"total cost ${cost:.4f} over {n_conv} conversations; "
                          f"{len(self.calls)} traced calls")
        if lats:
            self.lines.append(f"latency p50={percentile(lats, 50):.2f}s "
                              f"p95={percentile(lats, 95):.2f}s")
            span = 0.0
            tss = [r.get("ts") for r in self.calls
                   if isinstance(r.get("ts"), (int, float))]
            if len(tss) >= 2:
                span = max(tss) - min(tss)
            if span > 0:
                self.lines.append(f"rate={len(self.calls) / span * 60:.1f} "
                                  "calls/min")
        retries_total = sum(int(r.get("retries", 0) or 0) for r in self.calls)
        codes: Counter = Counter()
        for r in self.calls:
            for c in r.get("status_codes", []) or []:
                codes[str(c)] += 1
        self.lines.append(f"retries total={retries_total}; by status code: "
                          f"{dict(codes) if codes else '{}'}")
        slowest = sorted(self.calls,
                         key=lambda r: float(r.get("latency_s", 0.0) or 0.0),
                         reverse=True)[:10]
        self.lines.append("slowest calls:")
        for r in slowest:
            meta = r.get("meta") or {}
            self.lines.append(f"- {float(r.get('latency_s', 0.0) or 0.0):.2f}s "
                              f"model={r.get('model')} "
                              f"layer={meta.get('layer')} "
                              f"item={meta.get('item')} turn={meta.get('turn')} "
                              f"effort={r.get('effort')} "
                              f"error={r.get('error')}")
        if not self.calls:
            self.lines.append("- no calls.jsonl records")

    # -- raw traces ----------------------------------------------------
    def raw_traces(self):
        self.lines.append(f"## Raw traces (30 per genre, seed {self.seed})")
        genre_of = {cid: d.get("genre", "") for cid, d in self.items.items()}
        by_genre: dict = {}
        for r in self.calls:
            meta = r.get("meta") or {}
            g = genre_of.get(meta.get("item"), "")
            by_genre.setdefault(g, []).append(r)
        for genre in sorted(by_genre):
            recs = by_genre[genre]
            picks = (self.rng.sample(recs, min(30, len(recs))) if recs
                     else [])
            label = genre or "(no genre)"
            self.lines.append(f"### {label} (showing {len(picks)} "
                              f"of {len(recs)})")
            for r in picks:
                meta = r.get("meta") or {}
                self.lines.append(f"- layer={meta.get('layer')} "
                                  f"item={meta.get('item')} "
                                  f"turn={meta.get('turn')} "
                                  f"model={r.get('model')} "
                                  f"effort={r.get('effort')} "
                                  f"latency={float(r.get('latency_s', 0.0) or 0.0):.2f}s "
                                  f"error={r.get('error')}")
                raw = r.get("raw", "")
                if raw:
                    self.lines.append(f"  raw: {raw[:1500]}")
        if not self.calls:
            self.lines.append("- no calls.jsonl records")

# -- assembly ------------------------------------------------------
def build_report(out_dir, seed: int = 0) -> str:
    rep = Report(Path(out_dir), seed=seed)
    # Fixed order: table first, then drill-downs, Q5 detail, raw traces.
    rep.q1()
    rep.q2()
    rep.q3()
    rep.q4()
    rep.q5()
    head = ["# Overnight run 1 - morning report", ""]
    run_id = rep.run.get("run_id", rep.batch.get("run_id", "?"))
    head.append(f"run_id: {run_id}  commit: {rep.run.get('commit', '?')}")
    head.append(f"seed for sampling: {rep.seed}")
    head.append("")
    head.append("| # | Question | Value | Threshold | Verdict |")
    head.append("|---|---|---|---|---|")
    for row in rep.table:
        head.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} "
                    f"| {row[4]} |")
    head.append("")
    if rep.problems:
        head.append("PROBLEMs: " + ", ".join(rep.problems))
    else:
        head.append("All verdicts OK.")
    head.append("")
    rep.raw_traces()
    return "\n".join(head + rep.lines) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-analyze")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = Path(args.out)
    text = build_report(out, seed=args.seed)
    with open(out / "report.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out / 'report.md'}")

