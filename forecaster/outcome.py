"""Outcome validity of the position curve (offline, stdlib only)."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from . import board as B
from . import calibrate as C
from . import stats as S
from .batch import spearman

SATISFACTION = {
    "Extremely dissatisfied": -2,
    "Slightly dissatisfied": -1,
    "Undecided": 0,
    "Slightly satisfied": 1,
    "Extremely satisfied": 2,
}

LIKENESS = {
    "Extremely dislike": -2,
    "Slightly dislike": -1,
    "Undecided": 0,
    "Slightly like": 1,
    "Extremely like": 2,
}

WARM = {"empathize", "praise", "repair", "smalltalk"}
HOSTILE = {"pressure", "challenge", "reject"}

_AGENTS = (("A", "mturk_agent_1"), ("B", "mturk_agent_2"))


def _parse_points(value) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            try:
                return int(float(value.strip()))
            except ValueError:
                return None
    return None


def _map_likert(value, mapping: dict) -> int | None:
    if isinstance(value, str):
        text = value.strip()
        if text in mapping:
            return mapping[text]
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and -2 <= value <= 2:
        return value
    return None


def load_outcomes(casino_path) -> dict[int, dict]:
    with open(Path(casino_path), encoding="utf-8") as f:
        data = json.load(f)
    out: dict[int, dict] = {}
    for d in data or []:
        if not isinstance(d, dict):
            continue
        try:
            did = int(d.get("dialogue_id"))
        except (TypeError, ValueError):
            continue
        pinfo = d.get("participant_info")
        if not isinstance(pinfo, dict):
            continue
        record: dict[str, dict] = {}
        ok = True
        for side, agent in _AGENTS:
            agent_rec = pinfo.get(agent)
            if not isinstance(agent_rec, dict):
                ok = False
                break
            outcomes = agent_rec.get("outcomes")
            if not isinstance(outcomes, dict):
                ok = False
                break
            points = _parse_points(outcomes.get("points_scored"))
            sat = _map_likert(outcomes.get("satisfaction"), SATISFACTION)
            lik = _map_likert(outcomes.get("opponent_likeness"), LIKENESS)
            if points is None or sat is None or lik is None:
                ok = False
                break
            record[side] = {"points": points, "satisfaction": sat, "likeness": lik}
        if ok:
            out[did] = record
    return out


def dialogue_id_of(item_id) -> int | None:
    if not isinstance(item_id, str):
        return None
    if not item_id.startswith("casino-"):
        return None
    rest = item_id[len("casino-"):]
    if not rest:
        return None
    try:
        return int(rest)
    except ValueError:
        return None


def _effects_of(item) -> list:
    if not isinstance(item, dict):
        return []
    effects = item.get("effects", [])
    return [e for e in (effects or []) if isinstance(e, dict)]


def _moves_of(item) -> list:
    if not isinstance(item, dict):
        return []
    moves = item.get("moves", [])
    return [m for m in (moves or []) if isinstance(m, dict)]


def reception(item, mu) -> dict:
    mu = float(mu)
    out: dict[str, float | None] = {}
    for side in ("A", "B"):
        vals = [B._value_of(e) - mu for e in _effects_of(item) if e.get("speaker") == side]
        out[side] = (sum(vals) / len(vals)) if vals else None
    return out


def dyad_value(item, mu) -> float | None:
    mu = float(mu)
    vals = [B._value_of(e) - mu for e in _effects_of(item)]
    return (sum(vals) / len(vals)) if vals else None


def flagged(item, mu, sd, z=2.0) -> dict:
    mu = float(mu)
    try:
        positive = bool(sd > 0)
    except TypeError:
        positive = False
    out: dict[str, bool] = {}
    for side in ("A", "B"):
        if not positive:
            out[side] = False
            continue
        cutoff = -float(z) * float(sd)
        out[side] = any(
            (B._value_of(e) - mu) <= cutoff
            for e in _effects_of(item) if e.get("speaker") == side
        )
    return out


def tag_score(item) -> dict:
    out: dict[str, float | None] = {}
    for side in ("A", "B"):
        entries = [m for m in _moves_of(item) if m.get("speaker") == side]
        if not entries:
            out[side] = None
            continue
        warm = sum(1 for m in entries if set(m.get("moves", []) or []) & WARM)
        hostile = sum(1 for m in entries if set(m.get("moves", []) or []) & HOSTILE)
        out[side] = warm / len(entries) - hostile / len(entries)
    return out


def turn_count(item) -> dict:
    return {
        side: sum(1 for m in _moves_of(item) if m.get("speaker") == side)
        for side in ("A", "B")
    }


def move_value_table(fit_items, mu) -> dict[str, float]:
    mu = float(mu)
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for _, item in _pairs(fit_items):
        if not isinstance(item, dict):
            continue
        by_turn: dict[int, list] = {}
        for m in _moves_of(item):
            idx = m.get("turn_index")
            if not isinstance(idx, int) or idx in by_turn:
                continue
            moves = m.get("moves", []) or []
            if not isinstance(moves, list):
                try:
                    moves = list(moves)
                except TypeError:
                    moves = []
            by_turn[idx] = moves
        if not by_turn:
            continue
        for pos, e in enumerate(_effects_of(item)):
            idx = e.get("index", pos)
            idx = idx if isinstance(idx, int) else pos
            if idx not in by_turn:
                continue
            moves = by_turn[idx]
            if not moves:
                continue
            try:
                centered = float(B._value_of(e) - mu)
            except (TypeError, ValueError):
                continue
            seen: set = set()
            for mv in moves:
                if not isinstance(mv, str) or mv in seen:
                    continue
                seen.add(mv)
                sums[mv] = sums.get(mv, 0.0) + centered
                counts[mv] = counts.get(mv, 0) + 1
    return {k: sums[k] / counts[k] for k in sums}


def tag_value(item, table) -> dict:
    tbl = {} if table is None else table
    out: dict[str, float | None] = {}
    for side in ("A", "B"):
        entries = [m for m in _moves_of(item) if m.get("speaker") == side]
        if not entries:
            out[side] = None
            continue
        vals: list[float] = []
        for m in entries:
            moves = m.get("moves", []) or []
            if not moves:
                vals.append(0.0)
                continue
            total = 0.0
            for mv in moves:
                try:
                    total += float(tbl.get(mv, 0.0))
                except (TypeError, ValueError):
                    total += 0.0
            vals.append(total / len(moves))
        out[side] = sum(vals) / len(vals) if vals else None
    return out


def _pairs(items):
    if isinstance(items, dict):
        return list(items.items())
    return [(m.get("id") if isinstance(m, dict) else None, m) for m in (items or [])]


def participant_rows(items, outcomes, mu, sd, table=None) -> list[dict]:
    rows: list[dict] = []
    for item_id, item in _pairs(items):
        if not isinstance(item, dict):
            continue
        key = item_id if item_id is not None else item.get("id")
        did = dialogue_id_of(key)
        if did is None or outcomes is None or did not in outcomes:
            continue
        try:
            oc = outcomes[did]
            rec = reception(item, mu)
            tags = tag_score(item)
            turns = turn_count(item)
            flags = flagged(item, mu, sd)
            tvals = tag_value(item, table) if table is not None else None
            for side in ("A", "B"):
                if rec.get(side) is None:
                    continue
                opp = "B" if side == "A" else "A"
                row = {
                    "dialogue": did,
                    "side": side,
                    "reception": rec[side],
                    "tag": tags.get(side),
                    "turns": turns.get(side, 0),
                    "flagged": bool(flags.get(side, False)),
                    "opp_likeness": oc[opp]["likeness"],
                    "own_points": oc[side]["points"],
                    "own_satisfaction": oc[side]["satisfaction"],
                }
                if tvals is not None:
                    row["tag_value"] = tvals.get(side)
                rows.append(row)
        except (KeyError, TypeError):
            continue
    rows.sort(key=lambda r: (r["dialogue"], r["side"]))
    return rows


def dyad_rows(items, outcomes, mu) -> list[dict]:
    rows: list[dict] = []
    for item_id, item in _pairs(items):
        if not isinstance(item, dict):
            continue
        key = item_id if item_id is not None else item.get("id")
        did = dialogue_id_of(key)
        if did is None or outcomes is None or did not in outcomes:
            continue
        dy = dyad_value(item, mu)
        rec = reception(item, mu)
        if dy is None or rec.get("A") is None or rec.get("B") is None:
            continue
        try:
            oc = outcomes[did]
            mean_sat = (oc["A"]["satisfaction"] + oc["B"]["satisfaction"]) / 2
            points_diff = oc["A"]["points"] - oc["B"]["points"]
        except (KeyError, TypeError):
            continue
        rows.append({
            "dialogue": did,
            "dyad": dy,
            "mean_satisfaction": mean_sat,
            "rec_diff": rec["A"] - rec["B"],
            "points_diff": points_diff,
        })
    rows.sort(key=lambda r: r["dialogue"])
    return rows


def _pivot_entries(items, outcomes, mu) -> list[dict]:
    mu = float(mu)
    entries: list[dict] = []
    for item_id, item in _pairs(items):
        if not isinstance(item, dict):
            continue
        key = item_id if item_id is not None else item.get("id")
        did = dialogue_id_of(key)
        if did is None or outcomes is None or did not in outcomes:
            continue
        try:
            effects = _effects_of(item)
            if not effects:
                continue
            t = B.turning_point(effects)
            if t is None:
                continue
            pivot_effect = None
            for pos, e in enumerate(effects):
                idx = e.get("index", pos)
                idx = idx if isinstance(idx, int) else pos
                if idx == t:
                    pivot_effect = e
                    break
            if pivot_effect is None:
                continue
            side = pivot_effect.get("speaker")
            if side not in ("A", "B"):
                continue
            opp = "B" if side == "A" else "A"
            oc = outcomes[did]
            opp_like = oc[opp]["likeness"]
            gap = oc[opp]["likeness"] - oc[side]["likeness"]
            pivot = B._value_of(pivot_effect) - mu
            pool = [e for e in effects if e.get("speaker") == side]
            if not pool:
                continue
            entries.append({
                "key": key,
                "dialogue": did,
                "side": side,
                "pivot": pivot,
                "opp_likeness": opp_like,
                "own_likeness_gap": gap,
                "pool": pool,
            })
        except (KeyError, TypeError, ValueError):
            continue
    return entries


def pivot_rows(items, outcomes, mu) -> list[dict]:
    rows = [
        {"dialogue": e["dialogue"], "side": e["side"], "pivot": e["pivot"],
         "opp_likeness": e["opp_likeness"],
         "own_likeness_gap": e["own_likeness_gap"]}
        for e in _pivot_entries(items, outcomes, mu)
    ]
    rows.sort(key=lambda r: r["dialogue"])
    return rows


def random_turn_rho(items, outcomes, mu, draws=200, seed=0) -> float:
    mu = float(mu)
    entries = sorted(_pivot_entries(items, outcomes, mu),
                     key=lambda e: str(e["key"]))
    if not entries or draws <= 0:
        return 0.0
    rng = random.Random(seed)
    rhos: list[float] = []
    for _ in range(draws):
        xs: list[float] = []
        ys: list = []
        for e in entries:
            pick = rng.choice(e["pool"])
            try:
                xs.append(B._value_of(pick) - mu)
            except (TypeError, ValueError):
                continue
            ys.append(e["opp_likeness"])
        rhos.append(float(spearman(xs, ys)))
    return sum(rhos) / len(rhos) if rhos else 0.0


def pivot_direction(rows) -> dict:
    use = [r for r in (rows or []) if isinstance(r, dict)
           and r.get("pivot") is not None and r.get("own_likeness_gap") is not None
           and r["pivot"] != 0 and r["own_likeness_gap"] != 0]
    n = len(use)
    if n == 0:
        return {"agree": 0, "n": 0, "share": 0.0, "ci": S.wilson_ci(0, 0)}
    k = sum(1 for r in use if (r["pivot"] > 0) == (r["own_likeness_gap"] > 0))
    return {"agree": k, "n": n, "share": k / n, "ci": S.wilson_ci(k, n)}


def _cluster_pool(rows) -> tuple[list, dict]:
    by_dialogue: dict = {}
    for r in rows:
        by_dialogue.setdefault(r.get("dialogue"), []).append(r)
    return list(by_dialogue.keys()), by_dialogue


def boot_spearman(rows, x, y, b=2000, seed=0) -> list:
    use = [r for r in (rows or []) if isinstance(r, dict)
           and r.get(x) is not None and r.get(y) is not None]
    if len(use) < 3:
        return [0.0, 0.0, 0.0]
    point = float(spearman([r[x] for r in use], [r[y] for r in use]))
    keys, by_dialogue = _cluster_pool(use)
    rng = random.Random(seed)
    m = len(keys)
    reps: list[float] = []
    for _ in range(max(0, b)):
        pool = []
        for _ in range(m):
            pool.extend(by_dialogue[rng.choice(keys)])
        reps.append(float(spearman([r[x] for r in pool], [r[y] for r in pool])))
    if not reps:
        return [point, point, point]
    reps.sort()
    return [point, float(S._percentile(reps, 0.025)), float(S._percentile(reps, 0.975))]


def boot_spearman_diff(rows, x1, x2, y, b=2000, seed=0) -> list:
    use = [r for r in (rows or []) if isinstance(r, dict)
           and r.get(x1) is not None and r.get(x2) is not None and r.get(y) is not None]
    if len(use) < 3:
        return [0.0, 0.0, 0.0]
    point = float(spearman([r[x1] for r in use], [r[y] for r in use])
                  - spearman([r[x2] for r in use], [r[y] for r in use]))
    keys, by_dialogue = _cluster_pool(use)
    rng = random.Random(seed)
    m = len(keys)
    reps: list[float] = []
    for _ in range(max(0, b)):
        pool = []
        for _ in range(m):
            pool.extend(by_dialogue[rng.choice(keys)])
        rho1 = spearman([r[x1] for r in pool], [r[y] for r in pool])
        rho2 = spearman([r[x2] for r in pool], [r[y] for r in pool])
        reps.append(float(rho1 - rho2))
    if not reps:
        return [point, point, point]
    reps.sort()
    return [point, float(S._percentile(reps, 0.025)), float(S._percentile(reps, 0.975))]


def boot_mean_diff(rows, flag, y, b=2000, seed=0) -> list:
    use = [r for r in (rows or []) if isinstance(r, dict)
           and r.get(flag) is not None and r.get(y) is not None]
    group_t = [r[y] for r in use if r[flag]]
    group_f = [r[y] for r in use if not r[flag]]
    if not group_t or not group_f:
        return [0.0, 0.0, 0.0]
    point = float(sum(group_t) / len(group_t) - sum(group_f) / len(group_f))
    keys, by_dialogue = _cluster_pool(use)
    rng = random.Random(seed)
    m = len(keys)
    reps: list[float] = []
    for _ in range(max(0, b)):
        pool = []
        for _ in range(m):
            pool.extend(by_dialogue[rng.choice(keys)])
        pt = [r[y] for r in pool if r[flag]]
        pf = [r[y] for r in pool if not r[flag]]
        if not pt or not pf:
            continue
        reps.append(float(sum(pt) / len(pt) - sum(pf) / len(pf)))
    if not reps:
        return [point, point, point]
    reps.sort()
    return [point, float(S._percentile(reps, 0.025)), float(S._percentile(reps, 0.975))]


def analyze(fit_items, eval_items, outcomes, genre="negotiation", b=2000, seed=0) -> dict:
    if isinstance(fit_items, dict):
        fit_list = [v for v in fit_items.values() if isinstance(v, dict)]
    else:
        fit_list = [v for v in (fit_items or []) if isinstance(v, dict)]
    fit_eff = [d.get("effects", []) or [] for d in fit_list if d.get("genre") == genre]
    norm = C.fit_board_norm(fit_eff, method="z")
    mu = float(norm.get("mu", 0.0))
    sd = float(norm.get("sd", 0.0))
    table = move_value_table(
        [d for d in fit_list if d.get("genre") == genre], mu)

    prow = participant_rows(eval_items, outcomes, mu, sd, table=table)
    drow = dyad_rows(eval_items, outcomes, mu)
    ptag = [r for r in prow if r.get("tag") is not None]

    c1 = boot_spearman(prow, "reception", "opp_likeness", b=b, seed=seed)
    c2 = boot_spearman(drow, "dyad", "mean_satisfaction", b=b, seed=seed)
    c3 = boot_spearman(drow, "rec_diff", "points_diff", b=b, seed=seed)
    c4_tag = boot_spearman(ptag, "tag", "opp_likeness", b=b, seed=seed)
    c4_turns = boot_spearman(prow, "turns", "opp_likeness", b=b, seed=seed)
    c4_diff = boot_spearman_diff(ptag, "reception", "tag", "opp_likeness", b=b, seed=seed)
    c5 = boot_mean_diff(prow, "flagged", "opp_likeness", b=b, seed=seed)
    c6_tag = boot_spearman(prow, "tag_value", "opp_likeness", b=b, seed=seed)
    c6_diff = boot_spearman_diff(
        prow, "reception", "tag_value", "opp_likeness", b=b, seed=seed)
    n_flagged = sum(1 for r in prow if r.get("flagged"))

    pivot = pivot_rows(eval_items, outcomes, mu)
    d1_pivot = boot_spearman(pivot, "pivot", "opp_likeness", b=b, seed=seed)
    d1_rand = random_turn_rho(eval_items, outcomes, mu, draws=200, seed=0)
    d2 = pivot_direction(pivot)

    def _verdict(ci) -> str:
        return "SUPPORTED" if ci[1] > 0 else "NOT SUPPORTED"

    verdicts = {
        "C1": _verdict(c1),
        "C2": _verdict(c2),
        "C3": "REPORTED",
        "C4_tag": "REPORTED",
        "C4_turns": "REPORTED",
        "C4_board_minus_tag": _verdict(c4_diff),
        "C5": "REPORTED",
        "C6_tag_value": _verdict(c6_tag),
        "C6_board_minus_tag_value": _verdict(c6_diff),
        "D1_pivot": "REPORTED",
        "D1_random_turn_mean": "REPORTED",
        "D2": "REPORTED",
    }
    return {
        "mu": mu,
        "sd": sd,
        "n_dialogues": len(drow),
        "n_participants": len(prow),
        "n_tag": len(ptag),
        "C1": c1,
        "C2": c2,
        "C3": c3,
        "C4_tag": c4_tag,
        "C4_turns": c4_turns,
        "C4_board_minus_tag": c4_diff,
        "C5": c5,
        "C6_tag_value": c6_tag,
        "C6_board_minus_tag_value": c6_diff,
        "D1_pivot": d1_pivot,
        "D1_random_turn_mean": d1_rand,
        "D2": d2,
        "n_flagged": n_flagged,
        "n_pivot": len(pivot),
        "verdicts": verdicts,
    }


def _fmt_ci(ci) -> str:
    try:
        return f"{ci[0]:+.3f} [{ci[1]:+.3f}-{ci[2]:+.3f}]"
    except (TypeError, IndexError):
        return "n/a"


def _fmt_cell(value) -> str:
    if isinstance(value, dict):
        try:
            share = float(value.get("share", 0.0))
            ci = value.get("ci", (0.0, 1.0))
            return f"{share:.3f} [{float(ci[0]):.3f}-{float(ci[1]):.3f}]"
        except (TypeError, IndexError, ValueError):
            return "n/a"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):+.3f}"
    return _fmt_ci(value)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-outcome")
    ap.add_argument("--fit", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--casino", required=True)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    fit_items = C._load_items(Path(args.fit))
    outcomes = load_outcomes(args.casino)
    for edir in args.eval:
        epath = Path(edir)
        eval_items = C._load_items(epath)
        res = analyze(fit_items, eval_items, outcomes, b=args.b, seed=args.seed)
        payload = {"fit": str(args.fit), "eval": str(edir), "casino": str(args.casino),
                   "b": int(args.b), "seed": int(args.seed)}
        payload.update(res)
        with open(epath / "outcome.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        n_dialogues = res.get("n_dialogues", 0)
        n_participants = res.get("n_participants", 0)
        n_tag = res.get("n_tag", 0)
        n_flagged = res.get("n_flagged", 0)
        n_pivot = res.get("n_pivot", 0)
        verdicts = res.get("verdicts", {})
        d2 = res.get("D2", {}) or {}
        try:
            d2_n = f"{int(d2.get('agree', 0))}/{int(d2.get('n', 0))}"
        except (TypeError, ValueError):
            d2_n = "n/a"
        rows = [
            ("C1", "reception vs opp likeness", res.get("C1"), n_participants),
            ("C2", "dyad vs mean satisfaction", res.get("C2"), n_dialogues),
            ("C3", "rec diff vs points diff", res.get("C3"), n_dialogues),
            ("C4_tag", "tag vs opp likeness", res.get("C4_tag"), n_tag),
            ("C4_turns", "turns vs opp likeness", res.get("C4_turns"), n_participants),
            ("C4_board_minus_tag", "rho board - rho tag", res.get("C4_board_minus_tag"), n_tag),
            ("C5", "flagged vs unflagged liking", res.get("C5"),
             f"{n_participants} ({n_flagged} flagged)"),
            ("C6_tag_value", "tag-value vs opp likeness", res.get("C6_tag_value"), n_tag),
            ("C6_board_minus_tag_value", "rho board - rho tag-value",
             res.get("C6_board_minus_tag_value"), n_tag),
            ("D1_pivot", "pivot vs opp likeness", res.get("D1_pivot"), n_pivot),
            ("D1_random_turn_mean", "random turn mean rho (200 draws)",
             res.get("D1_random_turn_mean"), n_pivot),
            ("D2", "pivot direction agree share", d2, d2_n),
        ]
        lines: list[str] = []
        lines.append("# Outcome validity of the position curve")
        lines.append("")
        lines.append(f"fit={args.fit} eval={edir} casino={args.casino} b={args.b} seed={args.seed}")
        lines.append(
            f"mu={res.get('mu', 0.0):+.3f} sd={res.get('sd', 0.0):.3f}"
            f" n_dialogues={n_dialogues} n_participants={n_participants}"
            f" n_flagged={n_flagged} n_pivot={n_pivot}"
        )
        lines.append("")
        lines.append("| metric | value [CI] | verdict | n |")
        lines.append("|---|---|---|---|")
        for key, label, val, n in rows:
            lines.append(f"| {key} ({label}) | {_fmt_cell(val)} |"
                         f" {verdicts.get(key, 'REPORTED')} | {n} |")
        lines.append("")
        with open(epath / "outcome.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
