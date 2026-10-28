"""Value model prototype: position curve as predicted final result (offline, stdlib)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import board as B
from . import calibrate as C
from . import outcome as O

FEATURES = ["rec_self", "rec_other", "tagv_self", "tagv_other", "n_self"]


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


def _effect_index(effect: dict, pos: int) -> int:
    idx = effect.get("index", pos)
    return idx if isinstance(idx, int) else pos


def _pairs(items):
    if isinstance(items, dict):
        return list(items.items())
    return [(m.get("id") if isinstance(m, dict) else None, m) for m in (items or [])]


def prefix_features(item, t, side, mu, table) -> dict:
    mu = float(mu)
    tbl = {} if table is None else table
    other = "B" if side == "A" else "A"
    effects = _effects_of(item)
    vals_self: list[float] = []
    vals_other: list[float] = []
    n_self = 0
    for pos, e in enumerate(effects):
        idx = _effect_index(e, pos)
        try:
            if idx > t:
                continue
        except TypeError:
            continue
        if e.get("speaker") == side:
            try:
                vals_self.append(float(B._value_of(e) - mu))
            except (TypeError, ValueError):
                continue
            n_self += 1
        elif e.get("speaker") == other:
            try:
                vals_other.append(float(B._value_of(e) - mu))
            except (TypeError, ValueError):
                continue
    rec_self = sum(vals_self) / len(vals_self) if vals_self else 0.0
    rec_other = sum(vals_other) / len(vals_other) if vals_other else 0.0

    def _tagv(which: str) -> float:
        vals: list[float] = []
        for m in _moves_of(item):
            if m.get("speaker") != which:
                continue
            tidx = m.get("turn_index")
            if not isinstance(tidx, int):
                continue
            try:
                if tidx > t:
                    continue
            except TypeError:
                continue
            moves = m.get("moves", []) or []
            if not isinstance(moves, list):
                try:
                    moves = list(moves)
                except TypeError:
                    moves = []
            if not moves:
                vals.append(0.0)
                continue
            total = 0.0
            for mv in moves:
                try:
                    total += float(tbl.get(mv, 0.0))
                except (TypeError, ValueError, AttributeError):
                    total += 0.0
            vals.append(total / len(moves))
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "rec_self": float(rec_self),
        "rec_other": float(rec_other),
        "tagv_self": float(_tagv(side)),
        "tagv_other": float(_tagv(other)),
        "n_self": int(n_self),
    }


def _solve_linear(mat, vec) -> list[float]:
    n = len(mat)
    if n == 0:
        return []
    aug = [list(mat[i]) + [float(vec[i])] for i in range(n)]
    for col in range(n):
        piv = col
        best = abs(aug[col][col]) if col < len(aug) else 0.0
        for row in range(col + 1, n):
            v = abs(aug[row][col])
            if v > best:
                best = v
                piv = row
        if best < 1e-12:
            continue
        if piv != col:
            aug[col], aug[piv] = aug[piv], aug[col]
        denom = aug[col][col]
        for row in range(col + 1, n):
            factor = aug[row][col] / denom
            if factor == 0.0:
                continue
            for k in range(col, n + 1):
                aug[row][k] -= factor * aug[col][k]
    sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = aug[i][n] - sum(aug[i][j] * sol[j] for j in range(i + 1, n))
        d = aug[i][i]
        if abs(d) < 1e-12:
            sol[i] = 0.0
        else:
            sol[i] = s / d
    return [float(v) for v in sol]


def ridge_fit(X, y, lam=1.0) -> dict:
    lam = float(lam)
    rows = list(X) if X is not None else []
    targets = list(y) if y is not None else []
    if not rows or not targets:
        feats = list(FEATURES)
        p = len(feats)
        return {"w": [0.0] * p, "b": 0.0, "means": [0.0] * p,
                "sds": [1.0] * p, "lam": float(lam), "features": feats}
    if rows and isinstance(rows[0], dict):
        first_keys = list(rows[0].keys())
        if set(FEATURES).issubset(set(first_keys)):
            features = list(FEATURES)
        else:
            features = list(first_keys)
        mat: list[list[float]] = []
        for r in rows:
            if not isinstance(r, dict):
                continue
            row: list[float] = []
            for f in features:
                try:
                    row.append(float(r.get(f, 0.0)))
                except (TypeError, ValueError):
                    row.append(0.0)
            mat.append(row)
    else:
        try:
            p0 = len(rows[0])
        except TypeError:
            p0 = 0
        if p0 == len(FEATURES):
            features = list(FEATURES)
        else:
            features = [f"x{i}" for i in range(p0)]
        mat = []
        for r in rows:
            try:
                seq = list(r)
            except TypeError:
                continue
            row = []
            for v in seq:
                try:
                    row.append(float(v))
                except (TypeError, ValueError):
                    row.append(0.0)
            # pad/truncate to p0
            if len(row) < len(features):
                row.extend([0.0] * (len(features) - len(row)))
            else:
                row = row[:len(features)]
            mat.append(row)
    n = min(len(mat), len(targets))
    mat = mat[:n]
    targets = [float(v) for v in targets[:n]]
    p = len(features)
    if n == 0 or p == 0:
        feats = list(features) if features else list(FEATURES)
        pp = len(feats)
        return {"w": [0.0] * pp, "b": 0.0, "means": [0.0] * pp,
                "sds": [1.0] * pp, "lam": float(lam), "features": feats}
    means: list[float] = []
    sds: list[float] = []
    for j in range(p):
        col = [mat[i][j] for i in range(n)]
        m = sum(col) / n
        var = sum((v - m) ** 2 for v in col) / n
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd == 0.0:
            sd = 1.0
        means.append(float(m))
        sds.append(float(sd))
    Z = [[(mat[i][j] - means[j]) / sds[j] for j in range(p)] for i in range(n)]
    dim = p + 1
    M = [[0.0] * dim for _ in range(dim)]
    v = [0.0] * dim
    for i in range(n):
        arow = Z[i] + [1.0]
        yi = targets[i]
        for a in range(dim):
            v[a] += arow[a] * yi
            for b in range(dim):
                M[a][b] += arow[a] * arow[b]
    for j in range(p):
        M[j][j] += lam
    sol = _solve_linear(M, v)
    w = sol[:p] if len(sol) >= p else ([0.0] * p)
    b = sol[p] if len(sol) > p else 0.0
    return {"w": [float(x) for x in w], "b": float(b),
            "means": [float(x) for x in means], "sds": [float(x) for x in sds],
            "lam": float(lam), "features": list(features)}


def ridge_predict(model, x: dict) -> float:
    if not isinstance(model, dict):
        return 0.0
    features = model.get("features") or list(FEATURES)
    w = model.get("w", []) or []
    means = model.get("means", []) or []
    sds = model.get("sds", []) or []
    try:
        b = float(model.get("b", 0.0))
    except (TypeError, ValueError):
        b = 0.0
    total = 0.0
    for j, f in enumerate(features):
        if isinstance(x, dict):
            raw = x.get(f, 0.0)
        elif isinstance(x, (list, tuple)):
            raw = x[j] if j < len(x) else 0.0
        else:
            raw = 0.0
        try:
            raw_f = float(raw)
        except (TypeError, ValueError):
            raw_f = 0.0
        try:
            mj = float(means[j]) if j < len(means) else 0.0
        except (TypeError, ValueError):
            mj = 0.0
        try:
            sj = float(sds[j]) if j < len(sds) else 1.0
        except (TypeError, ValueError):
            sj = 1.0
        if sj == 0.0:
            sj = 1.0
        try:
            wj = float(w[j]) if j < len(w) else 0.0
        except (TypeError, ValueError):
            wj = 0.0
        total += wj * ((raw_f - mj) / sj)
    return float(b + total)


def _final_index(item) -> int | None:
    effects = _effects_of(item)
    if not effects:
        return None
    best = None
    for pos, e in enumerate(effects):
        idx = _effect_index(e, pos)
        if best is None or idx > best:
            best = idx
    return best


def fit_value(fit_items, outcomes, mu, table, lam=1.0) -> dict:
    mu = float(mu)
    lam = float(lam)
    X_rows: list[dict] = []
    y_list: list[float] = []
    for item_id, item in _pairs(fit_items):
        if not isinstance(item, dict):
            continue
        key = item_id if item_id is not None else item.get("id")
        did = O.dialogue_id_of(key)
        if did is None or outcomes is None or did not in outcomes:
            continue
        t_final = _final_index(item)
        if t_final is None:
            continue
        try:
            oc = outcomes[did]
        except (KeyError, TypeError):
            continue
        for side in ("A", "B"):
            opp = "B" if side == "A" else "A"
            try:
                opp_like = oc[opp]["likeness"]
            except (KeyError, TypeError):
                continue
            if opp_like is None:
                continue
            try:
                yv = float(opp_like)
            except (TypeError, ValueError):
                continue
            feats = prefix_features(item, t_final, side, mu, table)
            X_rows.append(feats)
            y_list.append(yv)
    return ridge_fit(X_rows, y_list, lam=lam)


def value_curves(item, model, mu, table) -> dict:
    mu = float(mu)
    effects = _effects_of(item)
    curves: dict[str, list[float]] = {"A": [], "B": []}
    for pos, e in enumerate(effects):
        t = _effect_index(e, pos)
        for side in ("A", "B"):
            feats = prefix_features(item, t, side, mu, table)
            curves[side].append(float(ridge_predict(model, feats)))
    return curves


def value_pivot(curves, item, model=None) -> dict | None:
    effects = _effects_of(item)
    if not effects or not isinstance(curves, dict):
        return None
    if model is not None:
        try:
            baseline = float(ridge_predict(model, {f: 0.0 for f in FEATURES}))
        except Exception:
            baseline = 0.0
    else:
        baseline = 0.0
    best = None
    best_abs = -1.0
    for pos, e in enumerate(effects):
        idx = _effect_index(e, pos)
        speaker = e.get("speaker")
        if speaker not in ("A", "B"):
            continue
        series = curves.get(speaker)
        if not isinstance(series, list) or pos >= len(series):
            continue
        try:
            val = float(series[pos])
        except (TypeError, ValueError):
            continue
        if pos == 0:
            prev = baseline
        else:
            try:
                prev = float(series[pos - 1])
            except (TypeError, ValueError):
                continue
        delta = float(val - prev)
        if best is None or abs(delta) > best_abs:
            best = {"index": idx, "speaker": speaker, "delta": float(delta)}
            best_abs = abs(delta)
    return best


def evaluate(model, eval_items, outcomes, mu, table, b=2000, seed=0) -> dict:
    mu = float(mu)
    b = int(b)
    seed = int(seed)
    rows: list[dict] = []
    for item_id, item in _pairs(eval_items):
        if not isinstance(item, dict):
            continue
        key = item_id if item_id is not None else item.get("id")
        did = O.dialogue_id_of(key)
        if did is None or outcomes is None or did not in outcomes:
            continue
        effects = _effects_of(item)
        if not effects:
            continue
        n_effects = len(effects)
        t_final = _final_index(item)
        if t_final is None:
            continue
        t_mid = max(0, (n_effects // 2) - 1)
        try:
            rec = O.reception(item, mu)
        except Exception:
            rec = {"A": None, "B": None}
        try:
            oc = outcomes[did]
        except (KeyError, TypeError):
            continue
        for side in ("A", "B"):
            opp = "B" if side == "A" else "A"
            try:
                opp_like = oc[opp]["likeness"]
            except (KeyError, TypeError):
                continue
            if opp_like is None:
                continue
            feats_final = prefix_features(item, t_final, side, mu, table)
            feats_mid = prefix_features(item, t_mid, side, mu, table)
            rows.append({
                "dialogue": did,
                "side": side,
                "pred_final": float(ridge_predict(model, feats_final)),
                "pred_mid": float(ridge_predict(model, feats_mid)),
                "rec": rec.get(side),
                "opp_likeness": opp_like,
            })
    rows.sort(key=lambda r: (r["dialogue"], r["side"]))
    V1 = O.boot_spearman(rows, "pred_final", "opp_likeness", b=b, seed=seed)
    V1mC1 = O.boot_spearman_diff(rows, "pred_final", "rec", "opp_likeness", b=b, seed=seed)
    V2 = O.boot_spearman(rows, "pred_mid", "opp_likeness", b=b, seed=seed)

    pivot_rows: list[dict] = []
    for item_id, item in _pairs(eval_items):
        if not isinstance(item, dict):
            continue
        key = item_id if item_id is not None else item.get("id")
        did = O.dialogue_id_of(key)
        if did is None or outcomes is None or did not in outcomes:
            continue
        if not _effects_of(item):
            continue
        curves = value_curves(item, model, mu, table)
        piv = value_pivot(curves, item, model)
        if piv is None:
            continue
        speaker = piv.get("speaker")
        if speaker not in ("A", "B"):
            continue
        opp = "B" if speaker == "A" else "A"
        try:
            opp_like = outcomes[did][opp]["likeness"]
        except (KeyError, TypeError):
            continue
        if opp_like is None:
            continue
        try:
            delta = float(piv.get("delta", 0.0))
        except (TypeError, ValueError):
            continue
        pivot_rows.append({"dialogue": did, "delta": delta, "opp_likeness": opp_like})
    pivot_rows.sort(key=lambda r: r["dialogue"])
    V3 = O.boot_spearman(pivot_rows, "delta", "opp_likeness", b=b, seed=seed)

    dialogues = {r["dialogue"] for r in rows}
    w = list(model.get("w", [])) if isinstance(model, dict) else []
    try:
        bb = float(model.get("b", 0.0)) if isinstance(model, dict) else 0.0
    except (TypeError, ValueError):
        bb = 0.0
    means = list(model.get("means", [])) if isinstance(model, dict) else []
    sds = list(model.get("sds", [])) if isinstance(model, dict) else []
    feats = list(model.get("features", FEATURES)) if isinstance(model, dict) else list(FEATURES)
    try:
        lam = float(model.get("lam", 1.0)) if isinstance(model, dict) else 1.0
    except (TypeError, ValueError):
        lam = 1.0
    return {
        "V1": [float(v) for v in V1],
        "V1_minus_C1": [float(v) for v in V1mC1],
        "V1-C1": [float(v) for v in V1mC1],
        "V2": [float(v) for v in V2],
        "V3": [float(v) for v in V3],
        "n": len(rows),
        "n_participants": len(rows),
        "n_dialogues": len(dialogues),
        "n_pivot": len(pivot_rows),
        "w": [float(v) for v in w],
        "b": float(bb),
        "means": [float(v) for v in means],
        "sds": [float(v) for v in sds],
        "lam": float(lam),
        "features": list(feats),
    }


def _fmt_ci(ci) -> str:
    try:
        return f"{ci[0]:+.3f} [{ci[1]:+.3f}-{ci[2]:+.3f}]"
    except (TypeError, IndexError):
        return "n/a"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-value")
    ap.add_argument("--fit", required=True)
    ap.add_argument("--eval", nargs="+", required=True)
    ap.add_argument("--casino", required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    fit_items = C._load_items(Path(args.fit))
    outcomes = O.load_outcomes(args.casino)
    fit_sel = {cid: d for cid, d in fit_items.items()
               if isinstance(d, dict) and d.get("genre") == "negotiation"}
    fit_eff = [d.get("effects", []) or [] for d in fit_sel.values()]
    norm = C.fit_board_norm(fit_eff, method="z")
    mu = float(norm.get("mu", 0.0))
    sd = float(norm.get("sd", 0.0))
    table = O.move_value_table(list(fit_sel.values()), mu)
    model = fit_value(fit_sel, outcomes, mu, table, lam=float(args.lam))

    for edir in args.eval:
        epath = Path(edir)
        eval_items = C._load_items(epath)
        res = evaluate(model, eval_items, outcomes, mu, table,
                       b=int(args.b), seed=int(args.seed))
        payload = {"fit": str(args.fit), "eval": str(edir),
                   "casino": str(args.casino), "lam": float(args.lam),
                   "b": int(args.b), "seed": int(args.seed),
                   "mu": float(mu), "sd": float(sd)}
        payload.update(res)
        with open(epath / "value.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        feats = res.get("features", FEATURES)
        w = res.get("w", [])
        means = res.get("means", [])
        sds = res.get("sds", [])
        lines: list[str] = []
        lines.append("# Value model: predicted final result")
        lines.append("")
        lines.append(f"fit={args.fit} eval={edir} casino={args.casino} "
                     f"lam={float(args.lam):.3f} b={int(args.b)} seed={int(args.seed)}")
        lines.append(f"mu={mu:+.3f} sd={sd:.3f} n={res.get('n', 0)} "
                     f"n_dialogues={res.get('n_dialogues', 0)} "
                     f"n_pivot={res.get('n_pivot', 0)} b_intercept={res.get('b', 0.0):+.3f}")
        lines.append("")
        lines.append("| feature | w | mean | sd |")
        lines.append("|---|---|---|---|")
        for j, f in enumerate(feats):
            try:
                wj = float(w[j])
            except (IndexError, TypeError, ValueError):
                wj = 0.0
            try:
                mj = float(means[j])
            except (IndexError, TypeError, ValueError):
                mj = 0.0
            try:
                sj = float(sds[j])
            except (IndexError, TypeError, ValueError):
                sj = 1.0
            lines.append(f"| {f} | {wj:+.3f} | {mj:+.3f} | {sj:.3f} |")
        lines.append("")
        lines.append("| metric | value [CI] | n |")
        lines.append("|---|---|---|")
        n_part = res.get("n_participants", res.get("n", 0))
        n_piv = res.get("n_pivot", 0)
        rows_md = [
            ("V1", "pred_final vs opp likeness", res.get("V1"), n_part),
            ("V1_minus_C1 (V1-C1)", "rho value - rho reception", res.get("V1_minus_C1"), n_part),
            ("V2", "pred_mid vs opp likeness", res.get("V2"), n_part),
            ("V3", "pivot delta vs opp likeness", res.get("V3"), n_piv),
        ]
        for key, label, val, n in rows_md:
            lines.append(f"| {key} ({label}) | {_fmt_ci(val)} | {n} |")
        lines.append("")
        with open(epath / "value.md", "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
