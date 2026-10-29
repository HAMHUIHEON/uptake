"""Scale-resolution pilot, residual test and power (Amendment 6, offline, stdlib)."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from . import board as B
from . import calibrate as C
from . import outcome as O
from . import stats as S
from . import usage as U
from . import valuemodel as V
from .batch import spearman

SCALE_BOUNDS = {
    "v2": ((-2, 2), (0, 2)),
    "fine100": ((0, 100), (0, 100)),
    "seven": ((-3, 3), (-3, 3)),
    "dist5": ((10, 90), (10, 90)),
}


def _num(value, default=0.0):
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        try:
            f = float(value)
        except (ValueError, OverflowError):
            return default
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    return default


def reading_raw(r):
    if not isinstance(r, dict):
        return (0, 0)
    if "rel_raw" in r or "face_raw" in r:
        rel = r.get("rel_raw", r.get("relationship_shift", 0))
        face = r.get("face_raw", r.get("face_threat", 0))
    else:
        rel = r.get("relationship_shift", 0)
        face = r.get("face_threat", 0)
    if isinstance(rel, bool):
        rel = 0
    if isinstance(face, bool):
        face = 0
    if not isinstance(rel, (int, float)):
        try:
            rel = float(rel)
        except (TypeError, ValueError):
            rel = 0
    if not isinstance(face, (int, float)):
        try:
            face = float(face)
        except (TypeError, ValueError):
            face = 0
    return (rel, face)


def reading_norm(r):
    if not isinstance(r, dict):
        return (0.0, 0.0)
    if "rel_norm" in r:
        rel = r.get("rel_norm", 0.0)
    else:
        rel = r.get("relationship_shift", 0.0)
    if "face_norm" in r:
        face = r.get("face_norm", 0.0)
    else:
        if "face_threat" in r:
            try:
                ft = r.get("face_threat", 0)
                if isinstance(ft, bool):
                    face = 0.0
                else:
                    face = -float(ft)
            except (TypeError, ValueError):
                face = 0.0
        else:
            face = 0.0
    return (float(_num(rel, 0.0)), float(_num(face, 0.0)))


def record_scale(record) -> str:
    if isinstance(record, dict):
        s = record.get("scale", "v2")
        if isinstance(s, str) and s:
            return s
    return "v2"


def load_dir(path, ids=None):
    out = {}
    base = Path(path)
    if ids is None:
        try:
            files = sorted(base.glob("casino-*.json"))
        except OSError:
            return {}
        for p in files:
            try:
                if not p.is_file():
                    continue
            except OSError:
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            out[p.stem] = data
        return out
    try:
        want = list(ids)
    except TypeError:
        return {}
    for cid in want:
        if not isinstance(cid, str) or not cid:
            continue
        p = base / f"{cid}.json"
        try:
            if not p.is_file():
                continue
        except OSError:
            continue
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        out[cid] = data
    return out


def _iter_records(records):
    if isinstance(records, dict):
        return list(records.items())
    items = []
    for r in (records or []):
        if isinstance(r, dict):
            items.append((r.get("id"), r))
        else:
            items.append((None, r))
    return items


def _effects_of(record):
    if not isinstance(record, dict):
        return []
    eff = record.get("effects", [])
    return [e for e in (eff or []) if isinstance(e, dict)]


def resolution(records, base_n=3) -> dict:
    try:
        base_n_f = float(base_n)
    except (TypeError, ValueError):
        base_n_f = 3.0
    pairs = _iter_records(records)
    turn_values = []
    reading_values = []
    spreads = []
    rel_min_c = rel_max_c = face_min_c = face_max_c = 0
    n_readings_total = 0
    sum_n = 0
    sum_exp = 0
    groups = []
    for _rid, record in pairs:
        scale = record_scale(record)
        bounds = SCALE_BOUNDS.get(scale, SCALE_BOUNDS["v2"])
        try:
            (rel_lo, rel_hi), (face_lo, face_hi) = bounds
        except (TypeError, ValueError):
            (rel_lo, rel_hi), (face_lo, face_hi) = (-2, 2), (0, 2)
        for e in _effects_of(record):
            try:
                n_r = int(e.get("n_readings", 0) or 0)
            except (TypeError, ValueError):
                n_r = 0
            sum_n += n_r
            if "n_expected" in e and e.get("n_expected") is not None:
                try:
                    exp = int(e.get("n_expected") or 0)
                except (TypeError, ValueError):
                    exp = 0
            else:
                try:
                    n_list = len(e.get("listeners", []) or [])
                except TypeError:
                    n_list = 0
                exp = int(base_n_f * n_list)
            sum_exp += exp
            if n_r > 0:
                try:
                    turn_values.append(float(B._value_of(e)))
                except (TypeError, ValueError):
                    continue
                sp = e.get("spread", None)
                if isinstance(sp, (int, float)) and not isinstance(sp, bool):
                    try:
                        fsp = float(sp)
                    except (ValueError, OverflowError):
                        pass
                    else:
                        if not (math.isnan(fsp) or math.isinf(fsp)):
                            spreads.append(fsp)
            readings = e.get("readings", []) or []
            vals = []
            for r in readings:
                if not isinstance(r, dict):
                    continue
                rn, fn = reading_norm(r)
                s = float(rn + fn)
                vals.append(s)
                reading_values.append(s)
                rr, fr = reading_raw(r)
                n_readings_total += 1
                try:
                    if float(rr) == float(rel_lo):
                        rel_min_c += 1
                except (TypeError, ValueError):
                    pass
                try:
                    if float(rr) == float(rel_hi):
                        rel_max_c += 1
                except (TypeError, ValueError):
                    pass
                try:
                    if float(fr) == float(face_lo):
                        face_min_c += 1
                except (TypeError, ValueError):
                    pass
                try:
                    if float(fr) == float(face_hi):
                        face_max_c += 1
                except (TypeError, ValueError):
                    pass
            if len(vals) >= 2:
                groups.append(vals)
    n_turns = len(turn_values)
    distinct_turn = len({round(v, 6) for v in turn_values})
    distinct_reading = len({round(v, 6) for v in reading_values})
    distinct_turn_01 = len({round(v, 2) for v in turn_values})
    distinct_turn_1 = len({round(v, 1) for v in turn_values})
    mean_spread = sum(spreads) / len(spreads) if spreads else 0.0
    if n_readings_total > 0:
        rel_min_share = rel_min_c / n_readings_total
        rel_max_share = rel_max_c / n_readings_total
        face_min_share = face_min_c / n_readings_total
        face_max_share = face_max_c / n_readings_total
    else:
        rel_min_share = rel_max_share = face_min_share = face_max_share = 0.0
    if n_turns == 0:
        mean = 0.0
        sd = 0.0
        z_ge2 = 0.0
        z_le = 0.0
        z_max = 0.0
        z_min = 0.0
    else:
        mean = sum(turn_values) / n_turns
        var = sum((v - mean) ** 2 for v in turn_values) / n_turns
        sd = math.sqrt(var) if var > 0 else 0.0
        if sd == 0:
            z_ge2 = 0.0
            z_le = 0.0
            z_max = 0.0
            z_min = 0.0
        else:
            zs = [(v - mean) / sd for v in turn_values]
            z_ge2 = sum(1 for z in zs if z >= 2) / n_turns
            z_le = sum(1 for z in zs if z <= -2) / n_turns
            z_max = max(zs)
            z_min = min(zs)
    if sum_exp == 0:
        parse_fail = 0.0
    else:
        parse_fail = 1.0 - sum_n / sum_exp
    g = len(groups)
    if g < 2:
        icc = 0.0
    else:
        n_list = [len(v) for v in groups]
        n_tot = sum(n_list)
        overall = sum(v for grp in groups for v in grp) / n_tot
        ssb = 0.0
        ssw = 0.0
        sum_sq = 0
        for grp in groups:
            ni = len(grp)
            mi = sum(grp) / ni
            ssb += ni * (mi - overall) ** 2
            ssw += sum((v - mi) ** 2 for v in grp)
            sum_sq += ni * ni
        msb = ssb / (g - 1)
        denom_w = n_tot - g
        msw = ssw / denom_w if denom_w != 0 else 0.0
        n0 = (n_tot - sum_sq / n_tot) / (g - 1)
        denom = msb + (n0 - 1.0) * msw
        if denom == 0:
            icc = 0.0
        else:
            icc = (msb - msw) / denom
    def _sb(m):
        d = 1.0 + (m - 1.0) * icc
        if d == 0:
            return 0.0
        return m * icc / d
    if isinstance(pairs, list):
        n_dialogues = len(pairs)
    else:
        n_dialogues = 0
    return {
        "n_turns": int(n_turns),
        "distinct_turn_values": int(distinct_turn),
        "distinct_reading_values": int(distinct_reading),
        "distinct_turn_values_01": int(distinct_turn_01),
        "distinct_turn_values_1": int(distinct_turn_1),
        "distinct_turn_values_0.01": int(distinct_turn_01),
        "distinct_turn_values_0.1": int(distinct_turn_1),
        "mean_spread": float(mean_spread),
        "rel_min_share": float(rel_min_share),
        "rel_max_share": float(rel_max_share),
        "face_min_share": float(face_min_share),
        "face_max_share": float(face_max_share),
        "z_ge2_share": float(z_ge2),
        "z_le_minus2_share": float(z_le),
        "z_max": float(z_max),
        "z_min": float(z_min),
        "mean": float(mean),
        "sd": float(sd),
        "parse_fail_rate": float(parse_fail),
        "icc1": float(icc),
        "sb3": float(_sb(3)),
        "sb5": float(_sb(5)),
        "n_dialogues": int(n_dialogues),
    }


def _turn_map(records):
    out = {}
    for item_id, record in _iter_records(records):
        if not isinstance(record, dict):
            continue
        rid = item_id if item_id is not None else record.get("id")
        if rid is None:
            continue
        for pos, e in enumerate(_effects_of(record)):
            try:
                n_r = int(e.get("n_readings", 0) or 0)
            except (TypeError, ValueError):
                n_r = 0
            if n_r <= 0:
                continue
            idx = e.get("index", pos)
            idx = idx if isinstance(idx, int) else pos
            try:
                v = float(B._value_of(e))
            except (TypeError, ValueError):
                continue
            out[(rid, idx)] = v
    return out


def convergent(records, base_records) -> dict:
    m1 = _turn_map(records)
    m2 = _turn_map(base_records)
    common = sorted(set(m1) & set(m2))
    n = len(common)
    if n == 0:
        return {"rho": 0.0, "n": 0}
    xs = [m1[k] for k in common]
    ys = [m2[k] for k in common]
    try:
        rho = float(spearman(xs, ys))
    except (ValueError, TypeError):
        rho = 0.0
    if math.isnan(rho) or math.isinf(rho):
        rho = 0.0
    return {"rho": float(rho), "n": int(n)}


def board_cost(calls_path, ids, layers=("board", "pairwise")) -> dict:
    zeros = {"calls": 0, "usd": 0.0, "usd_per_dialogue": 0.0,
             "mean_completion_tokens": 0.0}
    if isinstance(layers, str):
        layers_set = {layers}
    else:
        try:
            layers_set = set(layers or ())
        except TypeError:
            layers_set = {"board"}
    if ids is None:
        ids_set = None
        n_ids = 0
    else:
        try:
            ids_list = list(ids)
        except TypeError:
            ids_list = []
        ids_set = set(ids_list)
        n_ids = len(ids_list)
    try:
        p = Path(calls_path)
    except (TypeError, ValueError):
        return dict(zeros)
    try:
        if not p.is_file():
            return dict(zeros)
    except OSError:
        return dict(zeros)
    calls = 0
    usd = 0.0
    sum_ct = 0
    try:
        f = open(p, encoding="utf-8")
    except OSError:
        return dict(zeros)
    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(rec, dict):
                continue
            meta = rec.get("meta", {})
            if not isinstance(meta, dict):
                continue
            if meta.get("layer") not in layers_set:
                continue
            if ids_set is not None:
                if meta.get("item") in ids_set:
                    pass
                elif meta.get("layer") == "pairwise" and any(
                        d in ids_set
                        for d in (meta.get("dialogues", []) or [])):
                    pass
                else:
                    continue
            if rec.get("error", None) is not None:
                continue
            model = rec.get("model", "")
            try:
                price_in, price_out = U.PRICES.get(model, U._FALLBACK)
            except (TypeError, AttributeError):
                price_in, price_out = U._FALLBACK
            try:
                pt = rec.get("prompt_tokens", 0) or 0
                pt = int(pt)
            except (TypeError, ValueError):
                pt = 0
            try:
                ct = rec.get("completion_tokens", 0) or 0
                ct = int(ct)
            except (TypeError, ValueError):
                ct = 0
            usd += pt / 1_000_000 * float(price_in) + ct / 1_000_000 * float(price_out)
            sum_ct += ct
            calls += 1
    if calls > 0:
        mean_ct = sum_ct / calls
    else:
        mean_ct = 0.0
    if n_ids > 0:
        upd = usd / n_ids
    else:
        upd = 0.0
    return {"calls": int(calls), "usd": float(usd),
            "usd_per_dialogue": float(upd),
            "mean_completion_tokens": float(mean_ct)}


def rank(values):
    vals = list(values) if values is not None else []
    n = len(vals)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = float(avg)
        i = j + 1
    return out


def _solve(mat, vec):
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


def ols(X_rows, y):
    ys = [float(v) for v in (y or [])]
    n = len(ys)
    if X_rows is None:
        raw = []
    else:
        raw = [list(r) if isinstance(r, (list, tuple)) else [r] for r in X_rows]
    if n == 0:
        p = len(raw[0]) if raw else 0
        dim = p + 1
        return ([0.0] * dim, [], 0.0)
    if not raw:
        raw = [[] for _ in range(n)]
    if len(raw) != n:
        if len(raw) < n:
            width = len(raw[0]) if raw else 0
            raw = raw + [[0.0] * width for _ in range(n - len(raw))]
        else:
            raw = raw[:n]
    p = len(raw[0]) if raw else 0
    X = []
    for r in raw:
        row = []
        for v in r[:p]:
            try:
                if isinstance(v, bool):
                    row.append(0.0)
                else:
                    row.append(float(v))
            except (TypeError, ValueError):
                row.append(0.0)
        while len(row) < p:
            row.append(0.0)
        X.append([1.0] + row)
    dim = p + 1
    xtx = [[0.0] * dim for _ in range(dim)]
    xty = [0.0] * dim
    for i in range(n):
        xi = X[i]
        yi = ys[i]
        for a in range(dim):
            xty[a] += xi[a] * yi
            for b in range(dim):
                xtx[a][b] += xi[a] * xi[b]
    for d in range(dim):
        xtx[d][d] += 1e-9
    coef = _solve(xtx, xty)
    if len(coef) < dim:
        coef = coef + [0.0] * (dim - len(coef))
    preds = [sum(X[i][j] * coef[j] for j in range(dim)) for i in range(n)]
    resid = [ys[i] - preds[i] for i in range(n)]
    ssres = sum(r * r for r in resid)
    mean_y = sum(ys) / n
    sstot = sum((v - mean_y) ** 2 for v in ys)
    if sstot == 0:
        r2 = 0.0
    else:
        r2 = 1.0 - ssres / sstot
    return ([float(v) for v in coef], [float(v) for v in resid], float(r2))


def _pearson(a, b) -> float:
    n = len(a)
    if n == 0 or len(b) != n:
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = sum((x - ma) ** 2 for x in a)
    db = sum((y - mb) ** 2 for y in b)
    denom = math.sqrt(da * db)
    if denom == 0:
        return 0.0
    return num / denom


def partial_spearman(x, y, controls) -> float:
    xs = list(x or [])
    ys = list(y or [])
    if len(xs) != len(ys):
        raise ValueError("sequences must have equal length")
    n = len(xs)
    if n == 0:
        return 0.0
    if not controls:
        try:
            return float(spearman(xs, ys))
        except (ValueError, TypeError):
            return 0.0
    ctrl = [list(c) for c in (controls or [])]
    for c in ctrl:
        if len(c) != n:
            raise ValueError("sequences must have equal length")
    rx = rank(xs)
    ry = rank(ys)
    rc = [rank(c) for c in ctrl]
    x_rows = [[rc[j][i] for j in range(len(rc))] for i in range(n)]
    try:
        _, res_x, _ = ols(x_rows, rx)
        _, res_y, _ = ols(x_rows, ry)
    except (ValueError, TypeError):
        return 0.0
    return float(_pearson(res_x, res_y))


def _zscore(vals):
    n = len(vals)
    if n == 0:
        return []
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / n
    sd = math.sqrt(var) if var > 0 else 0.0
    if sd == 0:
        return [0.0] * n
    return [(v - m) / sd for v in vals]


def rank_regression(x, y, control):
    xs = list(x or [])
    ys = list(y or [])
    cs = list(control or [])
    n = len(xs)
    if len(ys) != n or len(cs) != n:
        raise ValueError("sequences must have equal length")
    if n == 0:
        return (0.0, 0.0)
    zx = _zscore(rank(xs))
    zy = _zscore(rank(ys))
    zc = _zscore(rank(cs))
    rows_full = [[zc[i], zx[i]] for i in range(n)]
    rows_c = [[zc[i]] for i in range(n)]
    try:
        coef_full, _, r2_full = ols(rows_full, zy)
    except (ValueError, TypeError):
        return (0.0, 0.0)
    try:
        _, _, r2_c = ols(rows_c, zy)
    except (ValueError, TypeError):
        r2_c = 0.0
    if len(coef_full) >= 3:
        beta_x = float(coef_full[2])
    elif len(coef_full) >= 2:
        beta_x = float(coef_full[1])
    else:
        beta_x = 0.0
    if math.isnan(beta_x) or math.isinf(beta_x):
        beta_x = 0.0
    d = float(r2_full - r2_c)
    if math.isnan(d) or math.isinf(d):
        d = 0.0
    return (float(beta_x), float(d))


def boot(rows, stat_fns, b, seed):
    names = list((stat_fns or {}).keys())
    out = {}
    base_rows = list(rows or [])
    points = {}
    for name in names:
        fn = stat_fns[name]
        try:
            v = float(fn(base_rows))
        except Exception:
            v = 0.0
        try:
            if math.isnan(v) or math.isinf(v):
                v = 0.0
        except TypeError:
            v = 0.0
        points[name] = float(v)
    try:
        b_int = int(b)
    except (TypeError, ValueError):
        b_int = 0
    if b_int < 0:
        b_int = 0
    try:
        keys, by_dialogue = O._cluster_pool(base_rows)
    except Exception:
        keys, by_dialogue = ([], {})
    if not keys or b_int == 0:
        for name in names:
            out[name] = {"point": float(points[name]), "lo": float(points[name]),
                         "hi": float(points[name]), "se": 0.0, "reps_n": 0}
        return out
    rng = random.Random(seed)
    m = len(keys)
    pools = []
    for _ in range(b_int):
        pool = []
        for _ in range(m):
            pool.extend(by_dialogue[rng.choice(keys)])
        pools.append(pool)
    for name in names:
        fn = stat_fns[name]
        reps = []
        for pool in pools:
            try:
                v = float(fn(pool))
            except Exception:
                continue
            try:
                if math.isnan(v):
                    continue
            except TypeError:
                continue
            reps.append(float(v))
        if not reps:
            out[name] = {"point": float(points[name]), "lo": float(points[name]),
                         "hi": float(points[name]), "se": 0.0, "reps_n": 0}
            continue
        srep = sorted(reps)
        try:
            lo = float(S._percentile(srep, 0.025))
            hi = float(S._percentile(srep, 0.975))
        except (IndexError, TypeError, ValueError):
            lo = hi = float(points[name])
        mean = sum(reps) / len(reps)
        var = sum((v - mean) ** 2 for v in reps) / len(reps)
        se = math.sqrt(var) if var > 0 else 0.0
        out[name] = {"point": float(points[name]), "lo": float(lo),
                     "hi": float(hi), "se": float(se),
                     "reps_n": int(len(reps))}
    return out


def _moves_list(m):
    if not isinstance(m, dict):
        return []
    mv = m.get("moves", []) or []
    if not isinstance(mv, list):
        try:
            mv = list(mv)
        except TypeError:
            return []
    return mv


def rows_for(eval_records, outcomes, fit_records, genre="negotiation"):
    if isinstance(fit_records, dict):
        fit_list = [v for v in fit_records.values() if isinstance(v, dict)]
    else:
        fit_list = [v for v in (fit_records or []) if isinstance(v, dict)]
    fit_genre = [d for d in fit_list if d.get("genre") == genre]
    fit_eff = [d.get("effects", []) or [] for d in fit_genre]
    norm = C.fit_board_norm(fit_eff, method="z")
    try:
        mu = float(norm.get("mu", 0.0))
    except (TypeError, ValueError):
        mu = 0.0
    try:
        sd = float(norm.get("sd", 0.0))
    except (TypeError, ValueError):
        sd = 0.0
    table = O.move_value_table(fit_genre, mu)
    rows = O.participant_rows(eval_records, outcomes, mu, sd, table=table)
    counts = {}
    total_tagged = 0
    for d in fit_genre:
        moves = d.get("moves", []) or []
        for m in moves:
            if not isinstance(m, dict):
                continue
            total_tagged += 1
            seen = set()
            for mv in _moves_list(m):
                if isinstance(mv, str) and mv not in seen:
                    seen.add(mv)
            for mv in seen:
                counts[mv] = counts.get(mv, 0) + 1
    if total_tagged > 0:
        share_moves = sorted([mv for mv, c in counts.items() if c / total_tagged >= 0.01])
    else:
        share_moves = []
    did_to_record = {}
    for item_id, record in _iter_records(eval_records):
        if not isinstance(record, dict):
            continue
        key = item_id if item_id is not None else record.get("id")
        did = O.dialogue_id_of(key)
        if did is not None:
            did_to_record[did] = record
    for row in rows:
        did = row.get("dialogue")
        side = row.get("side")
        record = did_to_record.get(did)
        if not isinstance(record, dict):
            for mv in share_moves:
                row[f"share_{mv}"] = 0.0
            row["resid"] = None
            continue
        entries = [m for m in (record.get("moves", []) or [])
                   if isinstance(m, dict) and m.get("speaker") == side]
        denom = len(entries)
        for mv in share_moves:
            if denom == 0:
                row[f"share_{mv}"] = 0.0
                continue
            c = 0
            for m in entries:
                ml = _moves_list(m)
                if mv in ml:
                    c += 1
            row[f"share_{mv}"] = float(c / denom)
        by_turn = {}
        for m in (record.get("moves", []) or []):
            if not isinstance(m, dict):
                continue
            idx = m.get("turn_index")
            if not isinstance(idx, int) or idx in by_turn:
                continue
            by_turn[idx] = _moves_list(m)
        terms = []
        for pos, e in enumerate(_effects_of(record)):
            if e.get("speaker") != side:
                continue
            idx = e.get("index", pos)
            idx = idx if isinstance(idx, int) else pos
            r = V.turn_residual(e, idx, by_turn, mu, table)
            if r is None:
                continue
            terms.append(r)
        if terms:
            row["resid"] = float(sum(terms) / len(terms))
        else:
            row["resid"] = None
    info = {"mu": float(mu), "sd": float(sd),
            "share_moves": list(share_moves), "n_rows": int(len(rows))}
    return (rows, info)


def analyze_eval(eval_records, outcomes, fit_records, compare_records=None,
                 b=2000, b_r2=1000, seed=0, same_scale=False) -> dict:
    rows_all, info = rows_for(eval_records, outcomes, fit_records)
    rows = [r for r in rows_all
            if r.get("reception") is not None
            and r.get("tag_value") is not None
            and r.get("opp_likeness") is not None]
    has_compare = compare_records is not None
    if has_compare:
        cmp_all, _ = rows_for(compare_records, outcomes, fit_records)
        cmp_map = {}
        for r in cmp_all:
            key = (r.get("dialogue"), r.get("side"))
            if key[0] is None or key[1] is None:
                continue
            if r.get("reception") is None:
                continue
            cmp_map[key] = r
        for r in rows:
            key = (r.get("dialogue"), r.get("side"))
            hit = cmp_map.get(key)
            if hit is None or hit.get("reception") is None:
                r["reception_cmp"] = None
            else:
                try:
                    r["reception_cmp"] = float(hit.get("reception"))
                except (TypeError, ValueError):
                    r["reception_cmp"] = None

    def _c1(rs):
        return float(spearman([r["reception"] for r in rs],
                              [r["opp_likeness"] for r in rs]))

    def _c6(rs):
        return float(spearman([r["tag_value"] for r in rs],
                              [r["opp_likeness"] for r in rs]))

    def _c6diff(rs):
        return float(spearman([r["reception"] for r in rs],
                              [r["opp_likeness"] for r in rs])
                     - spearman([r["tag_value"] for r in rs],
                                [r["opp_likeness"] for r in rs]))

    def _r1(rs):
        return float(partial_spearman([r["reception"] for r in rs],
                                      [r["opp_likeness"] for r in rs],
                                      [[r["tag_value"] for r in rs]]))

    def _r3b(rs):
        beta, _ = rank_regression([r["reception"] for r in rs],
                                  [r["opp_likeness"] for r in rs],
                                  [r["tag_value"] for r in rs])
        return float(beta)

    def _r3d(rs):
        _, d = rank_regression([r["reception"] for r in rs],
                               [r["opp_likeness"] for r in rs],
                               [r["tag_value"] for r in rs])
        return float(d)

    def _r4(rs):
        use = [r for r in rs if r.get("resid") is not None]
        if not use:
            return 0.0
        return float(spearman([r["resid"] for r in use],
                              [r["opp_likeness"] for r in use]))

    def _dc1(rs):
        use = [r for r in rs if r.get("reception_cmp") is not None]
        if not use:
            return 0.0
        return float(spearman([r["reception"] for r in use],
                              [r["opp_likeness"] for r in use])
                     - spearman([r["reception_cmp"] for r in use],
                                [r["opp_likeness"] for r in use]))

    def _r1cmp(rs):
        use = [r for r in rs if r.get("reception_cmp") is not None]
        if not use:
            return 0.0
        return float(partial_spearman([r["reception_cmp"] for r in use],
                                      [r["opp_likeness"] for r in use],
                                      [[r["tag_value"] for r in use]]))

    stat_fns = {
        "C1": _c1,
        "C6_tag": _c6,
        "C6diff": _c6diff,
        "R1": _r1,
        "R3_beta": _r3b,
        "R3_dr2": _r3d,
    }
    if same_scale:
        stat_fns["R4"] = _r4
    if has_compare:
        stat_fns["dC1"] = _dc1
        stat_fns["R1_cmp"] = _r1cmp
    try:
        b_int = int(b)
    except (TypeError, ValueError):
        b_int = 2000
    try:
        br2_int = int(b_r2)
    except (TypeError, ValueError):
        br2_int = 1000
    try:
        seed_int = int(seed)
    except (TypeError, ValueError):
        seed_int = 0
    main_stats = boot(rows, stat_fns, b_int, seed_int)
    share_moves = list(info.get("share_moves", []))

    def _r2(rs):
        ctrls = [[r["tag_value"] for r in rs],
                 [float(r.get("turns", 0) or 0) for r in rs]]
        for mv in share_moves:
            ctrls.append([float(r.get(f"share_{mv}", 0.0) or 0.0) for r in rs])
        return float(partial_spearman([r["reception"] for r in rs],
                                      [r["opp_likeness"] for r in rs],
                                      ctrls))

    r2_stats = boot(rows, {"R2": _r2}, br2_int, seed_int)
    stats = dict(main_stats)
    stats.update(r2_stats)
    verdicts = {}
    for name in stats:
        if name in ("C6diff", "R1"):
            try:
                lo = float(stats[name].get("lo", 0.0))
            except (TypeError, ValueError):
                lo = 0.0
            verdicts[name] = "SUPPORTED" if lo > 0 else "NOT SUPPORTED"
        else:
            verdicts[name] = "REPORTED"
    dialogues = {r.get("dialogue") for r in rows if r.get("dialogue") is not None}
    n_dial = len(dialogues)
    power = {}
    for name in ("C6diff", "R1"):
        if name in stats:
            try:
                se = float(stats[name].get("se", 0.0))
            except (TypeError, ValueError):
                se = 0.0
            power[name] = _power_entry(n_dial, se)
    if has_compare and "dC1" in stats:
        try:
            se = float(stats["dC1"].get("se", 0.0))
        except (TypeError, ValueError):
            se = 0.0
        power["dC1"] = _power_entry(n_dial, se)
    return {"n_dialogues": int(n_dial), "n_rows": int(len(rows)),
            "stats": stats, "verdicts": verdicts, "power": power,
            "info": info}


def _power_entry(n, se):
    try:
        nn = int(n)
    except (TypeError, ValueError):
        nn = 0
    try:
        s = float(se)
    except (TypeError, ValueError):
        s = 0.0
    if nn < 0:
        nn = 0
    if not (s >= 0):
        s = 0.0
    z = 1.959964
    out80 = {}
    out90 = {}
    for key, delta in (("0.03", 0.03), ("0.06", 0.06)):
        v80 = nn * (s * (z + 0.841621) / delta) ** 2 if delta else 0.0
        v90 = nn * (s * (z + 1.281552) / delta) ** 2 if delta else 0.0
        try:
            n80 = int(math.ceil(v80)) if v80 > 0 else 0
        except (ValueError, OverflowError):
            n80 = 0
        try:
            n90 = int(math.ceil(v90)) if v90 > 0 else 0
        except (ValueError, OverflowError):
            n90 = 0
        out80[key] = n80
        out90[key] = n90
    return {"se": float(s), "n80": out80, "n90": out90}


def _infer_base_n(records):
    ratios = []
    for _rid, record in _iter_records(records):
        for e in _effects_of(record):
            if "n_expected" not in e or e.get("n_expected") is None:
                continue
            try:
                ne = int(e.get("n_expected") or 0)
            except (TypeError, ValueError):
                continue
            try:
                nl = len(e.get("listeners", []) or [])
            except TypeError:
                nl = 0
            if nl > 0 and ne > 0:
                ratios.append(ne / nl)
    if not ratios:
        return 3
    mean = sum(ratios) / len(ratios)
    try:
        return max(1, int(round(mean)))
    except (TypeError, ValueError):
        return 3


def pilot(conditions, base_name, outcomes, fit_records, calls_paths, ids,
          b=2000, seed=0) -> dict:
    try:
        b_int = int(b)
    except (TypeError, ValueError):
        b_int = 2000
    try:
        seed_int = int(seed)
    except (TypeError, ValueError):
        seed_int = 0
    names = list((conditions or {}).keys())
    if base_name not in (conditions or {}):
        raise ValueError(f"unknown base: {base_name!r}")
    base_records = conditions[base_name]
    out = {}
    for name in names:
        records = conditions[name]
        if name == base_name:
            base_n = 3
        else:
            base_n = _infer_base_n(records)
        res = resolution(records, base_n=base_n)
        cpath = None
        if isinstance(calls_paths, dict):
            cpath = calls_paths.get(name)
        cost = board_cost(cpath, ids) if cpath is not None else board_cost("missing-calls.jsonl", ids)
        rows_all, info = rows_for(records, outcomes, fit_records)
        rows = [r for r in rows_all
                if r.get("reception") is not None
                and r.get("tag_value") is not None
                and r.get("opp_likeness") is not None]

        def _c1(rs, _n=name):
            if not rs:
                return 0.0
            return float(spearman([r["reception"] for r in rs],
                                  [r["opp_likeness"] for r in rs]))

        def _diff(rs, _n=name):
            if not rs:
                return 0.0
            return float(spearman([r["reception"] for r in rs],
                                  [r["opp_likeness"] for r in rs])
                         - spearman([r["tag_value"] for r in rs],
                                    [r["opp_likeness"] for r in rs]))

        bres = boot(rows, {"C1": _c1, "C6diff": _diff}, b_int, seed_int)
        dialogues = {r.get("dialogue") for r in rows if r.get("dialogue") is not None}
        entry = {
            "resolution": res,
            "cost": cost,
            "board_cost": cost,
            "C1": bres.get("C1", {"point": 0.0, "lo": 0.0, "hi": 0.0, "se": 0.0, "reps_n": 0}),
            "C6diff": bres.get("C6diff", {"point": 0.0, "lo": 0.0, "hi": 0.0, "se": 0.0, "reps_n": 0}),
            "n_rows": int(len(rows)),
            "n_dialogues": int(len(dialogues)),
            "base_n": int(base_n),
            "info": info,
        }
        if name != base_name:
            entry["convergent"] = convergent(records, base_records)
        out[name] = entry
    return out


def _fmt_ci(stat) -> str:
    try:
        if isinstance(stat, dict):
            return f"{float(stat['point']):+.3f} [{float(stat['lo']):+.3f}-{float(stat['hi']):+.3f}]"
        return f"{float(stat[0]):+.3f} [{float(stat[1]):+.3f}-{float(stat[2]):+.3f}]"
    except (TypeError, KeyError, IndexError, ValueError):
        return "n/a"


def _read_ids_file(path):
    ids = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cid = line.strip()
            if cid:
                ids.append(cid)
    return ids


def _pilot_table(result, order):
    header = ("| condition | n_dialogues | distinct_turn | distinct_reading |"
              " rel_min | rel_max | face_min | face_max | z_ge2 | z_le_minus2 |"
              " z_max | parse_fail | icc1 | sb3 | sb5 | conv_rho |"
              " usd_per_dialogue | mean_completion_tokens | C1 [CI] | C6diff [CI] |")
    sep = ("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lines = [header, sep]
    for name in order:
        entry = result.get(name, {})
        res = entry.get("resolution", {})
        cost = entry.get("cost", entry.get("board_cost", {}))
        conv = entry.get("convergent")
        c1 = entry.get("C1", {})
        diff = entry.get("C6diff", {})
        try:
            nd = int(res.get("n_dialogues", 0))
        except (TypeError, ValueError):
            nd = 0
        def _f(key, fmt=".3f"):
            try:
                return format(float(res.get(key, 0.0)), fmt)
            except (TypeError, ValueError):
                return "n/a"
        try:
            dt = int(res.get("distinct_turn_values", 0))
        except (TypeError, ValueError):
            dt = 0
        try:
            dr = int(res.get("distinct_reading_values", 0))
        except (TypeError, ValueError):
            dr = 0
        if conv is None:
            rho_txt = "--"
        else:
            try:
                rho_txt = f"{float(conv.get('rho', 0.0)):+.3f}"
            except (TypeError, ValueError):
                rho_txt = "n/a"
        try:
            upd = float(cost.get("usd_per_dialogue", 0.0))
            upd_txt = f"{upd:.4f}"
        except (TypeError, ValueError):
            upd_txt = "n/a"
        try:
            mct = float(cost.get("mean_completion_tokens", 0.0))
            mct_txt = f"{mct:.1f}"
        except (TypeError, ValueError):
            mct_txt = "n/a"
        lines.append(
            f"| {name} | {nd} | {dt} | {dr} | {_f('rel_min_share')} |"
            f" {_f('rel_max_share')} | {_f('face_min_share')} | {_f('face_max_share')} |"
            f" {_f('z_ge2_share')} | {_f('z_le_minus2_share')} | {_f('z_max')} |"
            f" {_f('parse_fail_rate')} | {_f('icc1')} | {_f('sb3')} | {_f('sb5')} |"
            f" {rho_txt} | {upd_txt} | {mct_txt} | {_fmt_ci(c1)} | {_fmt_ci(diff)} |"
        )
    return lines


def _residual_md(res, meta):
    lines = []
    lines.append("# Residual test and power")
    lines.append("")
    info = res.get("info", {})
    try:
        mu = float(info.get("mu", 0.0))
        sd = float(info.get("sd", 0.0))
    except (TypeError, ValueError):
        mu, sd = 0.0, 0.0
    lines.append(
        f"fit={meta.get('fit','')} eval={meta.get('eval','')}"
        f" casino={meta.get('casino','')} b={meta.get('b',0)}"
        f" b_r2={meta.get('b_r2',0)} seed={meta.get('seed',0)}"
        f" same_scale={bool(meta.get('same_scale', False))}"
    )
    lines.append(
        f"mu={mu:+.3f} sd={sd:.3f} n_dialogues={res.get('n_dialogues',0)}"
        f" n_rows={res.get('n_rows',0)}"
    )
    lines.append("")
    lines.append("| metric | value [CI] | verdict | n |")
    lines.append("|---|---|---|---|")
    stats = res.get("stats", {})
    verdicts = res.get("verdicts", {})
    n_rows = res.get("n_rows", 0)
    order = ["C1", "C6_tag", "C6diff", "R1", "R2", "R3_beta", "R3_dr2",
             "R4", "dC1", "R1_cmp"]
    for key in order:
        if key not in stats:
            continue
        lines.append(f"| {key} | {_fmt_ci(stats[key])} |"
                     f" {verdicts.get(key, 'REPORTED')} | {n_rows} |")
    for key in sorted(stats):
        if key in order:
            continue
        lines.append(f"| {key} | {_fmt_ci(stats[key])} |"
                     f" {verdicts.get(key, 'REPORTED')} | {n_rows} |")
    lines.append("")
    lines.append("## Power")
    lines.append("")
    lines.append("| metric | se | n80 (d=0.03) | n80 (d=0.06) | n90 (d=0.03) | n90 (d=0.06) |")
    lines.append("|---|---|---|---|---|---|")
    power = res.get("power", {})
    for key in ("C6diff", "R1", "dC1"):
        if key not in power:
            continue
        entry = power[key]
        try:
            se = float(entry.get("se", 0.0))
            se_txt = f"{se:.4f}"
        except (TypeError, ValueError):
            se_txt = "n/a"
        try:
            n80 = entry.get("n80", {})
            n90 = entry.get("n90", {})
            a = int(n80.get("0.03", 0))
            b_ = int(n80.get("0.06", 0))
            c = int(n90.get("0.03", 0))
            d = int(n90.get("0.06", 0))
        except (TypeError, ValueError, AttributeError):
            a = b_ = c = d = 0
        lines.append(f"| {key} | {se_txt} | {a} | {b_} | {c} | {d} |")
    lines.append("")
    return lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-scale")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("pilot")
    pp.add_argument("--fit", required=True)
    pp.add_argument("--casino", required=True)
    pp.add_argument("--ids-file", required=True)
    pp.add_argument("--cond", action="append", required=True)
    pp.add_argument("--base", required=True)
    pp.add_argument("--out", required=True)
    pp.add_argument("--b", type=int, default=2000)
    pp.add_argument("--seed", type=int, default=0)
    rp = sub.add_parser("residual")
    rp.add_argument("--fit", required=True)
    rp.add_argument("--casino", required=True)
    rp.add_argument("--eval", required=True)
    rp.add_argument("--compare", default=None)
    rp.add_argument("--ids-file", default=None)
    rp.add_argument("--same-scale", action="store_true", default=False)
    rp.add_argument("--out", required=True)
    rp.add_argument("--b", type=int, default=2000)
    rp.add_argument("--b-r2", type=int, default=1000, dest="b_r2")
    rp.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.cmd == "pilot":
        ids = _read_ids_file(args.ids_file)
        fit_records = load_dir(args.fit)
        outcomes = O.load_outcomes(args.casino)
        conditions = {}
        calls_paths = {}
        order = []
        for spec in (args.cond or []):
            if "=" not in spec:
                ap.error(f"bad --cond {spec!r}, want NAME=DIR")
            name, _, d = spec.partition("=")
            name = name.strip()
            d = d.strip()
            if not name or not d:
                ap.error(f"bad --cond {spec!r}, want NAME=DIR")
            conditions[name] = load_dir(d, ids=ids)
            calls_paths[name] = str(Path(d) / "calls.jsonl")
            if name not in order:
                order.append(name)
        if args.base not in conditions:
            ap.error(f"unknown --base {args.base!r}")
        result = pilot(conditions, args.base, outcomes, fit_records,
                       calls_paths, ids, b=args.b, seed=args.seed)
        payload = {"fit": str(args.fit), "casino": str(args.casino),
                   "ids_file": str(args.ids_file), "base": str(args.base),
                   "b": int(args.b), "seed": int(args.seed),
                   "conditions": result}
        out_prefix = str(args.out)
        Path(f"{out_prefix}.json").parent.mkdir(parents=True, exist_ok=True)
        with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        table = _pilot_table(result, order)
        md_lines = ["# Scale pilot", "",
                    f"fit={args.fit} casino={args.casino} base={args.base}"
                    f" b={args.b} seed={args.seed}", ""] + table + [""]
        with open(f"{out_prefix}.md", "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        for line in table:
            print(line)
        return
    ids = _read_ids_file(args.ids_file) if args.ids_file else None
    fit_records = load_dir(args.fit)
    eval_records = load_dir(args.eval, ids=ids)
    compare_records = load_dir(args.compare, ids=ids) if args.compare else None
    outcomes = O.load_outcomes(args.casino)
    res = analyze_eval(eval_records, outcomes, fit_records, compare_records,
                       b=args.b, b_r2=args.b_r2, seed=args.seed,
                       same_scale=bool(args.same_scale))
    payload = {"fit": str(args.fit), "eval": str(args.eval),
               "compare": str(args.compare) if args.compare else None,
               "casino": str(args.casino),
               "ids_file": str(args.ids_file) if args.ids_file else None,
               "same_scale": bool(args.same_scale),
               "b": int(args.b), "b_r2": int(args.b_r2),
               "seed": int(args.seed)}
    payload.update(res)
    out_prefix = str(args.out)
    Path(f"{out_prefix}.json").parent.mkdir(parents=True, exist_ok=True)
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    meta = {"fit": str(args.fit), "eval": str(args.eval),
            "casino": str(args.casino), "b": int(args.b),
            "b_r2": int(args.b_r2), "seed": int(args.seed),
            "same_scale": bool(args.same_scale)}
    md_lines = _residual_md(res, meta)
    with open(f"{out_prefix}.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    for line in md_lines:
        print(line)
