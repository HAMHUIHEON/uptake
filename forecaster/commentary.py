"""L7 commentary: one-line Go-broadcast lines on pivotal turns."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .board import _value_of, actual_class

LABEL_KO = {"!": "호수(좋은 수)", "?": "실수", "!?": "특이수",
            "!!": "묘수", "??": "악수"}
LABEL_EN = {"!": "good", "?": "mistake", "!?": "interesting",
            "!!": "brilliant", "??": "blunder"}

_YEAR_RE = re.compile(r"\b\d{4}\b")
_QUOTE_RES = (
    re.compile(r'"([^"]+)"'),
    re.compile(r"“([^”]+)”"),
    re.compile(r"‘([^’]+)’"),
    re.compile(r"『([^』]+)』"),
    re.compile(r"「([^」]+)」"),
)


def rel_mean_of(effect) -> float:
    """Mean relationship reading (rel_norm scale) behind one effect.

    Prefers the aggregated ``relationship`` mean, falls back to the mean of
    the readings' ``rel_norm``, then to the plain effect value (synthetic
    records that carry only ``value``).
    """
    if not isinstance(effect, dict):
        return 0.0
    rel = effect.get("relationship")
    if isinstance(rel, (list, tuple)) and len(rel) >= 1:
        try:
            return float(rel[0])
        except (TypeError, ValueError):
            pass
    readings = effect.get("readings")
    if isinstance(readings, list) and readings:
        vals: list[float] = []
        for r in readings:
            if isinstance(r, dict) and "rel_norm" in r:
                try:
                    vals.append(float(r["rel_norm"]))
                except (TypeError, ValueError):
                    pass
        if vals:
            return sum(vals) / len(vals)
    try:
        return float(_value_of(effect))
    except (TypeError, ValueError):
        return 0.0


def standing_of(rel_norm) -> float:
    """Side standing (잔고) on the 0–100 scale: 50 + 25·rel_norm."""
    try:
        return 50.0 + 25.0 * float(rel_norm)
    except (TypeError, ValueError):
        return 50.0


STANDING_SCALE_PAIRWISE = "pairwise→0-100 via z"


def pairwise_standing(value, mu, sd) -> float:
    """Side standing for a BT rating: 50 + 25·(value − mu)/sd, clipped 0–100."""
    try:
        z = (float(value) - float(mu)) / float(sd)
    except (TypeError, ValueError, ZeroDivisionError):
        return 50.0
    try:
        return max(0.0, min(100.0, 50.0 + 25.0 * float(z)))
    except (TypeError, ValueError):
        return 50.0


def _avg_ranks(vals) -> list[float]:
    """Average ranks (1-based) of a value list, ties averaged."""
    seq = list(vals) if vals is not None else []
    n = len(seq)
    order = sorted(range(n), key=lambda i: seq[i])
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and seq[order[j + 1]] == seq[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            out[order[k]] = float(avg)
        i = j + 1
    return out


def display_name_of(turn) -> str:
    """Speaker label for a turn: display name when provided, else the label.

    Never maps a side to 흑/백; the raw speaker label ("A", "B", ...) is used.
    """
    if not isinstance(turn, dict):
        return ""
    dn = turn.get("display_name")
    if isinstance(dn, str) and dn.strip():
        return dn.strip()
    sp = turn.get("speaker", "")
    return sp if isinstance(sp, str) else str(sp)


def _record_value_stats(item) -> tuple[float, float]:
    """Mean/population-sd of a record's raw effect values."""
    effects = item.get("effects", []) if isinstance(item, dict) else []
    vals: list[float] = []
    for e in effects or []:
        if not isinstance(e, dict):
            continue
        try:
            vals.append(float(_value_of(e)))
        except (TypeError, ValueError):
            continue
    if not vals:
        return (0.0, 0.0)
    mu = sum(vals) / len(vals)
    var = sum((v - mu) ** 2 for v in vals) / len(vals)
    return (float(mu), float(var ** 0.5) if var > 0 else 0.0)


def standing_pairs(item, norm=None) -> list[tuple[float, float]]:
    """Per-effect (before, after) standings in record order.

    Each side starts at 50; a side's standing after one of its own turns is
    50 + 25·rel_norm of that turn, carried forward until its next turn.
    Pairwise records carry BT ratings instead of readings, so their standing
    is 50 + 25·(value − mu)/sd clipped to 0–100, with mu/sd from the fit
    norm (or the record itself when no norm is given).
    """
    effects = item.get("effects", []) if isinstance(item, dict) else []
    pairwise_item = isinstance(item, dict) and item.get("scale") == "pairwise"
    mu = sd = None
    if pairwise_item:
        if isinstance(norm, dict) and "mu" in norm and "sd" in norm:
            try:
                mu = float(norm.get("mu", 0.0))
            except (TypeError, ValueError):
                mu = 0.0
            try:
                sd = float(norm.get("sd", 0.0))
            except (TypeError, ValueError):
                sd = 0.0
        else:
            mu, sd = _record_value_stats(item)
    last: dict[str, float] = {}
    out: list[tuple[float, float]] = []
    for e in effects or []:
        if not isinstance(e, dict):
            continue
        speaker = e.get("speaker", "")
        speaker = speaker if isinstance(speaker, str) else str(speaker)
        if pairwise_item:
            try:
                raw = float(_value_of(e))
            except (TypeError, ValueError):
                raw = 0.0
            after = pairwise_standing(raw, mu, sd)
        else:
            after = standing_of(rel_mean_of(e))
        before = last.get(speaker, 50.0)
        last[speaker] = after
        out.append((float(before), float(after)))
    return out


def load_references(path=None) -> list[dict]:
    if path is None:
        path = Path(__file__).with_name("references.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return list(data.get("references", []) or [])


def _moves_by_index(item) -> dict[int, list]:
    by_index: dict[int, list] = {}
    moves = item.get("moves", []) if isinstance(item, dict) else []
    for t in moves or []:
        if isinstance(t, dict):
            idx = t.get("turn_index")
            mv = t.get("moves", []) or []
        else:
            idx = getattr(t, "turn_index", None)
            mv = list(getattr(t, "moves", []) or [])
        if isinstance(idx, int) and idx not in by_index:
            by_index[idx] = list(mv or [])
    return by_index


def _tagv_of(moves, table) -> float:
    if not moves:
        return 0.0
    tot = 0.0
    for mv in moves:
        if isinstance(mv, str):
            try:
                tot += float(table.get(mv, 0.0))
            except (TypeError, ValueError, AttributeError):
                tot += 0.0
        else:
            tot += 0.0
    return tot / len(moves)


def _rank_entries(item, norm) -> list[dict] | None:
    """Rank-based turn values for cross-scale reads (--same-scale).

    Both the record's values and the table's per-turn predictions are
    converted to within-record fractional ranks; residuals live in that
    rank space, so a pairwise record can be read against a 0–100 table
    and vice versa.
    """
    table = norm.get("table") if isinstance(norm, dict) else None
    tbl = table if isinstance(table, dict) else {}
    by_index = _moves_by_index(item)
    effects = [e for e in (item.get("effects", [])
                           if isinstance(item, dict) else []) or []
               if isinstance(e, dict)]
    n = len(effects)
    if n == 0:
        return []
    obs: list[float] = []
    preds: list[float] = []
    metas: list[tuple] = []
    for pos, e in enumerate(effects):
        idx = e.get("index", pos)
        idx = idx if isinstance(idx, int) else pos
        speaker = e.get("speaker", "")
        text = e.get("text", "")
        moves = list(by_index.get(idx, []))
        try:
            obs.append(float(_value_of(e)))
        except (TypeError, ValueError):
            obs.append(0.0)
        preds.append(float(_tagv_of(moves, tbl)))
        try:
            dis = float(e.get("disagreement", 0.0))
        except (TypeError, ValueError):
            dis = 0.0
        metas.append((idx, speaker if isinstance(speaker, str) else str(speaker),
                      text if isinstance(text, str) else str(text),
                      moves, dis))
    qo = [(r - 0.5) / n for r in _avg_ranks(obs)]
    qp = [(r - 0.5) / n for r in _avg_ranks(preds)]
    resids = [a - b for a, b in zip(qo, qp)]
    mean_r = sum(resids) / n
    var_r = sum((v - mean_r) ** 2 for v in resids) / n
    sd_r = var_r ** 0.5 if var_r > 0 else 0.0
    if sd_r == 0:
        sd_r = 1.0
    mean_q = sum(qo) / n
    var_q = sum((v - mean_q) ** 2 for v in qo) / n
    sd_q = var_q ** 0.5 if var_q > 0 else 0.0
    out: list[dict] = []
    for (idx, speaker, text, moves, dis), a, b, r in zip(metas, qo, qp, resids):
        out.append({
            "index": idx,
            "speaker": speaker,
            "text": text,
            "moves": moves,
            "value": float(a - 0.5),
            "z": 0.0 if sd_q == 0 else float((a - mean_q) / sd_q),
            "disagreement": float(dis),
            "tagv": float(b - 0.5),
            "resid": float(r),
            "z_resid": float(r / sd_r),
        })
    return out


def turn_values(item, norm) -> list[dict]:
    if isinstance(norm, dict) and norm.get("rank_mode"):
        ranked = _rank_entries(item, norm)
        return ranked if ranked is not None else []
    mu = float(norm.get("mu", 0.0)) if isinstance(norm, dict) else 0.0
    try:
        sd = float(norm.get("sd", 0.0)) if isinstance(norm, dict) else 0.0
    except (TypeError, ValueError):
        sd = 0.0
    table = norm.get("table") if isinstance(norm, dict) else None
    use_resid = (isinstance(table, dict) and isinstance(norm, dict)
                 and "sd_resid" in norm)
    if use_resid:
        try:
            sd_resid = float(norm.get("sd_resid", 0.0))
        except (TypeError, ValueError):
            sd_resid = 0.0
        from .valuemodel import turn_residual
    by_index = _moves_by_index(item)
    effects = item.get("effects", []) if isinstance(item, dict) else []
    out: list[dict] = []
    for pos, e in enumerate(effects or []):
        if not isinstance(e, dict):
            continue
        idx = e.get("index", pos)
        idx = idx if isinstance(idx, int) else pos
        speaker = e.get("speaker", "")
        text = e.get("text", "")
        moves = list(by_index.get(idx, []))
        try:
            value = float(_value_of(e)) - mu
        except (TypeError, ValueError):
            value = 0.0 - mu
        z = 0.0 if sd == 0 else float(value) / float(sd)
        try:
            dis = float(e.get("disagreement", 0.0))
        except (TypeError, ValueError):
            dis = 0.0
        entry = {
            "index": idx,
            "speaker": speaker if isinstance(speaker, str) else str(speaker),
            "text": text if isinstance(text, str) else str(text),
            "moves": moves,
            "value": float(value),
            "z": float(z),
            "disagreement": float(dis),
        }
        if use_resid:
            if not moves:
                tagv = 0.0
            else:
                tot = 0.0
                for mv in moves:
                    if isinstance(mv, str):
                        try:
                            tot += float(table.get(mv, 0.0))
                        except (TypeError, ValueError, AttributeError):
                            tot += 0.0
                    else:
                        tot += 0.0
                tagv = tot / len(moves)
            r = turn_residual(e, idx, by_index, mu, table)
            resid = 0.0 if r is None else float(r)
            z_resid = 0.0 if sd_resid == 0 else float(resid) / float(sd_resid)
            entry["tagv"] = float(tagv)
            entry["resid"] = float(resid)
            entry["z_resid"] = float(z_resid)
        out.append(entry)
    return out


def side_curves(item, norm) -> dict:
    mu = float(norm.get("mu", 0.0)) if isinstance(norm, dict) else 0.0
    effects = item.get("effects", []) if isinstance(item, dict) else []
    a_curve: list[float] = []
    b_curve: list[float] = []
    bal: list[float] = []
    cum_a = 0.0
    cum_b = 0.0
    for e in effects or []:
        if not isinstance(e, dict):
            a_curve.append(float(cum_a))
            b_curve.append(float(cum_b))
            bal.append(float(cum_a - cum_b))
            continue
        try:
            v = float(_value_of(e)) - mu
        except (TypeError, ValueError):
            v = 0.0 - mu
        speaker = e.get("speaker", "")
        if speaker == "A":
            cum_a += v
        elif speaker == "B":
            cum_b += v
        a_curve.append(float(cum_a))
        b_curve.append(float(cum_b))
        bal.append(float(cum_a - cum_b))
    return {"A": a_curve, "B": b_curve, "balance": bal}


def mark_for(z, surprise, z_resid=None) -> str | None:
    if z_resid is not None:
        try:
            zf = float(z_resid)
        except (TypeError, ValueError):
            return None
        if zf >= 2.0:
            return "!!"
        if zf >= 1.0:
            return "!"
        if zf <= -2.0:
            return "??"
        if zf <= -1.0:
            return "?"
        if surprise:
            return "!?"
        return None
    try:
        zf = float(z)
    except (TypeError, ValueError):
        return None
    if zf >= 1.0:
        return "!"
    if zf <= -1.0:
        return "?"
    if surprise:
        return "!?"
    return None


def surprise(prev_class, cur_class, markov, threshold=0.05) -> bool:
    if markov is None:
        return False
    try:
        return float(markov[prev_class][cur_class]) < float(threshold)
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def _effective_norm(item, norm):
    """Norm for curves/standings; rank mode centers on the record itself.

    In rank mode the table's mu/sd live on the other scale, so curves and
    pairwise standings use the record's own value mean/sd instead. The
    returned dict keeps ``rank_mode`` so turn_values stays in rank space.
    """
    if not (isinstance(norm, dict) and norm.get("rank_mode")):
        return norm
    mu_r, sd_r = _record_value_stats(item)
    eff = dict(norm)
    eff["mu"] = float(mu_r)
    eff["sd"] = float(sd_r)
    return eff


def pick_turns(item, norm, markov=None, k=3) -> list[dict]:
    tv = turn_values(item, norm)
    eff = _effective_norm(item, norm)
    curves = side_curves(item, eff)
    stands = standing_pairs(item, eff)
    cands: list[dict] = []
    for pos, entry in enumerate(tv):
        cur_class = actual_class(list(entry.get("moves", []) or []))
        if pos == 0:
            prev_class = "proceed"
        else:
            prev_class = actual_class(list(tv[pos - 1].get("moves", []) or []))
        sur = surprise(prev_class, cur_class, markov)
        mark = mark_for(entry.get("z", 0.0), sur, entry.get("z_resid"))
        if mark is None:
            continue
        rec = dict(entry)
        rec["_pos"] = pos
        rec["mark"] = mark
        rec["label_ko"] = LABEL_KO[mark]
        rec["label_en"] = LABEL_EN[mark]
        speaker = entry.get("speaker", "")
        if speaker in ("A", "B"):
            side = curves.get(speaker, [])
            after = float(side[pos]) if pos < len(side) else 0.0
            before = float(side[pos - 1]) if pos > 0 and pos - 1 < len(side) else 0.0
        else:
            before, after = 0.0, 0.0
        rec["curve_before"] = float(before)
        rec["curve_after"] = float(after)
        if pos < len(stands):
            rec["standing_before"] = float(stands[pos][0])
            rec["standing_after"] = float(stands[pos][1])
        else:
            rec["standing_before"] = 50.0
            rec["standing_after"] = 50.0
        cands.append(rec)
    try:
        kk = int(k)
    except (TypeError, ValueError):
        kk = 3
    kk = max(0, kk)

    def _rank_val(e):
        zr = e.get("z_resid")
        if zr is None:
            zr = e.get("z", 0.0)
        return float(zr)

    ranked = sorted(cands, key=lambda e: (-abs(_rank_val(e)), int(e.get("index", 0))))
    picked = ranked[:kk]
    picked_ids = {id(e) for e in picked}
    extra = [e for e in sorted(cands, key=lambda e: int(e.get("index", 0)))
             if e.get("mark") == "!?" and id(e) not in picked_ids]
    if extra:
        picked.append(extra[0])
    picked.sort(key=lambda e: int(e.get("index", 0)))
    for rec in picked:
        rec.pop("_pos", None)
    return picked


def _genres_of(ref) -> list:
    """Genre tags of one reference; missing tags mean negotiation-only."""
    g = ref.get("genres") if isinstance(ref, dict) else None
    if isinstance(g, list) and g:
        return [str(x) for x in g]
    return ["negotiation"]


def choose_reference(turn, refs, used=(), genre=None) -> dict:
    refs = list(refs or [])
    if not refs:
        raise ValueError("empty references")
    if genre is None and isinstance(turn, dict):
        genre = turn.get("genre")
    if not genre or not isinstance(genre, str):
        genre = "negotiation"
    pool = [r for r in refs if isinstance(r, dict)
            and (genre in _genres_of(r) or "any" in _genres_of(r))]
    if not pool:
        pool = [r for r in refs if isinstance(r, dict)
                and "any" in _genres_of(r)]
    if not pool:
        pool = [r for r in refs if isinstance(r, dict)]
    if not pool:
        return refs[0]
    used_set = set(used or ())
    mark = turn.get("mark") if isinstance(turn, dict) else None
    tmoves = set(turn.get("moves", []) or []) if isinstance(turn, dict) else set()
    scored: list[tuple] = []
    for i, r in enumerate(pool):
        if mark not in (r.get("marks") or []):
            continue
        rmoves = r.get("moves") or []
        inter = set(rmoves) & tmoves
        if not inter:
            continue
        scored.append((r, len(inter), i))
    if scored:
        scored.sort(key=lambda x: (-x[1], (x[0].get("id") in used_set), x[2]))
        return scored[0][0]
    fall: list[tuple] = []
    for i, r in enumerate(pool):
        if mark not in (r.get("marks") or []):
            continue
        if "*" not in (r.get("moves") or []):
            continue
        fall.append((r, i))
    if fall:
        fall.sort(key=lambda x: ((x[0].get("id") in used_set), x[1]))
        return fall[0][0]
    any_mark = [(r, i) for i, r in enumerate(pool)
                if mark in (r.get("marks") or [])]
    if any_mark:
        any_mark.sort(key=lambda x: ((x[0].get("id") in used_set), x[1]))
        return any_mark[0][0]
    return pool[0]


def _line_numbers(turn) -> tuple[float, float]:
    """Standings quoted in a line; legacy curve fields as fallback."""
    turn = turn if isinstance(turn, dict) else {}
    if "standing_before" in turn or "standing_after" in turn:
        raw_b, raw_a = turn.get("standing_before", 50.0), turn.get("standing_after", 50.0)
    else:
        raw_b, raw_a = turn.get("curve_before", 0.0), turn.get("curve_after", 0.0)
    try:
        before = float(raw_b)
    except (TypeError, ValueError):
        before = 50.0
    try:
        after = float(raw_a)
    except (TypeError, ValueError):
        after = 50.0
    return before, after


def template_line(turn, ref) -> dict:
    idx = turn.get("index", 0) if isinstance(turn, dict) else 0
    name = display_name_of(turn)
    label_ko = turn.get("label_ko", "") if isinstance(turn, dict) else ""
    label_en = turn.get("label_en", "") if isinstance(turn, dict) else ""
    mark = turn.get("mark", "") if isinstance(turn, dict) else ""
    moves = turn.get("moves", []) or [] if isinstance(turn, dict) else []
    before, after = _line_numbers(turn)
    if moves:
        words = ", ".join(moves)
        words_ko = words
        words_en = words
    else:
        words_ko = "수"
        words_en = "a move"
    idea_ko = ref.get("idea_ko", "") if isinstance(ref, dict) else ""
    idea_en = ref.get("idea_en", "") if isinstance(ref, dict) else ""
    ko = (f"{idx}수 {name} {label_ko}({mark}) — {words_ko} 뒤 잔고 "
          f"{before:.0f} → {after:.0f}. {idea_ko}")
    en = (f"Turn {idx}, {name}: {label_en} ({mark}) — after {words_en}, "
          f"standing {before:.0f} → {after:.0f}. {idea_en}")
    return {"ko": ko, "en": en, "source": "template"}


def _call_llm(client, messages, model, meta=None):
    if meta is None or not getattr(client, "accepts_meta", False):
        return client.complete(messages, model, 1.0)
    try:
        return client.complete(messages, model, 1.0, meta=meta)
    except TypeError:
        return client.complete(messages, model, 1.0)


def _extract_json(raw: str) -> dict:
    if not isinstance(raw, str):
        raise ValueError("no JSON object")
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
        raise ValueError("not an object")
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    return obj


def _quoted_titles(text: str) -> list[str]:
    out: list[str] = []
    for rx in _QUOTE_RES:
        for m in rx.finditer(text or ""):
            s = (m.group(1) or "").strip()
            if s:
                out.append(s)
    return out


def llm_line(client, turn, ref, model, meta=None) -> dict:
    def _fallback():
        t = template_line(turn, ref)
        t["source"] = "template_fallback"
        return t

    idx = turn.get("index", 0) if isinstance(turn, dict) else 0
    name = display_name_of(turn)
    mark = turn.get("mark", "") if isinstance(turn, dict) else ""
    label_ko = turn.get("label_ko", "") if isinstance(turn, dict) else ""
    label_en = turn.get("label_en", "") if isinstance(turn, dict) else ""
    moves = turn.get("moves", []) or [] if isinstance(turn, dict) else []
    before, after = _line_numbers(turn)
    author = ref.get("author", "") if isinstance(ref, dict) else ""
    title = ref.get("title", "") if isinstance(ref, dict) else ""
    year = ref.get("year", "") if isinstance(ref, dict) else ""
    idea_en = ref.get("idea_en", "") if isinstance(ref, dict) else ""
    idea_ko = ref.get("idea_ko", "") if isinstance(ref, dict) else ""
    move_txt = ", ".join(moves) if moves else "a move"
    system = (
        "You are a Go/baduk TV commentator. Write one short broadcast-style line "
        "per language (Korean and English), each at most 120 characters, describing "
        "what the move did to the position and why, using ONLY the given reference "
        "idea. You may name the reference author in short form; no other names, "
        "books, papers or years. When you mention the score, quote the standing "
        "on the 0-100 scale (for example 잔고 53 → 45), never a cumulative value. "
        "Never call a side black/white (흑/백); use the speaker label. "
        'Reply with JSON only: {"ko": "...", "en": "..."}.'
    )
    user = (
        f"Turn {idx} by {name}: moves {move_txt}, mark {mark} "
        f"({label_en}/{label_ko}), standing {before:.0f} -> {after:.0f} "
        f"on the 0-100 scale (each side starts at 50).\n"
        f"Reference: {author} ({year}), {title}. "
        f"Idea EN: {idea_en} Idea KO: {idea_ko}\n"
        "Write one line per language as JSON."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    try:
        raw = _call_llm(client, messages, model, meta)
    except Exception:
        return _fallback()
    try:
        obj = _extract_json(raw)
    except (ValueError, json.JSONDecodeError):
        return _fallback()
    ko = obj.get("ko")
    en = obj.get("en")
    if not isinstance(ko, str) or not ko.strip():
        return _fallback()
    if not isinstance(en, str) or not en.strip():
        return _fallback()
    if len(ko) > 160 or len(en) > 160:
        return _fallback()
    allowed = set(_YEAR_RE.findall(str(year)))
    for y in _YEAR_RE.findall(ko) + _YEAR_RE.findall(en):
        if y not in allowed:
            return _fallback()
    ref_title = str(title or "")
    for q in _quoted_titles(ko) + _quoted_titles(en):
        if q != ref_title:
            return _fallback()
    return {"ko": ko, "en": en, "source": "llm"}


def comment(item, norm, markov=None, k=3, client=None,
            model="muse-spark-1.3-contributor", refs=None) -> dict:
    refs_list = list(refs) if refs is not None else load_references()
    eff = _effective_norm(item, norm)
    curves = side_curves(item, eff)
    picked = pick_turns(item, eff, markov, k)
    used: list = []
    turns_out: list[dict] = []
    item_id = item.get("id", "") if isinstance(item, dict) else ""
    item_genre = item.get("genre", "") if isinstance(item, dict) else ""
    if not item_genre or not isinstance(item_genre, str):
        item_genre = "negotiation"
    for turn in picked:
        ref = choose_reference(turn, refs_list, used, genre=item_genre)
        used.append(ref.get("id"))
        if client is None:
            line = template_line(turn, ref)
        else:
            meta = {"layer": "commentary", "item": item_id,
                    "turn": turn.get("index", 0)}
            try:
                line = llm_line(client, turn, ref, model, meta=meta)
            except Exception:
                line = template_line(turn, ref)
                line["source"] = "template_fallback"
        entry = dict(turn)
        entry["reference"] = {
            "id": ref.get("id"),
            "author": ref.get("author"),
            "title": ref.get("title"),
            "year": ref.get("year"),
            "venue": ref.get("venue"),
        }
        entry["line_ko"] = line.get("ko")
        entry["line_en"] = line.get("en")
        entry["source"] = line.get("source")
        turns_out.append(entry)
    if isinstance(norm, dict):
        try:
            mu = float(norm.get("mu", 0.0))
        except (TypeError, ValueError):
            mu = 0.0
        try:
            sd = float(norm.get("sd", 0.0))
        except (TypeError, ValueError):
            sd = 0.0
    else:
        mu, sd = 0.0, 0.0
    norm_out: dict = {"mu": mu, "sd": sd}
    if isinstance(norm, dict) and "scale" in norm:
        norm_out["scale"] = norm["scale"]
    if isinstance(norm, dict) and "table" in norm:
        norm_out["table"] = norm["table"]
    if isinstance(norm, dict) and "sd_resid" in norm:
        norm_out["sd_resid"] = norm["sd_resid"]
    if isinstance(norm, dict) and norm.get("rank_mode"):
        norm_out["rank_mode"] = True
    result: dict = {"id": item_id, "norm": norm_out,
                    "curves": curves, "turns": turns_out}
    if isinstance(item, dict) and item.get("scale") == "pairwise":
        result["standing_scale"] = STANDING_SCALE_PAIRWISE
    if isinstance(norm, dict) and norm.get("scale_note"):
        result["scale_note"] = norm["scale_note"]
    return result


def _load_fit_table(fit_dir) -> dict | None:
    """Precomputed fit table from DIR/table.json, or None when absent."""
    path = Path(fit_dir) / "table.json"
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"bad fit table {path}: {exc}")
    if not isinstance(data, dict) or "scale" not in data:
        raise SystemExit(f"bad fit table {path}: need an object with 'scale'")
    try:
        mu = float(data.get("mu", 0.0))
    except (TypeError, ValueError):
        mu = 0.0
    try:
        sd = float(data.get("sd", 0.0))
    except (TypeError, ValueError):
        sd = 0.0
    try:
        sd_resid = float(data.get("sd_resid", 1.0))
    except (TypeError, ValueError):
        sd_resid = 1.0
    raw_table = data.get("table", {})
    table: dict[str, float] = {}
    if isinstance(raw_table, dict):
        for key, val in raw_table.items():
            if isinstance(val, dict):
                val = val.get("mean", 0.0)
            try:
                table[str(key)] = float(val)
            except (TypeError, ValueError):
                table[str(key)] = 0.0
    return {"mu": mu, "sd": sd, "scale": data.get("scale", "v2"),
            "table": table, "sd_resid": sd_resid}


def main(argv=None) -> None:
    from . import calibrate as C
    from . import outcome as O
    from . import predict as P

    ap = argparse.ArgumentParser(prog="uptake-comment")
    ap.add_argument("--item", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--llm", action="store_true", default=False)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--model", default="muse-spark-1.3-contributor")
    ap.add_argument("--same-scale", action="store_true", default=False,
                    help="allow a pairwise table with a 0–100 record or "
                         "vice versa via rank-based residuals")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    with open(args.item, encoding="utf-8") as f:
        item = json.load(f)
    item_scale = item.get("scale", "v2") if isinstance(item, dict) else "v2"
    precomputed = _load_fit_table(args.fit)
    if precomputed is not None:
        fit_scale = precomputed["scale"]
        norm = dict(precomputed)
        fit_items = C._load_items(Path(args.fit))
        genre = item.get("genre", "") if isinstance(item, dict) else ""
        sel = {cid: d for cid, d in fit_items.items()
               if isinstance(d, dict) and d.get("genre") == genre}
    else:
        fit_items = C._load_items(Path(args.fit))
        genre = item.get("genre", "") if isinstance(item, dict) else ""
        sel = {cid: d for cid, d in fit_items.items()
               if isinstance(d, dict) and d.get("genre") == genre}
        fit_scales = sorted({d.get("scale", "v2") for d in sel.values()
                             if isinstance(d, dict)})
        if len(fit_scales) > 1:
            raise SystemExit(
                "scale mismatch: fit has mixed scales "
                + ",".join(str(s) for s in fit_scales))
        fit_scale = fit_scales[0] if fit_scales else item_scale
        eff_lists = [d.get("effects", []) or [] for d in sel.values()]
        norm = C.fit_board_norm(eff_lists, method="z")
        norm["scale"] = fit_scale
        try:
            mu_fit = float(norm.get("mu", 0.0))
        except (TypeError, ValueError):
            mu_fit = 0.0
        table = O.move_value_table(sel, mu_fit)
        resid = C.residual_norm(sel, mu_fit, table)
        norm["table"] = {str(k): float(v) for k, v in table.items()}
        norm["sd_resid"] = float(resid["sd_resid"])
    if item_scale != fit_scale:
        crosses_pairwise = (item_scale == "pairwise") != (fit_scale == "pairwise")
        if crosses_pairwise and args.same_scale:
            norm["rank_mode"] = True
            norm["scale_note"] = (
                f"rank-based residuals: item scale {item_scale} vs table "
                f"scale {fit_scale}; values and table predictions converted "
                f"to within-record ranks before computing residuals")
        else:
            hint = (" (pairwise BT ratings vs 0–100 readings; pass "
                    "--same-scale for rank-based residuals or fit a "
                    f"{item_scale} table)") if crosses_pairwise else ""
            raise SystemExit(
                f"scale mismatch: item={item_scale} fit={fit_scale}{hint}")
    fit_hits: list[dict] = []
    for cid in sorted(sel):
        try:
            fit_hits.extend(P.with_context(sel[cid]))
        except Exception:
            continue
    base = C.base_rates(fit_hits)
    markov = P.fit_markov(fit_hits, base)
    client = None
    if args.llm:
        from .reader import MuseClient
        from .usage import Meter
        client = MuseClient(meter=Meter(budget_usd=args.budget))
    result = comment(item, norm, markov, k=args.k, client=client,
                     model=args.model)
    if args.out:
        out_path = Path(args.out)
    else:
        item_id = item.get("id", "") if isinstance(item, dict) else ""
        out_path = Path(args.item).parent / f"commentary_{item_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    for t in result.get("turns", []):
        print(t.get("line_ko", ""))
        print(t.get("line_en", ""))
