"""L7 commentary: one-line Go-broadcast lines on pivotal turns."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from .board import _value_of, actual_class

LABEL_KO = {"!": "묘수", "?": "실수", "!?": "특이수"}
LABEL_EN = {"!": "strong move", "?": "mistake", "!?": "unusual move"}

_YEAR_RE = re.compile(r"\b\d{4}\b")
_QUOTE_RES = (
    re.compile(r'"([^"]+)"'),
    re.compile(r"“([^”]+)”"),
    re.compile(r"‘([^’]+)’"),
    re.compile(r"『([^』]+)』"),
    re.compile(r"「([^」]+)」"),
)


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


def turn_values(item, norm) -> list[dict]:
    mu = float(norm.get("mu", 0.0)) if isinstance(norm, dict) else 0.0
    try:
        sd = float(norm.get("sd", 0.0)) if isinstance(norm, dict) else 0.0
    except (TypeError, ValueError):
        sd = 0.0
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
        out.append({
            "index": idx,
            "speaker": speaker if isinstance(speaker, str) else str(speaker),
            "text": text if isinstance(text, str) else str(text),
            "moves": moves,
            "value": float(value),
            "z": float(z),
            "disagreement": float(dis),
        })
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


def mark_for(z, surprise) -> str | None:
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


def pick_turns(item, norm, markov=None, k=3) -> list[dict]:
    tv = turn_values(item, norm)
    curves = side_curves(item, norm)
    cands: list[dict] = []
    for pos, entry in enumerate(tv):
        cur_class = actual_class(list(entry.get("moves", []) or []))
        if pos == 0:
            prev_class = "proceed"
        else:
            prev_class = actual_class(list(tv[pos - 1].get("moves", []) or []))
        sur = surprise(prev_class, cur_class, markov)
        mark = mark_for(entry.get("z", 0.0), sur)
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
        cands.append(rec)
    try:
        kk = int(k)
    except (TypeError, ValueError):
        kk = 3
    kk = max(0, kk)
    ranked = sorted(cands, key=lambda e: (-abs(float(e.get("z", 0.0))), int(e.get("index", 0))))
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


def choose_reference(turn, refs, used=()) -> dict:
    refs = list(refs or [])
    if not refs:
        raise ValueError("empty references")
    used_set = set(used or ())
    mark = turn.get("mark") if isinstance(turn, dict) else None
    tmoves = set(turn.get("moves", []) or []) if isinstance(turn, dict) else set()
    scored: list[tuple] = []
    for i, r in enumerate(refs):
        if not isinstance(r, dict):
            continue
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
    for i, r in enumerate(refs):
        if not isinstance(r, dict):
            continue
        if mark not in (r.get("marks") or []):
            continue
        if "*" not in (r.get("moves") or []):
            continue
        fall.append((r, i))
    if fall:
        fall.sort(key=lambda x: ((x[0].get("id") in used_set), x[1]))
        return fall[0][0]
    any_mark = [(r, i) for i, r in enumerate(refs)
                if isinstance(r, dict) and mark in (r.get("marks") or [])]
    if any_mark:
        any_mark.sort(key=lambda x: ((x[0].get("id") in used_set), x[1]))
        return any_mark[0][0]
    return refs[0]


def template_line(turn, ref) -> dict:
    idx = turn.get("index", 0) if isinstance(turn, dict) else 0
    speaker = turn.get("speaker", "") if isinstance(turn, dict) else ""
    label_ko = turn.get("label_ko", "") if isinstance(turn, dict) else ""
    label_en = turn.get("label_en", "") if isinstance(turn, dict) else ""
    mark = turn.get("mark", "") if isinstance(turn, dict) else ""
    moves = turn.get("moves", []) or [] if isinstance(turn, dict) else []
    try:
        before = float(turn.get("curve_before", 0.0))
    except (TypeError, ValueError):
        before = 0.0
    try:
        after = float(turn.get("curve_after", 0.0))
    except (TypeError, ValueError):
        after = 0.0
    if moves:
        words = ", ".join(moves)
        words_ko = words
        words_en = words
    else:
        words_ko = "수"
        words_en = "a move"
    idea_ko = ref.get("idea_ko", "") if isinstance(ref, dict) else ""
    idea_en = ref.get("idea_en", "") if isinstance(ref, dict) else ""
    ko = (f"{idx}수 {speaker} {label_ko}({mark}) — {words_ko} 뒤 형세 "
          f"{before:+.1f}→{after:+.1f}. {idea_ko}")
    en = (f"Turn {idx}, {speaker}: {label_en} ({mark}) — after {words_en}, "
          f"position {before:+.1f}→{after:+.1f}. {idea_en}")
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
    speaker = turn.get("speaker", "") if isinstance(turn, dict) else ""
    mark = turn.get("mark", "") if isinstance(turn, dict) else ""
    label_ko = turn.get("label_ko", "") if isinstance(turn, dict) else ""
    label_en = turn.get("label_en", "") if isinstance(turn, dict) else ""
    moves = turn.get("moves", []) or [] if isinstance(turn, dict) else []
    try:
        before = float(turn.get("curve_before", 0.0))
    except (TypeError, ValueError):
        before = 0.0
    try:
        after = float(turn.get("curve_after", 0.0))
    except (TypeError, ValueError):
        after = 0.0
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
        "books, papers or years. "
        'Reply with JSON only: {"ko": "...", "en": "..."}.'
    )
    user = (
        f"Turn {idx} by {speaker}: moves {move_txt}, mark {mark} "
        f"({label_en}/{label_ko}), position {before:+.1f}->{after:+.1f}.\n"
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
    curves = side_curves(item, norm)
    picked = pick_turns(item, norm, markov, k)
    used: list = []
    turns_out: list[dict] = []
    item_id = item.get("id", "") if isinstance(item, dict) else ""
    for turn in picked:
        ref = choose_reference(turn, refs_list, used)
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
    return {"id": item_id, "norm": {"mu": mu, "sd": sd},
            "curves": curves, "turns": turns_out}


def main(argv=None) -> None:
    from . import calibrate as C
    from . import predict as P

    ap = argparse.ArgumentParser(prog="uptake-comment")
    ap.add_argument("--item", required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--llm", action="store_true", default=False)
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--model", default="muse-spark-1.3-contributor")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    with open(args.item, encoding="utf-8") as f:
        item = json.load(f)
    fit_items = C._load_items(Path(args.fit))
    genre = item.get("genre", "") if isinstance(item, dict) else ""
    sel = {cid: d for cid, d in fit_items.items()
           if isinstance(d, dict) and d.get("genre") == genre}
    eff_lists = [d.get("effects", []) or [] for d in sel.values()]
    norm = C.fit_board_norm(eff_lists, method="z")
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
