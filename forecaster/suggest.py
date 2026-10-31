"""Alternative wordings for blunder turns (task55, uptake-suggest).

For each blunder turn two rewrites keep the proposition and change only
the interpersonal wording (soft / firm-but-fair, ≤ 40 words, same
language), then each rewrite is scored against the original with the
pairwise vote reused from ``forecaster.perturb`` (both orders × 3
stances = 6 votes) → p_better.

Stdlib only plus sibling modules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import pairwise as PW
from . import perturb
from .commentary import standing_round

SOFT_STYLE = "soft"
FIRM_STYLE = "firm"

# Auditable rewrite prompts (module constants). Proposition kept, only the
# interpersonal wording changes.
SOFT_PROMPT = (
    "You rewrite a single dialogue turn into a softer wording.\n"
    "Keep the propositional content exactly: the same ask, offer, numbers, "
    "names and decisions.\n"
    "Change only the interpersonal wording: acknowledge the other side, "
    "then ask (soft: acknowledge + ask).\n"
    "Write at most 40 words, in the same language as the original.\n"
    "Reply with the rewritten turn only, no quotes, no explanation."
)
FIRM_PROMPT = (
    "You rewrite a single dialogue turn into a firm but fair wording.\n"
    "Keep the propositional content exactly: the same ask, offer, numbers, "
    "names and decisions.\n"
    "Change only the interpersonal wording: state your need plainly and "
    "offer a trade (firm but fair: state need + offer a trade).\n"
    "Write at most 40 words, in the same language as the original.\n"
    "Reply with the rewritten turn only, no quotes, no explanation."
)
SUGGEST_PROMPTS = {SOFT_STYLE: SOFT_PROMPT, FIRM_STYLE: FIRM_PROMPT}

MAX_WORDS = 40


def _as_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _cap_words(text, limit=MAX_WORDS) -> str:
    words = _as_text(text).split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit])


def build_rewrite_messages(text, style, speaker="") -> list[dict]:
    """Rewrite messages for one style (soft / firm)."""
    label = speaker if isinstance(speaker, str) and speaker else "TARGET"
    key = _as_text(style).strip().lower()
    if key in ("firm", "firm but fair", "firm_but_fair"):
        system = FIRM_PROMPT
        style_label = "FIRM BUT FAIR"
    else:
        system = SOFT_PROMPT
        style_label = "SOFT"
    user = (f"Style: {style_label}\n"
            f"TARGET turn {label}: {text}\n"
            "Rewrite:")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_rewrite(raw) -> str | None:
    """Clean one rewrite; caps at 40 words. None when blank."""
    if not isinstance(raw, str):
        return None
    s = " ".join(raw.split()).strip().strip('"').strip()
    if not s:
        return None
    # Strip a single wrapping quote pair left by chatty models.
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"', "“", "‘"):
        s = s[1:-1].strip()
    if not s:
        return None
    return _cap_words(s, MAX_WORDS)


def _rewrite_one(client, messages, model, meta):
    from . import board as _B

    for _ in range(2):
        try:
            raw = _B._call_board(client, messages, model, meta)
        except Exception:
            # BudgetExceeded propagates; anything else retries once.
            from .usage import BudgetExceeded as _BE

            try:
                raise
            except _BE:
                raise
            except Exception:
                continue
        parsed = parse_rewrite(raw)
        if parsed is not None:
            return parsed
    return None


def generate_alternatives(client, original, speaker="", model=None,
                          meter=None, dialogue_id="", turn=0) -> list[str | None]:
    """Two rewrites [soft, firm]; failures become None."""
    from .reader import DEFAULT_MODEL as _DM
    from .usage import BudgetExceeded

    model_name = model if model is not None else _DM
    out: list[str | None] = []
    for style in (SOFT_STYLE, FIRM_STYLE):
        if meter is not None:
            try:
                meter.check()
            except BudgetExceeded:
                out.append(None)
                continue
        try:
            text = _rewrite_one(
                client,
                build_rewrite_messages(original, style, speaker),
                model_name,
                {"layer": "suggest-write", "item": dialogue_id,
                 "dialogue": dialogue_id, "turn": turn, "style": style},
            )
        except BudgetExceeded:
            out.append(None)
            continue
        out.append(text)
    return out


# --- blunder turns --------------------------------------------------------------

def _is_blunder_turn(turn) -> bool:
    try:
        from .scenes import is_blunder as _ib
    except ImportError:
        _ib = None
    if _ib is not None:
        try:
            return bool(_ib(turn))
        except Exception:
            pass
    # Fallback: same rule without the scenes import.
    if not isinstance(turn, dict):
        return False
    try:
        zr = turn.get("z_resid")
        if zr is not None and float(zr) <= -1.5:
            return True
    except (TypeError, ValueError):
        pass
    try:
        b = float(turn.get("standing_before", turn.get("before", 50)))
        a = float(turn.get("standing_after", turn.get("after", 50)))
        return (b - a) >= 15
    except (TypeError, ValueError):
        return False


def find_blunders(item, commentary) -> list[dict]:
    """Blunder turns with standings, in chronological order.

    Prefers the commentary turns (z_resid + standings); falls back to the
    record effects with standings from ``commentary.standing_pairs``.
    Each entry: {turn, index, speaker, original, text, standing_before,
    standing_after, moves}.
    """
    item = item if isinstance(item, dict) else {}
    commentary = commentary if isinstance(commentary, dict) else {}
    raw_turns = commentary.get("turns", [])
    raw_turns = raw_turns if isinstance(raw_turns, list) else []
    effects = [e for e in (item.get("effects", []) or [])
               if isinstance(e, dict)]
    text_by_index: dict[int, str] = {}
    spk_by_index: dict[int, str] = {}
    moves_by_index: dict[int, list] = {}
    for pos, e in enumerate(effects):
        idx = e.get("index", pos)
        idx = idx if isinstance(idx, int) and not isinstance(idx, bool) else pos
        if idx not in text_by_index:
            tx = e.get("text", "")
            text_by_index[idx] = tx if isinstance(tx, str) else str(tx)
            sp = e.get("speaker", "")
            spk_by_index[idx] = sp if isinstance(sp, str) else str(sp)
    moves_raw = item.get("moves", []) or []
    for m in moves_raw:
        if isinstance(m, dict):
            idx = m.get("turn_index")
            mv = m.get("moves", []) or []
        else:
            idx = getattr(m, "turn_index", None)
            try:
                mv = list(getattr(m, "moves", []) or [])
            except TypeError:
                mv = []
        if isinstance(idx, int) and idx not in moves_by_index:
            moves_by_index[idx] = [x for x in mv if isinstance(x, str)]
    # Standings fallback from the record itself.
    pairs: list = []
    if effects:
        try:
            from .commentary import standing_pairs as _sp

            norm = commentary.get("norm")
            norm = norm if isinstance(norm, dict) else {}
            pairs = _sp(item, norm)
        except (TypeError, ValueError):
            pairs = []
    out: list[dict] = []
    if raw_turns:
        for c in raw_turns:
            if not isinstance(c, dict):
                continue
            idx = c.get("index")
            if isinstance(idx, bool) or not isinstance(idx, int):
                continue
            speaker = c.get("speaker", spk_by_index.get(idx, ""))
            speaker = speaker if isinstance(speaker, str) else str(speaker)
            text = c.get("text", text_by_index.get(idx, ""))
            text = text if isinstance(text, str) else str(text)
            moves = c.get("moves", moves_by_index.get(idx, []))
            moves = [m for m in (moves or []) if isinstance(m, str)]
            if "standing_before" in c or "standing_after" in c:
                try:
                    b = float(c.get("standing_before", 50.0))
                except (TypeError, ValueError):
                    b = 50.0
                try:
                    a = float(c.get("standing_after", 50.0))
                except (TypeError, ValueError):
                    a = 50.0
            else:
                pos = None
                for p, e in enumerate(effects):
                    ei = e.get("index", p)
                    ei = ei if isinstance(ei, int) else p
                    if ei == idx:
                        pos = p
                        break
                if pos is not None and pos < len(pairs):
                    b, a = pairs[pos]
                else:
                    b, a = 50.0, 50.0
            entry = {"turn": idx, "index": idx, "speaker": speaker,
                     "original": text, "text": text,
                     "standing_before": float(b), "standing_after": float(a),
                     "moves": moves}
            if c.get("z_resid") is not None:
                entry["z_resid"] = c.get("z_resid")
            if c.get("z") is not None:
                entry["z"] = c.get("z")
            if _is_blunder_turn(entry):
                out.append(entry)
        out.sort(key=lambda e: e["turn"])
        if out:
            return out
        # Commentary turns exist but none read as blunders: still check
        # every effect so a commentary without z_resid still yields drops.
    for pos, e in enumerate(effects):
        idx = e.get("index", pos)
        idx = idx if isinstance(idx, int) and not isinstance(idx, bool) else pos
        speaker = e.get("speaker", "")
        speaker = speaker if isinstance(speaker, str) else str(speaker)
        text = e.get("text", "")
        text = text if isinstance(text, str) else str(text)
        if pos < len(pairs):
            b, a = pairs[pos]
        else:
            b, a = 50.0, 50.0
        entry = {"turn": idx, "index": idx, "speaker": speaker,
                 "original": text, "text": text,
                 "standing_before": float(b), "standing_after": float(a),
                 "moves": list(moves_by_index.get(idx, []))}
        if _is_blunder_turn(entry):
            out.append(entry)
    # Dedupe by turn index (commentary + effects overlap).
    seen: set[int] = set()
    uniq: list[dict] = []
    for e in sorted(out, key=lambda x: x["turn"]):
        if e["turn"] in seen:
            continue
        seen.add(e["turn"])
        uniq.append(e)
    return uniq


# --- pairwise scoring (reuse perturb vote) ----------------------------------------

def _target_pos(record, turn) -> int | None:
    effects = record.get("effects", []) if isinstance(record, dict) else []
    for pos, e in enumerate(effects or []):
        if not isinstance(e, dict):
            continue
        idx = e.get("index", pos)
        idx = idx if isinstance(idx, int) and not isinstance(idx, bool) else pos
        if idx == turn:
            return pos
    return None


def score_alternative(client, record, turn, alt_text, model=None,
                      meter=None, dialogue_id="") -> dict:
    """Score one rewrite vs the original (6 votes, ties count half).

    Shared context up to ``turn``, candidates A = original / B = rewrite,
    both orders × 3 stances via ``perturb.build_perturb_messages``.
    Returns {p_better, votes, margin}.
    """
    from .reader import DEFAULT_MODEL as _DM

    model_name = model if model is not None else _DM
    record = record if isinstance(record, dict) else {}
    if isinstance(turn, bool) or not isinstance(turn, int):
        raise ValueError("unknown_turn")
    pos_in_record = _target_pos(record, turn)
    if pos_in_record is None:
        raise ValueError("unknown_turn")
    effects = [e for e in (record.get("effects", []) or [])
               if isinstance(e, dict)]
    target = effects[pos_in_record]
    orig_text = target.get("text", "")
    orig_text = orig_text if isinstance(orig_text, str) else str(orig_text)
    speaker = target.get("speaker", "")
    speaker = speaker if isinstance(speaker, str) else str(speaker)
    did = dialogue_id
    if not did:
        raw = record.get("id", "")
        did = raw if isinstance(raw, str) else str(raw or "")
    try:
        pool, contexts = PW.collect_pool({did: record} if did else {})
    except (TypeError, ValueError):
        pool, contexts = [], {}
    ctx = contexts.get(did, {}) if isinstance(contexts, dict) else {}
    turns = ctx.get("turns", []) if isinstance(ctx, dict) else []
    pos_of = ctx.get("pos_of", {}) if isinstance(ctx, dict) else {}
    if turns and isinstance(pos_of, dict) and turn in pos_of:
        pos = pos_of[turn]
        if not isinstance(pos, int) or pos < 0 or pos >= len(turns):
            pos = min(pos_in_record, max(0, len(turns) - 1))
    else:
        from .transcribe import Turn as _Turn

        turns = [_Turn(speaker=(e.get("speaker", "")
                               if isinstance(e.get("speaker"), str)
                               else str(e.get("speaker", ""))),
                       text=(e.get("text", "")
                             if isinstance(e.get("text"), str)
                             else str(e.get("text", ""))))
                 for e in effects]
        pos = min(pos_in_record, max(0, len(turns) - 1))
    listener = ""
    for e in pool or []:
        if (isinstance(e, dict) and e.get("dialogue") == did
                and e.get("turn") == turn):
            cand = e.get("listener", "")
            if isinstance(cand, str) and cand:
                listener = cand
                break
    if not listener:
        seen: list[str] = []
        for t in turns:
            if t.speaker not in seen:
                seen.append(t.speaker)
        others = [s for s in seen if s != speaker and s != "unknown"]
        listener = others[0] if others else "listener"
    context_lines = PW.snippet_lines(turns, pos, PW.WINDOW)
    m = meter if meter is not None else getattr(client, "meter", None)
    if m is not None:
        from .usage import BudgetExceeded

        try:
            m.check()
        except BudgetExceeded:
            raise
    try:
        pid = int(turn)
    except (TypeError, ValueError):
        pid = 0
    orders = (("orig_first", orig_text, alt_text),
              ("rewrite_first", alt_text, orig_text))
    votes: list[dict] = []
    for order_name, text_first, text_second in orders:
        for stance in PW.STANCES:
            try:
                s_idx = list(PW.STANCES).index(stance)
            except ValueError:
                s_idx = 0
            try:
                receivers = PW.pairwise_panel(pid)
            except (TypeError, ValueError):
                receivers = []
            if s_idx < len(receivers):
                rec = receivers[s_idx]
            else:
                from .types import Receiver as _Receiver

                rec = _Receiver(
                    id=f"pair{pid}-{s_idx}",
                    relationship="participant in this conversation",
                    power="equal", channel="conversation", state="",
                    stance=stance)
            messages = perturb.build_perturb_messages(
                context_lines, speaker, text_first, text_second,
                rec, listener)
            meta = {"layer": "pairwise", "pair": pid, "stance": stance,
                    "member": getattr(rec, "id", ""), "item": did,
                    "dialogue": did, "dialogues": [did], "turn": turn,
                    "task": "suggest"}
            parsed, _ = perturb._judge_one_perturb(
                client, messages, model_name, meta)
            better = parsed.get("better") if isinstance(parsed, dict) else None
            if better not in ("first", "second", "tie"):
                better = None
            if better is None or better == "tie":
                winner = "tie"
            elif order_name == "orig_first":
                winner = "orig" if better == "first" else "rewrite"
            else:
                winner = "rewrite" if better == "first" else "orig"
            votes.append({"stance": stance, "order": order_name,
                          "better": better, "winner": winner})
    n_re = sum(1 for v in votes if v["winner"] == "rewrite")
    n_orig = sum(1 for v in votes if v["winner"] == "orig")
    n_tie = sum(1 for v in votes if v["winner"] == "tie")
    denom = len(votes) or 6
    p_better = (n_re + 0.5 * n_tie) / denom
    margin = (n_re - n_orig) / denom
    return {"p_better": float(p_better), "votes": votes,
            "margin": float(margin)}


def score_text(*args, **kwargs) -> dict:
    """Alias for score_alternative."""
    return score_alternative(*args, **kwargs)


# --- notices ------------------------------------------------------------------------

def build_notices(item, commentary, scenes=None, culture="mix", client=None,
                  model=None, meter=None) -> list[dict]:
    """Notices for every blunder turn (alternatives scored, scene picked).

    Without a model/client the notices still build with alternatives [].
    """
    from .usage import BudgetExceeded

    item = item if isinstance(item, dict) else {}
    commentary = commentary if isinstance(commentary, dict) else {}
    raw_id = item.get("id", "")
    did = raw_id if isinstance(raw_id, str) else str(raw_id or "")
    blunders = find_blunders(item, commentary)
    pool: list[dict] = []
    if isinstance(scenes, (str, Path)):
        try:
            from .scenes import load_scenes as _load

            pool = _load(scenes)
        except SystemExit:
            pool = []
    elif isinstance(scenes, list):
        pool = [s for s in scenes if isinstance(s, dict)]
    scene_by_turn: dict[int, dict] = {}
    if pool:
        try:
            from .scenes import pick_for_page as _pfp

            scene_by_turn = _pfp(blunders, pool, did, culture=culture)
        except Exception:
            scene_by_turn = {}
    notices: list[dict] = []
    use_model = client is not None or model is not None
    for b in blunders:
        idx = b["turn"]
        try:
            sb = standing_round(b.get("standing_before", 50.0))
        except (TypeError, ValueError):
            sb = 50
        try:
            sa = standing_round(b.get("standing_after", 50.0))
        except (TypeError, ValueError):
            sa = 50
        scene = scene_by_turn.get(idx)
        scene_id = str(scene.get("id", "")) if isinstance(scene, dict) else None
        alternatives: list[dict] = []
        if use_model:
            if meter is not None:
                try:
                    meter.check()
                except BudgetExceeded:
                    pass
                else:
                    texts = generate_alternatives(
                        client, b.get("original", ""), b.get("speaker", ""),
                        model=model, meter=meter, dialogue_id=did, turn=idx)
                    for text in texts:
                        if not text:
                            continue
                        try:
                            scored = score_alternative(
                                client, item, idx, text, model=model,
                                meter=meter, dialogue_id=did)
                        except BudgetExceeded:
                            break
                        except (TypeError, ValueError):
                            continue
                        alternatives.append({
                            "text": text,
                            "p_better": float(scored["p_better"]),
                            "votes": list(scored["votes"]),
                        })
            else:
                texts = generate_alternatives(
                    client, b.get("original", ""), b.get("speaker", ""),
                    model=model, meter=meter, dialogue_id=did, turn=idx)
                for text in texts:
                    if not text:
                        continue
                    try:
                        scored = score_alternative(
                            client, item, idx, text, model=model,
                            meter=meter, dialogue_id=did)
                    except BudgetExceeded:
                        break
                    except (TypeError, ValueError):
                        continue
                    alternatives.append({
                        "text": text,
                        "p_better": float(scored["p_better"]),
                        "votes": list(scored["votes"]),
                    })
        notices.append({
            "turn": idx,
            "speaker": b.get("speaker", ""),
            "original": b.get("original", ""),
            "standing_before": int(sb),
            "standing_after": int(sa),
            "alternatives": alternatives,
            "scene_id": scene_id,
        })
    return notices


def suggest(*args, **kwargs) -> list[dict]:
    """Alias for build_notices."""
    return build_notices(*args, **kwargs)


def suggest_for_dialogue(*args, **kwargs) -> list[dict]:
    """Alias for build_notices."""
    return build_notices(*args, **kwargs)


# --- CLI ------------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-suggest")
    ap.add_argument("--item", required=True)
    ap.add_argument("--commentary", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--budget", type=float, default=5.0)
    ap.add_argument("--scenes", default=None)
    ap.add_argument("--culture", default="mix",
                    choices=["east", "west", "mix"])
    args = ap.parse_args(argv)
    with open(args.item, encoding="utf-8") as f:
        item = json.load(f)
    with open(args.commentary, encoding="utf-8") as f:
        commentary = json.load(f)
    raw_id = item.get("id", "") if isinstance(item, dict) else ""
    did = raw_id if isinstance(raw_id, str) else str(raw_id or "")
    if args.model is None:
        notices = build_notices(item, commentary, scenes=args.scenes,
                                culture=args.culture)
    else:
        from .reader import MuseClient
        from .usage import Meter

        try:
            b = float(args.budget) if args.budget is not None else None
        except (TypeError, ValueError):
            b = 5.0
        meter = Meter(budget_usd=b)
        client = MuseClient(meter=meter)
        notices = build_notices(item, commentary, scenes=args.scenes,
                                culture=args.culture, client=client,
                                model=args.model, meter=meter)
    out_path = Path(args.out)
    if str(out_path.parent) and str(out_path.parent) != ".":
        out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"dialogue": did, "notices": notices}, f,
                  ensure_ascii=False, indent=2)
    print(f"{len(notices)} notices")
