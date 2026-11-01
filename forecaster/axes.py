"""Per-turn four-axis reader (task70): what was wanted vs what happened.

For one target turn, the reader sees the prior three turns and the next
three turns and judges each stake -- outcome, relationship, face,
understanding -- after the reply is seen. One model call returns all
four stakes as JSON; the quote for each stake must occur verbatim in
its reply turn, else that stake reads ``unclear``.

Stdlib only. Caching and budget follow the report helpers: identical
inputs reuse the cached reading without re-querying, and a spent
budget raises before another call.
"""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path

from .v2 import STAKES, STAKE_TREE

#: Stakes in report order (imported from v2; never renamed here).
AXES = tuple(STAKES)

#: Allowed gap values.
GAPS = ("met", "partly", "missed", "cost", "not_relevant", "unclear")

#: How a mismatch arose (구상 v2).
GAP_KINDS = ("accident", "deliberate", "conflict", "aligned", "unclear")

#: Where a mismatch lives: the words, the receiver's frame, the situation.
LAYERS = ("say", "mind", "world")

#: The 23 misreading types (오해유형_분류표), or none.
MISREAD_TYPES = (
    "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5",
    "C1", "C2", "C3", "D1", "D2", "D3", "E1", "E2", "E3",
    "F1", "F2", "F3", "F4", "none",
)

#: Context windows: turns before and after the target shown to the model.
PRIOR_WINDOW = 3
NEXT_WINDOW = 3

#: Max characters kept per turn in the model prompt (cost control).
TURN_CHARS = 600

AXES_PROMPT = (
    "You read one turn of a conversation turn by turn.\n"
    "For the TARGET turn, judge what the speaker wanted on each of four "
    "stakes and what actually happened on that stake in the NEXT turns:\n"
    "- outcome: money, time, deliverable, commitment.\n"
    "- relationship: trust, warmth, continuity.\n"
    "- face: status, autonomy, competence.\n"
    "- understanding: facts, intent, obligations.\n"
    "Judge after the reply is seen: use only the target turn, the prior "
    "turns for context, and the next turns (up to 3 ahead) as evidence. "
    "The same turn may gain on one stake and lose on another; judge each "
    "stake separately.\n"
    "For each stake give wanted, sub_stakes, happened, gap, reply_turn "
    "and quote:\n"
    "- wanted: what the speaker wanted on this stake with this turn; "
    "write \"not_relevant\" when the turn wanted nothing on this stake.\n"
    "- sub_stakes: the list of sub-stakes of THIS stake touched by this "
    "turn (outcome: money, time, deliverable, commitment; relationship: "
    "trust, warmth, continuity; face: status, autonomy, competence; "
    "understanding: facts, intent, obligations); [] when none.\n"
    "- happened: what actually happened on this stake in the next turns.\n"
    "- gap: met (as wanted) | partly (partly) | missed (wanted but did "
    "not happen, whether intended or not) | cost (an unwanted loss on "
    "this stake) | not_relevant (this stake is not relevant here) | "
    "unclear (no evidence in the next turns).\n"
    "- reply_turn: the turn index (as numbered below) whose text proves "
    "happened, within the next turns; null when not_relevant or unclear.\n"
    "- quote: the verbatim span from that reply turn proving happened; "
    "empty when there is no evidence.\n"
    "Rules that keep the reading honest:\n"
    "- Mark a stake not_relevant unless this turn visibly reaches for it. "
    "A routine readback, a plain answer or a filler usually touches one "
    "stake at most. Do not invent a wish for warmth or status because any "
    "turn could have one.\n"
    "- relationship and face need a visible cue in the words of the target "
    "or the reply (warmth, coldness, deference, challenge, a threat to "
    "standing). Moving on to the next question is not evidence of either.\n"
    "- If the target is a fragment, cut off, overlapped or garbled by "
    "transcription so that what was wanted cannot be read, every stake is "
    "unclear. A reply that merely fails to pick up a fragment is not missed.\n"
    "- Give met only when the reply shows the wanted effect, not because "
    "nobody objected.\n"
    "Then for the turn as a whole give:\n"
    "- gap_kind: accident (an unintended mismatch) | deliberate (strategic "
    "ambiguity or deception) | conflict (understood correctly but interests "
    "clash) | aligned (it went as wanted) | unclear.\n"
    "- why, only when some stake is missed, partly or cost: layer, factor, "
    "type. layer: say (the words: implicature, speech act, reference, "
    "vagueness) | mind (the receiver's frame: emotion, prior belief, face, "
    "reading silence) | world (the situation: power distance, interests, "
    "relay path, audience, channel, timing). factor: one short phrase naming "
    "the specific cause. type: one code or none. A1 information missing, A2 "
    "arrived late, A3 overheard or wrong recipient, A4 distorted by a relay, "
    "A5 forged, B1 too little or too much said, B2 wrong referent, B3 irony "
    "misread, B4 indirect speech act missed, B5 ambiguity or translation, C1 "
    "power distance, C2 face threat, C3 hidden identity, D1 high or low "
    "context, D2 clashing values, D3 custom or ritual, E1 confirmation bias, "
    "E2 emotion projected, E3 silence misread, F1 strategic ambiguity, F2 "
    "designed deception, F3 staged context, F4 well-meant deceit gone wrong.\n"
    "Do not judge whether the turn was right or polite, or the person's "
    "character. Read the utterance, not the person. Write wanted, happened "
    "and factor in English; keep quote in the original language.\n"
    "Reply with JSON only: {\"axes\": {\"outcome\": {\"wanted\": ..., "
    "\"sub_stakes\": [...], \"happened\": ..., \"gap\": ..., "
    "\"reply_turn\": ..., \"quote\": ...}, \"relationship\": {...}, "
    "\"face\": {...}, \"understanding\": {...}}, \"gap_kind\": ..., "
    "\"why\": {\"layer\": ..., \"factor\": ..., \"type\": ...}}."
)


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _axes_cache_path(cache_dir, key):
    return Path(cache_dir) / f"axes-{key}.json"


def _cap(text, limit=TURN_CHARS) -> str:
    body = " ".join(str(text or "").split())
    if len(body) > limit:
        return body[: limit - 3].rstrip() + "..."
    return body


def _squash(text) -> str:
    """Compare quotes ignoring case, spacing and punctuation."""
    return re.sub(r"[^0-9a-z\u3131-\ud7a3\u4e00-\u9fff]+", " ",
                  str(text or "").lower()).strip()


def clean_turn_fields(data, axes) -> dict:
    """gap_kind and why for one turn, invalid values made honest."""
    data = data if isinstance(data, dict) else {}
    kind = str(data.get("gap_kind", "")).strip().lower()
    if kind not in GAP_KINDS:
        kind = "unclear"
    bad = any((axes.get(a) or {}).get("gap") in ("missed", "partly", "cost")
              for a in AXES)
    why = None
    if bad:
        raw = data.get("why") if isinstance(data.get("why"), dict) else {}
        layer = str(raw.get("layer", "")).strip().lower()
        code = str(raw.get("type", "")).strip()
        code = code if code in MISREAD_TYPES else "none"
        why = {"layer": layer if layer in LAYERS else "unclear",
               "factor": " ".join(str(raw.get("factor", "")).split()),
               "type": code}
    return {"gap_kind": kind, "why": why}


def unclear_axes() -> dict:
    """All four stakes as unclear (no evidence)."""
    return {
        axis: {"wanted": "", "sub_stakes": [], "happened": "",
               "gap": "unclear", "reply_turn": None, "quote": ""}
        for axis in AXES
    }


def _clean_sub_stakes(axis, value) -> list:
    """Model sub-stakes limited to this stake's STAKE_TREE names."""
    allowed = STAKE_TREE.get(axis, ())
    if not isinstance(value, (list, tuple)):
        return []
    out = []
    for item in value:
        name = str(item).strip() if not isinstance(item, str) else item.strip()
        if name in allowed and name not in out:
            out.append(name)
    return out


def _turn_text(turns, index):
    if not isinstance(index, int) or isinstance(index, bool):
        return None
    if 0 <= index < len(turns or []):
        turn = turns[index]
        if isinstance(turn, dict):
            return str(turn.get("text", ""))
    return None


def validate_axes(data, target, turns) -> dict | None:
    """Normalize one model reply to the four-stake shape, or None.

    Returns None when the reply has no usable ``axes`` object. Each
    stake keeps the model's wanted/happened text and its sub-stakes
    (limited to this stake's STAKE_TREE names), but the gap becomes
    ``unclear`` when the quote is not found verbatim in its reply
    turn (or the reply turn is missing/out of range), unless the gap
    is already ``not_relevant`` or ``unclear``.
    """
    if not isinstance(data, dict):
        return None
    axes = data.get("axes")
    if not isinstance(axes, dict):
        return None
    out = {}
    for axis in AXES:
        entry = axes.get(axis)
        if not isinstance(entry, dict):
            entry = {}
        wanted = entry.get("wanted", "")
        happened = entry.get("happened", "")
        wanted = "" if wanted is None else str(wanted)
        happened = "" if happened is None else str(happened)
        subs = _clean_sub_stakes(axis, entry.get("sub_stakes", []))
        gap = str(entry.get("gap", "")).strip().lower()
        if gap not in GAPS:
            gap = "unclear"
        reply = entry.get("reply_turn")
        if isinstance(reply, bool):
            reply = None
        elif isinstance(reply, float) and reply.is_integer():
            reply = int(reply)
        elif not isinstance(reply, int):
            reply = None
        quote = entry.get("quote", "")
        quote = "" if quote is None else str(quote)
        if gap in ("not_relevant", "unclear"):
            # not_relevant needs no evidence; keep the model text.
            out[axis] = {"wanted": wanted, "sub_stakes": subs,
                         "happened": happened, "gap": gap,
                         "reply_turn": reply, "quote": quote}
            continue
        # Evidential stakes need a reply turn within the next window
        # whose text contains the quote verbatim.
        text = _turn_text(turns, reply)
        in_window = (
            isinstance(reply, int)
            and isinstance(target, int)
            and target < reply <= target + NEXT_WINDOW
        )
        if not in_window or text is None or not _squash(quote) or _squash(quote) not in _squash(text):
            gap = "unclear"
        out[axis] = {"wanted": wanted, "sub_stakes": subs,
                     "happened": happened, "gap": gap,
                     "reply_turn": reply, "quote": quote}
    return out


def _parse_reply(raw):
    try:
        text = str(raw or "")
    except (TypeError, ValueError):
        return None
    start = text.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _context_lines(turns, target):
    lines = []
    lo = max(0, target - PRIOR_WINDOW)
    hi = min(len(turns or []), target + NEXT_WINDOW + 1)
    for i in range(lo, hi):
        turn = turns[i] if isinstance(turns[i], dict) else {}
        speaker = turn.get("speaker", "?")
        text = _cap(turn.get("text", ""))
        mark = "TARGET " if i == target else ""
        lines.append(f"{mark}#{i} {speaker}: {text}")
    return lines


def read_turn(target, turns, client=None, model=None, meter=None,
              cache_dir=None) -> dict:
    """axes + gap_kind + why for one turn. No client: all unclear."""
    empty = {"axes": unclear_axes(), "gap_kind": "unclear", "why": None}
    if not isinstance(target, int) or isinstance(target, bool):
        return empty
    n = len(turns or [])
    if not 0 <= target < n - 1 or client is None or not model \
            or cache_dir is None:
        return empty
    lines = _context_lines(turns, target)
    payload = json.dumps({"kind": "axes-v3", "model": model,
                          "target": int(target), "lines": lines,
                          "prompt": AXES_PROMPT},
                         ensure_ascii=False, sort_keys=True).encode("utf-8")
    path = _axes_cache_path(cache_dir, _sha_bytes(payload))
    data = None
    if path.exists():
        try:
            data = _read_json(path)
        except (OSError, ValueError):
            data = None
    if data is None:
        if meter is not None:
            meter.check()
        ctx = "\n".join(f"- {line}" for line in lines)
        messages = [{"role": "system", "content": AXES_PROMPT},
                    {"role": "user", "content":
                     f"Conversation (target #{target}):\n{ctx}\nReading:"}]
        data = _parse_reply(client.complete(messages, model=model,
                                            temperature=0.0))
        if data is None:
            return empty
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except OSError:
            pass
    axes = validate_axes(data, target, turns)
    if axes is None:
        return empty
    return {"axes": axes, **clean_turn_fields(data, axes)}


def read_axes(target, turns, client=None, model=None, meter=None,
              cache_dir=None) -> dict:
    """Four-stake reading for one target turn (cached, budget-aware).

    Sees the prior three turns and the next three turns; one model
    call returns all four stakes, each with wanted, sub_stakes,
    happened, gap, reply_turn and quote. Without a client/model the
    reading is all ``unclear`` (no evidence claimed). Unparseable
    model replies also read all ``unclear`` and are not cached.
    Quotes that do not occur verbatim in their reply turn read
    ``unclear`` on that stake. BudgetExceeded propagates so the run
    stops.
    """
    if not isinstance(target, int) or isinstance(target, bool):
        return unclear_axes()
    n = len(turns or [])
    if not 0 <= target < n:
        return unclear_axes()
    if n - 1 - target <= 0:
        # Last turn: no reply on record, nothing to judge against.
        return unclear_axes()
    if client is None or not model or cache_dir is None:
        return unclear_axes()
    lines = _context_lines(turns, target)
    target_line = ""
    if isinstance(turns[target], dict):
        target_line = (
            f"#{target} {turns[target].get('speaker', '?')}: "
            f"{_cap(turns[target].get('text', ''))}"
        )
    payload = json.dumps(
        {"kind": "axes-v2", "model": model, "target": int(target),
         "lines": lines, "target_line": target_line},
        ensure_ascii=False, sort_keys=True).encode("utf-8")
    key = _sha_bytes(payload)
    path = _axes_cache_path(cache_dir, key)
    if path.exists():
        try:
            cached = validate_axes(_read_json(path), target, turns)
            if cached is not None:
                return cached
        except (OSError, ValueError):
            pass
    if meter is not None:
        meter.check()
    ctx = "\n".join(f"- {line}" for line in lines) or "- (no turns)"
    messages = [{"role": "system", "content": AXES_PROMPT},
                {"role": "user",
                 "content": f"Conversation (target #{target}):\n{ctx}\nReading:"}]
    raw = client.complete(messages, model=model, temperature=0.0)
    data = _parse_reply(raw)
    axes = validate_axes(data, target, turns) if data is not None else None
    if axes is None:
        return unclear_axes()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"axes": axes}, f, ensure_ascii=False)
    except OSError:
        pass
    return axes


def build_uptake(turns, client=None, model=None, meter=None,
                 cache_dir=None, max_turns=None) -> list:
    """Per-turn uptake for every turn except the last.

    Each entry is ``{"turn", "speaker", "axes"}`` with the four-stake
    reading from :func:`read_axes`. ``max_turns`` caps the number of
    entries (small trials); the last turn is still never judged.
    """
    n = len(turns or [])
    if n <= 1:
        return []
    try:
        cap = int(max_turns) if max_turns is not None else None
    except (TypeError, ValueError):
        cap = None
    if cap is not None and cap < 0:
        cap = 0
    targets = list(range(n - 1))
    if cap is not None:
        targets = targets[:cap]

    def _one(t):
        turn = turns[t] if isinstance(turns[t], dict) else {}
        read = read_turn(t, turns, client=client, model=model,
                         meter=meter, cache_dir=cache_dir)
        return {"turn": int(t), "speaker": str(turn.get("speaker", "?")),
                **read}

    with ThreadPoolExecutor(max_workers=6) as pool:
        entries = list(pool.map(_one, targets))
    return entries
