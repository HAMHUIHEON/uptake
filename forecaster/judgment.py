"""One model judgment of a whole conversation, checked by code before use.

The model sees every turn's speaker and words, the per-turn readings (wanted,
actual, layers, alert level) and four conditions. It never sees a case id,
title, outcome, note or list of notable turns. It answers four things: the
turns where the game divided and the decisive one, how each turn was steered
and received, each speaker's goal and strategy shifts inside this conversation,
and a scene from the catalog or none. Code checks every field and drops what
fails. Scores are never taken from the model's words: the report copies them
from the readings, and copies the perspective line from the catalog.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .multilayer import Reader, _squash
from .usage import BudgetExceeded

SCENES_FILE = Path(__file__).with_name("scenes.json")
SOURCE_MUSE = "Muse"
#: A scene whose quoted words share this many runs of this many words with the
#: conversation is the same event, not a perspective on it.
SAME_EVENT_WORDS, SAME_EVENT_RUNS = 4, 2
MAX_TEXT = 600
#: A score in words. The model does not supply scores, whatever it writes.
SCORE_WORDS = re.compile(r"\d+(?:\.\d+)?\s*(?:/\s*100\b|%|percent\b)", re.I)
LABEL_SHAPED = re.compile(r"^[a-z_]+\s*:\s")
CONDITIONS = ("relationship", "power", "channel", "stakes_weight")

JUDGE_PROMPT = (
    "You read one whole conversation after it is over, one turn at a time. Each turn gives its number, "
    "speaker and words, then readings from another instrument: per item, w is how much the turn reached "
    "for that item and a is how far the wanted effect happened (0-100); then the layers and any alert "
    "level. The readings are estimates. Weigh them against the words.\n"
    "There is no right way to talk here. Each speaker came with a goal, changed strategy as the "
    "conversation went, and left with something gained or lost against that goal.\n"
    "Answer four things from what is shown:\n"
    "1. divides: the turns where the game divided, meaning what each speaker could still get changed "
    "there, with the reason. List as many as this conversation has; an empty list is a valid answer. "
    "Only a turn that another speaker later answers can be listed. decisive_turn is the one of them that "
    "mattered most for what each speaker was left with, and decisive_why says why; null when divides is "
    "empty.\n"
    "2. turns: one entry for every turn. opened says how this speaker steered the conversation at this "
    "moment. response_turn is the later turn by another speaker that answers it, and received says how "
    "that speaker took it; both null when nobody answers it.\n"
    "3. speakers: one entry for every listed speaker. goal is one sentence on what this speaker wanted "
    "in this conversation, written after reading all of their turns. Say nothing about character, "
    "personality or ability. shifts are the turns where this speaker's strategy changed: from, to, and "
    "quote, a span copied exactly from that turn's words in their original language. An empty list is a "
    "valid answer.\n"
    "4. scene: the catalog scene that truly fits this conversation, as scene_id with why; scene_id is "
    "null when none fits, with why saying so.\n"
    "Write English sentences, and write no scores or percentages: code adds the numbers. "
    "Reply with JSON only: {\"divides\": [{\"turn\": n, \"why\": \"\", \"wanted\": \"\", \"happened\": \"\"}], "
    "\"decisive_turn\": n|null, \"decisive_why\": \"\", "
    "\"turns\": [{\"turn\": n, \"opened\": \"\", \"response_turn\": n|null, \"received\": \"\"|null}], "
    "\"speakers\": {\"<speaker>\": {\"goal\": \"\", \"shifts\": [{\"turn\": n, \"from\": \"\", \"to\": \"\", "
    "\"quote\": \"\"}]}}, \"scene\": {\"scene_id\": \"\"|null, \"why\": \"\"}}.")


def _fold(text) -> str:
    return " ".join(str(text or "").split())


def _word_runs(text) -> set:
    words = _squash(text).split()
    return {" ".join(words[i:i + SAME_EVENT_WORDS]) for i in range(len(words) - SAME_EVENT_WORDS + 1)}


def load_scenes() -> list:
    """Catalog scenes that carry an English line. Unreadable file: no scenes."""
    try:
        data = json.loads(SCENES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [s for s in data if isinstance(s, dict) and s.get("id") and s.get("교훈_en")] if isinstance(data, list) else []


def offered_scenes(turns, scenes=None) -> list:
    """Scenes to offer: the catalog minus any scene recounting this same event."""
    said = _word_runs(" ".join(str(t.get("text", "")) for t in turns))
    return [s for s in (load_scenes() if scenes is None else scenes)
            if len(_word_runs(s.get("발화", "")) & said) < SAME_EVENT_RUNS]


def _reading_line(reading, levels) -> str:
    parts = []
    for name, e in (reading or {}).get("items", {}).items():
        wanted, actual = e.get("wanted"), e.get("actual")
        if wanted or actual:
            parts.append(f"{name} w{wanted['score']:.0f}" if wanted else f"{name} w-")
            parts[-1] += f" a{actual['score']:.0f}" if actual else " a-"
    line = "; ".join(parts) or "no item supported"
    layers = (reading or {}).get("layers")
    if layers:
        line += f" | layers {json.dumps(layers, ensure_ascii=False)}"
    return line + (f" | alert {'/'.join(levels)}" if levels else "")


def judgment_user(turns, uptake, alerts, conditions, scenes) -> str:
    """The whole input. Nothing about the case record may be added here."""
    by_turn = {r.get("turn"): r for r in (uptake or []) if isinstance(r, dict)}
    levels = {}
    for a in (alerts or []):
        levels.setdefault(a.get("turn"), []).append(str(a.get("level")))
    known = {k: (conditions or {}).get(k) for k in CONDITIONS}
    own = {}
    lines = []
    for i, t in enumerate(turns):
        own.setdefault(str(t.get("speaker", "?")), []).append(f"#{i}")
        lines.append(f"#{i} {t.get('speaker', '?')}: {_fold(t.get('text', ''))}")
        lines.append(f"   readings: {_reading_line(by_turn.get(i), levels.get(i))}")
    speakers = "\n".join(f"- {sp}: turns {', '.join(nums)}" for sp, nums in own.items())
    catalog = "\n".join(f"- {s['id']}: {_fold(s.get('상황_한줄', ''))} / {_fold(s['교훈_en'])}" for s in scenes)
    return (f"Conditions: {json.dumps(known, ensure_ascii=False)}\nSpeakers:\n{speakers}\n"
            "Conversation:\n" + "\n".join(lines) + f"\nScene catalog (id: situation / line):\n{catalog or '- (empty)'}")


def _turn_no(value, n):
    return value if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < n else None


def validate(data, turns, scene_ids) -> dict:
    """Keep the fields that pass. A list the model left empty stays empty;
    a missing field, or a list with nothing valid, comes back as None."""
    n, dropped = len(turns), []
    said = _fold(" ".join(str(t.get("text", "")) for t in turns))

    def text(value):
        body = _fold(value) if isinstance(value, str) else ""
        if not body or len(body) > MAX_TEXT:
            return None
        return None if any(m.group(0) not in said for m in SCORE_WORDS.finditer(body)) else body

    def speaker(t):
        return str(turns[t].get("speaker", "?"))

    moves = {}
    raw = data.get("turns")
    for e in raw if isinstance(raw, list) else []:
        t = _turn_no(e.get("turn"), n) if isinstance(e, dict) else None
        opened = text(e.get("opened")) if t is not None else None
        if opened is None or t in moves:
            dropped.append(f"turns: {e.get('turn') if isinstance(e, dict) else e!r}")
            continue
        move = {"turn": t, "speaker": speaker(t), "opened": opened, "response_turn": None, "received": None}
        reply = e.get("response_turn")
        if reply is not None or e.get("received"):
            r = _turn_no(reply, n)
            received = text(e.get("received"))
            if r is not None and r > t and speaker(r) != speaker(t) and received:
                move.update(response_turn=r, received_by=speaker(r), received=received)
            else:
                dropped.append(f"turns: #{t} reply {reply!r}")
        moves[t] = move
    has_turns = bool(moves)

    divides, seen = [], set()
    raw = data.get("divides")
    for e in raw if isinstance(raw, list) else []:
        t = _turn_no(e.get("turn"), n) if isinstance(e, dict) else None
        fields = {k: text(e.get(k)) for k in ("why", "wanted", "happened")} if t is not None else {}
        if t is None or t in seen or not all(fields.values()) or (moves.get(t) or {}).get("response_turn") is None:
            dropped.append(f"divides: {e.get('turn') if isinstance(e, dict) else e!r}")
            continue
        seen.add(t)
        divides.append({"turn": t, **fields})
    divides.sort(key=lambda d: d["turn"])
    if not isinstance(raw, list) or (raw and not divides):
        divides = None

    decisive = None
    picked, why = data.get("decisive_turn"), text(data.get("decisive_why"))
    if picked is not None:
        if divides and _turn_no(picked, n) in seen and why:
            decisive = {"turn": picked, "why": why}
        else:
            dropped.append(f"decisive_turn: {picked!r}")

    names = {}
    for t in range(n):
        names.setdefault(speaker(t), None)
    raw = data.get("speakers")
    raw = raw if isinstance(raw, dict) else {}
    dropped += [f"speakers: unknown {k!r}" for k in raw if k not in names]
    speakers = {}
    for name in names:
        e = raw.get(name) if isinstance(raw.get(name), dict) else {}
        goal = text(e.get("goal"))
        if goal and LABEL_SHAPED.match(goal):
            goal = None
        if goal is None and e.get("goal"):
            dropped.append(f"speakers: {name} goal")
        given, shifts = e.get("shifts"), []
        for s in given if isinstance(given, list) else []:
            t = _turn_no(s.get("turn"), n) if isinstance(s, dict) else None
            quote = _fold(s.get("quote")) if t is not None and isinstance(s.get("quote"), str) else ""
            frm, to = (text(s.get("from")), text(s.get("to"))) if t is not None else (None, None)
            if (t is None or speaker(t) != name or not quote or quote not in _fold(turns[t].get("text", ""))
                    or not frm or not to):
                dropped.append(f"speakers: {name} shift {s.get('turn') if isinstance(s, dict) else s!r}")
                continue
            shifts.append({"turn": t, "from": frm, "to": to, "evidence_quote": quote})
        if not isinstance(given, list) or (given and not shifts):
            shifts = None
        speakers[name] = {"goal": goal, "shifts": sorted(shifts, key=lambda s: s["turn"]) if shifts else shifts}

    scene = None
    raw = data.get("scene")
    if isinstance(raw, dict) and "scene_id" in raw:
        why = text(raw.get("why"))
        if why and (raw["scene_id"] is None or (isinstance(raw["scene_id"], str) and raw["scene_id"] in scene_ids)):
            scene = {"scene_id": raw["scene_id"], "why": why}
        else:
            dropped.append(f"scene: {raw['scene_id']!r}")
    return {"divides": divides, "decisive": decisive, "turns": sorted(moves.values(), key=lambda m: m["turn"]),
            "has_turns": has_turns, "speakers": speakers, "scene": scene, "dropped": dropped}


def _empty(reason) -> dict:
    return {"divides": None, "decisive": None, "turns": [], "has_turns": False, "speakers": {},
            "scene": None, "dropped": [], "reason": reason}


def judge_conversation(turns, uptake, alerts, conditions, client, model, meter, cache_dir, rep=0) -> dict:
    """Judge one conversation once. ``rep`` separates repeat judgments in the cache.

    Returns the checked fields with model, rep, source and n_scenes. A failure
    returns empty fields and a ``code fallback`` source. BudgetExceeded propagates.
    """
    turns = [t if isinstance(t, dict) else {} for t in (turns or [])]
    scenes = offered_scenes(turns)
    meta = {"model": model, "rep": rep, "n_scenes": len(scenes)}
    if client is None or not model:
        return {**_empty("no model for this run"), **meta, "source": "code fallback: no model for this run"}
    user = judgment_user(turns, uptake, alerts, conditions, scenes)
    try:
        data = Reader(client, model, meter, cache_dir, rep=rep).ask(JUDGE_PROMPT, user)
    except BudgetExceeded:
        raise
    except Exception as exc:
        reason = f"model call failed ({type(exc).__name__})"
        return {**_empty(reason), **meta, "source": f"code fallback: {reason}"}
    if data is None:
        reason = "model reply did not parse"
        return {**_empty(reason), **meta, "source": f"code fallback: {reason}"}
    checked = validate(data, turns, {s["id"] for s in scenes})
    usable = (checked["divides"] is not None or checked["has_turns"] or checked["scene"] is not None
              or any(v["goal"] or v["shifts"] is not None for v in checked["speakers"].values()))
    reason = "model reply had no valid field"
    return {**checked, "reason": "" if usable else reason, **meta,
            "source": SOURCE_MUSE if usable else f"code fallback: {reason}"}
