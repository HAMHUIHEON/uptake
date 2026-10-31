"""Per-turn misreading (의도 vs 해석) on three dials, shared by board2/game.

The speaker's move (의도) vs how the listener took it (해석): the majority
``next_action`` across the panel readings, whether it lands in the move's
conflict set (어긋남), and three listener dials — 관계 (pairwise standing
when available, else mean relationship shift), 체면 (mean face threat),
다음 행동 (majority label + share). The invited set (task60-widened) says
what counts as uptake; only a conflict-set reading counts as misread.
Stdlib only; reuses ``board.ACTION_TO_MOVES`` / ``board.MOVE_PRIORITY`` for
the expected-action mapping, ``labels.NEXT_ACTIONS`` for the candidate
actions, and ``viewer2.MOVE_KO`` for Korean move names.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .board import ACTION_TO_MOVES, MOVE_PRIORITY
from .labels import NEXT_ACTIONS
from .viewer2 import MOVE_KO

NEXT_ACTION_KO = {
    "comply": "따른다",
    "refuse": "거절한다",
    "negotiate": "흥정한다",
    "ask_clarification": "되묻는다",
    "escalate": "맞선다",
    "withdraw": "물러난다",
    "retaliate": "되갚는다",
    "repair": "달랜다",
    "proceed": "이어간다",
    "ignore": "무시한다",
}

_CANON = {a: i for i, a in enumerate(list(NEXT_ACTIONS))}
_CANON_LEN = len(_CANON)


def _canon_key(action):
    return (_CANON.get(action, _CANON_LEN), str(action))


def _invert_action_map() -> dict:
    out: dict = {}
    for action, moves in (ACTION_TO_MOVES or {}).items():
        for m in moves or []:
            if isinstance(m, str) and m:
                out.setdefault(m, set()).add(action)
    return {m: frozenset(v) for m, v in out.items()}


#: move kind -> actions it invites (inverse of board.ACTION_TO_MOVES).
MOVE_TO_ACTIONS = _invert_action_map()

#: task60: widened invited sets. ``board.ACTION_TO_MOVES`` was written for
#: negotiation chat; on ATC/medical/diplomacy records it marks ordinary
#: uptake as misread (complying with an instruction, answering a question).
#: These overrides replace the pure inverse for the listed moves; every
#: other move keeps the inverse mapping.
_INVITED_OVERRIDE = {
    "inform": frozenset({"proceed", "comply", "ask_clarification"}),
    "ask": frozenset({"ask_clarification", "comply", "proceed"}),
    "probe": frozenset({"ask_clarification", "comply", "proceed"}),
    "other": frozenset(list(NEXT_ACTIONS)),
    "smalltalk": frozenset(list(NEXT_ACTIONS)),
    "propose": frozenset({"negotiate", "comply", "refuse", "proceed"}),
    "reject": frozenset({"refuse", "negotiate", "ask_clarification",
                         "withdraw"}),
}

#: task60: move kind -> readings that conflict with it. Misread is true
#: only when the majority reading lands in this set; moves without an
#: entry (deflect, joke, cooperate, reveal_preference, smalltalk, other,
#: unknown) can never misread.
CONFLICTS = {
    "accept": frozenset({"refuse", "escalate", "retaliate", "ignore"}),
    "concede": frozenset({"refuse", "escalate", "retaliate", "ignore"}),
    "propose": frozenset({"ignore", "retaliate", "escalate"}),
    "counter": frozenset({"ignore", "retaliate", "escalate"}),
    "appeal_need": frozenset({"ignore", "retaliate", "escalate"}),
    "appeal_fairness": frozenset({"ignore", "retaliate", "escalate"}),
    "ask": frozenset({"ignore", "refuse", "escalate"}),
    "probe": frozenset({"ignore", "refuse", "escalate"}),
    "inform": frozenset({"refuse", "ignore", "retaliate", "escalate"}),
    "pressure": frozenset({"ignore", "withdraw"}),
    "challenge": frozenset({"ignore", "withdraw"}),
    "repair": frozenset({"retaliate", "escalate", "refuse", "ignore"}),
    "empathize": frozenset({"retaliate", "escalate", "refuse", "ignore"}),
    "praise": frozenset({"retaliate", "escalate", "refuse", "ignore"}),
    "reject": frozenset({"comply", "ignore"}),
}

#: Conflict set for appeal_* moves beyond the two known ones.
_APPEAL_CONFLICT = frozenset({"ignore", "retaliate", "escalate"})


def _representative_moves() -> dict:
    out: dict = {}
    for move, action in MOVE_PRIORITY or []:
        if (isinstance(move, str) and move and isinstance(action, str)
                and action and action not in out):
            out[action] = move
    for action in list(NEXT_ACTIONS) + list(ACTION_TO_MOVES or {}):
        if not isinstance(action, str) or not action or action in out:
            continue
        cands = sorted(ACTION_TO_MOVES.get(action, set()))
        if cands:
            out[action] = cands[0]
    return out


#: majority next_action -> representative move kind (for the strip quotes).
ACTION_TO_MOVE = _representative_moves()


def invited_actions(moves) -> set:
    """Actions invited by any of the turn's move kinds (may be empty).

    The task60-widened set: the board inverse plus overrides (an answer
    or a go-ahead counts as uptake; ``other``/``smalltalk`` invite every
    action, so those turns can never misread).
    """
    out: set = set()
    for m in moves or []:
        if isinstance(m, str) and m:
            if m in _INVITED_OVERRIDE:
                out |= set(_INVITED_OVERRIDE[m])
            else:
                out |= set(MOVE_TO_ACTIONS.get(m, frozenset()))
    return out


def conflict_actions(moves) -> set:
    """Readings that conflict with any of the turn's move kinds.

    Union of :data:`CONFLICTS` over the moves; empty when no move has a
    conflict set (then severity is 0 and misread is false).
    """
    out: set = set()
    for m in moves or []:
        if not (isinstance(m, str) and m):
            continue
        hit = CONFLICTS.get(m)
        if hit is None and m.startswith("appeal_"):
            hit = _APPEAL_CONFLICT
        if hit:
            out |= set(hit)
    return out


def representative_move(action):
    """One move kind standing for a next_action (strip quotes), or None."""
    if isinstance(action, str) and action:
        return ACTION_TO_MOVE.get(action)
    return None


def face_word(mean) -> str:
    """Mean face_threat (0-2) -> 없음 / 약간 / 큼 (missing reads as 없음)."""
    try:
        v = float(mean)
    except (TypeError, ValueError):
        return "없음"
    if not math.isfinite(v):
        return "없음"
    if v < 0.5:
        return "없음"
    if v < 1.5:
        return "약간"
    return "큼"


def _is_num(value) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(float(value)))


def _clamp(value, lo, hi) -> float:
    return max(lo, min(hi, float(value)))


def counts_from_readings(readings) -> Counter:
    """next_action counts across one turn's readings (unknown kept raw)."""
    counts: Counter = Counter()
    for r in readings or []:
        if not isinstance(r, dict):
            continue
        action = r.get("next_action")
        if isinstance(action, str) and action:
            counts[action] += 1
    return counts


def majority_of_counts(counts) -> tuple:
    """(majority action or None, share); ties break by NEXT_ACTIONS order."""
    total = sum(counts.values()) if counts else 0
    if not total:
        return (None, 0.0)
    best = sorted(counts, key=lambda a: (-counts[a], *_canon_key(a)))[0]
    return (best, counts[best] / total)


def majority_of_dist(dist) -> tuple:
    """(majority action or None, share) from an action_dist mapping."""
    shares: dict = {}
    if isinstance(dist, dict):
        for action, share in dist.items():
            if isinstance(action, str) and action and _is_num(share):
                shares[action] = max(0.0, float(share))
    total = sum(shares.values())
    if total <= 0:
        return (None, 0.0)
    best = sorted(shares, key=lambda a: (-shares[a], *_canon_key(a)))[0]
    return (best, shares[best] / total)


def threat_of_reading(reading):
    """Face threat (0-2) behind one reading, or None when absent.

    Prefers ``face_threat``; falls back to ``face_raw`` (0-100, 50 = no
    effect) and ``face_norm`` (positive affirms), which the 0-100/dist5
    panel scales carry instead.
    """
    if not isinstance(reading, dict):
        return None
    v = reading.get("face_threat")
    if _is_num(v):
        return _clamp(v, 0.0, 2.0)
    v = reading.get("face_raw")
    if _is_num(v):
        return _clamp((50.0 - float(v)) / 25.0, 0.0, 2.0)
    v = reading.get("face_norm")
    if _is_num(v):
        return _clamp(-float(v), 0.0, 2.0)
    return None


def shift_of_reading(reading):
    """Relationship shift (-2..2) behind one reading, or None when absent."""
    if not isinstance(reading, dict):
        return None
    v = reading.get("relationship_shift")
    if _is_num(v):
        return float(v)
    v = reading.get("rel_norm")
    if _is_num(v):
        return float(v)
    for key in ("relationship_after", "rel_raw"):
        v = reading.get(key)
        if _is_num(v):
            return (float(v) - 50.0) / 25.0
    return None


def reading_for_turn(moves, effect) -> dict:
    """Full misreading of one turn: intent, majority takeaway, dials."""
    moves = [m for m in (moves or []) if isinstance(m, str) and m]
    effect = effect if isinstance(effect, dict) else {}
    readings = effect.get("readings")
    if not isinstance(readings, list):
        readings = []
    counts = counts_from_readings(readings)
    dist: dict = {}
    if counts:
        total = sum(counts.values())
        dist = {a: counts[a] / total for a in counts}
        majority, share = majority_of_counts(counts)
        n = total
    else:
        majority, share = majority_of_dist(effect.get("action_dist"))
        raw = effect.get("action_dist")
        if isinstance(raw, dict):
            tot = sum(float(s) for s in raw.values() if _is_num(s))
            if tot > 0:
                dist = {a: max(0.0, float(s)) / tot
                        for a, s in raw.items()
                        if isinstance(a, str) and a and _is_num(s)}
        try:
            n = int(effect.get("n_readings", 0))
        except (TypeError, ValueError):
            n = 0
        n = max(0, n)
    invited = invited_actions(moves)
    conflict = conflict_actions(moves)
    misread = bool(majority) and majority in conflict
    if dist and conflict:
        severity = _clamp(sum(dist.get(a, 0.0) for a in conflict),
                           0.0, 1.0)
    else:
        severity = 0.0
    threats = [t for t in (threat_of_reading(r) for r in readings)
               if t is not None]
    if threats:
        face_mean = sum(threats) / len(threats)
    else:
        face_ci = effect.get("face")
        if (isinstance(face_ci, (list, tuple)) and face_ci
                and _is_num(face_ci[0])):
            face_mean = _clamp(-float(face_ci[0]), 0.0, 2.0)
        else:
            face_mean = 0.0
    shifts = [s for s in (shift_of_reading(r) for r in readings)
              if s is not None]
    if shifts:
        rel_mean = sum(shifts) / len(shifts)
    else:
        rel_ci = effect.get("relationship")
        if (isinstance(rel_ci, (list, tuple)) and rel_ci
                and _is_num(rel_ci[0])):
            rel_mean = float(rel_ci[0])
        else:
            rel_mean = 0.0
    rep = representative_move(majority)
    intent = [MOVE_KO.get(m, m) for m in moves]
    try:
        pct = int(round(float(share) * 100))
    except (TypeError, ValueError):
        pct = 0
    return {
        "moves": list(moves),
        "intent": intent,
        "majority": majority,
        "majority_ko": (NEXT_ACTION_KO.get(majority, majority)
                        if majority else ""),
        "share": float(share),
        "pct": pct,
        "n": n,
        "dist": dist,
        "invited": sorted(invited, key=_canon_key),
        "conflict": sorted(conflict, key=_canon_key),
        "misread": bool(misread),
        "severity": float(severity),
        "face_mean": float(face_mean),
        "face": face_word(face_mean),
        "rel_mean": float(rel_mean),
        "rep_move": rep,
        "rep_ko": MOVE_KO.get(rep, "") if rep else "",
    }


def panel_turns(panel) -> list:
    """Turns of a panel record as [{index, speaker, text, moves, effect}]."""
    if not isinstance(panel, dict):
        return []
    moves_by_index: dict = {}
    for t in panel.get("moves", []) or []:
        if not isinstance(t, dict):
            continue
        idx = t.get("turn_index")
        if isinstance(idx, bool) or not isinstance(idx, int):
            continue
        if idx in moves_by_index:
            continue
        moves_by_index[idx] = t
    out: list = []
    effects = panel.get("effects", [])
    if not isinstance(effects, list):
        return []
    for pos, e in enumerate(effects):
        if not isinstance(e, dict):
            continue
        idx = e.get("index", pos)
        if isinstance(idx, bool) or not isinstance(idx, int):
            idx = pos
        tag = moves_by_index.get(idx, {})
        moves = tag.get("moves", []) if isinstance(tag, dict) else []
        if not moves:
            moves = e.get("moves", []) or []
        speaker = e.get("speaker", "")
        if not speaker and isinstance(tag, dict):
            speaker = tag.get("speaker", "")
        if not isinstance(speaker, str):
            speaker = str(speaker)
        text = e.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        out.append({"index": idx, "speaker": speaker, "text": text,
                    "moves": [m for m in moves if isinstance(m, str)],
                    "effect": e})
    out.sort(key=lambda t: t["index"])
    return out


def analyze_panel(panel) -> list:
    """Per-turn readings for a panel record, in index order."""
    out: list = []
    for t in panel_turns(panel):
        reading = reading_for_turn(t["moves"], t["effect"])
        listeners = t["effect"].get("listeners", [])
        if not isinstance(listeners, list):
            listeners = []
        reading.update({
            "index": t["index"],
            "speaker": t["speaker"],
            "text": t["text"],
            "listeners": [s for s in listeners if isinstance(s, str)],
        })
        out.append(reading)
    return out


def top_misread(analyses, k=5) -> list:
    """Worst-k misread turns by severity, in chronological order."""
    try:
        kk = int(k)
    except (TypeError, ValueError):
        kk = 5
    flagged = [a for a in (analyses or [])
               if isinstance(a, dict) and a.get("misread")]
    flagged.sort(key=lambda a: (-float(a.get("severity", 0.0)),
                                a.get("index", 0)))
    return sorted(flagged[:max(0, kk)], key=lambda a: a.get("index", 0))


def listener_of(effect_or_listeners, speaker, speakers=None):
    """The turn's listener: first non-speaker listener, else other side."""
    if isinstance(effect_or_listeners, dict):
        listeners = effect_or_listeners.get("listeners", [])
    elif isinstance(effect_or_listeners, list):
        listeners = effect_or_listeners
    else:
        listeners = []
    if not isinstance(listeners, list):
        listeners = []
    for cand in listeners:
        if isinstance(cand, str) and cand and cand != speaker:
            return cand
    for cand in listeners:
        if isinstance(cand, str) and cand:
            return cand
    for cand in speakers or []:
        if cand != speaker:
            return cand
    return ""


def counts_as_hit(pick, key, invited=None) -> bool:
    """Lenient game key: exact, or in-invited when the key is in it too."""
    if pick == key:
        return True
    try:
        inv = set(invited or [])
    except TypeError:
        return False
    return key in inv and pick in inv


def options_for(majority, invited=None, dist=None) -> list:
    """5-6 (value, Korean label) buttons with the key included.

    The key and the invited set come first, then the remaining readings by
    share, padded from NEXT_ACTIONS; the final order is canonical so the
    key's position gives nothing away.
    """
    shares: dict = {}
    if isinstance(dist, dict):
        for action, share in dist.items():
            if isinstance(action, str) and action and _is_num(share):
                shares[action] = max(0.0, float(share))
    inv = [a for a in (invited or []) if isinstance(a, str) and a]
    cands = set(shares) | set(inv)
    if isinstance(majority, str) and majority:
        cands.add(majority)
    out: list = []
    if isinstance(majority, str) and majority:
        out.append(majority)
    for a in sorted(set(inv) - set(out), key=_canon_key):
        out.append(a)
    rest = sorted(cands - set(out),
                  key=lambda a: (-shares.get(a, 0.0), *_canon_key(a)))
    out.extend(rest)
    for a in NEXT_ACTIONS:
        if len(out) >= 5:
            break
        if a not in out:
            out.append(a)
    out = sorted(out[:6], key=_canon_key)
    return [(a, NEXT_ACTION_KO.get(a, a)) for a in out]


def intent_str(moves) -> str:
    """·-joined Korean intent ("발언" when the turn has no moves)."""
    if isinstance(moves, dict):
        kos = moves.get("intent")
        if isinstance(kos, list) and kos:
            return "·".join(str(k) for k in kos)
        moves = moves.get("moves", [])
    kos = [MOVE_KO.get(m, m) for m in (moves or [])
           if isinstance(m, str) and m]
    return "·".join(kos) if kos else "발언"


def _standing_int(value):
    if isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return int(math.floor(v + 0.5))


def _has_jongseong(text) -> bool:
    if not text:
        return False
    code = ord(str(text)[-1])
    return 0xAC00 <= code <= 0xD7A3 and (code - 0xAC00) % 28 != 0


def card_line(reading, before=None, after=None) -> str:
    """Decisive-card three-dial line for one analyzed turn."""
    reading = reading if isinstance(reading, dict) else {}
    intent = intent_str(reading)
    ko = reading.get("majority_ko") or ""
    if not (isinstance(ko, str) and ko):
        ko = "?"
    try:
        pct = int(round(float(reading.get("share", 0.0)) * 100))
    except (TypeError, ValueError):
        pct = 0
    face = reading.get("face", "약간")
    if not (isinstance(face, str) and face):
        face = "약간"
    bi, ai = _standing_int(before), _standing_int(after)
    if bi is not None and ai is not None:
        rel = f"{bi} → {ai}"
    else:
        try:
            rel = f"{float(reading.get('rel_mean', 0.0)):+.2f}"
        except (TypeError, ValueError):
            rel = "?"
    return (f"의도: {intent} → 상대가 읽은 것: {ko}({pct}%) · "
            f"체면 위협 {face} · 관계 {rel}")


def strip_flag(index, speaker, intent_first, listener, rep_ko) -> str:
    """One 오해 지점 flag: #1 B 거절 → A는 '압박'으로 읽음."""
    try:
        idx_s = str(int(index)) if not isinstance(index, bool) else str(index)
    except (TypeError, ValueError):
        idx_s = str(index)
    sp = speaker if isinstance(speaker, str) and speaker else "?"
    it = intent_first if isinstance(intent_first, str) and intent_first else "발언"
    li = listener if isinstance(listener, str) and listener else "상대"
    rep = rep_ko if isinstance(rep_ko, str) and rep_ko else "?"
    eun = "은" if _has_jongseong(li) else "는"
    ro = "으로" if _has_jongseong(rep) or not rep else "로"
    if rep and not (0xAC00 <= ord(rep[-1]) <= 0xD7A3):
        ro = "으로"
    return f"#{idx_s} {sp} {it} → {li}{eun} '{rep}'{ro} 읽음"


def reveal_line(reading, before=None, after=None) -> str:
    """Game reveal line for one analyzed turn."""
    reading = reading if isinstance(reading, dict) else {}
    ko = reading.get("majority_ko") or ""
    if not (isinstance(ko, str) and ko):
        ko = "?"
    try:
        pct = int(round(float(reading.get("share", 0.0)) * 100))
    except (TypeError, ValueError):
        pct = 0
    face = reading.get("face", "약간")
    if not (isinstance(face, str) and face):
        face = "약간"
    bi, ai = _standing_int(before), _standing_int(after)
    rel = (f"{bi} → {ai}" if bi is not None and ai is not None else "?")
    return f"상대 판정: {ko}({pct}%) · 체면 위협 {face} · 관계 {rel}"


def load_panel(source):
    """Panel record dict from a dict, a JSON path, or None (None if bad)."""
    if source is None:
        return None
    if isinstance(source, dict):
        return source
    if isinstance(source, (str, Path)):
        try:
            with open(source, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None
    return None
