"""Uptake report: where a high-stakes talk did not land (uptake-report).

Reads one case turn by turn: for each turn, what the speaker wanted on
each of four stakes (outcome, relationship, face, understanding) and
what the later turns show actually happened on that stake, judged
after the reply is seen. Rules are general: they read the turns and
the per-turn stake readings, never case names or quoted phrases.
Relationship movement alone never raises an alert. Alerts come only
from stake gaps of missed, partly or cost, one per (turn, stake).

Optional ``--model`` scoring reads every eligible turn with cached
four-stake judgments, then writes two scene-specific rewrites of the
decisive turn and scores each against the original with cached paired
votes on the turn's criterion (see pick_criterion) from one panel (both
orders, ties count half) under a USD budget; identical inputs are never
re-queried. The same panel also judges two negative controls -- the
original against itself (~0.5) and against a deliberately worse wording
(<0.5) -- so a judge that always prefers the new wording cannot pass,
and a middle wording only a little better. Without ``--model``
the stake readings stay unclear (no evidence claimed) and the run
spends $0.00.

The decisive turn, turning points, goals, shifts and perspective come
from one model judgment of the whole conversation (see judgment.py),
marked model estimate. The code rules below remain as the contrast when
the model is absent or a field fails its check, marked code fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from pathlib import Path

from .board import actual_class
from .judgment import SOURCE_MUSE, judge_conversation
from .labels import NEXT_ACTIONS
from .usage import BudgetExceeded, Meter
from .v2 import STAKES

REPORT_VERSION = "uptake-report/29"

#: Product sentence: the first paragraph of the README and the lead of
#: every HTML report.
PRODUCT_SENTENCE = (
    "Reads important conversations turn by turn, showing where what "
    "each speaker wanted in outcome, relationship, face, and "
    "understanding diverged from what actually happened, and what "
    "remains against each speaker's own goals."
)

#: Pre-registered reader measurements (C7). These describe the reader on
#: CaSiNo held-out dialogues and are identical in every case report;
#: they are never recomputed from the case at hand.
#: Split-half 0.863 (k=8, 3 stances, n=486 turns on 40 held-out-2
#: dialogues); outcome R1 +0.277 on held-out 1 (134 dialogues) and
#: +0.344 on held-out 2 (300 dialogues); perturbation ordering 16/20.
PREREG_SPLIT_HALF = 0.863
PREREG_K = 8
PREREG_OUTCOME_R1_HO1 = 0.277
PREREG_OUTCOME_R1_HO2 = 0.344
#: Single shown value: the held-out-2 R1 (larger set, 300 dialogues).
PREREG_OUTCOME_R1 = PREREG_OUTCOME_R1_HO2
PREREG_PERTURB = 0.8
PREREG_MEASURED_ON = (
    "CaSiNo held-out dialogues (pairwise reader, "
    "muse-spark-1.3-contributor): split-half k=8 on 40 held-out-2 "
    "dialogues (n=486 turns); outcome R1 on held-out 1 "
    "(134 dialogues, +0.277) and held-out 2 (300 dialogues, +0.344); "
    "perturbation ordering on 20 held-out-2 dialogues (16/20)."
)
PREREG_LIMIT = (
    "Reliability figures are pre-registered measurements on CaSiNo "
    "held-out sets, not values computed from this case: split-half "
    "0.863 (k=8, n=486 turns); outcome R1 +0.277 on held-out 1 and "
    "+0.344 on held-out 2 (report shows held-out 2); perturbation "
    "ordering 16/20 (0.80). Same values in every case report."
)

#: Panel next_action values that count as an ordinary continuation.
ROUTINE_READ = frozenset({"proceed", "comply", "ask_clarification"})

#: Move labels that carry an urgent or warning intent (general set).
URGENT_INTENT = frozenset({
    "appeal_need", "appeal_fairness", "pressure",
    "challenge", "probe", "reject",
})

#: Order in which urgent moves describe the turn's intent.
URGENT_ORDER = ("appeal_need", "appeal_fairness", "pressure",
                "challenge", "probe", "reject")

#: Response moves that show the other side changed course (engaged,
#: negotiated, repaired or pushed back). Anything else is "unchanged".
ENGAGED_RESPONSE = frozenset({
    "repair", "propose", "counter", "concede", "challenge", "reject",
    "pressure", "probe", "appeal_need", "appeal_fairness",
    "empathize", "reveal_preference",
})

#: General English gloss per move label (used for intended/read text).
MOVE_GLOSS = {
    "appeal_need": "ask for priority or help with an urgent need",
    "appeal_fairness": "appeal to fairness",
    "pressure": "press the other side to act",
    "challenge": "flag that something seems wrong",
    "probe": "test whether the other shares the concern",
    "reject": "refuse or push back",
    "inform": "state information",
    "ask": "ask for information or confirmation",
    "accept": "accept what was said or offered",
    "propose": "propose a course of action",
    "counter": "counter with a different course",
    "concede": "give in",
    "cooperate": "cooperate",
    "repair": "repair the misunderstanding",
    "deflect": "deflect or stall",
    "empathize": "show empathy",
    "praise": "praise",
    "joke": "joke",
    "smalltalk": "make small talk",
    "reveal_preference": "reveal a preference",
    "other": "say something else",
}

def load_case_meta() -> dict:
    """Per-case display metadata (domain, type, English glosses).

    Read from the ``report_meta.json`` sidecar at run time so this
    module embeds no case names or quoted phrases. Detection rules
    never read this table; it only labels the report. Returns {}
    when the file is missing or unreadable; callers then fall back
    to the case record and generic defaults.
    """
    path = Path(__file__).resolve().parent / "report_meta.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if not str(k).startswith("_")}


def load_stakes_records() -> dict:
    """Recorded stakes weights by case id.

    Read from the ``stakes_records.json`` sidecar at run time, the
    same way load_case_meta reads report_meta.json. Returns {}
    when the file is missing or unreadable; callers then fall back
    to the reading's own value.
    """
    path = Path(__file__).resolve().parent / "stakes_records.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {k: v for k, v in data.items() if not str(k).startswith("_")}

#: Severity rank for alert levels (higher = worse).
LEVEL_RANK = {"advisory": 0, "caution": 1, "warning": 2, "critical": 3}

#: Moves that can resolve an earlier alert when they appear later. Yielding
#: alone ("concede") is not uptake: in a warning context it is often the
#: dismissal of the concern ("maybe it is"), not a new course of action.
RESOLVE_MOVES = frozenset({"repair", "propose", "empathize"})

#: Look-ahead window (turns) for the resolved check.
RESOLVE_WINDOW = 5

#: Moves proving a turn does not continue the previous appeal (uptake,
#: repair or topic shift). A same-speaker turn carrying any of these
#: never inherits urgency from the turn before it.
APPEAL_BREAK = frozenset({"concede", "accept", "repair", "empathize",
                          "smalltalk"})

#: Moves by which the addressee withholds or contradicts (R2 counter).
COUNTER_MOVES = frozenset({"deflect", "repair", "challenge", "reject"})

#: Moves that contradict uptake: a turn carrying any of these is still
#: pressing or refusing, so a repair/propose/empathize label on the same
#: turn -- or a panel repair guess about it -- does not count as the
#: other side taking the warning up.
HOSTILE_MOVES = frozenset({"reject", "pressure", "challenge", "deflect"})

#: Look-ahead window (turns) for the R2 counter check.
COUNTER_WINDOW = 3

#: Genres where turns address a party by spoken designator (callsign).
#: A designator that changes mid-dialogue is presumptive confusion
#: only in addressed radio speech; elsewhere numbers vary routinely,
#: so the rule is inert for every other genre.
DESIGNATOR_GENRES = frozenset({"aviation_atc"})

#: Alphas that never establish a designator: articles, and instruction
#: nouns whose numbers change as a matter of course (a newly assigned
#: heading is not confusion). General vocabulary, no case content.
DESIGNATOR_STOPLIST = frozenset({
    "the", "a", "an",
    "heading", "headings", "runway", "runways", "altitude",
    "speed", "speeds", "frequency", "frequencies", "squawk",
    "flight level", "flight levels",
})

_INITIAL_DESIGNATOR_RE = re.compile(
    r"^\s*([A-Za-z]+(?:\s+[A-Za-z]+)?)\s+(\d[\d\-]*)")

BEFORE_PREFIX = ("Could have been flagged before the reply from the "
                 "utterance and prior turns: ")
AFTER_PREFIX = "Seen in hindsight after the reply: "


def _prefix_reason(prefix, reason) -> str:
    text = str(reason or "")
    if len(text) > 1 and text[0].isupper() and text[1].islower():
        return prefix + text[0].lower() + text[1:]
    return prefix + text


# --- small statistics ---------------------------------------------------------

def _mean(values) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def pearson(xs, ys):
    """Pearson r, or None when undefined (too few points / no variance)."""
    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    n = min(len(xs), len(ys))
    if n < 3:
        return None
    xs, ys = xs[:n], ys[:n]
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return None
    return cov / math.sqrt(vx * vy)


def wilson(k, n, z=1.96):
    """Wilson 95% interval for k successes in n trials."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


# --- record loading -----------------------------------------------------------

def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_inputs(case_id, data_root):
    """Load turns, case file and panel record for a case id.

    Returns (turns, case, panel_or_None, input_paths). Turns are
    ``{"speaker": str, "text": str}`` in file order (turn n = index n).
    """
    root = Path(data_root)
    hs = root / "highstakes"
    paths = []
    turns_path = hs / f"casino-{case_id}_turns.jsonl"
    case_path = hs / f"casino-{case_id}_case.json"
    panel_path = hs / "cases" / f"casino-{case_id}" / "panel_record.json"
    if turns_path.exists() and case_path.exists():
        paths = [turns_path, case_path]
        raw_turns = _read_jsonl(turns_path)
        turns = [{"speaker": str(t.get("speaker", "?")),
                  "text": str(t.get("text", ""))} for t in raw_turns]
        case = _read_json(case_path)
        panel = None
        if panel_path.exists():
            panel = _read_json(panel_path)
            paths.append(panel_path)
        return turns, case, panel, paths
    # Control dialogues live in the CaSiNo dataset by dialogue_id.
    casino_path = root / "datasets" / "casino.json"
    try:
        wanted = int(case_id)
    except (TypeError, ValueError):
        wanted = None
    if wanted is not None and casino_path.exists():
        data = _read_json(casino_path)
        for dlg in data if isinstance(data, list) else []:
            if dlg.get("dialogue_id") == wanted:
                logs = dlg.get("chat_logs", [])
                turns = [{"speaker": str(m.get("id", "?")),
                          "text": str(m.get("text", ""))} for m in logs]
                case = {}
                pinfo = dlg.get("participant_info")
                if isinstance(pinfo, dict):
                    case["participant_info"] = pinfo
                return turns, case, None, [casino_path]
    raise FileNotFoundError(f"no inputs for case {case_id!r} under {root}")


def _moves_by_turn(panel):
    out = {}
    if not panel:
        return out
    moves = panel.get("moves", [])
    if isinstance(moves, list):
        for m in moves:
            if isinstance(m, dict) and isinstance(m.get("turn_index"), int):
                out[m["turn_index"]] = m
    return out


def _effects_by_turn(panel):
    out = {}
    if not panel:
        return out
    effects = panel.get("effects", [])
    if isinstance(effects, list):
        for e in effects:
            if isinstance(e, dict) and isinstance(e.get("index"), int):
                out[e["index"]] = e
    return out


def _action_dist(effect) -> dict:
    dist = (effect or {}).get("action_dist", {})
    if isinstance(dist, dict):
        return {str(k): float(v) for k, v in dist.items()
                if isinstance(v, (int, float)) and math.isfinite(v)}
    readings = (effect or {}).get("readings", [])
    counts: dict[str, int] = {}
    if isinstance(readings, list):
        for r in readings:
            if isinstance(r, dict) and r.get("next_action"):
                key = str(r["next_action"])
                counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def routine_share(effect) -> float:
    """Share of panel readings that continue routinely."""
    dist = _action_dist(effect)
    if not dist:
        return 0.0
    return sum(v for k, v in dist.items() if k in ROUTINE_READ)


def modal_read(effect):
    """Most common panel next_action and its share."""
    dist = _action_dist(effect)
    if not dist:
        return "unknown", 0.0
    best = max(sorted(dist), key=lambda k: dist[k])
    return best, dist[best]


# --- general alert rule: urgent warning read as routine ------------------------

def next_other_speaker(moves_by_turn, turns, turn):
    """Next turn index after `turn` spoken by someone else, or None."""
    speaker = turns[turn]["speaker"] if 0 <= turn < len(turns) else None
    for j in range(turn + 1, len(turns)):
        if turns[j]["speaker"] != speaker:
            return j
    return None


def prev_other_speaker(turns, turn):
    """Most recent turn index before `turn` spoken by someone else, or None."""
    if not 0 <= turn < len(turns or []):
        return None
    me = turns[turn].get("speaker") if isinstance(turns[turn], dict) else None
    for j in range(turn - 1, -1, -1):
        other = turns[j].get("speaker") if isinstance(turns[j], dict) else None
        if other != me:
            return j
    return None


def score_turn(move_labels, effect, response_labels, repeated=False):
    """Alert level for one turn under the general rule.

    An urgent or warning intent (appeal / pressure / challenge / probe /
    reject) that the panel reads as an ordinary continuation is a
    mismatch. When the next reply shows no change of course the level
    is ``warning``; when the other side engages it is ``caution``.
    When the same speaker has to press again soon (``repeated``), the
    first raising did not land, so a mismatch is ``warning`` even if
    the reply engaged. Relationship scores never enter this rule.
    """
    labels = set(move_labels or [])
    urgent = [m for m in URGENT_ORDER if m in labels]
    share = routine_share(effect)
    if not urgent:
        return None, share, False, False
    mismatch = share >= 0.5
    engaged = bool(set(response_labels or []) & ENGAGED_RESPONSE)
    unchanged = not engaged
    if mismatch and unchanged:
        return "warning", share, True, True
    if mismatch and repeated:
        return "warning", share, True, False
    if mismatch:
        return "caution", share, True, False
    if share >= 1.0 / 3.0:
        return "advisory", share, False, unchanged
    return None, share, False, unchanged


def first_repeat(moves_by_turn, turns, turn, window):
    """First later turn where the same speaker presses again.

    Returns the turn index when the speaker of `turn` re-raises an
    urgent intent within `window`, else None. A repeated press is the
    record's own trace that the first raising was not taken up.
    """
    speaker = turns[turn]["speaker"] if 0 <= turn < len(turns) else None
    for j in range(turn + 1, min(turn + 1 + window, len(turns))):
        if turns[j]["speaker"] != speaker:
            continue
        labels = set(moves_by_turn.get(j, {}).get("moves", []) or [])
        if labels & URGENT_INTENT:
            return j
    return None


def inherited_urgent(moves_by_turn, effects_by_turn, turns, turn):
    """Urgent labels carried over from an adjacent same-speaker appeal.

    Returns (prev_turn, labels) when `turn` itself carries no urgent
    intent but directly continues the same speaker's urgent appeal
    that the panel read as routine, while `turn` is also read as
    routine and neither takes up nor shifts topic (see APPEAL_BREAK).
    Depth one: the previous turn must be originally urgent, so a
    concession or acknowledgement never inherits. Returns None when
    the turn stands on its own.
    """
    if turn < 1 or turn >= len(turns):
        return None
    labels = set(moves_by_turn.get(turn, {}).get("moves", []) or [])
    if labels & URGENT_INTENT:
        return None
    if labels & APPEAL_BREAK:
        return None
    prev = turn - 1
    if turns[prev]["speaker"] != turns[turn]["speaker"]:
        return None
    plabels = set(moves_by_turn.get(prev, {}).get("moves", []) or [])
    urgent = [m for m in URGENT_ORDER if m in plabels]
    if not urgent:
        return None
    if routine_share(effects_by_turn.get(prev)) < 0.5:
        return None
    if routine_share(effects_by_turn.get(turn)) < 0.5:
        return None
    return prev, urgent


def addressed_counter(moves_by_turn, turns, turn, window):
    """Later turn where the addressee withholds or contradicts.

    Returns (counter_turn, moves) for the first turn within `window`
    spoken by the addressed party -- the recorded addressee when the
    record names one, else any other speaker -- that deflects,
    repairs, challenges or rejects. Returns None when the addressee
    never pushes back in the window.
    """
    if turn < 0 or turn >= len(turns):
        return None
    addr = moves_by_turn.get(turn, {}).get("addressed_to")
    if isinstance(addr, list):
        addrs = {str(a) for a in addr if a}
    elif addr:
        addrs = {str(addr)}
    else:
        addrs = set()
    speaker = turns[turn]["speaker"]
    for j in range(turn + 1, min(turn + 1 + window, len(turns))):
        if addrs:
            if turns[j]["speaker"] not in addrs:
                continue
        elif turns[j]["speaker"] == speaker:
            continue
        clabels = set(moves_by_turn.get(j, {}).get("moves", []) or [])
        hit = sorted(clabels & COUNTER_MOVES)
        if hit:
            return j, hit
    return None


def refused_minority(effect) -> bool:
    """True when at least one panel reader would refuse the turn."""
    return _action_dist(effect).get("refuse", 0.0) > 0.0


def _repeat_within(mismatch, turn, speaker, window):
    for j in range(turn + 1, min(turn + 1 + window, len(mismatch))):
        cand = mismatch[j]
        if cand is not None and cand["speaker"] == speaker and cand["mismatch"]:
            return True
    return False


def _recovery_within(moves_by_turn, effects_by_turn, turn, window,
                     speaker=None):
    """True when someone else takes the warning up within `window`.

    A later turn counts as uptake only when it carries a resolving move
    (repair, a new proposal, empathy) with no hostile move on the same
    turn, and -- for move labels -- when it is spoken by someone other
    than `speaker` (the warner's own continued proposing is not uptake).
    A panel repair guess counts only for a non-hostile turn; against a
    hostile turn the record contradicts the guess.
    """
    for j in range(turn + 1, min(turn + 1 + window, max(
            len(moves_by_turn), len(effects_by_turn)) + 1)):
        mv = moves_by_turn.get(j, {})
        labels = set(mv.get("moves", []) or [])
        if labels & RESOLVE_MOVES and not labels & HOSTILE_MOVES:
            other = mv.get("speaker")
            if speaker is None or other is None or other != speaker:
                return True
        dist = _action_dist(effects_by_turn.get(j))
        if dist.get("repair", 0.0) >= 0.5:
            if not labels & HOSTILE_MOVES:
                return True
    return False


def _digits_norm(raw) -> str:
    return re.sub(r"\D", "", str(raw or ""))


def _initial_designator(text):
    """Turn-initial ALPHA + DIGITS mention, or None.

    Only a turn-initial mention establishes a new designator alpha;
    instruction numbers live mid-turn and never qualify. Returns
    (alpha, digits) with normalized digits.
    """
    m = _INITIAL_DESIGNATOR_RE.match(str(text or ""))
    if not m:
        return None
    alpha = " ".join(str(m.group(1)).split())
    if alpha.lower() in DESIGNATOR_STOPLIST:
        return None
    digits = _digits_norm(m.group(2))
    if len(digits) < 3:
        return None
    return alpha, digits


def _known_mentions(text, canonical):
    """Mentions of established designator alphas in one turn.

    ``canonical`` maps lower-alpha to {"digits", "alpha"}. Each
    established alpha is matched case-insensitively anywhere in the
    text; returns [(display alpha, digits)] deduplicated.
    """
    body = str(text or "")
    out = []
    seen = set()
    for key in sorted(canonical, key=len, reverse=True):
        pat = re.compile(r"\b%s\s+(\d[\d\-]*)" % re.escape(key),
                         re.IGNORECASE)
        for m in pat.finditer(body):
            digits = _digits_norm(m.group(1))
            if len(digits) < 3:
                continue
            if (key, digits) not in seen:
                seen.add((key, digits))
                out.append((canonical[key]["alpha"], digits))
    return out


def designator_timeline(turns):
    """Per-turn designator mentions and the first-seen form per alpha.

    Returns (mentions, canonical): mentions[t] is a list of
    {"alpha", "digits", "key"} for the turn, and canonical maps
    lower-alpha to {"digits", "alpha"} of first sight. A turn-initial
    mention establishes an unknown alpha; known alphas match anywhere
    in the text, so a mid-turn variant still counts.
    """
    canonical: dict = {}
    mentions: list = []
    for turn in turns or []:
        text = turn.get("text", "") if isinstance(turn, dict) else ""
        per: dict = {}
        for alpha, digits in _known_mentions(text, canonical):
            per[(alpha.lower(), digits)] = alpha
        init = _initial_designator(text)
        if init is not None:
            alpha, digits = init
            key = alpha.lower()
            if key not in canonical:
                canonical[key] = {"digits": digits, "alpha": alpha}
            per.setdefault((key, digits), alpha)
        mentions.append([{"alpha": a, "digits": d, "key": k}
                         for (k, d), a in per.items()])
    return mentions, canonical


def build_designator_alerts(turns, genre, outcome_turn):
    """R3 alerts: a spoken designator changes mid-dialogue (ATC only).

    The first-seen form per alpha is canonical; a later turn using
    the same alpha with different digits is confusion (``warning``).
    It resolves when the canonical form is used again in a later
    turn -- alignment restored -- and stays open when it never
    returns. Inert for every genre outside :data:`DESIGNATOR_GENRES`.
    The variant is visible from the
    utterance and the first-seen form, while the return check only
    sets ``resolved``.
    """
    if genre not in DESIGNATOR_GENRES:
        return []
    mentions, canonical = designator_timeline(turns)
    sightings: dict = {}
    for t, per in enumerate(mentions):
        for m in per:
            sightings.setdefault((m["key"], m["digits"]), []).append(t)
    n = len(turns or [])
    alerts = []
    for t, per in enumerate(mentions):
        for m in per:
            canon = canonical.get(m["key"], {})
            if not canon or m["digits"] == canon["digits"]:
                continue
            disp_canon = f"{canon['alpha']} {canon['digits']}"
            disp_var = f"{m['alpha']} {m['digits']}"
            later = [p for p in sightings.get((m["key"],
                                              canon["digits"]), [])
                     if p > t]
            if later:
                resolved = True
                tail = f"'{disp_canon}' returns at #{later[0]}."
            else:
                resolved = False
                tail = f"'{disp_canon}' never returns."
            reason = (f"Designator '{disp_canon}' appears as "
                      f"'{disp_var}' here; {tail}")
            reason = _prefix_reason(BEFORE_PREFIX, reason)
            alerts.append({"turn": t, "level": "warning",
                           "axis": "designator-inconsistency",
                           "reason": reason, "resolved": bool(resolved)})
            break  # one alert per turn; the first variant wins
    return alerts


def build_alerts(turns, moves_by_turn, effects_by_turn, outcome_turn,
                 genre=None):
    """Alert timeline under the general uptake-failure rules.

    R1 (urgent-read-as-routine): an urgent intent the panel reads as
    an ordinary continuation. The level is ``warning`` when the next
    reply shows no uptake or the speaker has to press again; a
    same-speaker continuation of such an appeal inherits its urgency.
    R2 (routine-read-then-countered): a routine-read turn one panel
    reader would refuse, which the addressee then withholds or
    contradicts -- the turn ran ahead of alignment.
    R3 (designator-inconsistency): in addressed radio speech, a
    spoken designator that changes mid-dialogue warns until the
    first-seen form returns (see :func:`build_designator_alerts`).
    """
    n = len(turns)
    scored = []
    for t in range(n):
        mv = moves_by_turn.get(t, {})
        labels = [str(m) for m in (mv.get("moves", []) or [])]
        inherit = inherited_urgent(moves_by_turn, effects_by_turn,
                                   turns, t)
        eff_labels = list(labels)
        if inherit is not None:
            for u in inherit[1]:
                if u not in eff_labels:
                    eff_labels.append(u)
        nxt = next_other_speaker(moves_by_turn, turns, t)
        resp = (moves_by_turn.get(nxt, {}).get("moves", []) or []) if nxt is not None else []
        rep = None
        if set(eff_labels) & URGENT_INTENT:
            rep = first_repeat(moves_by_turn, turns, t, RESOLVE_WINDOW)
        level, share, mismatch, unchanged = score_turn(
            eff_labels, effects_by_turn.get(t), resp,
            repeated=rep is not None)
        scored.append({"turn": t, "level": level, "share": share,
                       "mismatch": mismatch, "unchanged": unchanged,
                       "repeat": rep,
                       "inherited_from": inherit[0] if inherit else None,
                       "speaker": turns[t]["speaker"] if t < n else "?",
                       "moves": [str(m) for m in labels],
                       "eff_moves": [str(m) for m in eff_labels],
                       "response": [str(m) for m in resp]})
    alerts = []
    for row in scored:
        t = row["turn"]
        if row["level"] is not None:
            urgent = [m for m in URGENT_ORDER if m in row["eff_moves"]]
            intent = MOVE_GLOSS.get(urgent[0], urgent[0]) if urgent else "speak"
            modal, prob = modal_read(effects_by_turn.get(t))
            if modal in ROUTINE_READ:
                read_bit = f"read as routine ({modal} {prob:.0%})"
            else:
                read_bit = (f"partly read as routine ({row['share']:.0%}; "
                            f"modal read {modal} {prob:.0%})")
            if row["unchanged"]:
                tail = "the next reply stays on the same course"
            else:
                tail = "the next reply engages but the read stays routine"
            core = f"urgent intent ({intent}) {read_bit}; {tail}"
            if row["inherited_from"] is not None:
                core = (f"same-speaker continuation of "
                        f"#{row['inherited_from']} with {core}")
            reason = core[0].upper() + core[1:] + "."
            if row["repeat"] is not None:
                reason += (f" The same speaker has to press again "
                           f"at #{row['repeat']}.")
            reason = _prefix_reason(BEFORE_PREFIX, reason)
            if row["repeat"] is not None:
                # The repeat proves this raising did not land at once; the
                # issue is resolved only when the repeat itself finds
                # uptake (delayed uptake). The reason already discloses
                # the delay ("has to press again at #N").
                resolved = _recovery_within(moves_by_turn,
                                            effects_by_turn, row["repeat"],
                                            RESOLVE_WINDOW,
                                            speaker=row["speaker"])
            else:
                resolved = _recovery_within(moves_by_turn, effects_by_turn,
                                            t, RESOLVE_WINDOW,
                                            speaker=row["speaker"])
                if _repeat_within(scored, t, row["speaker"],
                                  RESOLVE_WINDOW):
                    resolved = False
            alerts.append({"turn": t, "level": row["level"],
                           "axis": "urgent-read-as-routine",
                           "reason": reason, "resolved": bool(resolved)})
            continue
        hit = countered_routine(moves_by_turn, effects_by_turn, turns,
                                t, row)
        if hit is None:
            continue
        cj, cmoves, modal, prob = hit
        reason = (f"Read as routine ({modal} {prob:.0%}) while one panel "
                  f"reader would refuse; the addressee then pushes back "
                  f"at #{cj} ({'/'.join(cmoves)}). The turn ran ahead of "
                  f"alignment.")
        reason = _prefix_reason(AFTER_PREFIX, reason)
        resolved = _recovery_within(moves_by_turn, effects_by_turn, t,
                                    RESOLVE_WINDOW,
                                    speaker=row["speaker"])
        if _repeat_within(scored, t, row["speaker"], RESOLVE_WINDOW):
            resolved = False
        alerts.append({"turn": t, "level": "warning",
                       "axis": "routine-read-then-countered",
                       "reason": reason, "resolved": bool(resolved)})
    alerts.extend(build_designator_alerts(turns, genre, outcome_turn))
    alerts.sort(key=lambda a: (a.get("turn", 0), a.get("axis", "")))
    return alerts, scored


def countered_routine(moves_by_turn, effects_by_turn, turns, turn, row):
    """R2 gate: routine read, refused minority, addressed counter.

    Returns (counter_turn, counter_moves, modal, prob) when `turn`
    carries no urgent intent, the panel mostly reads it as routine
    while at least one reader would refuse, and the addressee
    withholds or contradicts soon after. Else None.
    """
    labels = set(moves_by_turn.get(turn, {}).get("moves", []) or [])
    if labels & URGENT_INTENT:
        return None
    modal, prob = modal_read(effects_by_turn.get(turn))
    if modal not in ROUTINE_READ or row["share"] < 0.5:
        return None
    if not refused_minority(effects_by_turn.get(turn)):
        return None
    hit = addressed_counter(moves_by_turn, turns, turn, COUNTER_WINDOW)
    if hit is None:
        return None
    cj, cmoves = hit
    return cj, cmoves, modal, prob


#: Alert level per axis gap (task70). Only missed, partly and cost
#: raise alerts; met, not_relevant and unclear never do.
AXIS_ALERT_LEVEL = {"partly": "caution", "missed": "warning",
                    "cost": "critical"}

#: Look-ahead window (turns) for the axis resolved check.
AXIS_RESOLVE_WINDOW = 5


def build_axis_alerts(uptake, resolve_window=AXIS_RESOLVE_WINDOW):
    """Alert timeline from the four-stake reading (task70).

    One alert per (turn, stake) whose gap is missed, partly or cost.
    Each reason states what was wanted on that stake and what
    happened instead, with the reply-turn evidence. An alert is
    resolved when the same stake reads ``met`` within
    ``resolve_window`` turns. General rule: no case names, no quoted
    phrases, no relationship-only triggers.
    """
    try:
        window = int(resolve_window)
    except (TypeError, ValueError):
        window = AXIS_RESOLVE_WINDOW
    entries = [u for u in (uptake or []) if isinstance(u, dict)]
    by_turn = {}
    for u in entries:
        if isinstance(u.get("turn"), int):
            by_turn[u["turn"]] = u
    alerts = []
    for u in entries:
        turn = u.get("turn")
        if not isinstance(turn, int) or isinstance(turn, bool):
            continue
        axes = u.get("axes", {}) or {}
        if not isinstance(axes, dict):
            continue
        for axis in STAKES:
            entry = axes.get(axis)
            if not isinstance(entry, dict):
                continue
            gap = str(entry.get("gap", ""))
            level = AXIS_ALERT_LEVEL.get(gap)
            if level is None:
                continue
            wanted = " ".join(str(entry.get("wanted", "")).split())
            happened = " ".join(str(entry.get("happened", "")).split())
            reply = entry.get("reply_turn")
            quote = " ".join(str(entry.get("quote", "")).split())
            if wanted and happened:
                core = (f"On {axis}, wanted {wanted} but "
                        f"{happened} ({gap}).")
            elif happened:
                core = f"On {axis}, {happened} ({gap})."
            elif wanted:
                core = f"On {axis}, wanted {wanted} ({gap})."
            else:
                core = f"On {axis}, the gap reads {gap}."
            if isinstance(reply, int) and not isinstance(reply, bool) \
                    and quote:
                core += f" Evidence #{reply}: {quote}."
            elif isinstance(reply, int) and not isinstance(reply, bool):
                core += f" Evidence #{reply}."
            resolved = False
            for j in range(turn + 1, turn + 1 + max(0, window)):
                later = by_turn.get(j)
                if not isinstance(later, dict):
                    continue
                later_axes = later.get("axes", {}) or {}
                if not isinstance(later_axes, dict):
                    continue
                later_entry = later_axes.get(axis)
                if isinstance(later_entry, dict) and \
                        str(later_entry.get("gap", "")) == "met":
                    resolved = True
                    break
            alerts.append({"turn": int(turn), "level": level,
                           "axis": axis, "reason": core,
                           "resolved": bool(resolved)})
    alerts.sort(key=lambda a: (a.get("turn", 0), a.get("axis", "")))
    return alerts


#: Turn score per stake gap. A first rule to test, not a measurement.
GAP_SCORE = {"met": 1.0, "partly": 0.5, "missed": -1.0, "cost": -1.0}

#: Turns over which a shift in standing is measured.
SPLIT_WINDOW = 3


def build_standing(uptake, window=SPLIT_WINDOW):
    """Standing per stake: running score per speaker, never merged.

    Returns {"series": {stake: {speaker: [{"turn", "value"}]}},
    "end": {speaker: {stake: {"score", "judged", "mean"}}},
    "splits": {stake: {"turn", "shift", "between"} or None}}. A split is
    the turn where the margin between the two most-read speakers on that
    stake moves most over ``window`` turns; a stake read for only one
    speaker uses that speaker's own running score.
    """
    entries = [u for u in (uptake or []) if isinstance(u, dict)
               and isinstance(u.get("turn"), int)]
    speakers = []
    for u in entries:
        sp = str(u.get("speaker", "?"))
        if sp not in speakers:
            speakers.append(sp)
    series = {a: {sp: [] for sp in speakers} for a in STAKES}
    end = {sp: {a: {"score": 0.0, "judged": 0, "mean": None}
                for a in STAKES} for sp in speakers}
    steps = {a: {} for a in STAKES}
    for u in entries:
        sp, turn = str(u.get("speaker", "?")), u["turn"]
        axes = u.get("axes") or {}
        for a in STAKES:
            gap = str((axes.get(a) or {}).get("gap", ""))
            if gap in GAP_SCORE:
                end[sp][a]["score"] += GAP_SCORE[gap]
                end[sp][a]["judged"] += 1
                steps[a].setdefault(turn, {})[sp] = GAP_SCORE[gap]
            for other in speakers:
                series[a][other].append(
                    {"turn": turn, "value": round(end[other][a]["score"], 2)})
    for sp in speakers:
        for a in STAKES:
            row = end[sp][a]
            row["score"] = round(row["score"], 2)
            if row["judged"]:
                row["mean"] = round(row["score"] / row["judged"], 3)
    splits = {}
    for a in STAKES:
        ranked = sorted(speakers, key=lambda s: -end[s][a]["judged"])
        ranked = [s for s in ranked if end[s][a]["judged"]]
        if not ranked:
            splits[a] = None
            continue
        pair = ranked[:2]
        margin = {}
        for t, row in steps[a].items():
            first = row.get(pair[0], 0.0)
            second = row.get(pair[1], 0.0) if len(pair) > 1 else 0.0
            margin[t] = first - second
        best = None
        for t in sorted(margin):
            shift = sum(v for k, v in margin.items() if t <= k < t + window)
            if best is None or abs(shift) > abs(best[1]):
                best = (t, shift)
        splits[a] = {"turn": best[0], "shift": round(best[1], 2),
                     "between": pair}
    return {"series": series, "end": end, "splits": splits}


def turn_scales(effect) -> dict:
    """Three scales from the panel readings of one turn (no panel: None)."""
    if not isinstance(effect, dict):
        return {"next_action": None, "relationship": None, "face": None}
    dist = effect.get("action_dist") or {}
    nxt = max(dist, key=dist.get) if dist else None
    def mean(key):
        vals = [v for v in (effect.get(key) or [])
                if isinstance(v, (int, float))]
        return round(sum(vals) / len(vals), 3) if vals else None
    return {"next_action": nxt, "relationship": mean("relationship"),
            "face": mean("face")}


#: Gap values that count as a judged reading (evidence on record).
JUDGED_GAPS = ("met", "partly", "missed", "cost")


def stake_tally(uptake):
    """Per-stake gap counts from the per-turn readings.

    Returns {stake: {"met": n, "partly": n, "missed": n, "cost": n,
    "turns": [...]}} with one row per stake in STAKES order. Stakes
    stay separate and are never merged into one number. ``turns``
    lists the judged turns (gap met, partly, missed or cost) in
    order; not_relevant and unclear readings count nowhere.
    """
    tally = {axis: {"met": 0, "partly": 0, "missed": 0, "cost": 0,
                   "turns": []}
             for axis in STAKES}
    for u in (uptake or []):
        if not isinstance(u, dict):
            continue
        turn = u.get("turn")
        axes = u.get("axes", {}) or {}
        if not isinstance(axes, dict):
            continue
        for axis in STAKES:
            entry = axes.get(axis)
            if not isinstance(entry, dict):
                continue
            gap = str(entry.get("gap", ""))
            if gap not in JUDGED_GAPS:
                continue
            tally[axis][gap] += 1
            if isinstance(turn, int) and not isinstance(turn, bool):
                tally[axis]["turns"].append(int(turn))
    return tally


def speaker_stake_tally(uptake):
    """Per-speaker, per-stake gap counts from the readings.

    Returns {speaker: {stake: {"met": n, "partly": n, "missed": n,
    "cost": n}}} in first-appearance speaker order. Each speaker's
    stakes stay separate and are never merged into one number;
    not_relevant and unclear readings count nowhere.
    """
    out: dict = {}
    for u in (uptake or []):
        if not isinstance(u, dict):
            continue
        speaker = str(u.get("speaker", "?"))
        axes = u.get("axes", {}) or {}
        if not isinstance(axes, dict):
            continue
        row = out.setdefault(
            speaker, {axis: {"met": 0, "partly": 0, "missed": 0,
                             "cost": 0} for axis in STAKES})
        for axis in STAKES:
            entry = axes.get(axis)
            if not isinstance(entry, dict):
                continue
            gap = str(entry.get("gap", ""))
            if gap in JUDGED_GAPS:
                row[axis][gap] += 1
    return out


# --- decisive turn --------------------------------------------------------------

def pick_decisive(turns, moves_by_turn, effects_by_turn, scored):
    """Panel-only decisive turn: strongest urgent-read-as-routine mismatch.

    Used when no per-item reading shows a measured gap (see
    :func:`pick_decisive_layered`).

    Severity is the routine share plus panel disagreement (both from the
    record); ties go to the earliest turn. With no mismatch anywhere,
    the most disagreed-about turn wins; with no panel at all, turn 0
    stands in and the limits say so.
    """
    best = None
    for row in (scored or []):
        if not row["mismatch"]:
            continue
        eff = effects_by_turn.get(row["turn"], {})
        try:
            dis = float(eff.get("disagreement", 0.0))
        except (TypeError, ValueError):
            dis = 0.0
        if not math.isfinite(dis):
            dis = 0.0
        key = (row["share"] + dis, -row["turn"])
        if best is None or key > best[0]:
            best = (key, row["turn"])
    if best is not None:
        return best[1], "strongest urgent-read-as-routine mismatch"
    peak, peak_dis = None, None
    for t in range(len(turns)):
        eff = effects_by_turn.get(t, {})
        try:
            dis = float(eff.get("disagreement", 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(dis):
            continue
        if peak is None or dis > peak_dis:
            peak, peak_dis = t, dis
    if peak is not None:
        return peak, "highest panel disagreement (no urgent mismatch found)"
    return 0, "turn 0 (no panel record)"


def top_measured_item(reading, use):
    """Measured item with the largest gap in one turn (None: no such item).

    The gap is the one severity is built from: wanted x (100 - actual) / 100.
    Ties go to the first item, as in the layers above the items.
    """
    best, best_gap = None, 0.0
    for name, e in (reading.get("items") or {}).items():
        if name in use and e.get("wanted") and e.get("actual"):
            gap = e["wanted"]["score"] * (100 - e["actual"]["score"]) / 100
            if gap > best_gap:
                best, best_gap = name, gap
    return best


def pick_decisive_layered(uptake, use):
    """Turn with the largest layers.severity, ties to the earliest turn.

    Returns (turn, item), or None when no reading shows a measured gap
    (offline, every reading is blank) so the panel rule applies.
    """
    best = None
    for r in (uptake or []):
        severity = (r.get("layers") or {}).get("severity") or 0.0
        item = top_measured_item(r, use)
        if item is None or severity <= 0.0:
            continue
        key = (severity, -r["turn"])
        if best is None or key > best[0]:
            best = (key, r["turn"], item)
    return (best[1], best[2]) if best else None


NO_BROKEN_ITEM = "no broken item of measured or reference grade"


def pick_criterion(reading, use, grades=None, turn=None):
    """The one item the alternatives of a decisive turn are compared on.

    Among the turn's broken items, measured grade first, then reference
    grade, the largest wanted - actual; ties go to the first item. With
    neither, ``item`` is None and the speaker's intended sentence stands in.
    ``grades`` maps item to grade; an item absent from it counts as reference.
    """
    from .multilayer import broken_items
    items = (reading or {}).get("items") or {}
    broken = broken_items(items)
    measured = [i for i in broken if i in use]
    reference = [i for i in broken if i not in use
                 and (grades or {}).get(i, "reference") == "reference"]
    pool, grade = (measured, "measured") if measured else (reference, "reference")
    if not pool:
        return {"item": None, "grade": None, "wanted": None, "actual": None,
                "source": (f"{NO_BROKEN_ITEM} on turn #{turn}; the criterion "
                           f"is the speaker's intended sentence")}
    item = max(pool, key=lambda i: items[i]["wanted"]["score"] - items[i]["actual"]["score"])
    where = ("among measured broken items" if grade == "measured"
             else "among reference-grade broken items (no measured item broke)")
    return {"item": item, "grade": grade, "wanted": items[item]["wanted"]["score"],
            "actual": items[item]["actual"]["score"],
            "source": f"largest wanted minus actual on turn #{turn}, {where}"}


def _grade_note(criterion):
    """Words that mark a criterion as reference grade; empty when measured."""
    return ", reference grade" if criterion.get("grade") == "reference" else ""


def _layered_intent(reading, item, why, grade_note=""):
    """Intended, read and basis of one turn from one item's reading."""
    from .multilayer import ITEM_CLASS, STAKE_ITEMS
    e = reading["items"][item]
    wanted, actual = e["wanted"]["score"], e["actual"]["score"]
    gloss = STAKE_ITEMS[ITEM_CLASS[item]][item]
    read = f"{item}: the wanted effect landed {actual:.0f}/100{grade_note}"
    if e.get("quote"):
        read += f" (reply #{e['reply_turn']}: {e['quote']})"
    source = (f"model estimate: per-item reading of turn #{reading['turn']}, "
              f"{why}; item readings are model judgments, not record facts")
    return f"{item}: reached for {wanted:.0f}/100 ({gloss})", read, source


def _judged_values(reading, use):
    """(item, wanted, actual) copied from one turn's reading, or Nones."""
    item = top_measured_item(reading, use) if reading else None
    if item is None:
        return None, None, None
    e = reading["items"][item]
    return item, e["wanted"]["score"], e["actual"]["score"]


def build_decisive(case_id, turns, case, moves_by_turn, effects_by_turn,
                   scored, uptake=None, use=(), judged=None, grades=None):
    """Decisive turn. ``judged`` (turn, why, opened, received, source) wins
    when given; else the largest severity, else the panel rule. The item,
    its values and the read's bracket follow :func:`pick_criterion`."""
    layered = pick_decisive_layered(uptake, use)
    if judged:
        layered = None
        turn, why = judged["turn"], judged["why"]
    elif layered:
        turn, why = layered[0], "largest severity in the per-item readings"
    else:
        turn, why = pick_decisive(turns, moves_by_turn, effects_by_turn, scored)
    mv = moves_by_turn.get(turn, {})
    labels = [str(m) for m in (mv.get("moves", []) or [])]
    quote = str(mv.get("quote", "") or "")
    said = turns[turn]["text"] if 0 <= turn < len(turns) else ""
    if not quote or (layered and quote not in said):
        quote = said[:160]
    urgent = [m for m in URGENT_ORDER if m in labels]
    if urgent:
        intended = f"{urgent[0]}: {MOVE_GLOSS.get(urgent[0], urgent[0])}"
    elif labels:
        intended = f"{labels[0]}: {MOVE_GLOSS.get(labels[0], labels[0])}"
    else:
        intended = "unknown"
    if moves_by_turn and labels:
        intended_source = (
            f"model estimate: panel move label(s) "
            f"{'/'.join(labels)} on turn #{turn} "
            f"(cases/casino-{case_id}/panel_record.json); "
            f"move labels are model judgments, not record facts")
    elif moves_by_turn:
        intended_source = (
            f"model estimate: no move labels on turn #{turn} "
            f"(cases/casino-{case_id}/panel_record.json); "
            f"move labels are model judgments, not record facts")
    else:
        intended_source = ("model estimate: no panel record for this case; "
                           "needs a model pass")
    modal, prob = modal_read(effects_by_turn.get(turn))
    readings = (effects_by_turn.get(turn, {}) or {}).get("readings", [])
    rels, faces = [], []
    if isinstance(readings, list):
        for r in readings:
            if not isinstance(r, dict):
                continue
            if isinstance(r.get("relationship_shift"), (int, float)):
                rels.append(float(r["relationship_shift"]))
            if isinstance(r.get("face_threat"), (int, float)):
                faces.append(float(r["face_threat"]))
    scales = {"next_action": modal,
              "relationship": round(_mean(rels), 3) if rels else 0.0,
              "face": round(-_mean(faces), 3) if faces else 0.0}
    read, read_prob = modal, round(float(prob), 3)
    reading = next((r for r in (uptake or []) if r.get("turn") == turn), None)
    criterion = pick_criterion(reading, use, grades, turn)
    item = criterion["item"]
    extra = {"item": item, "grade": criterion["grade"], "wanted": criterion["wanted"],
             "actual": criterion["actual"]} if item else {}
    if layered:
        # An item reading is a score, not a vote share: no probability to give.
        if item:
            intended, read, intended_source = _layered_intent(
                reading, item, f"{item}: {criterion['source']}", _grade_note(criterion))
        else:
            intended, read, intended_source = _layered_intent(
                reading, layered[1], f"{layered[1]} has the largest gap among measured "
                f"items; {criterion['source']}")
            read += f" ({NO_BROKEN_ITEM})"
        read_prob = None
    if judged:
        if quote not in said:
            quote = said[:160]
        intended, read, read_prob = judged["opened"], judged["received"], None
        if item:
            read += (f" (reading: {item} landed {criterion['actual']:.0f}/100"
                     f"{_grade_note(criterion)})")
        else:
            read += f" (reading: {NO_BROKEN_ITEM})"
        intended_source = f"{judged['source']}: {why}"
        extra["source"] = judged["source"]
    return {"turn": int(turn), "quote": quote, "intended": intended,
            "intended_source": intended_source, "said": said,
            "read": read, "read_prob": read_prob,
            "scales": scales, "pick_reason": why, "criterion": criterion, **extra}


# --- speakers: goals, shifts, end balance ----------------------------------------

def _primary_strategy(labels):
    """One strategy per turn: first urgent move, else first move, else other."""
    lab = [str(m) for m in (labels or [])]
    for u in URGENT_ORDER:
        if u in lab:
            return u
    if lab:
        return lab[0]
    return "other"


def _strategy_gloss(strategy):
    return f"{strategy}: {MOVE_GLOSS.get(strategy, strategy)}"


def _parse_points(value):
    """Deal points as int, or None when missing/unparseable."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
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


#: Participant satisfaction labels to [-1, 1] (general Likert map).
_SAT_TO_BALANCE = {
    "Extremely dissatisfied": -1.0,
    "Slightly dissatisfied": -0.5,
    "Undecided": 0.0,
    "Slightly satisfied": 0.5,
    "Extremely satisfied": 1.0,
}


def _satisfaction_to_balance(value):
    """Satisfaction label (or -2..2 int) to [-1, 1], else None."""
    if isinstance(value, str):
        return _SAT_TO_BALANCE.get(value.strip())
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and -2 <= value <= 2:
        return value / 2.0
    return None


def _value_goal_sentence(speaker, value2issue):
    """Record-based goal sentence from a value -> issue record.

    The High/Medium/Low ranking becomes one sentence ("wants Food
    most, then Water, then Firewood"); unknown levels read as plain
    clauses. Returns None when the record names no values.
    """
    if not isinstance(value2issue, dict) or not value2issue:
        return None
    ranked = [value2issue.get(level)
              for level in ("High", "Medium", "Low")]
    ranked = [str(v) for v in ranked if v]
    extra = [f"{level} value on {value2issue[level]}"
             for level in sorted(str(k) for k in value2issue)
             if level not in ("High", "Medium", "Low")
             and value2issue.get(level)]
    if not ranked and not extra:
        return None
    if ranked:
        want = f"wants {ranked[0]} most"
        if len(ranked) > 1:
            want += ", then " + ", then ".join(ranked[1:])
        clauses = [want, *[f"places {e}" for e in extra]]
    else:
        clauses = [f"places {e}" for e in extra]
    return f"Speaker {speaker} {' and '.join(clauses)} in this negotiation."


#: Infinitive verb phrases that read better than the bare move gloss
#: inside a goal sentence. General English; only the label's gloss
#: varies, never case content.
_GOAL_VERBS = {
    "inform": "share information",
    "praise": "praise the other side",
    "joke": "joke with the other side",
}


def _goal_verb(strategy):
    if strategy in _GOAL_VERBS:
        return _GOAL_VERBS[strategy]
    return MOVE_GLOSS.get(strategy, str(strategy).replace("_", " "))


def _goal_sentence(speaker, strategy):
    """Offline one-sentence goal from a first-move label (general rule)."""
    return (f"Speaker {speaker} wants to {_goal_verb(strategy)} "
            f"in this conversation.")


def _strategy_phrase(strategy):
    """Shift endpoint as a plain phrase (no 'label:' prefix)."""
    return MOVE_GLOSS.get(strategy, str(strategy).replace("_", " "))


#: General goal-writing brief: one sentence per speaker, grounded in
#: the speaker's own turns in this conversation only.
GOAL_PROMPT = (
    "You read one conversation and state each listed speaker's goal.\n"
    "For each speaker, write exactly one English sentence saying what "
    "that speaker wanted in THIS conversation: their aim in the "
    "exchange itself, not their character, personality or ability.\n"
    "Ground each sentence in the speaker's own turns shown below. "
    "Read the utterances, not the person.\n"
    "Reply with JSON only: "
    "{\"goals\": {\"<speaker>\": \"<one sentence>\", ...}}."
)

#: Context cap for the goal prompt (chars per turn; every turn goes in, so a
#: speaker who first talks late is still seen).
GOAL_TURN_CHARS = 400

_MOVE_TOKEN_RE = re.compile(
    r"\b(appeal_need|appeal_fairness|reveal_preference)\b")
_LABEL_PREFIX_RE = re.compile(r"^[a-z_]+:\s")


def _goals_cache_path(cache_dir, key):
    return Path(cache_dir) / f"goal-{key}.json"


def _valid_goals(data, speakers):
    """Model goal sentences worth keeping, keyed by speaker.

    Keeps one-line sentences only: entries shaped like move labels
    ("ask: ...") or carrying raw move tokens are dropped so the
    caller falls back to offline sentences for those speakers.
    """
    if not isinstance(data, dict):
        return {}
    goals = data.get("goals")
    if not isinstance(goals, dict):
        return {}
    out = {}
    for sp in speakers:
        text = goals.get(sp)
        if not isinstance(text, str):
            continue
        s = " ".join(text.split()).strip().strip('"').strip()
        if not (8 <= len(s) <= 400) or " " not in s:
            continue
        if "\n" in s or _LABEL_PREFIX_RE.match(s):
            continue
        if _MOVE_TOKEN_RE.search(s):
            continue
        if not s.endswith("."):
            s += "."
        out[sp] = s
    return out


def model_speaker_goals(turns, hints, client, model, meter, cache_dir):
    """One-sentence per-speaker goals from the model (cached, budget-aware).

    One call per distinct (model, context, hints); identical inputs
    reuse the cached goals without re-querying. Returns {} when no
    usable goals parse (callers fall back to offline sentences).
    BudgetExceeded propagates so the run stops spending.
    """
    speakers = list(hints)
    if not speakers:
        return {}
    context = []
    for turn in (turns or []):
        if isinstance(turn, dict):
            text = " ".join(str(turn.get("text", "")).split())
            if len(text) > GOAL_TURN_CHARS:
                text = text[:GOAL_TURN_CHARS - 3].rstrip() + "..."
            context.append(f"{turn.get('speaker', '?')}: {text}")
    payload = json.dumps(
        {"kind": "goals-v1", "model": model, "context": context,
         "hints": {sp: str(hints[sp]) for sp in speakers}},
        ensure_ascii=False, sort_keys=True).encode("utf-8")
    key = _sha_bytes(payload)
    path = _goals_cache_path(cache_dir, key)
    if path.exists():
        try:
            kept = _valid_goals(_read_json(path), speakers)
            if kept:
                return kept
        except (OSError, ValueError):
            pass
    if meter is not None:
        meter.check()
    ctx = "\n".join(f"- {line}" for line in context) or "- (no turns)"
    hint_lines = "\n".join(f"- {sp}: {hints[sp]}" for sp in speakers)
    messages = [{"role": "system", "content": GOAL_PROMPT},
                {"role": "user",
                 "content": f"Conversation:\n{ctx}\nSpeakers:\n{hint_lines}\n"
                            f"Goals:"}]
    for _ in range(2):
        try:
            raw = client.complete(messages, model=model, temperature=0.0)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        try:
            text = str(raw or "")
            start = text.find("{")
            if start < 0:
                continue
            data, _ = json.JSONDecoder().raw_decode(text[start:])
            kept = _valid_goals(data, speakers)
            if kept:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"goals": kept}, f, ensure_ascii=False)
                return kept
        except (ValueError, TypeError):
            continue
    return {}


def build_speakers(turns, moves_by_turn, alerts, effects_by_turn=None,
                   case=None, client=None, model=None, meter=None,
                   cache_dir=None):
    """Per-speaker goals, strategy shifts and goal-relative end balance.

    General rules over the record: speakers in order of first
    appearance; the initial goal is one sentence saying what the
    speaker wanted in this conversation -- a model-written sentence
    grounded in the speaker's own turns when a model is available,
    else a sentence built from the primary strategy of the
    speaker's first labeled turn, or the recorded value ranking
    when no move labels exist but the participant record names one;
    a strategy shift is a change of primary strategy between two
    consecutive turns of the same speaker. Every goal and shift
    carries a verbatim quote from the turn's own text. The end
    balance is measured against the speaker's own goals in this
    conversation only, never as a score of the person:

    - when any urgent press exists on record, each urgent turn
      scores +1 (no alert, or a resolved alert: the urgency
      landed), -0.5 (unresolved caution/advisory: the reply
      engaged but the read stayed routine -- partial uptake) or
      -1 (unresolved warning/critical: the reply stayed on the
      same course -- missed), averaged over the speaker's urgent
      turns, 0.0 when the speaker never pressed;
    - when no urgent press exists but move labels do, each turn
      with a panel read scores +1 when the read matches the
      intent (routine bids read routinely, urgent bids read
      urgently) and -1 otherwise, averaged over read turns;
    - when neither exists (control dialogues without a panel
      record), the participant record's deal points decide:
      (own - others) / (own + others), or the satisfaction label
      when points are missing.
    """
    order = []
    own: dict = {}
    for t, turn in enumerate(turns or []):
        sp = turn.get("speaker", "?") if isinstance(turn, dict) else "?"
        if sp not in own:
            own[sp] = []
            order.append(sp)
        own[sp].append(t)
    alerts_by_turn: dict = {}
    for a in (alerts or []):
        if isinstance(a, dict) and isinstance(a.get("turn"), int):
            alerts_by_turn.setdefault(a["turn"], []).append(a)
    moves = moves_by_turn or {}
    effects = effects_by_turn or {}
    total_urgent = 0
    for t in range(len(turns or [])):
        labs = set(str(m) for m in (moves.get(t, {}).get("moves", [])
                                    or []))
        if labs & URGENT_INTENT:
            total_urgent += 1
    has_moves = bool(moves)
    pinfo = {}
    if isinstance(case, dict) and isinstance(
            case.get("participant_info"), dict):
        pinfo = case["participant_info"]
    points: dict = {}
    sat_raw: dict = {}
    val_goal: dict = {}
    for sp in order:
        rec = pinfo.get(sp)
        if not isinstance(rec, dict):
            continue
        outcomes = rec.get("outcomes")
        if isinstance(outcomes, dict):
            pts = _parse_points(outcomes.get("points_scored"))
            if pts is not None:
                points[sp] = pts
            sat_raw[sp] = outcomes.get("satisfaction")
        sentence = _value_goal_sentence(sp, rec.get("value2issue"))
        if sentence:
            val_goal[sp] = sentence
    strategies_by: dict = {}
    first_by: dict = {}
    for sp in order:
        strategies = []
        for t in own[sp]:
            labels = moves.get(t, {}).get("moves", []) or []
            strategies.append((_primary_strategy(labels), list(labels)))
        strategies_by[sp] = strategies
        first_labeled = None
        for (t, (strat, labels)) in zip(own[sp], strategies):
            if labels:
                first_labeled = (t, strat, labels)
                break
        first_by[sp] = first_labeled
    model_goals: dict = {}
    if client is not None and model and cache_dir is not None:
        hints = {}
        for sp in order:
            if sp in val_goal:
                continue  # the value record already grounds this goal
            first_labeled = first_by[sp]
            if first_labeled is not None:
                t0, _strat0, labels0 = first_labeled
                hints[sp] = (
                    f"first labeled move "
                    f"{'/'.join(str(m) for m in labels0)} on turn #{t0}")
            else:
                hints[sp] = ("no move labels for this speaker; judge "
                             "from the speaker's own turns")
        if hints:
            try:
                model_goals = model_speaker_goals(
                    turns, hints, client, model, meter, cache_dir)
            except (BudgetExceeded, OSError, ValueError):
                model_goals = {}
    out = []
    for sp in order:
        strategies = strategies_by[sp]
        first_labeled = first_by[sp]
        if sp in model_goals:
            if first_labeled is not None:
                t0, _strat0, labels0 = first_labeled
                first_note = (
                    f"first labeled move "
                    f"{'/'.join(str(m) for m in labels0)} on turn #{t0}; "
                    "move labels are model judgments, not record facts")
            else:
                t0 = own[sp][0]
                first_note = "no move labels for this speaker"
            initial_goal = model_goals[sp]
            evidence = turns[t0].get("text", "") if isinstance(
                turns[t0], dict) else ""
            goal_source = (
                f"model estimate: one-sentence goal written by {model} "
                f"from this speaker's turns in this conversation "
                f"({first_note})")
        elif first_labeled is not None:
            t0, strat0, labels0 = first_labeled
            initial_goal = _goal_sentence(sp, strat0)
            evidence = turns[t0].get("text", "") if isinstance(
                turns[t0], dict) else ""
            goal_source = (
                "model estimate: first labeled move "
                f"{'/'.join(str(m) for m in labels0)} on turn #{t0}; "
                "move labels are model judgments, not record facts")
        elif sp in val_goal:
            t0 = own[sp][0]
            initial_goal = val_goal[sp]
            evidence = turns[t0].get("text", "") if isinstance(
                turns[t0], dict) else ""
            goal_source = ("record: participant_info value ranking for "
                           "this speaker (negotiation dataset)")
        else:
            t0 = own[sp][0]
            initial_goal = (
                f"Speaker {sp}'s goal in this conversation is unclear "
                f"from the record.")
            evidence = turns[t0].get("text", "") if isinstance(
                turns[t0], dict) else ""
            goal_source = ("model estimate: no move labels for this "
                           "speaker; needs a model pass")
        shifts = []
        prev = strategies[0][0] if strategies else "other"
        for t, (strat, _labels) in zip(own[sp], strategies):
            if t == own[sp][0]:
                continue
            if strat != prev:
                text = turns[t].get("text", "") if isinstance(
                    turns[t], dict) else ""
                shifts.append({"turn": int(t),
                               "from": _strategy_phrase(prev),
                               "to": _strategy_phrase(strat),
                               "evidence_quote": str(text)})
            prev = strat
        if total_urgent > 0:
            up = n_warn = n_part = n_urgent = 0
            for t, (_strat, labels) in zip(own[sp], strategies):
                if not (set(str(m) for m in labels) & URGENT_INTENT):
                    continue
                n_urgent += 1
                urgent_alerts = [
                    a for a in alerts_by_turn.get(t, [])
                    if a.get("axis") == "urgent-read-as-routine"]
                if not urgent_alerts:
                    up += 1  # read as non-routine: the urgency landed
                elif any(a.get("resolved") for a in urgent_alerts):
                    up += 1
                else:
                    levels = [str(a.get("level", "")) for a in
                              urgent_alerts if not a.get("resolved")]
                    if any(lv in ("warning", "critical")
                           for lv in levels):
                        n_warn += 1
                    else:
                        n_part += 1
            if n_urgent:
                score = up - n_warn - 0.5 * n_part
                balance = round(score / n_urgent, 3)
                balance = max(-1.0, min(1.0, balance))
                basis = (
                    f"{up} of {n_urgent} urgent turns found uptake; "
                    f"{n_warn + n_part} did not ({n_warn} warnings, "
                    f"{n_part} cautions/advisories). Balance = (found - "
                    f"warnings - 0.5*cautions) / urgent turns, against "
                    f"this speaker's own goals in this conversation only.")
            else:
                balance = 0.0
                basis = ("No urgent turns on record; balance 0.0 (nothing "
                         "pressed, nothing missed).")
        elif has_moves:
            match = mis = 0
            for t in own[sp]:
                eff = effects.get(t)
                if eff is None:
                    continue
                labs = set(str(m) for m in (moves.get(t, {}).get(
                    "moves", []) or []))
                urgent = bool(labs & URGENT_INTENT)
                routine = routine_share(eff) >= 0.5
                if urgent:
                    if routine:
                        mis += 1
                    else:
                        match += 1
                elif routine:
                    match += 1
                else:
                    mis += 1
            counted = match + mis
            if counted:
                balance = round((match - mis) / counted, 3)
                balance = max(-1.0, min(1.0, balance))
                basis = (
                    f"{match} of {counted} turns read as expected "
                    f"(routine bids read routinely); {mis} misread. "
                    f"Balance = (matched - mismatched) / turns, against "
                    f"this speaker's own intents in this conversation "
                    f"only.")
            else:
                balance = 0.0
                basis = ("No panel reads for this speaker; balance 0.0.")
        elif sp in points and len(points) >= 2:
            own_pts = points[sp]
            others = [v for k, v in points.items() if k != sp]
            others_avg = sum(others) / len(others) if others else 0.0
            denom = own_pts + others_avg
            if denom > 0:
                balance = round((own_pts - others_avg) / denom, 3)
            else:
                balance = 0.0
            balance = max(-1.0, min(1.0, balance))
            sat = sat_raw.get(sp, "?")
            basis = (
                f"Points {own_pts} vs others avg {others_avg:.1f} from "
                f"the participant record; balance = (own - others) / "
                f"(own + others), against own deal values in this "
                f"conversation only. Satisfaction: {sat}.")
        else:
            sat_bal = _satisfaction_to_balance(sat_raw.get(sp))
            if sat_bal is not None:
                balance = round(float(sat_bal), 3)
                basis = (
                    f"Satisfaction {sat_raw.get(sp)!r} from the "
                    f"participant record; balance {balance}, against own "
                    f"deal values in this conversation only.")
            else:
                balance = 0.0
                basis = ("No urgent turns, no move record and no outcome "
                         "record for this speaker; balance 0.0.")
        out.append({"speaker": sp, "initial_goal": initial_goal,
                    "goal_source": goal_source,
                    "evidence_quote": str(evidence),
                    "shifts": shifts, "end_balance": balance,
                    "end_basis": basis})
    return out


# --- turning points -------------------------------------------------------------

#: Moves that steer the conversation (proposals, decisions, pushes,
#: repairs). General vocabulary; detection never reads case content.
STEERING_MOVES = frozenset({
    "propose", "counter", "concede", "accept", "cooperate",
    "empathize", "reveal_preference", "repair", "deflect",
    "pressure", "appeal_need", "appeal_fairness", "challenge",
    "probe", "reject",
})

#: Order for listing steering moves (urgent first, then deal moves).
STEERING_ORDER = ("appeal_need", "appeal_fairness", "pressure",
                  "challenge", "probe", "reject", "propose", "counter",
                  "concede", "accept", "cooperate", "empathize",
                  "reveal_preference", "repair", "deflect")

#: Response moves that count as uptake in the fallback (agreement and
#: cooperation join the engaged set; the alert rule's ENGAGED_RESPONSE
#: is unchanged).
FALLBACK_ENGAGED = ENGAGED_RESPONSE | frozenset({"accept", "cooperate"})

#: General English negotiation cues for the no-panel fallback. Domain
#: vocabulary only (offer / deal / agree / refuse / ask / need); no
#: case names, no quoted phrases, no issue nouns.
_PROPOSE_CUES = ("would like", "how about", "what about", "how does",
                 "i will give", "i'll give", "i give you", "you can have",
                 "you get", "you would get", "i can give", "let's",
                 "let us", "offer", "trade", "split", "compromise",
                 "middle ground", "final offer", "take it or leave",
                 "deal")
_REJECT_CUES = ("reject", "can't", "cannot", "won't", "will not",
                "walk away", "not fair", "unfair", "disagree",
                "doesn't look like")
_ACCEPT_CUES = ("agree", "accept", "sounds good", "sounds fair",
                "sounds fine", "thank you", "i will agree", "i agree")
_ACCEPT_NEGATED = ("cannot accept", "can't accept", "won't accept",
                   "not accept", "never accept")
_INFORM_CUES = ("i need", "i want", "my highest", "priority", "i have",
                "bringing")
_QUESTION_STARTS = ("what ", "how ", "would you", "are you", "do you",
                    "can you", "are there", "is there", "could you")
_NO_RE = re.compile(r"\bno\b")


def infer_text_moves(text) -> list:
    """Minimal negotiation acts from raw text (no-panel fallback).

    General English cues only: deal-protocol markers (submit/accept/
    reject-deal), offer language, refusal language (standalone "no"
    via word boundary, so "know" never fires), agreement language
    (suppressed under negation, so "cannot accept" is not an
    accept), questions, and need statements. Returns at least
    ["other"]. Never reads case names or quoted phrases.
    """
    t = str(text or "").lower().strip()
    if not t:
        return ["other"]
    if "submit-deal" in t:
        return ["propose"]
    if "accept-deal" in t:
        return ["accept"]
    if "reject-deal" in t:
        return ["reject"]
    found: list = []
    if any(c in t for c in _PROPOSE_CUES):
        found.append("propose")
    if any(c in t for c in _REJECT_CUES) or _NO_RE.search(t):
        found.append("reject")
    if any(c in t for c in _ACCEPT_CUES) and not any(
            n in t for n in _ACCEPT_NEGATED):
        found.append("accept")
    if "?" in t or t.startswith(_QUESTION_STARTS):
        found.append("ask")
    if any(c in t for c in _INFORM_CUES):
        found.append("inform")
    if not found:
        return ["other"]
    order = ("propose", "reject", "accept", "ask", "inform")
    found.sort(key=lambda m: order.index(m) if m in order else 99)
    return found


def _ordered_steering(moves) -> list:
    have = set(str(m) for m in (moves or [])) & STEERING_MOVES
    return [m for m in STEERING_ORDER if m in have]


def has_forward_reply(turns, turn) -> bool:
    """True when another speaker replies after `turn` (general rule).

    A turning point pairs steering with an observed reply; without a
    later turn by another speaker there is no reception on record.
    """
    nxt = next_other_speaker({}, turns or [], turn)
    return nxt is not None and 0 <= nxt < len(turns or [])


def _fallback_entry(turn, turns, moves_lists, effects_by_turn, inferred):
    """One fallback turning point for a steering turn.

    Returns None when no other speaker replies later: the close a
    final turn makes is recorded as the reception half of the
    proposal it answers, never as its own entry.
    """
    nxt = next_other_speaker({}, turns or [], turn)
    if nxt is None or not 0 <= nxt < len(turns or []):
        return None
    steering = _ordered_steering(moves_lists.get(turn, []))
    if not steering:
        primary = _primary_strategy(moves_lists.get(turn, []))
        steering = [primary]
    intent = "; ".join(_strategy_gloss(m) for m in steering)
    opened = intent + " (text-inferred)" if inferred else intent
    text = turns[turn].get("text", "") if 0 <= turn < len(turns or []) \
        and isinstance(turns[turn], dict) else ""
    speaker = turns[turn].get("speaker", "?") if 0 <= turn < len(
        turns or []) and isinstance(turns[turn], dict) else "?"
    received_by = turns[nxt].get("speaker", "") if isinstance(
        turns[nxt], dict) else ""
    resp = [str(m) for m in (moves_lists.get(nxt, []) or [])]
    if inferred or not (effects_by_turn or {}).get(turn):
        if resp:
            received_as = (f"reply carried {'/'.join(resp)} "
                           f"(no panel record; text-inferred)")
        else:
            received_as = "no labeled reply on record (no panel record)"
        effect = "uptake" if set(resp) & FALLBACK_ENGAGED else "no uptake"
        resp_gloss = ("; ".join(_strategy_gloss(m) for m in resp)
                      if resp else "no reply on record")
        if inferred and resp:
            resp_gloss += " (text-inferred)"
    else:
        modal, prob = modal_read((effects_by_turn or {}).get(turn))
        received_as = (f"read as {modal} ({prob:.0%}); reply carried "
                       f"{'/'.join(resp) if resp else 'no labeled moves'}")
        effect = "uptake" if set(resp) & FALLBACK_ENGAGED else "no uptake"
        resp_gloss = ("; ".join(_strategy_gloss(m) for m in resp)
                      if resp else "no reply on record")
    # response_turn is the observed forward reply. It never points
    # back and is never null: entries without a reply are not built.
    return {
        "turn": int(turn), "speaker": str(speaker),
        "opened": opened, "quote": str(text),
        "received_by": str(received_by),
        "received_as": received_as,
        "response_turn": int(nxt),
        "effect": effect,
        "wanted_vs_happened": (f"wanted: {intent}; "
                               f"happened: {resp_gloss}"),
        "intentional": "unclear",
    }


def _ultimate_fallback_turn(turns, moves_by_turn, effects_by_turn):
    """One turn that must exist when no steering turn does.

    With a panel record, the most disagreed-about turn; without one,
    the longest substantive text (word count). General signals only.
    """
    if effects_by_turn:
        best, best_dis = None, None
        for t in range(len(turns or [])):
            eff = effects_by_turn.get(t, {})
            try:
                dis = float(eff.get("disagreement", 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(dis):
                continue
            if best is None or dis > best_dis:
                best, best_dis = t, dis
        if best is not None:
            return best
    best, best_len = 0, -1
    for t, turn in enumerate(turns or []):
        text = turn.get("text", "") if isinstance(turn, dict) else ""
        n = len(str(text).split())
        if n > best_len:
            best, best_len = t, n
    return best


def _reading_reply(reading, item, turns):
    """(response_turn, words on the reply) for one item, or None without a reply.

    The item's reply_turn stands when another speaker said it. A reading
    may cite nothing, or the speaker's own later turn; the reply is then
    the next turn by someone else, and the words say the reading did not
    cite it.
    """
    turn, e = reading["turn"], reading["items"][item]
    reply, quote = e.get("reply_turn"), " ".join(str(e.get("quote") or "").split())
    cited = isinstance(reply, int) and not isinstance(reply, bool) and turn < reply < len(turns)
    if cited and turns[reply]["speaker"] != turns[turn]["speaker"]:
        return reply, f"reply #{reply}: {quote}"
    nxt = next_other_speaker({}, turns, turn)
    if nxt is None or not 0 <= nxt < len(turns):
        return None
    note = (f"the reading cites only the speaker's own #{reply}: {quote}" if cited
            else "the reading cites no reply")
    text = " ".join(str(turns[nxt].get("text", "")).split())
    return nxt, f"{note}; next reply #{nxt}: {text}"


def build_reading_turning_points(turns, uptake, alerts, standing, use):
    """Turning points from the per-item readings, or None without any.

    A turn enters when an alert fired on it, or when the standing on a
    stake split there and an item of that stake fell short on that turn.
    ``opened`` is the most wanted of those items; ``received_as``,
    ``wanted_vs_happened`` and ``effect`` come from its wanted and actual
    values and its reply. Only measured items count, as in the alerts and
    the standing. Same reception rule as the panel path: no reply by
    another speaker, no entry.

    ``intentional`` is always "unclear": nothing here measures
    deliberateness, so neither "yes" nor "no" is stated.

    None (no measured item holds both values) leaves the panel rule in
    place; an empty list means readings exist and nothing fell short.
    """
    from .multilayer import ITEM_CLASS, STAKE_ITEMS, broken_items
    readings = [r for r in (uptake or []) if isinstance(r, dict)]
    if not any(top_measured_item(r, use) for r in readings):
        return None
    by_turn = {r["turn"]: r for r in readings}
    pool, triggers = {}, {}
    for a in (alerts or []):
        if a.get("turn") in by_turn and a.get("items"):
            pool.setdefault(a["turn"], []).extend(a["items"])
            triggers.setdefault(a["turn"], []).append(f"alert:{a.get('level')}")
    for cls, split in ((standing or {}).get("splits") or {}).items():
        reading = by_turn.get(split["turn"]) if split else None
        if reading is None:
            continue
        items = reading["items"]
        short = [i for i in broken_items(items) if i in use and ITEM_CLASS[i] == cls
                 and items[i]["actual"]["score"] < items[i]["wanted"]["score"]]
        if short:
            pool.setdefault(split["turn"], []).extend(short)
            triggers.setdefault(split["turn"], []).append(f"split:{cls}")
    out = []
    for turn in sorted(pool):
        reading = by_turn[turn]
        item = max(pool[turn], key=lambda i: reading["items"][i]["wanted"]["score"])
        reply = _reading_reply(reading, item, turns)
        if reply is None:
            continue
        response_turn, reply_words = reply
        e = reading["items"][item]
        wanted, actual = e["wanted"]["score"], e["actual"]["score"]
        gloss = STAKE_ITEMS[ITEM_CLASS[item]][item]
        gap = round(wanted) - round(actual)
        out.append({
            "turn": int(turn), "speaker": str(turns[turn]["speaker"]),
            "opened": f"{item}: reached for {wanted:.0f}/100 ({gloss})",
            "quote": str(turns[turn].get("text", "")),
            "received_by": str(turns[response_turn]["speaker"]),
            "received_as": f"{item} landed {actual:.0f}/100; {reply_words}",
            "response_turn": int(response_turn),
            "effect": (f"{gap} points short of what was wanted" if gap > 0
                       else f"no shortfall on {item}, but the alert stands"),
            "wanted_vs_happened": (f"wanted: {item} at {wanted:.0f}/100; "
                                   f"happened: {actual:.0f}/100; {reply_words}"),
            "intentional": "unclear",
            "item": item, "wanted": wanted, "actual": actual,
            "triggers": triggers[turn],
            "source": (f"model estimate: per-item reading of turn #{turn} "
                       f"({', '.join(triggers[turn])}); item readings are "
                       f"model judgments, not record facts"),
        })
    return out


def build_turning_points(turns, moves_by_turn, effects_by_turn, scored):
    """Turning points without readings: steering and reception from the panel.

    One entry per mismatch turn (urgent intent read as routine):
    ``opened`` states the steering in the move-label vocabulary,
    ``received_as`` states the panel's modal read plus the reply's
    moves, and ``wanted_vs_happened`` contrasts the two. ``quote``
    is the turn's verbatim text. ``intentional`` is always "unclear":
    nothing here measures deliberateness.

    Fallback (general, no case content): when no mismatch exists,
    steering moves (proposals, decisions, pushes, repairs) are the
    turning points; without a panel record the acts come from
    :func:`infer_text_moves` and the entries say so. When even that
    finds nothing, one ultimate turn (most disagreement, else
    longest text) stands in when it has a reply.

    Reception rule (general, every path): an entry exists only with
    an observed forward reply from another speaker, so
    ``response_turn`` is always a strictly later turn spoken by
    ``received_by`` -- never null, never backward. The close a
    final turn makes is recorded as the reception half of the
    proposal it answers, never as its own entry. A dialogue where
    no steering turn is ever answered (a single speaker, a single
    turn) reports no turning points.
    """
    out = []
    for row in (scored or []):
        if not row.get("mismatch"):
            continue
        t = row["turn"]
        nxt = next_other_speaker(moves_by_turn, turns or [], t)
        if nxt is None or not 0 <= nxt < len(turns or []):
            continue
        urgent = [m for m in URGENT_ORDER if m in row.get("eff_moves", [])]
        intent = ("; ".join(_strategy_gloss(u) for u in urgent)
                  if urgent else "speak")
        if row.get("inherited_from") is not None:
            opened = (f"continued the appeal from "
                      f"#{row['inherited_from']}: {intent}")
        else:
            opened = intent
        text = turns[t].get("text", "") if 0 <= t < len(turns or []) and isinstance(
            turns[t], dict) else ""
        received_by = turns[nxt].get("speaker", "") if isinstance(
            turns[nxt], dict) else ""
        resp = [str(m) for m in (row.get("response", []) or [])]
        modal, prob = modal_read((effects_by_turn or {}).get(t))
        received_as = (f"read as {modal} ({prob:.0%}); reply carried "
                       f"{'/'.join(resp) if resp else 'no labeled moves'}")
        effect = ("uptake" if set(resp) & ENGAGED_RESPONSE
                  else "no uptake")
        resp_gloss = ("; ".join(_strategy_gloss(m) for m in resp)
                      if resp else "no reply on record")
        out.append({
            "turn": int(t), "speaker": row.get("speaker", "?"),
            "opened": opened, "quote": str(text),
            "received_by": str(received_by),
            "received_as": received_as,
            "response_turn": int(nxt),
            "effect": effect,
            "wanted_vs_happened": (f"wanted: {intent}; "
                                   f"happened: {resp_gloss}"),
            "intentional": "unclear",
        })
    if out:
        return out
    if not (turns or []):
        return []
    if moves_by_turn:
        moves_lists = {t: [str(m) for m in (
            moves_by_turn.get(t, {}).get("moves", []) or [])]
            for t in range(len(turns))}
        entries = []
        for t in range(len(turns)):
            if not _ordered_steering(moves_lists.get(t, [])):
                continue
            entry = _fallback_entry(t, turns, moves_lists,
                                    effects_by_turn or {}, False)
            if entry is not None:
                entries.append(entry)
        if entries:
            return entries
        solo = _ultimate_fallback_turn(turns, moves_by_turn,
                                       effects_by_turn or {})
        entry = _fallback_entry(solo, turns, moves_lists,
                                effects_by_turn or {}, False)
        return [entry] if entry is not None else []
    inferred_lists = {t: infer_text_moves(
        turns[t].get("text", "") if isinstance(turns[t], dict) else "")
        for t in range(len(turns))}
    entries = []
    for t in range(len(turns)):
        if not _ordered_steering(inferred_lists.get(t, [])):
            continue
        entry = _fallback_entry(t, turns, inferred_lists, {}, True)
        if entry is not None:
            entries.append(entry)
    if entries:
        return entries
    solo = _ultimate_fallback_turn(turns, {}, {})
    entry = _fallback_entry(solo, turns, inferred_lists, {}, True)
    return [entry] if entry is not None else []


# --- fields from the model judgment ----------------------------------------------

def build_judged_turning_points(turns, judged, uptake, use, source):
    """Turning points from the judgment: words from the model, values from the reading."""
    moves = {m["turn"]: m for m in judged["turns"]}
    by_turn = {r.get("turn"): r for r in (uptake or []) if isinstance(r, dict)}
    out = []
    for d in judged["divides"]:
        t, move = d["turn"], moves[d["turn"]]
        item, wanted, actual = _judged_values(by_turn.get(t), use)
        values = (f"reading: {item} wanted {wanted:.0f}/100, landed {actual:.0f}/100" if item
                  else "reading: no measured item on this turn")
        out.append({
            "turn": t, "speaker": move["speaker"], "opened": move["opened"],
            "quote": str(turns[t].get("text", "")),
            "received_by": move["received_by"], "received_as": move["received"],
            "response_turn": move["response_turn"], "effect": d["happened"],
            "wanted_vs_happened": f"wanted: {d['wanted']}; happened: {d['happened']}; {values}",
            "intentional": "unclear", "why": d["why"],
            "item": item, "wanted": wanted, "actual": actual,
            "source": f"{source}: {d['why']}",
        })
    return out


def apply_judged_speakers(speakers, judged, source, fallback):
    """Goals and shifts from the judgment; a field it lacks keeps the code rule, marked."""
    for sp in speakers:
        got = judged["speakers"].get(str(sp.get("speaker"))) or {}
        if got.get("goal"):
            sp["initial_goal"], sp["goal_source"] = got["goal"], f"{source}: read from the whole conversation"
        else:
            sp["goal_source"] = f"{sp['goal_source']}; {fallback}"
        if got.get("shifts") is not None:
            sp["shifts"], sp["shifts_source"] = got["shifts"], source
        else:
            sp["shifts_source"] = fallback
    return speakers


def judged_perspective(scene, n_scenes, source):
    """Perspective for a checked scene pick: every shown word is copied by code."""
    if scene["scene_id"] is None:
        return {"line": "No scene in the catalog fits this conversation.", "work": "none",
                "source": f"{source}: no fitting scene among {n_scenes}",
                "url": "", "scene_id": None, "why": scene["why"]}
    pick = next(s for s in _load_scene_list() if str(s.get("id")) == scene["scene_id"])
    return {**_perspective_of(pick), "why": scene["why"], "basis": source}


# --- alternatives -----------------------------------------------------------------

COMPARE_PROMPT = (
    "You compare two wordings of one high-stakes turn.\n"
    "Wording A and wording B carry the same facts. Judge only which one "
    "would better produce the effect the speaker wanted on the given "
    "criterion, for the listener in this situation.\n"
    "Reply with JSON only: {\"clearer\": \"A\"}, {\"clearer\": \"B\"}, "
    "or {\"clearer\": \"Tie\"} when they read about the same."
)

COMPARE_STANCES = ("neutral", "trusting", "suspicious", "tired", "impatient")

#: Minimum parsed paired votes per alternative (both orders x stances).
MIN_PAIRED_VOTES = 8

#: General rewrite brief: two scene-tailored wordings, facts kept.
REWRITE_PROMPT = (
    "You rewrite one high-stakes dialogue turn in two different ways.\n"
    "Keep the facts exactly: the same ask, numbers, names and decisions, "
    "in the same language as the original.\n"
    "Change only the wording so the turn better produces the effect the "
    "speaker wanted on the given criterion.\n"
    "Tailor each rewrite to this situation (the speakers and what just "
    "happened); the two rewrites must differ from each other and from "
    "the original. At most 40 words each.\n"
    "Reply with JSON only: "
    "{\"alternatives\": [\"<rewrite 1>\", \"<rewrite 2>\"]}."
)

MAX_REWRITE_WORDS = 40


def _criterion_line(criterion, intended, speaker):
    """The criterion as prompts and limits state it (also a cache key part)."""
    from .multilayer import ITEM_CLASS, STAKE_ITEMS
    who = speaker or "the speaker"
    item = (criterion or {}).get("item")
    if item:
        return f"the effect {who} wanted on {item} ({STAKE_ITEMS[ITEM_CLASS[item]][item]})"
    aim = " ".join(str(intended or "").split()) or "unknown"
    return f"the effect {who} wanted, as stated in this sentence: {aim}"


def alternative_texts(quote):
    """Two general explicit restatements of the decisive quote."""
    short = _short_quote(quote)
    return [
        f"To be explicit: {short} Please read back what you understood.",
        (f"{short} — this is urgent, and I need action now. "
         f"Please confirm how you will handle it."),
    ]


def _short_quote(quote):
    """Quote as embedded in an alternative (whitespace-folded, capped)."""
    short = " ".join(str(quote or "").split())
    if len(short) > 220:
        short = short[:217].rstrip() + "..."
    if not short:
        short = "what was just said"
    return short


def split_verbatim(text, quote):
    """Split alternative text into (framing_before, verbatim, framing_after).

    The verbatim span is the decisive quote as embedded by
    :func:`alternative_texts` (full or capped form). Returns
    ``(text, "", "")`` when the quote is not embedded, e.g. for
    model-written rewordings that paraphrase instead of quoting.
    """
    body = str(text or "")
    full = " ".join(str(quote or "").split())
    for cand in (full, _short_quote(quote)):
        if cand and cand in body:
            pre, _, post = body.partition(cand)
            return pre, cand, post
    return body, "", ""


def _alt_cache_path(cache_dir, key):
    return Path(cache_dir) / f"alt-{key}.json"


def _clean_rewrite(text):
    if not isinstance(text, str):
        return ""
    s = " ".join(text.split()).strip().strip('"').strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"', "\u201c", "\u2018"):
        s = s[1:-1].strip()
    words = s.split()
    if len(words) > MAX_REWRITE_WORDS:
        s = " ".join(words[:MAX_REWRITE_WORDS])
    return s


def _valid_rewrites(texts, original):
    """First two rewrites when usable, else None (general checks only)."""
    if not isinstance(texts, list) or len(texts) < 2:
        return None
    cleaned = [_clean_rewrite(t) for t in texts[:2]]
    if any(not c for c in cleaned):
        return None
    norm = [" ".join(c.lower().split()) for c in cleaned]
    if norm[0] == norm[1]:
        return None
    orig = " ".join(str(original or "").lower().split())
    if any(n == orig for n in norm):
        return None
    return cleaned


def generate_alternative_texts(original, context_lines, speaker, intended,
                               client, model, meter, cache_dir, criterion_line=""):
    """Two model-written rewrites for this scene (cached, budget-aware).

    One call per distinct (model, speaker, original, context, aim,
    criterion); identical inputs reuse the cached pair without re-querying.
    Raises BudgetExceeded before spending past the budget, RuntimeError
    when no usable pair parses.
    """
    payload = json.dumps(
        {"kind": "alts-v2", "model": model, "speaker": speaker,
         "original": original, "context": list(context_lines or []),
         "intended": intended, "criterion": criterion_line}, ensure_ascii=False,
        sort_keys=True).encode("utf-8")
    key = _sha_bytes(payload)
    path = _alt_cache_path(cache_dir, key)
    if path.exists():
        try:
            valid = _valid_rewrites(_read_json(path).get("texts"), original)
            if valid is not None:
                return valid
        except (OSError, ValueError):
            pass
    if meter is not None:
        meter.check()
    if context_lines:
        ctx = "\n".join(f"- {line}" for line in context_lines)
    else:
        ctx = "- (no earlier turns)"
    messages = [{"role": "system", "content": REWRITE_PROMPT},
                {"role": "user",
                 "content": (f"Speaker: {speaker}\nJust before:\n{ctx}\n"
                             f"Original turn: {original}\n"
                             f"Speaker's aim: {intended}\n"
                             f"Criterion: {criterion_line}\nRewrite:")}]
    for _ in range(2):
        try:
            raw = client.complete(messages, model=model, temperature=0.7)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        try:
            text = str(raw or "")
            start = text.find("{")
            if start < 0:
                continue
            data, _ = json.JSONDecoder().raw_decode(text[start:])
            items = data.get("alternatives") if isinstance(data, dict) else None
            valid = _valid_rewrites(items, original)
            if valid is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"texts": valid}, f, ensure_ascii=False)
                return valid
        except (ValueError, TypeError):
            continue
    raise RuntimeError("no model rewrites parsed")


def _vote_cache_path(cache_dir, key):
    return Path(cache_dir) / f"vote-{key}.json"


def _parse_clearer(raw):
    text = str(raw or "")
    start = text.find("{")
    if start >= 0:
        try:
            data, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(data, dict):
                val = str(data.get("clearer", "")).strip().lower()
                if val == "a":
                    return "A"
                if val == "b":
                    return "B"
                if val in ("tie", "t", "same", "equal", "neither"):
                    return "TIE"
        except (ValueError, TypeError):
            pass
    upper = text.strip().upper()
    if upper.startswith("TIE") or upper in ("T", "SAME"):
        return "TIE"
    if upper.startswith("B"):
        return "B"
    if upper.startswith("A"):
        if "TIE" in upper or "SAME" in upper or "EQUAL" in upper:
            return "TIE"
        return "A"
    return None


def _cached_vote(client, model, meter, cache_dir, stance, first, second,
                 context_lines=None, criterion_line=""):
    payload = json.dumps({"model": model, "stance": stance,
                          "context": list(context_lines or []),
                          "criterion": criterion_line,
                          "a": first, "b": second}, ensure_ascii=False,
                         sort_keys=True).encode("utf-8")
    key = _sha_bytes(payload)
    path = _vote_cache_path(cache_dir, key)
    if path.exists():
        try:
            pick = _read_json(path).get("clearer")
            if pick in ("A", "B", "TIE"):
                return pick
        except (OSError, ValueError):
            pass
    if meter is not None:
        meter.check()
    if context_lines:
        ctx = "\n".join(f"- {line}" for line in context_lines)
    else:
        ctx = "(no earlier turns)"
    messages = [{"role": "system", "content": COMPARE_PROMPT},
                {"role": "user",
                 "content": (f"Stance: {stance}\nCriterion: {criterion_line}\n"
                             f"Just before:\n{ctx}\n"
                             f"A: {first}\nB: {second}")}]
    clearer = None
    for _ in range(2):
        try:
            raw = client.complete(messages, model=model, temperature=0.0)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        clearer = _parse_clearer(raw)
        if clearer is not None:
            break
    if clearer is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"clearer": clearer}, f)
    return clearer


def model_p_better(original, alternative, context_lines, client, model,
                   meter, cache_dir, criterion_line=""):
    """Paired votes on the criterion: both orders x five stances (10 votes).

    One panel judges original vs rewrite in both orders; ties count
    half. Returns (p_smooth, n, k_eff, p_empirical): the Laplace
    point estimate (k+1)/(n+2), which never asserts 0% or 100% from
    finite votes, the parsed vote count, effective wins, and the raw
    paired share. Raises when fewer than MIN_PAIRED_VOTES parse.
    """
    better, n = 0.0, 0
    for order in (0, 1):
        first, second = (original, alternative) if order == 0 else (
            alternative, original)
        for stance in COMPARE_STANCES:
            pick = _cached_vote(client, model, meter, cache_dir, stance,
                                first, second, context_lines, criterion_line)
            if pick is None:
                continue
            n += 1
            if pick == "TIE":
                better += 0.5
            else:
                alt_is_first = order == 1
                if (pick == "A") == alt_is_first:
                    better += 1
    if n < MIN_PAIRED_VOTES:
        raise RuntimeError(f"too few paired votes ({n})")
    return (better + 1.0) / (n + 2.0), n, better, better / n


def proxy_p_better(scored, effects_by_turn):
    """Offline proxy from the record (Laplace-smoothed, never 0/1).

    Evidence base: panel readings on urgent-mismatch turns. The share
    read as routine is the room an explicit restatement could improve.
    """
    routine, total = 0, 0
    for row in (scored or []):
        if not row["mismatch"]:
            continue
        readings = (effects_by_turn.get(row["turn"], {}) or {}).get(
            "readings", [])
        if not isinstance(readings, list):
            continue
        for r in readings:
            if isinstance(r, dict) and r.get("next_action"):
                total += 1
                if str(r["next_action"]) in ROUTINE_READ:
                    routine += 1
    k, n = routine + 1, total + 2
    return k / n, n, k


def build_alternatives(quote, scored, effects_by_turn, client=None,
                       model=None, meter=None, cache_dir=None,
                       turns=None, decisive_turn=None, speaker="",
                       intended="", original=None, criterion=None):
    full = " ".join(str(original if original is not None else quote or "")
                    .split()) or "(the original turn)"
    context_lines = []
    if isinstance(turns, list) and isinstance(decisive_turn, int) \
            and not isinstance(decisive_turn, bool):
        for t in turns[max(0, decisive_turn - 3):decisive_turn]:
            if isinstance(t, dict):
                context_lines.append(
                    f"{t.get('speaker', '?')}: {t.get('text', '')}")
    criterion = criterion or pick_criterion(None, (), turn=decisive_turn)
    line = _criterion_line(criterion, intended, speaker)
    texts = None
    if client is not None and model and cache_dir is not None:
        try:
            texts = generate_alternative_texts(
                full, context_lines, speaker, intended, client, model,
                meter, cache_dir, line)
        except (BudgetExceeded, RuntimeError, OSError, ValueError):
            texts = None
    if texts is None:
        texts = alternative_texts(quote)
    out = []
    for text in texts:
        p_emp = None
        if client is not None and model and cache_dir is not None:
            try:
                p, n, k_eff, p_emp = model_p_better(
                    full, text, context_lines, client, model, meter,
                    cache_dir, line)
                method = "paired"
                k = k_eff
            except (BudgetExceeded, RuntimeError, OSError, ValueError):
                p, n, k = proxy_p_better(scored, effects_by_turn)
                method = "offline-proxy"
        else:
            p, n, k = proxy_p_better(scored, effects_by_turn)
            method = "offline-proxy"
        lo, hi = wilson(k, n)
        entry = {"text": text, "p_better": round(float(p), 3),
                 "n_compared": int(n), "ci_low": round(float(lo), 3),
                 "ci_high": round(float(hi), 3), "method": method,
                 "criterion": criterion}
        if p_emp is not None:
            entry["p_empirical"] = round(float(p_emp), 3)
        out.append(entry)
    return out


#: General worse-rewrite brief: one scene-tailored wording that produces the
#: speaker's wanted effect on the criterion less well. Same facts, no new topic.
WORSE_PROMPT = (
    "You rewrite one high-stakes dialogue turn in one WORSE way.\n"
    "Keep the underlying facts (the same ask, numbers, names) and the "
    "same language as the original.\n"
    "Change only the wording so the turn produces the effect the speaker "
    "wanted on the given criterion LESS well than the original does.\n"
    "Do not add any topic that is not already in the original.\n"
    "Tailor the rewrite to this situation (the speakers and what just "
    "happened); it must differ from the original. At most 40 words.\n"
    "Reply with JSON only: {\"worse\": \"<rewrite>\"}."
)

#: General middle-rewrite brief: one scene-tailored wording only a little
#: better than the original on the criterion. Same facts, no new topic.
MIDDLE_PROMPT = (
    "You rewrite one high-stakes dialogue turn in one SLIGHTLY better way.\n"
    "Keep the underlying facts (the same ask, numbers, names) and the "
    "same language as the original.\n"
    "Change only a little of the wording so the turn produces the effect "
    "the speaker wanted on the given criterion only a little better than "
    "the original does: a modest gain, well short of the best rewrite you "
    "could write.\n"
    "Do not add any topic that is not already in the original.\n"
    "Tailor the rewrite to this situation (the speakers and what just "
    "happened); it must differ from the original. At most 40 words.\n"
    "Reply with JSON only: {\"middle\": \"<rewrite>\"}."
)


def worse_template_text(quote):
    """General vague restatement of the decisive quote (offline fallback)."""
    short = _short_quote(quote)
    return (f"{short} Just mentioning it in passing; "
            f"no rush at all.")


def _valid_worse(text, original):
    """Cleaned worse rewrite when usable, else None (general checks)."""
    cleaned = _clean_rewrite(text)
    if not cleaned:
        return None
    norm = " ".join(cleaned.lower().split())
    orig = " ".join(str(original or "").lower().split())
    if norm == orig:
        return None
    return cleaned


def generate_worse_text(original, context_lines, speaker, intended,
                        client, model, meter, cache_dir, criterion_line=""):
    """One model-written worse wording for this scene."""
    return _generate_control_text("worse", WORSE_PROMPT, original, context_lines,
                                  speaker, intended, client, model, meter,
                                  cache_dir, criterion_line)


def generate_middle_text(original, context_lines, speaker, intended,
                         client, model, meter, cache_dir, criterion_line=""):
    """One model-written slightly better wording for this scene."""
    return _generate_control_text("middle", MIDDLE_PROMPT, original, context_lines,
                                  speaker, intended, client, model, meter,
                                  cache_dir, criterion_line)


def _generate_control_text(field, prompt, original, context_lines, speaker,
                           intended, client, model, meter, cache_dir,
                           criterion_line):
    """One control wording, JSON key ``field`` (cached, budget-aware).

    One call per distinct (kind, model, speaker, original, context, aim,
    criterion); identical inputs reuse the cached wording without re-querying.
    Raises BudgetExceeded before spending past the budget, RuntimeError
    when no usable wording parses.
    """
    payload = json.dumps(
        {"kind": f"{field}-v2", "model": model, "speaker": speaker,
         "original": original, "context": list(context_lines or []),
         "intended": intended, "criterion": criterion_line}, ensure_ascii=False,
        sort_keys=True).encode("utf-8")
    key = _sha_bytes(payload)
    path = _alt_cache_path(cache_dir, key)
    if path.exists():
        try:
            valid = _valid_worse(_read_json(path).get("text"), original)
            if valid is not None:
                return valid
        except (OSError, ValueError):
            pass
    if meter is not None:
        meter.check()
    if context_lines:
        ctx = "\n".join(f"- {line}" for line in context_lines)
    else:
        ctx = "- (no earlier turns)"
    messages = [{"role": "system", "content": prompt},
                {"role": "user",
                 "content": (f"Speaker: {speaker}\nJust before:\n{ctx}\n"
                             f"Original turn: {original}\n"
                             f"Speaker's aim: {intended}\n"
                             f"Criterion: {criterion_line}\n"
                             f"{field.capitalize()} rewrite:")}]
    for _ in range(2):
        try:
            raw = client.complete(messages, model=model, temperature=0.7)
        except BudgetExceeded:
            raise
        except Exception:
            continue
        try:
            text = str(raw or "")
            start = text.find("{")
            if start < 0:
                continue
            data, _ = json.JSONDecoder().raw_decode(text[start:])
            item = data.get(field) if isinstance(data, dict) else None
            valid = _valid_worse(item, original)
            if valid is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"text": valid}, f, ensure_ascii=False)
                return valid
        except (ValueError, TypeError):
            continue
    raise RuntimeError(f"no model {field} rewrite parsed")


def build_alternative_controls(quote, scored, effects_by_turn, client=None,
                               model=None, meter=None, cache_dir=None,
                               turns=None, decisive_turn=None, speaker="",
                               intended="", original=None, criterion=None):
    """Controls for the alternative judgment (C5b), on one criterion.

    The same panel that scores the better-wording alternatives also
    judges two negative controls in both orders (ties count half): the
    original against itself (``identical``), which must land near
    0.5, and the original against a deliberately worse wording
    (``worse``), which must land below 0.5. A judge that always
    prefers the new wording fails the worse control. The worse
    wording is model-written for the scene when a model is
    available, else a general vague restatement; scoring is paired
    whenever a client exists. Fully offline, identical wordings tie
    by identity (p 0.5, n 0, no votes cast) and the worse wording
    takes the mirrored offline proxy (smoothed non-routine share).

    ``middle`` is a model-written wording only a little better than the
    original, voted the same way. It exists only as a scene-written,
    voted sentence: without one, its score stays None (method
    "unavailable"), never a stand-in value.
    """
    full = " ".join(str(original if original is not None else quote or "")
                    .split()) or "(the original turn)"
    context_lines = []
    if isinstance(turns, list) and isinstance(decisive_turn, int) \
            and not isinstance(decisive_turn, bool):
        for t in turns[max(0, decisive_turn - 3):decisive_turn]:
            if isinstance(t, dict):
                context_lines.append(
                    f"{t.get('speaker', '?')}: {t.get('text', '')}")
    criterion = criterion or pick_criterion(None, (), turn=decisive_turn)
    line = _criterion_line(criterion, intended, speaker)
    use_model = client is not None and model and cache_dir is not None
    worse_text = middle_text = None
    if use_model:
        try:
            worse_text = generate_worse_text(
                full, context_lines, speaker, intended, client, model,
                meter, cache_dir, line)
        except (BudgetExceeded, RuntimeError, OSError, ValueError):
            worse_text = None
        try:
            middle_text = generate_middle_text(
                full, context_lines, speaker, intended, client, model,
                meter, cache_dir, line)
        except (BudgetExceeded, RuntimeError, OSError, ValueError):
            middle_text = None
    if worse_text is None:
        worse_text = worse_template_text(quote)
    controls = {}
    pairs = (("identical", full), ("worse", worse_text))
    for name, candidate in pairs:
        p_emp = None
        if use_model:
            try:
                p, n, k_eff, p_emp = model_p_better(
                    full, candidate, context_lines, client, model,
                    meter, cache_dir, line)
                method = "paired"
                k = k_eff
            except (BudgetExceeded, RuntimeError, OSError, ValueError):
                use_model = False
        if not use_model:
            if name == "identical":
                p, n, k = 0.5, 0, 0.0
                method = "offline-identity"
            else:
                p, n, k = proxy_p_better(scored, effects_by_turn)
                p, k = 1.0 - p, n - k
                method = "offline-proxy"
        lo, hi = wilson(k, n)
        entry = {"p_better": round(float(p), 3),
                 "n": int(n), "n_compared": int(n),
                 "ci_low": round(float(lo), 3),
                 "ci_high": round(float(hi), 3), "method": method}
        if p_emp is not None:
            entry["p_empirical"] = round(float(p_emp), 3)
        if name == "worse":
            entry["text"] = candidate
        controls[name] = entry
    middle = {"text": middle_text or "", "p_better": None, "n": 0, "n_compared": 0,
              "ci_low": None, "ci_high": None, "method": "unavailable"}
    if use_model and middle_text:
        try:
            p, n, k, p_emp = model_p_better(
                full, middle_text, context_lines, client, model, meter,
                cache_dir, line)
            lo, hi = wilson(k, n)
            middle.update(p_better=round(float(p), 3), n=int(n), n_compared=int(n),
                          ci_low=round(float(lo), 3), ci_high=round(float(hi), 3),
                          method="paired", p_empirical=round(float(p_emp), 3))
        except (BudgetExceeded, RuntimeError, OSError, ValueError):
            pass
    controls["middle"] = middle
    controls["criterion"] = criterion
    return controls


def alternative_ordering(alternatives, controls):
    """Whether worse < middle < mean alternative, from the p_better values.

    ``holds`` is None unless all of them are paired votes: an offline proxy
    or a missing middle has no order to report.
    """
    worse, middle = controls.get("worse") or {}, controls.get("middle") or {}
    alts = alternatives or []
    mean = _mean([a["p_better"] for a in alts]) if alts else None
    paired = bool(alts) and all(x.get("method") == "paired" for x in (worse, middle, *alts))
    return {"worse": worse.get("p_better"), "middle": middle.get("p_better"),
            "alternatives_mean": None if mean is None else round(mean, 3),
            "holds": worse["p_better"] < middle["p_better"] < mean if paired else None}


# --- perspective --------------------------------------------------------------------

def _scene_codes(scene) -> list[str]:
    raw = str(scene.get("주유형", "") or "")
    codes = [c for c in raw.replace(",", " ").split() if c]
    if codes:
        return codes
    return []


def _scene_work_gloss() -> dict:
    """(Korean work, author) -> (English work, author), from sidecar data."""
    path = Path(__file__).resolve().parent / "scenes_en.json"
    try:
        data = _read_json(path)
    except (OSError, ValueError):
        return {}
    works = data.get("works", []) if isinstance(data, dict) else []
    out = {}
    for entry in works if isinstance(works, list) else []:
        if not isinstance(entry, dict):
            continue
        key = (str(entry.get("work", "") or ""),
               str(entry.get("author", "") or ""))
        out[key] = (str(entry.get("work_en", "") or ""),
                    str(entry.get("author_en", "") or ""))
    return out


def perspective_for(move_labels):
    """One line of counsel from the classics (deterministic, cached text).

    The decisive move kind selects candidate scene types through the
    general blunder map; the first scene (by id) with an English counsel
    line wins. No model call: the line is the recorded counsel text.
    """
    try:
        from .scenes import TYPE_FOR_BLUNDER
    except (ImportError, AttributeError):
        TYPE_FOR_BLUNDER = {}
    wanted: list[str] = []
    for label in (move_labels or []):
        for entry in TYPE_FOR_BLUNDER.get(str(label), []):
            code = str(entry).split()[0]
            if code not in wanted:
                wanted.append(code)
    ordered = sorted(_load_scene_list(), key=lambda s: str(s.get("id", "")))
    with_counsel = [s for s in ordered if s.get("교훈_en")]
    pick = None
    if wanted:
        for scene in with_counsel:
            if any(c in wanted for c in _scene_codes(scene)):
                pick = scene
                break
    if pick is None and with_counsel:
        pick = with_counsel[0]
    if pick is None:
        return {"line": "State the need plainly, then ask for a read-back.",
                "work": "uptake fallback (no scene file)",
                "source": ""}
    return _perspective_of(pick)


def _load_scene_list() -> list:
    scenes_path = Path(__file__).resolve().parent / "scenes.json"
    try:
        scenes = _read_json(scenes_path)
    except (OSError, ValueError):
        return []
    return [s for s in scenes if isinstance(s, dict)] if isinstance(scenes, list) else []


def _perspective_of(pick) -> dict:
    """Line, work and source of one scene, copied from the catalog."""
    work = str(pick.get("작품", "") or "")
    author = str(pick.get("작가", "") or "")
    gloss = _scene_work_gloss().get((work, author))
    if gloss is not None:
        work_en, author_en = gloss
    else:  # stay English-first even without a gloss entry
        work_en = f"classic scene {pick.get('id', '')}".strip()
        author_en = ""
    if author_en and work_en:
        work_en = f"{work_en} ({author_en})"
    sources = pick.get("원문출처", []) or []
    source = str(sources[0]) if sources else ""
    return {"line": str(pick.get("교훈_en", "") or ""),
            "work": work_en, "source": source,
            "scene_id": str(pick.get("id", "") or "")}


# --- reliability ----------------------------------------------------------------------

def split_half_routine(effects_by_turn):
    """Split-half r of the routine-read share (even vs odd readings)."""
    xs, ys = [], []
    for key in sorted(effects_by_turn):
        readings = (effects_by_turn.get(key, {}) or {}).get("readings", [])
        if not isinstance(readings, list) or len(readings) < 2:
            continue
        halves = [readings[0::2], readings[1::2]]
        shares = []
        for half in halves:
            actions = [str(r.get("next_action")) for r in half
                       if isinstance(r, dict) and r.get("next_action")]
            if not actions:
                shares = []
                break
            shares.append(sum(1 for a in actions if a in ROUTINE_READ)
                          / len(actions))
        if len(shares) == 2:
            xs.append(shares[0])
            ys.append(shares[1])
    return pearson(xs, ys)


def outcome_alignment(scored, effects_by_turn, outcome_turn, n_turns):
    """r between mismatch severity and proximity to the outcome."""
    sev, prox = [], []
    for row in (scored or []):
        t = row["turn"]
        eff = effects_by_turn.get(t, {})
        try:
            dis = float(eff.get("disagreement", 0.0))
        except (TypeError, ValueError):
            dis = 0.0
        if not math.isfinite(dis):
            dis = 0.0
        sev.append(row["share"] + dis if row["mismatch"] else 0.0)
        if outcome_turn is not None:
            lead = max(0, outcome_turn - t)
        else:
            lead = max(0, (n_turns - 1) - t)
        prox.append(1.0 / (1.0 + lead))
    return pearson(sev, prox)


def _decisive_without(turns, moves_by_turn, effects_by_turn, drop):
    kept_effects = {}
    for key, eff in effects_by_turn.items():
        readings = (eff or {}).get("readings", [])
        if isinstance(readings, list):
            left = [r for i, r in enumerate(readings)
                    if not drop(i, r)]
        else:
            left = readings
        total = len(left) if isinstance(left, list) else 0
        counts: dict[str, int] = {}
        if isinstance(left, list):
            for r in left:
                if isinstance(r, dict) and r.get("next_action"):
                    name = str(r["next_action"])
                    counts[name] = counts.get(name, 0) + 1
        dist = {k: v / total for k, v in counts.items()} if total else {}
        kept_effects[key] = {"action_dist": dist, "readings": left,
                             "disagreement": (eff or {}).get("disagreement", 0.0)}
    rescored = []
    for t in range(len(turns)):
        mv = moves_by_turn.get(t, {})
        labels = mv.get("moves", []) or []
        nxt = next_other_speaker(moves_by_turn, turns, t)
        resp = (moves_by_turn.get(nxt, {}).get("moves", []) or []
                if nxt is not None else [])
        level, share, mismatch, _ = score_turn(labels,
                                               kept_effects.get(t), resp)
        rescored.append({"turn": t, "level": level, "share": share,
                         "mismatch": mismatch,
                         "speaker": turns[t]["speaker"]})
    turn, _ = pick_decisive(turns, moves_by_turn, kept_effects, rescored)
    return turn


def perturb_stability(turns, moves_by_turn, effects_by_turn, decisive_turn):
    """Share of leave-one-stance-out runs keeping the decisive turn."""
    stances = set()
    for eff in effects_by_turn.values():
        readings = (eff or {}).get("readings", [])
        if isinstance(readings, list):
            for r in readings:
                if isinstance(r, dict) and r.get("stance"):
                    stances.add(str(r["stance"]))
    if len(stances) >= 2:
        drops = [lambda i, r, s=s: isinstance(r, dict)
                 and str(r.get("stance")) == s for s in sorted(stances)]
    else:
        k = max((len((eff or {}).get("readings", []) or [])
                 for eff in effects_by_turn.values()), default=0)
        if k < 2:
            return None
        drops = [lambda i, r, j=j: i == j for j in range(k)]
    same, total = 0, 0
    for drop in drops:
        try:
            kept = _decisive_without(turns, moves_by_turn,
                                     effects_by_turn, drop)
        except (ValueError, TypeError, KeyError):
            continue
        total += 1
        if kept == decisive_turn:
            same += 1
    if total == 0:
        return None
    return same / total


def build_reliability(scored, effects_by_turn, decisive_turn, outcome_turn,
                      n_turns, limits):
    """Pre-registered reliability block (C7).

    The numbers describe the reader on CaSiNo held-out dialogues and
    are identical for every case; the per-case arguments are accepted
    for signature stability and deliberately ignored.
    """
    return {"split_half": PREREG_SPLIT_HALF,
            "k": PREREG_K,
            "outcome_r1": PREREG_OUTCOME_R1,
            "outcome_r1_ho1": PREREG_OUTCOME_R1_HO1,
            "outcome_r1_ho2": PREREG_OUTCOME_R1_HO2,
            "perturb": PREREG_PERTURB,
            "measured_on": PREREG_MEASURED_ON,
            "limits": [PREREG_LIMIT, *list(limits)]}


# --- report assembly ---------------------------------------------------------------------

def outcome_turn_of(case, n_turns):
    try:
        value = int(case.get("outcome_turn"))
    except (TypeError, ValueError):
        return n_turns - 1 if n_turns else 0
    if n_turns and not 0 <= value < n_turns:
        return n_turns - 1
    return value


#: Stated whenever a report uses the model judgment.
JUDGMENT_LIMIT = (
    "The decisive turn, turning points, goals, strategy shifts and perspective "
    "are one model judgment of the whole conversation, a model estimate and not "
    "a record fact. The model was given no case title, id, outcome or note, but "
    "if the words of the conversation identify a known event, its judgment may "
    "be influenced by what it knows of that event. The relationship, channel "
    "and stakes weight it saw were read earlier by a step that did see the "
    "record's title and note.")
#: The same limit when the stakes weight the judgment saw came from the case record.
JUDGMENT_LIMIT_RECORD = JUDGMENT_LIMIT.replace(
    "relationship, channel and stakes weight it saw were", "relationship and channel it saw were"
).replace("record's title and note.", "record's title and note; the stakes weight it saw is the case record's.")
#: Stated in every report: how far the stakes weight in the alert level has been tested.
STAKES_LIMIT = (
    "Alert cuts were set on development turns whose stakes weight was a model "
    "estimate. The stakes weight condition test currently fails; this is to be "
    "checked in the locked evaluation.")
ESTIMATE = "model estimate"


def judgment_limit(conditions) -> str:
    """The judgment limit, naming where the stakes weight it saw came from."""
    source = str((conditions or {}).get("stakes_source") or "")
    return JUDGMENT_LIMIT_RECORD if source.startswith("record: ") else JUDGMENT_LIMIT


def judgment_grade_limit(grade) -> str:
    """The measured standing of the model judgment; empty until it is measured."""
    if not (grade or {}).get("grade"):
        return ""
    return (f"The model judgment is graded {grade['grade']}, not measurement: "
            f"{grade.get('repeat', '')}; {grade.get('swap', '')}; {grade.get('mask', '')}; "
            f"{grade.get('divides', '')}. Read the per-turn values first.")


def render_criterion(criterion, intended, esc) -> str:
    """The sentence naming what section 6 compares the wordings on."""
    if not criterion:
        return ""
    if not criterion.get("item"):
        return (f"<p>Wordings are compared on the speaker's intended sentence: "
                f"{esc(str(intended or 'unknown'))}<br>"
                f"Basis: {esc(str(criterion.get('source', '')))}.</p>")
    return ("<p>Wordings are compared on {} ({} grade): wanted {:.0f}/100, landed "
            "{:.0f}/100.<br>Basis: {}.</p>".format(
                esc(str(criterion["item"])), esc(str(criterion.get("grade", ""))),
                criterion["wanted"], criterion["actual"],
                esc(str(criterion.get("source", "")))))


def render_middle(middle, ordering, esc) -> str:
    """The middle wording and whether worse, middle and alternatives keep order."""
    if not middle:
        return ""
    if middle.get("p_better") is None:
        return ("<p>Middle wording: not scored in this run, so the order of worse, "
                "middle and alternatives is not measured.</p>")
    order = ordering or {}
    verdict = {True: "holds", False: "does not hold"}.get(order.get("holds"), "is not measured")
    return ("<p>A wording written to be only a little better than the original reads {} (n={}, {}). "
            "Worse {}, middle {}, mean alternative {}: the expected order {}.</p>"
            "<blockquote>{}</blockquote>".format(
                esc(str(middle["p_better"])), esc(str(middle.get("n_compared", ""))),
                esc(str(middle.get("method", ""))), esc(str(order.get("worse", "—"))),
                esc(str(order.get("middle", "—"))),
                esc(str(order.get("alternatives_mean", "—"))), verdict,
                esc(str(middle.get("text", "")))))


def top_severity_line(uptake, n=3) -> str:
    """The highest-severity turns, from the values alone, before any model words."""
    ranked = sorted(((r["layers"]["severity"], r["turn"]) for r in (uptake or [])
                     if (r.get("layers") or {}).get("severity")), key=lambda x: (-x[0], x[1]))[:n]
    return ", ".join(f"#{turn} ({severity:.1f})" for severity, turn in ranked)


def build_report(case_id, data_root="data/private", model=None, meter=None,
                 cache_dir=None, budget_note=None, max_turns=None, judgment_rep=0):
    turns, case, panel, _ = load_inputs(case_id, data_root)
    meta = load_case_meta().get(case_id, {})
    moves_by_turn = _moves_by_turn(panel)
    effects_by_turn = _effects_by_turn(panel)
    n_turns = len(turns)
    outcome_turn = outcome_turn_of(case, n_turns)
    genre = panel.get("genre") if isinstance(panel, dict) else None
    panel_alerts, scored = build_alerts(turns, moves_by_turn, effects_by_turn,
                                        outcome_turn, genre=genre)
    client = None
    if model:
        from .reader import MuseClient
        client = MuseClient(meter=meter)
    from .multilayer import build_multilayer, measured_items, turn_effect
    stakes_record = case.get("stakes_record")
    if stakes_record is None:
        records = load_stakes_records()
        stakes_record = records.get(case_id, records.get(str(case_id)))
    layered = build_multilayer(turns, client=client, model=model, meter=meter,
                               cache_dir=cache_dir, max_turns=max_turns,
                               record={"title": case.get("title"), "note": case.get("note")},
                               stakes_record=stakes_record)
    uptake, alerts, standing = layered["uptake"], layered["alerts"], layered["standing"]
    use = measured_items(layered["grades"])
    for u in uptake:
        effect = effects_by_turn.get(u["turn"])
        u["scales"] = turn_scales(effect)
        u["mark"] = effect.get("annotation") if isinstance(effect, dict) else None
        u["effect"] = turn_effect(u, use)
    judged = judge_conversation(turns, uptake, alerts, layered["conditions"], client, model,
                                meter, cache_dir, rep=judgment_rep)
    estimate = f"{ESTIMATE} ({model}, rep {judgment_rep})"

    def fallback(field):
        return f"code fallback: {judged.get('reason') or f'no valid {field} from the model'}"

    decisive_pick = None
    if judged["decisive"]:
        move = next(m for m in judged["turns"] if m["turn"] == judged["decisive"]["turn"])
        decisive_pick = {**judged["decisive"], "opened": move["opened"],
                         "received": move["received"], "source": estimate}
    decisive = build_decisive(case_id, turns, case, moves_by_turn,
                              effects_by_turn, scored, uptake=uptake,
                              use=use, judged=decisive_pick,
                              grades=layered["grades"].get("items"))
    if not decisive_pick:
        decisive["source"] = fallback("decisive turn")
        decisive["intended_source"] += f"; {decisive['source']}"
    d_moves = moves_by_turn.get(decisive["turn"], {}).get("moves", []) or []
    d_turn = decisive["turn"]
    if isinstance(d_turn, int) and 0 <= d_turn < len(turns):
        d_speaker = turns[d_turn]["speaker"]
        d_original = turns[d_turn]["text"]
    else:
        d_speaker, d_original = "", decisive["quote"]
    alternatives = build_alternatives(
        decisive["quote"], scored, effects_by_turn, client=client,
        model=model, meter=meter, cache_dir=cache_dir, turns=turns,
        decisive_turn=d_turn, speaker=d_speaker,
        intended=decisive.get("intended", ""), original=d_original,
        criterion=decisive["criterion"])
    alternatives_control = build_alternative_controls(
        decisive["quote"], scored, effects_by_turn, client=client,
        model=model, meter=meter, cache_dir=cache_dir, turns=turns,
        decisive_turn=d_turn, speaker=d_speaker,
        intended=decisive.get("intended", ""), original=d_original,
        criterion=decisive["criterion"])
    alternatives_control["ordering"] = alternative_ordering(alternatives, alternatives_control)
    if judged["scene"]:
        perspective = judged_perspective(judged["scene"], judged["n_scenes"], estimate)
    else:
        perspective = perspective_for(d_moves)
        perspective["url"] = perspective["source"]
        perspective["basis"] = fallback("scene")
        perspective["source"] = "; ".join(x for x in (perspective["basis"], perspective["url"]) if x)
    # The goal path spends a model call only for what the judgment left blank.
    all_goals = all((judged["speakers"].get(str(t["speaker"])) or {}).get("goal") for t in turns)
    speakers = build_speakers(turns, moves_by_turn, panel_alerts,
                              effects_by_turn, case,
                              client=None if all_goals else client,
                              model=model, meter=meter,
                              cache_dir=cache_dir)
    apply_judged_speakers(speakers, judged, estimate, fallback("goal or shifts"))
    if judged["divides"] is not None:
        turning_points = build_judged_turning_points(turns, judged, uptake, use, estimate)
        turning_points_source = "model judgment"
    else:
        turning_points = build_reading_turning_points(
            turns, uptake, alerts, standing, use)
        turning_points_source = "per-item readings"
        if turning_points is None:
            turning_points_source = "panel or text"
            turning_points = build_turning_points(turns, moves_by_turn,
                                                  effects_by_turn, scored)
    judgment = {
        "model": model, "rep": judgment_rep, "source": judged["source"],
        "grade": (layered["grades"].get("judgment") or {}).get("grade"),
        "fields": {
            "decisive": estimate if decisive_pick else decisive["source"],
            "turning_points": estimate if judged["divides"] is not None else fallback("turning points"),
            "turns": estimate if judged["turns"] else fallback("turn explanations"),
            "speakers": {str(sp["speaker"]): {
                "goal": estimate if (judged["speakers"].get(str(sp["speaker"])) or {}).get("goal")
                else fallback("goal"), "shifts": sp["shifts_source"]} for sp in speakers},
            "perspective": estimate if judged["scene"] else perspective["basis"],
        },
        "turns": judged["turns"], "n_scenes": judged["n_scenes"], "dropped": judged["dropped"],
    }
    limits = [
        "Alerts come from per-item readings made after the reply is seen: "
        "how much a turn reached for an item, how far it landed, weighted "
        "by what the conversation puts at stake. Blank means the words give "
        "no support; blank is not zero.",
        STAKES_LIMIT,
    ]
    if judged["source"] == SOURCE_MUSE:
        limits.append(judgment_limit(layered["conditions"]))
        if judgment_grade_limit(layered["grades"].get("judgment")):
            limits.append(judgment_grade_limit(layered["grades"].get("judgment")))
    if layered["grades_status"] == "provisional":
        limits.append("Item grades and alert cuts are provisional until the "
                      "measurement run is complete.")
    else:
        counted = sorted(k for k, g in layered["grades"]["items"].items() if g == "measured")
        limits.append("Alerts and standing count only items that passed the measurement run ("
                      + ", ".join(counted) + "). Every other item is shown as reference.")
    if not moves_by_turn:
        limits.append("No panel record for this case: scales are unknown."
                      if decisive["read_prob"] is None else
                      "No panel record for this case: decisive "
                      "turn is a placeholder, scales are unknown.")
    if any(a.get("method") == "offline-proxy" for a in alternatives):
        limits.append("Alternative scores are an offline proxy "
                      "(Laplace-smoothed room for improvement from panel "
                      "readings), not model votes; pass --model for votes.")
    if any(a.get("method") == "paired" for a in alternatives):
        criterion_words = _criterion_line(decisive["criterion"],
                                          decisive.get("intended", ""), d_speaker)
        limits.append("Alternative scores are paired votes on which wording "
                      f"better produces {criterion_words}: one "
                      "five-stance panel judges the original against each "
                      "model-written rewrite in both orders (ties count "
                      "half); p_better is Laplace-smoothed (k+1)/(n+2) so "
                      "finite votes never assert 0% or 100%, raw share "
                      "in p_empirical.")
    if alternatives_control.get("identical", {}).get("method") == "paired":
        limits.append("Negative controls use the same panel: the "
                      "original against itself (must read ~0.5) and "
                      "against a deliberately worse wording (must read "
                      "below 0.5), so a judge that always prefers the "
                      "new wording cannot pass.")
    if budget_note:
        limits.append(budget_note)
    reliability = build_reliability(scored, effects_by_turn,
                                    decisive["turn"], outcome_turn,
                                    n_turns, limits)
    report = {
        "id": case_id,
        "title": meta.get("title_en", str(case.get("title", case_id))),
        "record_title": str(case.get("title", "")),
        "domain": meta.get("domain", "negotiation"),
        "result_type": meta.get("result_type", "control"),
        "outcome": str(case.get("outcome", "")),
        "outcome_en": meta.get("outcome_en", ""),
        "source_url": str(case.get("source_url", "")),
        "n_turns": n_turns,
        "conditions": layered["conditions"],
        "grades": {"items": layered["grades"].get("items", {}),
                   "causes": layered["grades"].get("causes", {})},
        "uptake": uptake,
        "alerts": alerts,
        "panel_alerts": [{k: a.get(k) for k in ("turn", "level", "axis",
                                                 "reason", "resolved")}
                         for a in panel_alerts],
        "decisive": {k: decisive[k] for k in (
            "turn", "quote", "intended", "intended_source", "said",
            "read", "read_prob", "scales", "source", "item", "grade", "wanted",
            "actual") if k in decisive},
        "judgment": judgment,
        "alternatives": alternatives,
        "alternatives_control": alternatives_control,
        "perspective": perspective,
        "reliability": reliability,
        "speakers": speakers,
        "turning_points": turning_points,
        "turning_points_source": turning_points_source,
        "standing": standing,
    }
    for sp in report["speakers"]:
        row = standing["end"].get(str(sp.get("speaker")))
        if row:
            sp["end_by_stake"] = row
            means = [v["mean"] for v in row.values() if v["mean"] is not None]
            if means:
                sp["end_balance"] = round(sum(means) / len(means), 3)
                sp["end_basis"] = ("Average of the per-stake means, kept for "
                                   "sorting only; read end_by_stake. "
                                   "Each reading is how far the wanted effect "
                                   "landed, -1 to +1, weighted by how much it "
                                   "was wanted.")
    return report


# --- HTML -----------------------------------------------------------------------------------

def _top_causes(cause, n=3, grades=None):
    rows = [(v, f"{layer}: {k}" + (" (reference)" if grades and grades.get(f"{layer}.{k}") != "measured" else ""))
            for layer in ("say", "mind", "world")
            for k, v in (cause.get(layer) or {}).items()]
    return [name for _, name in sorted(rows, key=lambda x: -x[0])[:n]]


#: What the mark next to a turn number is, shown with the alert timeline.
TURN_EFFECT_NOTE = (
    "The mark next to a turn number is the turn effect computed from measured "
    "items only; a turn with no measured item value stays blank, and the mark "
    "does not rate the person.")


def _effect_marks(uptake) -> dict:
    return {u.get("turn"): (u.get("effect") or {}).get("mark")
            for u in uptake if isinstance(u, dict)}


def _turn_label(turn, marks, esc) -> str:
    """A turn number with its effect mark; a turn without one shows the number alone."""
    mark = marks.get(turn)
    return f"#{esc(str(turn))} {esc(str(mark))}" if mark else f"#{esc(str(turn))}"


def _effect_cell(effect, esc) -> str:
    """The turn effect and the items it was added from; a dash when it has no value."""
    effect = effect or {}
    if effect.get("value") is None:
        return "–"
    parts = ", ".join(f"{esc(str(k))} {v:+.3f}" for k, v in (effect.get("items") or {}).items())
    return f"{esc(str(effect.get('mark')))} {effect['value']:+.3f} ({parts})"


def render_readings(uptake, conditions, esc, grades=None) -> str:
    """Per-turn readings: classes in a row, items and causes one click away."""
    readings = [u for u in uptake if isinstance(u, dict)]
    if not readings:
        return "<p>No per-turn reading on record.</p>"
    grades = grades or {}
    item_grades = grades.get("items") or {}
    known = [f"{k}: {esc(str(conditions[k]))}" for k in (
        "relationship", "power", "channel", "culture", "stakes_weight")
        if conditions.get(k) is not None]
    if conditions.get("stakes_weight") is not None and conditions.get("stakes_source"):
        source = str(conditions["stakes_source"])
        if source.startswith("record: "):
            model_read = conditions.get("stakes_weight_model")
            source += f"; model read {'–' if model_read is None else model_read}"
        known[-1] += f" ({esc(source)})"
    head = (f"<p>Conditions ({esc(str(conditions.get('source', '')))}): "
            + "; ".join(known) + "</p>") if known else ""
    rows = []
    for u in readings:
        classes = (u.get("layers") or {}).get("classes") or {}
        cells = "".join("<td>{}</td>".format(
            "–" if classes.get(c) is None else f"{classes[c]:.0f}") for c in STAKES)
        detail = []
        for name, e in (u.get("items") or {}).items():
            if not (e.get("wanted") or e.get("actual")):
                continue
            exp = (u.get("expected") or {}).get(name) or {}
            detail.append("{}{}: wanted {}, expected {}, landed {}{}".format(
                esc(name), "" if not item_grades or item_grades.get(name) == "measured" else " (reference)",
                "–" if not e.get("wanted") else f"{e['wanted']['score']:.0f}",
                "–" if exp.get("score") is None else f"{exp['score']:.0f}",
                "–" if not e.get("actual") else f"{e['actual']['score']:.0f}",
                f" (#{e['reply_turn']}: {esc(e['quote'])})" if e.get("quote") else ""))
        cause = u.get("cause") or {}
        if cause:
            kind = cause.get("kind") or {}
            top_kind = max(kind, key=kind.get) if kind else None
            stages = {k: v for k, v in (cause.get("stages") or {}).items() if v is not None}
            weak = min(stages, key=stages.get) if stages else None
            line = "Where (estimate): " + ", ".join(esc(x) for x in _top_causes(cause, grades=grades.get("causes")))
            if top_kind:
                line += f"; kind: {esc(top_kind)}"
            if weak:
                line += f"; weakest link: {esc(weak)}"
            detail.append(line)
        body = "<br>".join(detail) or "No item supported by the words."
        rows.append(
            "<tr><td>#{}</td><td>{}</td>{}<td>{}</td><td>{}</td><td><details><summary>items</summary>{}"
            "</details></td></tr>".format(
                esc(str(u.get("turn", ""))), esc(str(u.get("speaker", ""))), cells,
                esc(", ".join(u.get("broken") or [])) or "–", _effect_cell(u.get("effect"), esc), body))
    heads = "".join(f"<th>{esc(c)}</th>" for c in STAKES)
    return (head + "<p>How far the wanted effect landed, 0 to 100, read after the reply. "
            "A dash means the words give no support.</p>"
            f"<table><tr><th>Turn</th><th>Speaker</th>{heads}<th>Fell short</th>"
            "<th>Turn effect (measured items)</th><th></th></tr>"
            + "".join(rows) + "</table>")


#: What the per-stake end values cover, shown with them.
END_SCOPE_SENTENCE = (
    "These values add up, inside this conversation, how far each speaker's "
    "words landed as read through up to 3 later turns; they do not include "
    "the outcome of the event.")
NO_STAKE_VALUES = ("No per-stake values on record: no turn in this report has "
                   "a judged value on a measured item. Stakes are never "
                   "merged into one number.")


def _end_stake_cell(entry) -> str:
    """One speaker's end value on one stake; a dash when nothing was judged."""
    judged = (entry or {}).get("judged")
    if not judged:
        return "–"
    return "{:+.3f} (mean {:+.3f}, {} turn{})".format(
        entry["score"], entry["mean"], judged, "" if judged == 1 else "s")


def _is_estimate(source) -> bool:
    return str(source or "").startswith(ESTIMATE)


def render_turn_moves(moves, esc, marks=None) -> str:
    """How each turn steered and how it was received, from the judgment."""
    rows = "".join(
        "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            _turn_label(m["turn"], marks or {}, esc), esc(str(m["speaker"])), esc(str(m["opened"])),
            f"#{m['response_turn']} {esc(str(m['received_by']))}: {esc(str(m['received']))}"
            if m.get("response_turn") is not None else "no reply")
        for m in moves)
    return ("<p>Turn by turn: how each speaker steered and how it was received "
            "(model estimate).</p><table><tr><th>Turn</th><th>Speaker</th>"
            f"<th>Steered</th><th>Received</th></tr>{rows}</table>") if moves else ""


def render_html(report) -> str:
    esc = html.escape
    judgment = report.get("judgment") or {}
    judged_fields = judgment.get("fields") or {}
    is_reference = bool(judgment.get("grade")) and judgment.get("source") == SOURCE_MUSE
    reference_mark = f" ({esc(str(judgment['grade']))}, model judgment)" if is_reference else ""
    reference_html = ("<p><strong>Reference, not measurement.</strong> The scene choices and "
                      "explanations below are a model judgment that did not pass its measurement "
                      "run; see the limits at the end.</p>") if is_reference else ""
    alerts = report.get("alerts", []) or []
    uptake = report.get("uptake", []) or []
    standing = report.get("standing") or {}
    decisive = report.get("decisive", {}) or {}
    scales = decisive.get("scales", {}) or {}
    alternatives = report.get("alternatives", []) or []
    perspective = report.get("perspective", {}) or {}
    reliability = report.get("reliability", {}) or {}
    speakers = report.get("speakers", []) or []
    turning_points = report.get("turning_points", []) or []
    stake_rows = [sp for sp in speakers if isinstance(sp, dict)]
    has_stake_values = any((sp.get("end_by_stake") or {}).get(a, {}).get("judged")
                           for sp in stake_rows for a in STAKES)
    marks = _effect_marks(uptake)
    rows = []
    for a in alerts:
        rows.append(
            "<tr><td>{turn}</td><td>{level}</td><td>{axis}</td>"
            "<td>{reason}</td><td>{resolved}</td></tr>".format(
                turn=_turn_label(a.get("turn", ""), marks, esc),
                level=esc(str(a.get("level", ""))),
                axis=esc(str(a.get("axis", ""))),
                reason=esc(str(a.get("reason", ""))),
                resolved="yes" if a.get("resolved") else "no"))
    if not rows:
        rows.append("<tr><td colspan=\"5\">No alerts: no axis reads "
                    "missed, partly or cost.</td></tr>")
    uptake_html = render_readings(uptake, report.get("conditions") or {}, esc, report.get("grades"))
    goal_blocks = []
    for sp in speakers:
        if not isinstance(sp, dict):
            continue
        shift_items = []
        for sh in (sp.get("shifts", []) or []):
            if not isinstance(sh, dict):
                continue
            shift_items.append(
                "<li>Turn #{turn}{mark}: {frm} → {to}"
                "<blockquote>{quote}</blockquote></li>".format(
                    mark=" (model estimate)" if _is_estimate(sp.get("shifts_source")) else "",
                    turn=esc(str(sh.get("turn", ""))),
                    frm=esc(str(sh.get("from", ""))),
                    to=esc(str(sh.get("to", ""))),
                    quote=esc(str(sh.get("evidence_quote", "")))))
        if shift_items:
            shift_html = "<ul>" + "".join(shift_items) + "</ul>"
        else:
            shift_html = ("<p>No strategy shift in the model judgment "
                          "(model estimate).</p>"
                          if _is_estimate(sp.get("shifts_source")) else
                          "<p>No strategy shift detected in the move "
                          "record.</p>")
        goal_blocks.append(
            "<h3>{speaker}</h3>"
            "<p>Initial goal: {goal}<br>"
            "Basis: {source}</p>"
            "<blockquote>{quote}</blockquote>"
            "{shifts}".format(
                speaker=esc(str(sp.get("speaker", "?"))),
                goal=esc(str(sp.get("initial_goal", ""))),
                source=esc(str(sp.get("goal_source", ""))),
                quote=esc(str(sp.get("evidence_quote", ""))),
                shifts=shift_html))
    if not goal_blocks:
        goal_blocks.append("<p>No speakers on record.</p>")
    tp_blocks = []
    for tp in turning_points:
        if not isinstance(tp, dict):
            continue
        resp = tp.get("response_turn")
        try:
            turn_no = int(tp.get("turn", -1))
        except (TypeError, ValueError):
            turn_no = -1
        if resp is None:
            reply = "no later reply"
        elif isinstance(resp, int) and not isinstance(resp, bool) \
                and resp < turn_no:
            reply = f"no later reply (answers #{esc(str(resp))})"
        else:
            reply = f"#{esc(str(resp))}"
        tp_blocks.append(
            "<h3>Turn {turn} ({speaker})</h3>"
            "<p>Steered: {opened}</p>"
            "<blockquote>{quote}</blockquote>"
            "<p>Received by {by} as: {received}<br>"
            "Reply at {reply}; effect: {effect}<br>"
            "Wanted vs happened: {wanted}<br>"
            "Deliberate: {intent}.{basis}</p>".format(
                basis=(f"<br>Basis: {esc(str(tp['source']))}"
                       if tp.get("source") else ""),
                turn=_turn_label(tp.get("turn", ""), marks, esc),
                speaker=esc(str(tp.get("speaker", "?"))),
                opened=esc(str(tp.get("opened", ""))),
                quote=esc(str(tp.get("quote", ""))),
                by=esc(str(tp.get("received_by", "") or "—")),
                received=esc(str(tp.get("received_as", ""))),
                reply=reply,
                effect=esc(str(tp.get("effect", ""))),
                wanted=esc(str(tp.get("wanted_vs_happened", ""))),
                intent=esc(str(tp.get("intentional", "")))))
    if not tp_blocks:
        tp_blocks.append("<p>No turning points with an observed reply "
                         "on record.</p>")
    if report.get("turning_points_source") == "model judgment":
        tp_intro = ("Each turning point is a turn the model judged to divide "
                    "the conversation, reading it whole. The words are a model "
                    "estimate; the values are copied from the per-item reading.")
    elif report.get("turning_points_source") == "per-item readings":
        tp_intro = ("Each turning point is a turn where an alert fired, or "
                    "where the standing on a stake split and an item of that "
                    "stake fell short. Values are per-item readings, which "
                    "are model estimates.")
    else:
        tp_intro = ("Each turning point is a turn where an urgent intent was "
                    "read as routine; when no such mismatch exists, steering "
                    "moves (proposals, decisions, pushes) stand in, "
                    "text-inferred where no panel record exists.")
    position = stake_tally(uptake)
    judged_total = sum(position[a][g] for a in STAKES
                       for g in ("met", "partly", "missed", "cost"))
    if judged_total:
        pos_rows = []
        for axis in STAKES:
            row = position[axis]
            turns_txt = ", ".join(f"#{t}" for t in row["turns"]) or "—"
            pos_rows.append(
                "<tr><td>{stake}</td><td>{met}</td><td>{partly}</td>"
                "<td>{missed}</td><td>{cost}</td>"
                "<td>{turns}</td></tr>".format(
                    stake=esc(axis), met=row["met"],
                    partly=row["partly"], missed=row["missed"],
                    cost=row["cost"], turns=esc(turns_txt)))
        position_html = (
            "<p>Position by stake, from the per-turn readings: each "
            "stake stays separate and is never merged into one "
            "number.</p>"
            "<table><tr><th>Stake</th><th>Met</th><th>Partly</th>"
            "<th>Missed</th><th>Cost</th><th>Judged turns</th></tr>"
            + "".join(pos_rows) + "</table>")
    elif has_stake_values:
        position_html = ("<p>Position by stake: section 8 gives each "
                         "speaker's value on each stake. Each stake stays "
                         "separate and is never merged into one number.</p>")
    else:
        position_html = f"<p>{esc(NO_STAKE_VALUES)}</p>"
    balance_rows = []
    for sp in speakers:
        if not isinstance(sp, dict):
            continue
        balance_rows.append(
            "<tr><td>{speaker}</td><td>{balance}</td>"
            "<td>{basis}</td></tr>".format(
                speaker=esc(str(sp.get("speaker", "?"))),
                balance=esc(str(sp.get("end_balance", ""))),
                basis=esc(str(sp.get("end_basis", "")))))
    if not balance_rows:
        balance_rows.append("<tr><td colspan=\"3\">No speakers on "
                            "record.</td></tr>")
    per_speaker = speaker_stake_tally(uptake)
    judged_any = any(per_speaker[sp][a][g]
                     for sp in per_speaker for a in STAKES
                     for g in ("met", "partly", "missed", "cost"))
    if judged_any:
        sp_rows = []
        for speaker in per_speaker:
            for axis in STAKES:
                row = per_speaker[speaker][axis]
                sp_rows.append(
                    "<tr><td>{speaker}</td><td>{stake}</td>"
                    "<td>{met}</td><td>{partly}</td><td>{missed}</td>"
                    "<td>{cost}</td></tr>".format(
                        speaker=esc(speaker), stake=esc(axis),
                        met=row["met"], partly=row["partly"],
                        missed=row["missed"], cost=row["cost"]))
        speaker_position_html = (
            "<p>End position by speaker and stake, from the per-turn "
            "readings: each stake stays separate and is never merged "
            "into one number.</p>"
            "<table><tr><th>Speaker</th><th>Stake</th><th>Met</th>"
            "<th>Partly</th><th>Missed</th><th>Cost</th></tr>"
            + "".join(sp_rows) + "</table>")
    else:
        speaker_position_html = ""
    if has_stake_values:
        heads = "".join(f"<th>{esc(a)}</th>" for a in STAKES)
        end_rows = "".join(
            "<tr><td>{}</td>{}</tr>".format(
                esc(str(sp.get("speaker", "?"))),
                "".join("<td>{}</td>".format(esc(_end_stake_cell(
                    (sp.get("end_by_stake") or {}).get(a)))) for a in STAKES))
            for sp in stake_rows)
        split_rows = []
        for a in STAKES:
            sp = (standing.get("splits") or {}).get(a)
            if sp:
                split_rows.append("<li>{}: #{} ({:+.1f} over 3 turns)</li>".format(
                    esc(a), sp["turn"], sp["shift"]))
        end_stake_html = (
            "<table><tr><th>Speaker</th>" + heads + "</tr>" + end_rows + "</table>"
            f"<p>{esc(END_SCOPE_SENTENCE)} Each cell is the running total, "
            "its mean per judged turn, and the judged turns. A dash means no "
            "judged turn on this stake for this speaker. Each stake stays "
            "separate and is never merged into one number.</p>")
        if split_rows:
            end_stake_html += ("<p>Where each stake shifted most:</p><ul>"
                               + "".join(split_rows) + "</ul>")
        balance_intro = ""
        balance_table = ""
    else:
        end_stake_html = f"<p>{esc(NO_STAKE_VALUES)}</p>"
        balance_intro = ("<p>Without per-stake values, each row's basis says "
                         "what its number comes from.</p>")
        balance_table = ("<table>"
                         "<tr><th>Speaker</th><th>Balance</th><th>Basis</th></tr>"
                         + "".join(balance_rows) + "</table>")
    alt_blocks = []
    quote = str(decisive.get("quote", ""))
    read_prob = decisive.get("read_prob")
    read_share = "" if read_prob is None else f" ({esc(str(read_prob))})"
    for i, alt in enumerate(alternatives, 1):
        pre, span, post = split_verbatim(str(alt.get("text", "")), quote)
        if span:
            lead = f"Alternative {i}:" + (f" {pre.strip()}"
                                          if pre.strip() else "")
            wording = (f"<p>{esc(lead)}</p>"
                       f"<blockquote>{esc(span)}</blockquote>")
            if post.strip():
                wording += f"<p>{esc(post.strip())}</p>"
        else:
            wording = "<p>Alternative {i}: {text}</p>".format(
                i=i, text=esc(pre))
        alt_blocks.append(
            "<li>{wording}"
            "<p>Better-read chance: {p} "
            "(n={n}, 95% CI {lo}–{hi}, {method}).</p></li>".format(
                wording=wording,
                p=esc(str(alt.get("p_better", ""))),
                n=esc(str(alt.get("n_compared", ""))),
                lo=esc(str(alt.get("ci_low", ""))),
                hi=esc(str(alt.get("ci_high", ""))),
                method=esc(str(alt.get("method", "")))))
    control = report.get("alternatives_control", {}) or {}
    same = control.get("identical", {}) or {}
    worse = control.get("worse", {}) or {}
    if same or worse:
        control_html = (
            "<p>Negative controls (same panel): the original against "
            "itself reads {sp} (n={sn}, {sm}); a deliberately worse "
            "wording reads {wp} (n={wn}, {wm}). A judge "
            "that always prefers the new wording fails the worse "
            "control.</p>"
            "<blockquote>{wt}</blockquote>".format(
                sp=esc(str(same.get("p_better", "—"))),
                sn=esc(str(same.get("n_compared",
                                    same.get("n", "—")))),
                sm=esc(str(same.get("method", "—"))),
                wt=esc(str(worse.get("text", "—"))),
                wp=esc(str(worse.get("p_better", "—"))),
                wn=esc(str(worse.get("n_compared",
                                     worse.get("n", "—")))),
                wm=esc(str(worse.get("method", "—")))))
    else:
        control_html = ""
    criterion_html = render_criterion(control.get("criterion"), decisive.get("intended"), esc)
    middle_html = render_middle(control.get("middle"), control.get("ordering"), esc)
    if any(_is_estimate(v.get("goal")) for v in (judged_fields.get("speakers") or {}).values()):
        goals_intro = ("Goals and strategy shifts marked model estimate come from one "
                       "model judgment of the whole conversation and cover this "
                       "conversation only. Quotes are verbatim from the record.")
    else:
        goals_intro = ("Goals are read from this conversation only: each speaker's first "
                       "move sets the initial goal, and later moves show strategy shifts. "
                       "Move labels are model judgments; quotes are verbatim from the "
                       "record.")
    tp_fallback = judged_fields.get("turning_points")
    tp_source_html = (f"<p>Source: {esc(str(tp_fallback))}</p>"
                      if tp_fallback and not _is_estimate(tp_fallback) else "")
    p_url = perspective.get("url", perspective.get("source"))
    p_link = f'<a href="{esc(str(p_url))}">source</a>' if str(p_url or "").startswith("http") else ""
    p_notes = []
    if perspective.get("why"):
        p_notes.append(f"<br>Why (model estimate): {esc(str(perspective['why']))}")
    p_basis = perspective.get("basis") or ("" if p_link else perspective.get("source"))
    if p_basis:
        p_notes.append(f"<br>Basis: {esc(str(p_basis))}")
    judgment_html = ("<p>Judgment: {} (model {}, rep {}). Every field it supplies is "
                     "marked model estimate; fields from code rules are marked code "
                     "fallback.</p>".format(esc(str(judgment.get("source", ""))),
                                            esc(str(judgment.get("model"))),
                                            esc(str(judgment.get("rep"))))
                     if judgment else "")
    limit_items = "".join(
        f"<li>{esc(str(item))}</li>"
        for item in (reliability.get("limits", []) or []))
    source = str(report.get("source_url", "") or "")
    if source:
        source_html = f"<p>Source: <a href=\"{esc(source)}\">{esc(source)}</a></p>"
    else:
        source_html = "<p>Source: case record (no URL).</p>"
    top_turns = top_severity_line(uptake)
    numbers_html = f"<p>By the values alone, highest severity: {esc(top_turns)}.</p>" if top_turns else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Uptake report: {esc(str(report.get("title", "")))}</title>
<style>
body {{ font-family: sans-serif; max-width: 72ch; margin: 2em auto;
padding: 0 1em; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.3em 0.6em; text-align: left; }}
.notice {{ background: #f6f8fa; border: 1px solid #d0d7de;
padding: 0.6em 1em; }}
blockquote {{ border-left: 3px solid #ccc; margin-left: 0; padding-left: 1em; }}
</style>
</head>
<body>
<p class="notice">Uptake reads the utterance, not the person.</p>
<h1>Uptake report: {esc(str(report.get("title", "")))}</h1>
<p>{esc(PRODUCT_SENTENCE)}</p>
<section id="event">
<h2>1. What happened and how it ended</h2>
<p>{esc(str(report.get("outcome_en", "") or report.get("outcome", "")))}</p>
{source_html}
</section>
<section id="goals">
<h2>2. Each speaker's goals and strategy shifts</h2>
<p>{goals_intro}</p>
{"".join(goal_blocks)}
</section>
<section id="turning-points">
<h2>3. Position and turning points</h2>
<p>How each turn steered the conversation, and how that was received.
{tp_intro}</p>
{numbers_html}
{position_html}
{reference_html}
{tp_source_html}
{"".join(tp_blocks)}
{render_turn_moves(judgment.get("turns") or [], esc, marks)}
</section>
<section id="alerts">
<h2>4. Alert timeline</h2>
<p>Each alert is one stake where what the speaker wanted and what
happened diverged (missed, partly, or cost), judged after the reply
is seen. For each turn: what was wanted on this stake, and what the
later turns show happened.</p>
<p>{esc(TURN_EFFECT_NOTE)}</p>
<table>
<tr><th>Turn</th><th>Level</th><th>Axis</th><th>Reason</th>
<th>Resolved</th></tr>
{"".join(rows)}
</table>
{uptake_html}
</section>
<section id="intent">
<h2>5. Intended, said, read</h2>
<p>Decisive turn #{esc(str(decisive.get("turn", "")))}{reference_mark}.</p>
<p>Intended: {esc(str(decisive.get("intended", "")))}<br>
Basis: {esc(str(decisive.get("intended_source", "")))}</p>
<blockquote>{esc(str(decisive.get("said", "")))}</blockquote>
<p>Read as: {esc(str(decisive.get("read", "")))}{read_share}{" (model estimate)" if _is_estimate(decisive.get("source")) else ""}</p>
<p>Scales — next action: {esc(str(scales.get("next_action", "")))}; \
relationship: {esc(str(scales.get("relationship", "")))}; \
face: {esc(str(scales.get("face", "")))}.</p>
</section>
<section id="alternatives">
<h2>6. If it had been said differently</h2>
{criterion_html}
<ul>
{"".join(alt_blocks)}
</ul>
{control_html}
{middle_html}
</section>
<section id="perspective">
<h2>7. One line of perspective{reference_mark}</h2>
<p>{esc(str(perspective.get("line", "")))}<br>
— {esc(str(perspective.get("work", "")))}
{p_link}{"".join(p_notes)}</p>
</section>
<section id="balance">
<h2>8. Plus and minus at the end</h2>
<p>Each speaker's balance against their own goals in this
conversation, stake by stake — not a score of the person. Plus means the
wanted effect happened; minus means it did not.</p>
{end_stake_html}
{speaker_position_html}
{balance_intro}
{balance_table}
</section>
<section id="reliability">
<h2>9. How much to trust this</h2>
<p>Pre-registered reader measurements (same in every case report, not
computed from this case). Split-half: {esc(str(reliability.get("split_half", "")))}; \
comparison partners per turn (k): {esc(str(reliability.get("k", "")))}; \
outcome R1: {esc(str(reliability.get("outcome_r1", "")))} \
(held-out 1: {esc(str(reliability.get("outcome_r1_ho1", "")))}, \
held-out 2: {esc(str(reliability.get("outcome_r1_ho2", "")))}); \
perturbation ordering: {esc(str(reliability.get("perturb", "")))} (16/20).</p>
<p>Measured on: {esc(str(reliability.get("measured_on", "")))}</p>
{judgment_html}
<ul>
{limit_items}
</ul>
</section>
<p class="notice">This report reads the utterance, not the person.</p>
</body>
</html>
"""


# --- CLI ------------------------------------------------------------------------------------

def cache_key(case_id, paths, model, max_turns=None, judgment_rep=0) -> str:
    h = hashlib.sha256()
    h.update(REPORT_VERSION.encode("utf-8"))
    h.update(b"\x00" + str(judgment_rep).encode("utf-8"))
    h.update(b"\x00" + case_id.encode("utf-8"))
    h.update(b"\x00" + (model or "").encode("utf-8"))
    h.update(b"\x00" + ("" if max_turns is None else str(max_turns)
                        ).encode("utf-8"))
    for path in paths:
        h.update(b"\x00" + str(path).encode("utf-8"))
        h.update(b"\x00" + _file_sha(Path(path)).encode("utf-8"))
    return h.hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="uptake-report")
    ap.add_argument("--case", required=True,
                    help="case id (see data/private/highstakes for available cases)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--budget", type=float, default=None,
                    help="USD budget for model calls")
    ap.add_argument("--model", default=None,
                    help="model for alternative scoring (default: offline)")
    ap.add_argument("--max-turns", type=int, default=None,
                    help="read only the first N eligible turns (small trial)")
    ap.add_argument("--judgment-rep", type=int, default=0,
                    help="repeat number of the model judgment; a new number asks again")
    ap.add_argument("--data-root", default="data/private",
                    help="data root (default: data/private)")
    args = ap.parse_args(argv)
    spend = 0.0

    def emit_spend():
        print(f"SPEND={spend:.2f}")

    out_dir = Path(args.out)
    try:
        budget = float(args.budget) if args.budget is not None else None
    except (TypeError, ValueError):
        budget = None
    meter = Meter(budget_usd=budget)
    try:
        _, _, _, paths = load_inputs(args.case, args.data_root)
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"error: {exc}")
        emit_spend()
        raise SystemExit(1)
    key = cache_key(args.case, [str(p) for p in paths], args.model or "",
                    args.max_turns, args.judgment_rep)
    cache_path = out_dir / ".uptake_cache.json"
    report_path = out_dir / "report.json"
    html_path = out_dir / "report.html"
    if cache_path.exists() and report_path.exists() and html_path.exists():
        try:
            cached = _read_json(cache_path)
        except (OSError, ValueError):
            cached = {}
        if cached.get("key") == key:
            print(f"cached {args.case}")
            emit_spend()
            return
    cache_dir = Path(args.data_root) / "goal_run" / "_model_cache"
    budget_note = None
    try:
        report = build_report(args.case, data_root=args.data_root,
                              model=args.model, meter=meter,
                              cache_dir=cache_dir,
                              max_turns=args.max_turns,
                              judgment_rep=args.judgment_rep)
        spend = meter.cost()
    except BudgetExceeded as exc:
        budget_note = f"Budget stop: {exc}"
        try:
            report = build_report(args.case, data_root=args.data_root,
                                  model=None, meter=None, cache_dir=None,
                                  budget_note=budget_note,
                                  max_turns=args.max_turns,
                                  judgment_rep=args.judgment_rep)
        except (OSError, ValueError) as exc2:
            print(f"error: {exc2}")
            emit_spend()
            raise SystemExit(1)
        spend = meter.cost()
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(render_html(report))
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump({"key": key, "spend": spend}, f)
        print(f"budget stop {args.case}")
        emit_spend()
        raise SystemExit(2)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_html(report))
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"key": key, "spend": spend}, f)
    print(f"wrote {report_path} {html_path}")
    emit_spend()


if __name__ == "__main__":
    main()
