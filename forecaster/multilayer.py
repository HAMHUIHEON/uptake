"""Multilayer reading of one turn: many indicators, each read the way it can be read.

Goal document (harness/GOAL.md) is the source. Five dimensions: what is at
stake, where it broke, what kind of break, at which stage, under which
conditions. Items are scored from five-bin probabilities (mean = score,
spread = confidence). Reading before the reply never sees later turns.
A value with no support in the text stays None; None is not zero.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .board import annotate_turn
from .usage import BudgetExceeded

#: What is at stake: big class -> {item: what shows it in the text}.
STAKE_ITEMS = {
    "outcome": {
        "money": "offers, figures, acceptance, who gets which resource",
        "time": "deadlines, schedule changes, delay, urgency acted on",
        "deliverable": "whether the requested thing or action was produced",
        "commitment": "acceptance, refusal, conditions, promises",
    },
    "relationship": {
        "trust": "statements of belief or doubt, demands to verify",
        "warmth": "address, thanks, empathy, cold or curt words",
        "continuity": "whether the exchange keeps going or breaks off",
        "emotion": ("the feeling the speaker wanted to produce in the other. High only if the later turns "
                    "show that feeling; low if they show anger, hurt or another feeling the speaker did not want"),
    },
    "face": {
        "status": "titles, authority accepted or challenged",
        "autonomy": "commands, choice respected or overridden",
        "competence": "ability praised, doubted or corrected",
    },
    "understanding": {
        "facts": "corrections, repeated questions, readbacks",
        "intent": "whether the other restates the meaning correctly",
        "obligations": "whether the duty implied is accepted",
    },
}
ITEM_CLASS = {i: c for c, d in STAKE_ITEMS.items() for i in d}
ITEMS = tuple(ITEM_CLASS)

#: Where it broke. Asked only on turns that broke. Weak ones are labeled.
CAUSES = {
    "say": ("amount", "reference", "speech_act", "implicature", "irony", "ambiguity"),
    "mind": ("emotion", "face", "confirmation", "values", "history", "silence"),
    "world": ("power", "interests", "relay", "audience", "channel_timing", "culture", "stakes"),
}
#: Indicators the text alone cannot support: shown as estimates only.
WEAK = {"continuity", "competence", "irony", "ambiguity", "confirmation", "values",
        "history", "silence", "culture", "accident_vs_deliberate", "meant_to_said"}

KINDS = ("accident", "deliberate", "conflict", "aligned")
STAGES = ("meant_to_said", "said_to_received", "received_to_done")
MISREAD_TYPES = tuple(f"{a}{i}" for a, n in zip("ABCDEF", (5, 5, 3, 3, 3, 4))
                      for i in range(1, n + 1))
BIN_MID = (10.0, 30.0, 50.0, 70.0, 90.0)
PRIOR, NEXT = 3, 3
TURN_CHARS = 600
#: A stake item counts as broken when it was wanted and did not happen.
BROKEN_WANTED, BROKEN_ACTUAL = 40.0, 60.0
RECOVERY_WINDOW = 5
POINTING = re.compile(r"\b(?:this|that|it|they|them|he|she|there|here)\b|\uc774\uac70|\uadf8\uac70|\uc800\uac70|\uac70\uae30", re.I)
ASKS_WHO = re.compile(r"\b(?:who|what|which|say again|pardon|sorry\?)\b|\ub204\uad6c|\ubb50\ub77c\uace0|\ubb34\uc2a8", re.I)


def bins_to_score(value):
    """Five probabilities -> {score, spread}. Anything else -> None."""
    if not isinstance(value, (list, tuple)) or len(value) != 5:
        return None
    try:
        p = [float(x) for x in value]
    except (TypeError, ValueError):
        return None
    total = sum(p)
    if total <= 0 or any(x < 0 for x in p):
        return None
    p = [x / total for x in p]
    mean = sum(a * b for a, b in zip(p, BIN_MID))
    var = sum(a * (b - mean) ** 2 for a, b in zip(p, BIN_MID))
    return {"score": round(mean, 1), "spread": round(math.sqrt(var), 1)}


def _squash(text) -> str:
    return re.sub(r"[^0-9a-zㄱ-힣一-鿿]+", " ", str(text or "").lower()).strip()


def _cap(text) -> str:
    body = " ".join(str(text or "").split())
    return body if len(body) <= TURN_CHARS else body[:TURN_CHARS - 3] + "..."


def _line(turns, i, mark="") -> str:
    t = turns[i] if isinstance(turns[i], dict) else {}
    return f"{mark}#{i} {t.get('speaker', '?')}: {_cap(t.get('text', ''))}"


def lines_before(turns, target) -> list:
    """Context for a reading made before the reply: never a later turn."""
    lo = max(0, target - PRIOR)
    return [_line(turns, i, "TARGET " if i == target else "") for i in range(lo, target + 1)]


def lines_after(turns, target) -> list:
    lo, hi = max(0, target - PRIOR), min(len(turns), target + NEXT + 1)
    return [_line(turns, i, "TARGET " if i == target else "") for i in range(lo, hi)]


def _parse(raw):
    text = str(raw or "")
    start = text.find("{")
    if start < 0:
        return None
    try:
        data, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


BINS_NOTE = ("Give a score as five probabilities for the bins 0-20, 20-40, 40-60, 60-80, 80-100, "
             "summing to 1, spread out when unsure. Use null when the words give no support; "
             "null is not zero and is better than a guess.")
STANCES = ("neutral", "trusting", "suspicious")


def wanted_actual_prompt(cls) -> str:
    items = "\n".join(f"- {k}: {v}" for k, v in STAKE_ITEMS[cls].items())
    return ("You read one turn of a real conversation. You see earlier turns, the TARGET and up to "
            f"three later turns. Stake class: {cls}. Items and what shows them in the text:\n{items}\n"
            "For each item give:\n"
            "- wanted: how much this turn reached for this item, for the speaker.\n"
            "- actual: how far the wanted effect really happened in the later turns, counting replies, "
            "actions and the lack of a response.\n"
            "- reply_turn and quote: optional, a later turn and verbatim span if one shows it.\n"
            f"{BINS_NOTE} No wanted when the turn does not visibly reach for the item. No actual "
            "when the later turns do not bear on the item. If the target is a fragment or garbled, use null throughout. Do not "
            "judge whether the turn was right or polite, or the person. Keep quote in the original "
            "language. Reply with JSON only: "
            '{"items": {"<item>": {"wanted": [...]|null, "actual": [...]|null, "reply_turn": n|null, '
            '"quote": ""}}, "emotion_intensity": 0-100|null}.')


#: Conditions a reading made before the reply may see. Stakes weight and the
#: evidence quote can carry what happens later in the conversation.
STATIC_CONTEXT = ("relationship", "power", "channel", "translated", "culture", "audience", "stakes")


def static_context(context) -> dict:
    return {k: context[k] for k in STATIC_CONTEXT
            if isinstance(context, dict) and context.get(k) is not None}


def expected_prompt(context, stance) -> str:
    items = ", ".join(ITEMS)
    context = static_context(context)
    return ("You are the person this turn is addressed to, and you have heard only what is below; "
            "what comes next has not happened. "
            f"Conditions: {json.dumps(context, ensure_ascii=False)}. Stance: {stance}.\n"
            f"For each item ({items}) estimate how far the speaker's wish on that item is likely to "
            f"come true with you. {BINS_NOTE} Reply with JSON only: "
            '{"items": {"<item>": [...]|null}}.')


#: Asked one group at a time: a single prompt filled every layer and history did not repeat (-0.07).
CAUSE_HEAD = ("One turn of a real conversation was flagged as possibly not producing what its speaker wanted. "
              "You see earlier turns, the TARGET and later turns. ")
CAUSE_GROUPS = {
    "say": ("amount (too little or too much said), reference (who or what is meant), speech_act (request vs "
            "order vs report), implicature, irony, ambiguity (including translation)"),
    "mind": ("emotion, face, confirmation (the listener keeps the position they already held and reads the new "
             "words as support for it), values, history (an earlier exchange between these people changes how "
             "this turn reads), silence (reading no response)"),
    "world": ("power, interests, relay (passed through someone), audience, channel_timing, culture (situation "
              "only, never a nationality), stakes"),
}


def cause_group_prompt(layer) -> str:
    return (CAUSE_HEAD + "Rate only the causes in this group that the words support, 0-100 for how much each "
            f"contributed. Omit the rest; omitted is not zero.\nGroup: {CAUSE_GROUPS[layer]}.\n"
            'Reply with JSON only: {"causes": {}}.')


CAUSE_KIND_PROMPT = (
    CAUSE_HEAD + "Judge from the words only.\nkind: shares summing to 100 for accident, deliberate, conflict "
    "(understood but interests clash), aligned (it worked as wanted).\nstages: 0-100 link strength, null when "
    "unseen: meant_to_said (only if a record shows what was meant), said_to_received, received_to_done.\n"
    'Reply with JSON only: {"kind": {}, "stages": {}}.')
CAUSE_TYPES_PROMPT = (
    CAUSE_HEAD + "Name up to three misreading codes with 0-100: A1 information missing, A2 late, A3 overheard, A4 "
    "distorted by a relay, A5 forged, B1 too little or too much, B2 wrong referent, B3 irony, B4 indirect act "
    "missed, B5 ambiguity, C1 power distance, C2 face threat, C3 hidden identity, D1 high or low context, D2 "
    "values, D3 custom, E1 confirmation, E2 emotion projected, E3 silence, F1 strategic ambiguity, F2 deception, "
    "F3 staged context, F4 well-meant deceit.\n"
    'Reply with JSON only: {"types": {}}.')


CONTEXT_PROMPT = (
    "Read this whole conversation once and describe its conditions. Give each value only if the "
    "text or the given record supports it; else null. Reply with JSON only: {\"relationship\": "
    "text, \"power\": -100..100 (first speaker over the other, 0 equal), \"channel\": text, "
    "\"translated\": true|false|null, \"culture\": situation text or null, never a nationality, "
    "\"stakes_weight\": 0-100 (0 everyday, 50 reputation or money, 100 life and death), "
    "\"evidence\": one verbatim quote}.")


class Reader:
    """Cached model calls under a budget. ``rep`` separates repeat readings."""

    def __init__(self, client, model, meter=None, cache_dir=None, rep=0):
        self.client, self.model, self.meter = client, model, meter
        self.cache, self.rep = Path(cache_dir) if cache_dir else None, rep
        self.failures = 0

    def ask(self, system, user, temperature=0.7):
        key = hashlib.sha256(json.dumps([self.model, self.rep, system, user],
                                        ensure_ascii=False).encode()).hexdigest()
        path = self.cache / f"ml-{key}.json" if self.cache else None
        if path and path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        if self.meter is not None:
            self.meter.check()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        data = None
        for _ in range(2):
            data = _parse(self.client.complete(messages, model=self.model, temperature=temperature))
            if data is not None:
                break
        if data is None:
            self.failures += 1  # a failed reply is not a blank reading
        if data is not None and path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data


def code_features(turns, target) -> dict:
    """Indicators code can count. These need no model."""
    text = str((turns[target] or {}).get("text", ""))
    nxt = str((turns[target + 1] or {}).get("text", "")) if target + 1 < len(turns) else ""
    a, b = set(_squash(text).split()), set(_squash(nxt).split())
    speakers = {str((t or {}).get("speaker", "?")) for t in turns}
    return {"words": len(text.split()), "is_question": text.rstrip().endswith("?"),
            "pointing_words": len(POINTING.findall(text)), "speakers": len(speakers),
            "reply_asks_who": bool(ASKS_WHO.search(nxt)), "reply_words": len(nxt.split()),
            "reply_overlap": round(len(a & b) / len(a), 3) if a else None,
            "reply_is_question": nxt.rstrip().endswith("?")}


def read_stakes(reader, turns, target, classes=None) -> dict:
    user = "Conversation:\n" + "\n".join(lines_after(turns, target))
    out, emotion = {}, None
    for cls, members in STAKE_ITEMS.items():
        if classes is not None and cls not in classes:
            continue
        data = reader.ask(wanted_actual_prompt(cls), user) or {}
        if cls == "relationship" and isinstance(data.get("emotion_intensity"), (int, float)):
            emotion = float(data["emotion_intensity"])
        got = data.get("items") if isinstance(data.get("items"), dict) else {}
        for item in members:
            e = got.get(item) if isinstance(got.get(item), dict) else {}
            wanted, actual = bins_to_score(e.get("wanted")), bins_to_score(e.get("actual"))
            reply, quote = e.get("reply_turn"), str(e.get("quote") or "")
            ok = (isinstance(reply, int) and not isinstance(reply, bool) and target < reply <= target + NEXT and reply < len(turns)
                  and _squash(quote) and _squash(quote) in _squash(turns[reply].get("text", "")))
            if not ok:  # the quote is display evidence, not a condition on the score
                reply, quote = None, ""
            out[item] = {"class": cls, "wanted": wanted, "actual": actual,
                         "reply_turn": reply, "quote": quote}
    return {"items": out, "emotion_intensity": emotion}


def read_expected(reader, turns, target, context) -> dict:
    user = "Heard so far:\n" + "\n".join(lines_before(turns, target))
    runs = [(reader.ask(expected_prompt(context, s), user) or {}).get("items") or {} for s in STANCES]
    out = {}
    for item in ITEMS:
        scores = [bins_to_score(r.get(item)) for r in runs if isinstance(r, dict)]
        scores = [s["score"] for s in scores if s]
        out[item] = ({"score": round(sum(scores) / len(scores), 1), "n": len(scores),
                      "range": round(max(scores) - min(scores), 1)} if scores else None)
    return out


def _bounded(value):
    return float(value) if isinstance(value, (int, float)) and 0 <= value <= 100 else None


def read_cause(reader, turns, target) -> dict:
    user = "Conversation:\n" + "\n".join(lines_after(turns, target))
    out = {}
    for layer, names in CAUSES.items():
        d = reader.ask(cause_group_prompt(layer), user) or {}
        got = d.get("causes") if isinstance(d.get("causes"), dict) else {}
        out[layer] = {n: _bounded(got[n]) for n in names if n in got and _bounded(got[n]) is not None}
    d = reader.ask(CAUSE_TYPES_PROMPT, user) or {}
    types = d.get("types") if isinstance(d.get("types"), dict) else {}
    out["types"] = {k: _bounded(v) for k, v in types.items()
                    if k in MISREAD_TYPES and _bounded(v) is not None}
    d = reader.ask(CAUSE_KIND_PROMPT, user) or {}
    kind = d.get("kind") if isinstance(d.get("kind"), dict) else {}
    vals = {k: _bounded(kind.get(k)) for k in KINDS}
    total = sum(v for v in vals.values() if v is not None)
    out["kind"] = ({k: round((v or 0) * 100 / total, 1) for k, v in vals.items()} if total else None)
    stages = d.get("stages") if isinstance(d.get("stages"), dict) else {}
    out["stages"] = {k: _bounded(stages.get(k)) for k in STAGES}
    return out


def read_context(reader, turns, record=None) -> dict:
    head = [_line(turns, i) for i in range(min(len(turns), 40))]
    user = f"Record: {json.dumps(record or {}, ensure_ascii=False)}\nConversation:\n" + "\n".join(head)
    d = reader.ask(CONTEXT_PROMPT, user) or {}
    power = d.get("power")
    return {"relationship": d.get("relationship"),
            "power": float(power) if isinstance(power, (int, float)) and -100 <= power <= 100 else None,
            "channel": d.get("channel"), "translated": d.get("translated") if isinstance(d.get("translated"), bool) else None,
            "culture": d.get("culture"), "stakes_weight": _bounded(d.get("stakes_weight")),
            "evidence": d.get("evidence"), "source": "model estimate"}


def apply_stakes_record(conditions, stakes_record=None) -> dict:
    """Conditions with the weight from a valid case record, else the reading's own, and where it came from."""
    out = {**conditions, "stakes_weight_model": conditions.get("stakes_weight"),
           "stakes_source": conditions.get("source")}
    rec = stakes_record if isinstance(stakes_record, dict) else {}
    weight, basis = rec.get("weight"), rec.get("basis")
    if (not isinstance(weight, bool) and _bounded(weight) is not None
            and isinstance(basis, str) and basis.strip()):
        out["stakes_weight"], out["stakes_source"] = float(weight), f"record: {basis.strip()}"
    return out


def broken_items(items) -> list:
    out = []
    for name, e in items.items():
        w, a = e.get("wanted"), e.get("actual")
        if w and a and w["score"] >= BROKEN_WANTED and a["score"] <= BROKEN_ACTUAL:
            out.append(name)
    return out


def read_turn(reader, turns, target, context) -> dict:
    stakes = read_stakes(reader, turns, target)
    broken = broken_items(stakes["items"])
    return {"turn": target, "speaker": str((turns[target] or {}).get("speaker", "?")),
            "features": code_features(turns, target), **stakes,
            "expected": read_expected(reader, turns, target, context),
            "broken": broken,
            "cause": read_cause(reader, turns, target) if broken else None}


def read_turns(reader, turns, targets, context, workers=6) -> list:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda t: read_turn(reader, turns, t, context), targets))


def combine(reading, stakes_weight=None, use=None) -> dict:
    """Layers above the items, from the values only; a blank stays out.

    ``use`` limits the alert inputs to items that earned a measured grade."""
    classes, miss, conf = {}, 0.0, []
    for cls in STAKE_ITEMS:
        rows = [e for e in reading["items"].values() if e["class"] == cls and e["wanted"] and e["actual"]]
        total = sum(e["wanted"]["score"] for e in rows)
        classes[cls] = (round(sum(e["wanted"]["score"] * e["actual"]["score"] for e in rows) / total, 1)
                        if total else None)
    for name, e in reading["items"].items():
        if use is not None and name not in use:
            continue
        if e["wanted"] and e["actual"]:
            gap = e["wanted"]["score"] * (100 - e["actual"]["score"]) / 100
            if gap > miss:
                miss = gap
                conf = [e["wanted"]["spread"], e["actual"]["spread"]]
    confidence = round(1 - sum(conf) / (2 * 40.0), 3) if conf else None
    weight = (stakes_weight if stakes_weight is not None else 50.0) / 100
    severity = round(miss * weight * (confidence if confidence is not None else 0.5), 1)
    return {"classes": classes, "miss": round(miss, 1), "confidence": confidence, "severity": severity}


def recovery(readings) -> dict:
    """Per broken turn: did the same speaker later land the same class? From values only."""
    out = {}
    for pos, r in enumerate(readings):
        for item in r.get("broken") or []:
            cls = ITEM_CLASS[item]
            later = readings[pos + 1:pos + 1 + RECOVERY_WINDOW]
            same = [e["actual"]["score"] for x in later if x["speaker"] == r["speaker"]
                    for e in x["items"].values() if e["class"] == cls and e["actual"]]
            out[f"{r['turn']}:{item}"] = max(same) if same else None
    return out


GRADES_FILE = Path(__file__).with_name("multilayer_grades.json")
#: First-pass cuts, replaced by the development distribution in GRADES_FILE.
PROVISIONAL_CUTS = (10.0, 20.0, 30.0, 40.0)
LEVELS = ("advisory", "caution", "warning", "critical")
RESOLVE_ACTUAL = 60.0


def load_grades() -> dict:
    """Measured grades and alert cuts. Absent file: provisional, stated in the report."""
    try:
        data = json.loads(GRADES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "provisional", "items": {}, "cuts": list(PROVISIONAL_CUTS)}
    return {"status": data.get("status", "measured"), "items": data.get("items", {}),
            "causes": data.get("causes", {}), "cuts": data.get("cuts", list(PROVISIONAL_CUTS)),
            "judgment": data.get("judgment", {})}


def measured_items(grades) -> set | None:
    if grades.get("status") == "provisional":
        return {i for i in ITEMS if i not in WEAK}
    return {i for i, g in grades.get("items", {}).items() if g == "measured"}


def level_of(severity, cuts):
    level = None
    for cut, name in zip(cuts, LEVELS):
        if severity >= cut:
            level = name
    return level


def signed(e):
    """How far a wanted effect landed, -1 to +1, weighted by how much it was wanted."""
    if not (e.get("wanted") and e.get("actual")):
        return None
    return (e["actual"]["score"] - 50.0) / 50.0 * e["wanted"]["score"] / 100.0


def turn_effect(reading, use) -> dict:
    """One turn's signed effect, added over ``use`` items with both values; no such item, no value.

    Scores carry one decimal, so six places hold each cell exactly and keep float noise off a cut."""
    items = {name: round(signed(e), 6) for name, e in reading["items"].items()
             if name in use and signed(e) is not None}
    if not items:
        return {"value": None, "mark": None, "items": {}}
    value = round(sum(items.values()), 6)
    return {"value": value, "mark": annotate_turn(value), "items": items}


def build_standing_items(readings, window=3, use=None) -> dict:
    """Running standing per speaker and class, never merged across classes.

    ``use`` limits the standing to items that earned a measured grade."""
    speakers = []
    for r in readings:
        if r["speaker"] not in speakers:
            speakers.append(r["speaker"])
    end = {sp: {c: {"score": 0.0, "judged": 0, "mean": None, "items": {}} for c in STAKE_ITEMS}
           for sp in speakers}
    series = {c: {sp: [] for sp in speakers} for c in STAKE_ITEMS}
    for r in readings:
        for name, e in r["items"].items():
            v = signed(e)
            if v is None or (use is not None and name not in use):
                continue
            row = end[r["speaker"]][e["class"]]
            row["score"] += v
            row["judged"] += 1
            row["items"][name] = round(row["items"].get(name, 0.0) + v, 3)
        for c in STAKE_ITEMS:
            for sp in speakers:
                series[c][sp].append({"turn": r["turn"], "value": round(end[sp][c]["score"], 3)})
    for sp in speakers:
        for row in end[sp].values():
            row["score"] = round(row["score"], 3)
            if row["judged"]:
                row["mean"] = round(row["score"] / row["judged"], 3)
    splits = {}
    for c in STAKE_ITEMS:
        ranked = [s for s in sorted(speakers, key=lambda x: -end[x][c]["judged"]) if end[s][c]["judged"]][:2]
        best = None
        if ranked:
            def margin(k):
                a = series[c][ranked[0]][k]["value"]
                return a - series[c][ranked[1]][k]["value"] if len(ranked) > 1 else a
            n = len(readings)
            for k in range(n):
                d = margin(k) - margin(max(0, k - window))
                if d and (best is None or abs(d) > abs(best["shift"])):
                    best = {"turn": readings[k]["turn"], "shift": round(d, 3), "between": ranked}
        splits[c] = best
    return {"series": series, "end": end, "splits": splits}


#: Calm reference in points for end gaps: mean (actual - wanted) on 40 calm
#: wiki talk conversations, not a sample of well-understood conversations.
CALM_REFERENCE = {
    "deliverable": -30.481,
    "warmth": -22.482,
    "competence": -25.66,
    "facts": -28.932,
}
#: Alias kept for readability at call sites.
CALM_BASELINE = CALM_REFERENCE
END_CALM_REFERENCE = CALM_REFERENCE


def build_end_gaps(readings, use=None) -> dict:
    """End value per speaker and item, against a calm reference.

    For each speaker and each item in ``use``, the mean of
    (actual - wanted) over turns with both values, minus the calm
    reference for that item, divided by 100. An item with no values
    is absent, never zero. No merged number per stake.
    """
    if use is None:
        use = set(CALM_REFERENCE)
    else:
        use = set(use)
    speakers: list = []
    for r in readings or []:
        sp = (r or {}).get("speaker", "?")
        if sp not in speakers:
            speakers.append(sp)
    out: dict = {sp: {} for sp in speakers}
    for sp in speakers:
        for item in use:
            diffs = []
            for r in readings or []:
                if (r or {}).get("speaker") != sp:
                    continue
                e = ((r or {}).get("items") or {}).get(item) or {}
                w, a = e.get("wanted"), e.get("actual")
                if not (isinstance(w, dict) and isinstance(a, dict)):
                    continue
                ws, ac = w.get("score"), a.get("score")
                if not (isinstance(ws, (int, float)) and isinstance(ac, (int, float))):
                    continue
                if isinstance(ws, bool) or isinstance(ac, bool):
                    continue
                diffs.append(float(ac) - float(ws))
            if not diffs:
                continue
            ref = CALM_REFERENCE.get(item, 0.0)
            out[sp][item] = (sum(diffs) / len(diffs) - float(ref)) / 100.0
    return out


def build_alerts_items(readings, conditions, grades, window=RECOVERY_WINDOW) -> list:
    """One alert per broken turn; the level comes from size, stakes and confidence."""
    use, cuts, weight, alerts = measured_items(grades), grades["cuts"], (conditions or {}).get("stakes_weight"), []
    for pos, r in enumerate(readings):
        layers = combine(r, weight, use)
        r["layers"] = layers
        items = [i for i in r.get("broken") or [] if i in use]
        level = level_of(layers["severity"], cuts)
        if not items or level is None:
            continue
        top = max(items, key=lambda i: r["items"][i]["wanted"]["score"] - r["items"][i]["actual"]["score"])
        e = r["items"][top]
        reason = (f"On {top}, this turn reached for {e['wanted']['score']:.0f}/100 and the wanted "
                  f"effect landed {e['actual']['score']:.0f}/100.")
        if e.get("quote"):
            reason += f" Reply #{e['reply_turn']}: {e['quote']}."
        cls = ITEM_CLASS[top]
        later = [x for x in readings[pos + 1:pos + 1 + window] if x["speaker"] == r["speaker"]]
        resolved = any(v["class"] == cls and v["actual"] and v["actual"]["score"] >= RESOLVE_ACTUAL
                       for x in later for v in x["items"].values())
        alerts.append({"turn": r["turn"], "level": level, "axis": cls, "items": items,
                       "severity": layers["severity"], "reason": reason, "resolved": resolved})
    return alerts


def blank_reading(turns, target) -> dict:
    t = turns[target] if isinstance(turns[target], dict) else {}
    return {"turn": target, "speaker": str(t.get("speaker", "?")), "features": code_features(turns, target),
            "items": {i: {"class": ITEM_CLASS[i], "wanted": None, "actual": None, "reply_turn": None, "quote": ""}
                      for i in ITEMS},
            "emotion_intensity": None, "expected": {i: None for i in ITEMS}, "broken": [], "cause": None}


def build_multilayer(turns, client=None, model=None, meter=None, cache_dir=None, record=None,
                     max_turns=None, grades=None, stakes_record=None) -> dict:
    """Everything the report needs from the multilayer reading of a conversation.

    ``stakes_record`` applies after the reading, so no reading prompt or cache key depends on it."""
    grades = grades or load_grades()
    last = max(0, len(turns) - 1)
    targets = list(range(last if max_turns is None else min(last, max(0, int(max_turns)))))
    if client is None or not model or cache_dir is None:
        conditions = {"relationship": None, "power": None, "channel": None, "translated": None,
                      "culture": None, "stakes_weight": None, "evidence": None, "source": "not read"}
        readings = [blank_reading(turns, t) for t in targets]
    else:
        reader = Reader(client, model, meter, cache_dir)
        conditions = read_context(reader, turns, record)
        readings = read_turns(reader, turns, targets, conditions, workers=12)
    conditions = apply_stakes_record(conditions, stakes_record)
    alerts = build_alerts_items(readings, conditions, grades)
    return {"conditions": conditions, "uptake": readings, "alerts": alerts,
            "standing": build_standing_items(readings, use=measured_items(grades)),
            "grades_status": grades["status"], "grades": grades,
            "recovery": recovery(readings)}


END_STAKES = ("outcome", "relationship", "face", "understanding")


def end_read_prompt(with_goal: bool) -> str:
    goal = ("Judge the outcome stake against the goal given for this speaker, not against "
            "what they seemed to ask for in a single turn. " if with_goal else "")
    return (
        "You read a whole finished conversation for one named speaker. "
        "For each of the four stakes (outcome, relationship, face, understanding) give how "
        "the conversation left this speaker on that stake: 0 means far worse off than they "
        "wanted, 50 means no change either way, 100 means fully got what they wanted. "
        "outcome means what they were trying to get or get done; relationship means trust "
        "and warmth with the others; face means standing, autonomy and being seen as "
        "capable; understanding means whether what they meant and needed to know got across. "
        f"{goal}"
        f"{BINS_NOTE} "
        "Null when the conversation does not bear on the stake. "
        "Optional quote per stake: a verbatim span from the conversation, empty when none. "
        "Do not judge whether the speaker was right or polite, and do not judge the person. "
        "Keep quotes in the original language. "
        'Reply with JSON only: {"stakes": {"<stake>": {"score": [...]|null, "quote": ""}}}.'
    )


def read_end(reader, turns, speaker, goal=None) -> dict:
    lines = [f"Read for speaker: {speaker}"]
    if goal is not None:
        lines.append(f"Goal at the start: {goal}")
    lines.append("Conversation:")
    try:
        for t in (turns or []):
            if isinstance(t, dict):
                lines.append(f"{t.get('speaker', '?')}: {t.get('text', '')}")
            else:
                lines.append(str(t or ""))
    except Exception:
        pass
    user = "\n".join(lines)
    try:
        data = reader.ask(end_read_prompt(goal is not None), user)
    except BudgetExceeded:
        raise
    except Exception:
        data = None
    try:
        blob = _squash("\n".join(
            str(t.get("text", "")) if isinstance(t, dict) else str(t or "")
            for t in (turns or [])))
    except Exception:
        blob = ""
    got = {}
    try:
        if isinstance(data, dict):
            inner = data.get("stakes")
            if not isinstance(inner, dict):
                inner = data.get("items")
            got = inner if isinstance(inner, dict) else {}
    except Exception:
        got = {}
    stakes = {}
    for stake in END_STAKES:
        score, spread, quote = None, None, ""
        try:
            e = got.get(stake)
            bins = None
            if isinstance(e, (list, tuple)):
                bins = e
            elif isinstance(e, dict):
                bins = e.get("score", e.get("bins"))
                quote = str(e.get("quote") or "")
            parsed = bins_to_score(bins)
            if parsed:
                score, spread = parsed["score"], parsed["spread"]
            if not (_squash(quote) and _squash(quote) in blob):
                quote = ""
        except Exception:
            score, spread, quote = None, None, ""
        stakes[stake] = {"score": score, "spread": spread, "quote": quote}
    return {"speaker": speaker, "goal": goal, "stakes": stakes}


def end_read2_prompt(with_goal: bool) -> str:
    goal = ("Judge the outcome stake by comparing the settled list against the goal given "
            "for this speaker. " if with_goal else "")
    return (
        "You read a whole finished conversation for one named speaker. "
        "First list what the conversation settled, as settled: one row per thing at stake, "
        "with what this speaker ends up with and what the others end up with, in the amounts "
        "the conversation states. "
        "Use null for an amount the conversation does not state; never infer an amount from "
        "what a speaker seemed to want. "
        "An empty list is correct when nothing was settled. "
        "Then score the four stakes using that list. "
        "For each of the four stakes (outcome, relationship, face, understanding) give how "
        "the conversation left this speaker on that stake: 0 means far worse off than they "
        "wanted, 50 means no change either way, 100 means fully got what they wanted. "
        "outcome means what they were trying to get or get done; relationship means trust "
        "and warmth with the others; face means standing, autonomy and being seen as "
        "capable; understanding means whether what they meant and needed to know got across. "
        f"{goal}"
        f"{BINS_NOTE} "
        "Null when the conversation does not bear on the stake. "
        "Optional quote per stake: a verbatim span from the conversation, empty when none. "
        "Do not judge whether the speaker was right or polite, and do not judge the person. "
        "Keep quotes in the original language. "
        'Reply with JSON only: {"settled": [{"item": "", "speaker_gets": n|null, "others_get": '
        'n|null}], "stakes": {"<stake>": {"score": [...]|null, "quote": ""}}}.'
    )


def _end_amount(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def read_end2(reader, turns, speaker, goal=None) -> dict:
    lines = [f"Read for speaker: {speaker}"]
    if goal is not None:
        lines.append(f"Goal at the start: {goal}")
    lines.append("Conversation:")
    try:
        for t in (turns or []):
            if isinstance(t, dict):
                lines.append(f"{t.get('speaker', '?')}: {t.get('text', '')}")
            else:
                lines.append(str(t or ""))
    except Exception:
        pass
    user = "\n".join(lines)
    try:
        data = reader.ask(end_read2_prompt(goal is not None), user)
    except BudgetExceeded:
        raise
    except Exception:
        data = None
    try:
        blob = _squash("\n".join(
            str(t.get("text", "")) if isinstance(t, dict) else str(t or "")
            for t in (turns or [])))
    except Exception:
        blob = ""
    got = {}
    try:
        if isinstance(data, dict):
            inner = data.get("stakes")
            if not isinstance(inner, dict):
                inner = data.get("items")
            got = inner if isinstance(inner, dict) else {}
    except Exception:
        got = {}
    stakes = {}
    for stake in END_STAKES:
        score, spread, quote = None, None, ""
        try:
            e = got.get(stake)
            bins = None
            if isinstance(e, (list, tuple)):
                bins = e
            elif isinstance(e, dict):
                bins = e.get("score", e.get("bins"))
                quote = str(e.get("quote") or "")
            parsed = bins_to_score(bins)
            if parsed:
                score, spread = parsed["score"], parsed["spread"]
            if not (_squash(quote) and _squash(quote) in blob):
                quote = ""
        except Exception:
            score, spread, quote = None, None, ""
        stakes[stake] = {"score": score, "spread": spread, "quote": quote}
    settled = []
    try:
        raw = data.get("settled") if isinstance(data, dict) else None
        if isinstance(raw, list):
            for row in raw:
                try:
                    if not isinstance(row, dict):
                        continue
                    item = row.get("item")
                    if not isinstance(item, str):
                        continue
                    settled.append({"item": item,
                                    "speaker_gets": _end_amount(row.get("speaker_gets")),
                                    "others_get": _end_amount(row.get("others_get"))})
                except Exception:
                    continue
    except Exception:
        settled = []
    return {"speaker": speaker, "goal": goal, "stakes": stakes, "settled": settled}
