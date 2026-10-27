"""Scoring v2: effect gap by stake, not label match."""
from __future__ import annotations

import json
from dataclasses import dataclass

from .labels import NEXT_ACTIONS, SPEECH_ACTS
from .stats import cluster_bootstrap_ci, wilson_ci
from .types import Receiver

STAKES = ["outcome", "relationship", "face", "understanding"]

REL_MIN, REL_MAX = -2, 2
FACE_MIN, FACE_MAX = 0, 2

_JUDGE_MODEL = "muse-spark-1.3"

_SYNONYMS = {
    "order": "command",
    "instruction": "command",
    "ask": "request",
    "rejection": "refusal",
    "compliment": "praise",
    "complaint": "criticism",
    "test": "probe",
    "clarify": "ask_clarification",
}


@dataclass
class Goal:
    acceptable_actions: list[str]
    relationship_floor: int = 0
    face_ceiling: int = 1
    core_meaning: str = ""

    def __post_init__(self):
        if not self.acceptable_actions:
            raise ValueError("acceptable_actions must be non-empty")
        for a in self.acceptable_actions:
            if a not in NEXT_ACTIONS:
                raise ValueError(f"bad acceptable_action: {a!r}")
        if (isinstance(self.relationship_floor, bool)
                or not isinstance(self.relationship_floor, int)
                or not (REL_MIN <= self.relationship_floor <= REL_MAX)):
            raise ValueError(f"bad relationship_floor: {self.relationship_floor!r}")
        if (isinstance(self.face_ceiling, bool)
                or not isinstance(self.face_ceiling, int)
                or not (FACE_MIN <= self.face_ceiling <= FACE_MAX)):
            raise ValueError(f"bad face_ceiling: {self.face_ceiling!r}")


@dataclass
class ReadingV2:
    receiver_id: str
    sample: int
    next_action: str
    relationship_shift: int
    face_threat: int
    understood: str
    speech_act: str = ""
    trigger: str = ""

    def __post_init__(self):
        if self.next_action not in NEXT_ACTIONS:
            raise ValueError(f"bad next_action: {self.next_action!r}")
        if (isinstance(self.relationship_shift, bool)
                or not isinstance(self.relationship_shift, int)
                or not (REL_MIN <= self.relationship_shift <= REL_MAX)):
            raise ValueError(f"bad relationship_shift: {self.relationship_shift!r}")
        if (isinstance(self.face_threat, bool)
                or not isinstance(self.face_threat, int)
                or not (FACE_MIN <= self.face_threat <= FACE_MAX)):
            raise ValueError(f"bad face_threat: {self.face_threat!r}")


def _extract_obj(raw: str) -> dict:
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    return obj


def _coerce_bounded(value, lo: int, hi: int, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return default
        rounded = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, rounded))


def parse_reading_v2(raw: str, receiver_id: str, sample: int,
                     message: str) -> ReadingV2 | None:
    try:
        data = _extract_obj(raw)
    except (ValueError, json.JSONDecodeError):
        return None
    nxt = str(data.get("next_action", "")).strip().lower()
    nxt = _SYNONYMS.get(nxt, nxt)
    if nxt not in NEXT_ACTIONS:
        return None
    understood = data.get("understood", "")
    understood = "" if understood is None else str(understood).strip()
    if not understood:
        return None
    rel = _coerce_bounded(data.get("relationship_shift", 0), REL_MIN, REL_MAX)
    face = _coerce_bounded(data.get("face_threat", 0), FACE_MIN, FACE_MAX)
    speech_act = data.get("speech_act", "")
    speech_act = "" if speech_act is None else str(speech_act)
    trigger = data.get("trigger", "")
    trigger = "" if trigger is None else str(trigger)
    if trigger not in message:
        trigger = ""
    return ReadingV2(
        receiver_id=receiver_id,
        sample=sample,
        next_action=nxt,
        relationship_shift=rel,
        face_threat=face,
        understood=understood,
        speech_act=speech_act,
        trigger=trigger,
    )


def gaps(reading: ReadingV2, goal: Goal,
         same_meaning: bool | None) -> dict[str, bool]:
    return {
        "outcome": reading.next_action not in goal.acceptable_actions,
        "relationship": reading.relationship_shift < goal.relationship_floor,
        "face": reading.face_threat > goal.face_ceiling,
        "understanding": same_meaning is False,
    }


def any_gap(g: dict[str, bool]) -> bool:
    return any(g.values())


def judge_same_meaning(client, core_meaning: str, understood: str,
                       model: str = _JUDGE_MODEL) -> bool | None:
    if not core_meaning or not str(core_meaning).strip():
        return None
    messages = [
        {"role": "system",
         "content": "Decide whether the receiver's understanding preserves "
                    "the core meaning of the sender's message. Reply with JSON "
                    'only (no other text) like {"same": true, "why": "..."}.'},
        {"role": "user",
         "content": f"Core meaning: {core_meaning}\nUnderstood: {understood}"},
    ]
    try:
        raw = client.complete(messages, model, 0.0)
    except TypeError:
        raw = client.complete(messages, model=model, temperature=0.0)
    try:
        data = _extract_obj(raw)
    except (ValueError, json.JSONDecodeError):
        return None
    same = data.get("same")
    if isinstance(same, bool):
        return same
    return None


def score_v2(readings: list[ReadingV2], goal: Goal,
             same: dict[tuple[str, int], bool | None],
             b: int = 2000, seed: int = 0) -> dict:
    out: dict = {"n_readings": len(readings)}
    for stake in STAKES + ["any"]:
        values_by_cluster: dict[str, list[float]] = {}
        for r in readings:
            g = gaps(r, goal, same.get((r.receiver_id, r.sample)))
            gap = any_gap(g) if stake == "any" else g[stake]
            values_by_cluster.setdefault(r.receiver_id, []).append(
                1.0 if gap else 0.0)
        out[stake] = cluster_bootstrap_ci(values_by_cluster, b=b, seed=seed)
    return out


def false_alarm_rate(results: list[dict]) -> tuple[float, float, float]:
    n = len(results)
    if n == 0:
        low, high = wilson_ci(0, 0)
        return (0.0, low, high)
    alarms = sum(1 for r in results if r["any"][0] > 0.5)
    rate = alarms / n
    low, high = wilson_ci(alarms, n)
    return (rate, low, high)


def build_messages_v2(message: str, receiver: Receiver) -> list[dict]:
    system = (
        "You are this receiver reading a message in context:\n"
        f"- relationship: {receiver.relationship}\n"
        f"- power: {receiver.power}\n"
        f"- channel: {receiver.channel}\n"
        f"- state: {receiver.state}\n"
        f"- stance: {receiver.stance}\n\n"
        "You do not know what the sender meant. "
        "Read the message below the way this receiver would, then reply with JSON only "
        "(no other text) with keys next_action, relationship_shift, face_threat, "
        "understood, speech_act, trigger.\n"
        f"- next_action must be one of: {', '.join(NEXT_ACTIONS)}\n"
        "- relationship_shift must be an integer from -2 (much colder) to 2 (much warmer)\n"
        "- face_threat must be an integer from 0 (none) to 2 (severe)\n"
        "- understood must be one sentence saying what the receiver thinks the message means\n"
        f"- speech_act must be one of: {', '.join(SPEECH_ACTS)}\n"
        "- trigger must be the exact phrase in the message that drove this reading"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]
