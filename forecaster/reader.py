import json
import os
import ssl
import threading
from concurrent.futures import ThreadPoolExecutor

import truststore

from .labels import NEXT_ACTIONS, SPEECH_ACTS, TONE_MAX, TONE_MIN
from .types import Reading, Receiver

DEFAULT_MODEL = "muse-spark-1.3-contributor"
BASE_URL = "https://api.meta.ai/v1"

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


def build_messages(message: str, receiver: Receiver) -> list[dict]:
    system = (
        "You are this receiver reading a message in context:\n"
        f"- relationship: {receiver.relationship}\n"
        f"- power: {receiver.power}\n"
        f"- channel: {receiver.channel}\n"
        f"- state: {receiver.state}\n"
        f"- stance: {receiver.stance}\n\n"
        "You do not know what the sender meant. "
        "Read the message below the way this receiver would, then reply with JSON only "
        "(no other text) with keys speech_act, tone, next_action, trigger, gloss.\n"
        f"- speech_act must be one of: {', '.join(SPEECH_ACTS)}\n"
        f"- next_action must be one of: {', '.join(NEXT_ACTIONS)}\n"
        f"- tone must be an integer from {TONE_MIN} (cold) to {TONE_MAX} (warm)\n"
        "- trigger must be the exact phrase in the message that drove this reading\n"
        "- gloss must be one sentence saying what the receiver thinks the sender meant"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": message},
    ]


def _extract_obj(raw: str) -> dict:
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(raw[start:])
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    return obj


def _coerce_tone(value) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        rounded = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(TONE_MIN, min(TONE_MAX, rounded))


def parse_reading(raw: str, receiver_id: str, sample: int, message: str) -> Reading | None:
    try:
        data = _extract_obj(raw)
    except (ValueError, json.JSONDecodeError):
        return None
    act = str(data.get("speech_act", "")).strip().lower()
    nxt = str(data.get("next_action", "")).strip().lower()
    act = _SYNONYMS.get(act, act)
    nxt = _SYNONYMS.get(nxt, nxt)
    if act not in SPEECH_ACTS or nxt not in NEXT_ACTIONS:
        return None
    tone = _coerce_tone(data.get("tone", 0))
    if tone is None:
        return None
    trigger = data.get("trigger", "")
    trigger = "" if trigger is None else str(trigger)
    if trigger not in message:
        trigger = ""
    gloss = data.get("gloss", "")
    gloss = "" if gloss is None else str(gloss)
    return Reading(
        receiver_id=receiver_id,
        sample=sample,
        speech_act=act,
        tone=tone,
        next_action=nxt,
        trigger=trigger,
        gloss=gloss,
    )


def _read_one(client, message, receiver, s, model, temperature) -> Reading | None:
    raw = client.complete(build_messages(message, receiver), model, temperature)
    reading = parse_reading(raw, receiver.id, s, message)
    if reading is None:
        raw = client.complete(build_messages(message, receiver), model, temperature)
        reading = parse_reading(raw, receiver.id, s, message)
    return reading


# Upper bound on how long a read waits for the previous (receiver, sample)
# slot to finish before starting its own client calls. Instant clients finish
# far below this, so their calls stay in panel/sample order (including the
# retry); slow clients exceed it, so their calls overlap across workers.
_ORDER_TIMEOUT = 0.005


def read_panel(client, message, panel, k=5, model=DEFAULT_MODEL,
               temperature=1.0, max_workers: int = 8) -> list[Reading]:
    tasks = [(receiver, s) for receiver in panel for s in range(k)]
    if not tasks:
        return []
    if max_workers is None or max_workers < 1:
        max_workers = 1
    done: list[threading.Event] = [threading.Event() for _ in tasks]

    def _run(j: int, receiver, s, prev: threading.Event | None) -> Reading | None:
        if prev is not None:
            prev.wait(timeout=_ORDER_TIMEOUT)
        try:
            return _read_one(client, message, receiver, s, model, temperature)
        finally:
            done[j].set()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_run, j, receiver, s, done[j - 1] if j else None)
                   for j, (receiver, s) in enumerate(tasks)]
        results = [f.result() for f in futures]
    return [r for r in results if r is not None]


class MuseClient:
    def __init__(self, api_key: str | None = None,
                 base_url: str = BASE_URL):
        key = api_key or os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
        from openai import OpenAI
        try:
            import httpx
        except ModuleNotFoundError:
            import httpx2 as httpx
        self.ssl_context = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        http_client = httpx.Client(verify=self.ssl_context)
        self._client = OpenAI(api_key=key, base_url=base_url, http_client=http_client)

    def complete(self, messages, model=DEFAULT_MODEL, temperature=1.0,
                 reasoning_effort=None) -> str:
        kwargs: dict = {"model": model, "messages": messages, "temperature": temperature}
        if reasoning_effort is not None:
            kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""
