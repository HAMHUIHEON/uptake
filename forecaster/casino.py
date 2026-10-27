"""CaSiNo dialogue loader mapped onto move labels."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .moves import CASINO_MAP
from .transcribe import Turn

DEAL_ACTIONS = {"Submit-Deal", "Accept-Deal", "Reject-Deal", "Walk-Away"}
SPEAKERS = {"mturk_agent_1": "A", "mturk_agent_2": "B"}


def load_casino(path, limit=None, seed=0, exclude_ids=()) -> list[dict]:
    with open(Path(path), encoding="utf-8") as f:
        data = json.load(f)
    dialogues: list[dict] = []
    for d in data:
        dialogue_id = d.get("dialogue_id")
        annotations = d.get("annotations") or []
        if not annotations:
            continue
        kept: list[Turn] = []
        for entry in d.get("chat_logs") or []:
            text = entry.get("text", "")
            if isinstance(text, str) and text.strip() in DEAL_ACTIONS:
                continue
            speaker = SPEAKERS.get(entry.get("id"), entry.get("id"))
            kept.append(Turn(speaker=speaker, text=text))
        turns: list[Turn] = []
        gold: list[set[str]] = []
        ai = 0
        n_ann = len(annotations)
        for t in kept:
            while ai < n_ann and annotations[ai][0] != t.text:
                ai += 1
            if ai >= n_ann:
                continue
            label_str = annotations[ai][1] if len(annotations[ai]) > 1 else ""
            ai += 1
            mapped: set[str] = set()
            if isinstance(label_str, str):
                for part in label_str.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if part in CASINO_MAP:
                        mapped.add(CASINO_MAP[part])
            turns.append(t)
            gold.append(mapped)
        if not turns:
            continue
        dialogues.append({"dialogue_id": dialogue_id, "turns": turns, "gold": gold})
    excluded = set(exclude_ids or ())
    if excluded:
        dialogues = [d for d in dialogues if d.get("dialogue_id") not in excluded]
    if limit is not None and len(dialogues) > limit:
        rng = random.Random(seed)
        dialogues = rng.sample(dialogues, limit)
        dialogues.sort(key=lambda x: x["dialogue_id"])
    return dialogues
