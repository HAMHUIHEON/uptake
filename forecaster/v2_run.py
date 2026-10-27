"""End-to-end v2 pipeline: panel reads, meaning judge, effect-gap score."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .panel import build_panel
from .reader import DEFAULT_MODEL
from .types import Receiver
from .v2 import (
    Goal,
    ReadingV2,
    build_messages_v2,
    judge_same_meaning,
    parse_reading_v2,
    score_v2,
)


def annotate(score: dict) -> str:
    p = score["any"][0]
    if p < 0.15:
        return "!"
    if p < 0.35:
        return "="
    if p < 0.60:
        return "?!"
    if p < 0.85:
        return "?"
    return "??"


def load_case_v2(path: str | Path) -> tuple[str, Goal, Receiver, dict | None]:
    with open(path, encoding="utf-8") as f:
        case = json.load(f)
    receiver_data = dict(case["receiver"])
    stance_mix = receiver_data.pop("stance_mix", None)
    return (
        case["message"],
        Goal(**case["goal"]),
        Receiver(**receiver_data),
        stance_mix,
    )


def _read_one_v2(client, message, receiver, sample, model,
                 subs=()) -> ReadingV2 | None:
    raw = client.complete(build_messages_v2(message, receiver, subs=subs), model, 1.0)
    reading = parse_reading_v2(raw, receiver.id, sample, message)
    if reading is None:
        raw = client.complete(build_messages_v2(message, receiver, subs=subs), model, 1.0)
        reading = parse_reading_v2(raw, receiver.id, sample, message)
    return reading


def run_forecast_v2(client, message: str, goal: Goal,
                    base_receiver: Receiver, n: int = 6, k: int = 5,
                    seed: int = 0, model: str = DEFAULT_MODEL,
                    judge_model: str = "muse-spark-1.3", b: int = 2000,
                    max_workers: int = 8,
                    stance_mix: dict | None = None) -> dict:
    panel = build_panel(base_receiver, n=n, seed=seed, stance_mix=stance_mix)
    if max_workers is None or max_workers < 1:
        max_workers = 1
    subs = goal.sub_stakes
    tasks = [(receiver, s) for receiver in panel for s in range(k)]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_read_one_v2, client, message, receiver, s, model, subs)
                   for receiver, s in tasks]
        results = [f.result() for f in futures]
    readings = [r for r in results if r is not None]

    def _judge(reading: ReadingV2):
        return judge_same_meaning(client, goal.core_meaning,
                                  reading.understood, model=judge_model)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        judged = list(pool.map(_judge, readings))
    same = {(r.receiver_id, r.sample): v for r, v in zip(readings, judged)}

    score = score_v2(readings, goal, same, b=b, seed=seed)
    return {
        "panel": panel,
        "readings": readings,
        "same": same,
        "score": score,
        "annotation": annotate(score),
    }
