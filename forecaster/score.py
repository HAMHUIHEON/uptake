from collections import Counter

from .types import Intent, Reading


def is_miss(reading: Reading, intent: Intent) -> bool:
    return (
        reading.speech_act != intent.speech_act
        or reading.next_action != intent.next_action
        or abs(reading.tone - intent.tone) >= 2
    )


def miss_rate(readings: list[Reading], intent: Intent) -> float:
    if not readings:
        return 0.0
    return sum(1.0 for r in readings if is_miss(r, intent)) / len(readings)


def per_receiver_rates(readings: list[Reading], intent: Intent) -> dict[str, float]:
    by_receiver: dict[str, list[Reading]] = {}
    for r in readings:
        by_receiver.setdefault(r.receiver_id, []).append(r)
    return {rid: miss_rate(rs, intent) for rid, rs in by_receiver.items()}


def top_triggers(readings: list[Reading], intent: Intent, n: int = 3) -> list[tuple[str, int]]:
    counts = Counter(
        r.trigger for r in readings if is_miss(r, intent) and r.trigger)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return ranked[:n]
