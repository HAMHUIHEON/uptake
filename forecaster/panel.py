from .types import Receiver

STANCE_ORDER = ["neutral", "trusting", "suspicious"]
DEFAULT_MIX = {"neutral": 0.5, "trusting": 0.3, "suspicious": 0.2}

_STANCES = ["neutral", "suspicious", "trusting"]


def allocate(n: int, mix: dict[str, float]) -> dict[str, int]:
    for key in mix:
        if key not in STANCE_ORDER:
            raise ValueError(f"bad stance: {key!r}")
    try:
        total = sum(float(v) for v in mix.values())
    except (TypeError, ValueError):
        raise ValueError(f"bad stance_mix weights: {mix!r}")
    if total <= 0:
        raise ValueError(f"bad stance_mix weights: {mix!r}")
    if any(float(v) < 0 for v in mix.values()):
        raise ValueError(f"bad stance_mix weights: {mix!r}")
    quotas = {key: n * float(value) / total for key, value in mix.items()}
    floors = {key: int(q // 1) for key, q in quotas.items()}
    remainders = {key: quotas[key] - floors[key] for key in mix}
    left = n - sum(floors.values())
    order = sorted(mix.keys(), key=lambda k: (-remainders[k], STANCE_ORDER.index(k)))
    counts = dict(floors)
    for key in order[:left]:
        counts[key] += 1
    return counts


def build_panel(base: Receiver, n: int = 6, seed: int = 0,
                stance_mix: dict[str, float] | None = None) -> list[Receiver]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if stance_mix is None:
        stances = [_STANCES[i % len(_STANCES)] for i in range(n)]
    else:
        counts = allocate(n, stance_mix)
        stances = []
        for stance in STANCE_ORDER:
            stances.extend([stance] * counts.get(stance, 0))
    return [
        Receiver(
            id=f"{base.id}-{i}",
            relationship=base.relationship,
            power=base.power,
            channel=base.channel,
            state=base.state,
            stance=stance,
        )
        for i, stance in enumerate(stances)
    ]
