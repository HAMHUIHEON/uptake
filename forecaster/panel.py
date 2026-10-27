from .types import Receiver

_STANCES = ["neutral", "suspicious", "trusting"]


def build_panel(base: Receiver, n: int = 6, seed: int = 0) -> list[Receiver]:
    if n < 1:
        raise ValueError("n must be >= 1")
    return [
        Receiver(
            id=f"{base.id}-{i}",
            relationship=base.relationship,
            power=base.power,
            channel=base.channel,
            state=base.state,
            stance=_STANCES[i % len(_STANCES)],
        )
        for i in range(n)
    ]
