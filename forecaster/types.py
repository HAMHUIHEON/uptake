from dataclasses import dataclass, field

from .labels import NEXT_ACTIONS, SPEECH_ACTS, TONE_MAX, TONE_MIN

_POWERS = {"sender_higher", "receiver_higher", "equal"}
_STANCES = {"trusting", "neutral", "suspicious"}


@dataclass
class Intent:
    speech_act: str
    next_action: str
    tone: int

    def __post_init__(self):
        if self.speech_act not in SPEECH_ACTS:
            raise ValueError(f"bad speech_act: {self.speech_act!r}")
        if self.next_action not in NEXT_ACTIONS:
            raise ValueError(f"bad next_action: {self.next_action!r}")
        if not isinstance(self.tone, int) or isinstance(self.tone, bool):
            raise ValueError(f"bad tone: {self.tone!r}")
        if not (TONE_MIN <= self.tone <= TONE_MAX):
            raise ValueError(f"bad tone: {self.tone!r}")


@dataclass
class Receiver:
    id: str
    relationship: str
    power: str
    channel: str
    state: str
    stance: str = "neutral"

    def __post_init__(self):
        if self.power not in _POWERS:
            raise ValueError(f"bad power: {self.power!r}")
        if self.stance not in _STANCES:
            raise ValueError(f"bad stance: {self.stance!r}")


@dataclass
class Reading:
    receiver_id: str
    sample: int
    speech_act: str
    tone: int
    next_action: str
    trigger: str
    gloss: str


@dataclass
class Forecast:
    message: str
    intent: Intent
    readings: list[Reading] = field(default_factory=list)
    miss_rate: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    per_receiver: dict[str, float] = field(default_factory=dict)
    top_triggers: list[tuple[str, int]] = field(default_factory=list)
