"""Usage metering for metered Muse API calls."""
from __future__ import annotations

import threading

PRICES = {
    "muse-spark-1.3": (1.25, 4.25),
    "muse-spark-1.2": (1.25, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.20),
    "muse-spark-1.2-contributor": (0.10, 0.20),
}

_FALLBACK = (1.25, 4.25)

# Audio-input price in USD per hour of audio, by model (docs rate card).
AUDIO_USD_PER_HOUR = {
    "muse-spark-1.2": 1.50,
    "muse-spark-1.3": 1.50,
    "muse-spark-1.2-contributor": 0.15,
    "muse-spark-1.3-contributor": 0.15,
}

_AUDIO_FALLBACK = 1.50


class BudgetExceeded(RuntimeError):
    pass


class Meter:
    def __init__(self, budget_usd: float | None = None):
        self._budget = budget_usd
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = {}
        self._audio_seconds = 0.0
        self._audio_usd = 0.0

    def add(self, model, prompt_tokens, completion_tokens) -> None:
        price_in, price_out = PRICES.get(model, _FALLBACK)
        usd = prompt_tokens / 1_000_000 * price_in + completion_tokens / 1_000_000 * price_out
        with self._lock:
            row = self._rows.get(model)
            if row is None:
                row = {"calls": 0, "prompt_tokens": 0,
                       "completion_tokens": 0, "usd": 0.0}
                self._rows[model] = row
            row["calls"] += 1
            row["prompt_tokens"] += prompt_tokens
            row["completion_tokens"] += completion_tokens
            row["usd"] += usd

    def add_audio(self, model, seconds: float) -> None:
        """Book `seconds` of audio input on `model` (priced per hour)."""
        rate = AUDIO_USD_PER_HOUR.get(model, _AUDIO_FALLBACK)
        with self._lock:
            self._audio_seconds += max(0.0, float(seconds))
            self._audio_usd += max(0.0, float(seconds)) / 3600.0 * rate

    def audio_seconds(self) -> float:
        with self._lock:
            return self._audio_seconds

    def audio_cost(self) -> float:
        with self._lock:
            return self._audio_usd

    def cost(self) -> float:
        with self._lock:
            return (sum(row["usd"] for row in self._rows.values())
                    + self._audio_usd)

    def by_model(self) -> dict:
        with self._lock:
            return {model: dict(row) for model, row in self._rows.items()}

    def check(self) -> None:
        if self._budget is not None and self.cost() >= self._budget:
            raise BudgetExceeded(
                f"budget ${self._budget:.2f} exceeded (spent ${self.cost():.4f})")
