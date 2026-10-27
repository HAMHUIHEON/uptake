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


class BudgetExceeded(RuntimeError):
    pass


class Meter:
    def __init__(self, budget_usd: float | None = None):
        self._budget = budget_usd
        self._lock = threading.Lock()
        self._rows: dict[str, dict] = {}

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

    def cost(self) -> float:
        with self._lock:
            return sum(row["usd"] for row in self._rows.values())

    def by_model(self) -> dict:
        with self._lock:
            return {model: dict(row) for model, row in self._rows.items()}

    def check(self) -> None:
        if self._budget is not None and self.cost() >= self._budget:
            raise BudgetExceeded(
                f"budget ${self._budget:.2f} exceeded (spent ${self.cost():.4f})")
