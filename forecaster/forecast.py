from . import score, stats
from .panel import build_panel
from .reader import DEFAULT_MODEL, read_panel
from .types import Forecast, Intent, Receiver


def run_forecast_on_panel(client, message: str, intent: Intent, panel: list[Receiver],
                          k: int = 5, seed: int = 0, model: str = DEFAULT_MODEL,
                          b: int = 2000, max_workers: int = 8) -> Forecast:
    readings = read_panel(client, message, panel, k=k, model=model,
                          max_workers=max_workers)
    rate = score.miss_rate(readings, intent)
    per = score.per_receiver_rates(readings, intent)
    triggers = score.top_triggers(readings, intent)
    values: dict[str, list[float]] = {}
    for r in readings:
        values.setdefault(r.receiver_id, []).append(
            1.0 if score.is_miss(r, intent) else 0.0)
    if not values:
        return Forecast(message, intent, readings, rate, rate, rate, per, triggers)
    _, low, high = stats.cluster_bootstrap_ci(values, b=b, seed=seed)
    return Forecast(message, intent, readings, rate, low, high, per, triggers)


def run_forecast(client, message: str, intent: Intent, base_receiver: Receiver,
                 n: int = 6, k: int = 5, seed: int = 0,
                 model: str = DEFAULT_MODEL, b: int = 2000,
                 max_workers: int = 8) -> Forecast:
    panel = build_panel(base_receiver, n=n, seed=seed)
    return run_forecast_on_panel(client, message, intent, panel, k=k, seed=seed,
                                 model=model, b=b, max_workers=max_workers)
