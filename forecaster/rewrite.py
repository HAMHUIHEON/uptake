from . import score
from .forecast import run_forecast_on_panel
from .panel import build_panel
from .reader import DEFAULT_MODEL
from .stats import paired_bootstrap_diff
from .types import Forecast, Intent, Receiver


def _values_by_receiver(readings, intent) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    for r in readings:
        values.setdefault(r.receiver_id, []).append(
            1.0 if score.is_miss(r, intent) else 0.0)
    return values


def run_rewrite(client, original: str, rewrite: str, intent: Intent,
                base_receiver: Receiver, n: int = 6, k: int = 5, seed: int = 0,
                b: int = 2000, model: str = DEFAULT_MODEL,
                max_workers: int = 8) -> dict:
    panel = build_panel(base_receiver, n=n, seed=seed)
    before: Forecast = run_forecast_on_panel(client, original, intent, panel,
                                             k=k, seed=seed, model=model, b=b,
                                             max_workers=max_workers)
    after: Forecast = run_forecast_on_panel(client, rewrite, intent, panel,
                                            k=k, seed=seed, model=model, b=b,
                                            max_workers=max_workers)
    before_values = _values_by_receiver(before.readings, intent)
    after_values = _values_by_receiver(after.readings, intent)
    keep = [rid for rid in before_values if rid in after_values]
    diff = paired_bootstrap_diff(
        {rid: before_values[rid] for rid in keep},
        {rid: after_values[rid] for rid in keep},
        b=b, seed=seed,
    )
    return {"panel": panel, "before": before, "after": after, "diff": diff}
