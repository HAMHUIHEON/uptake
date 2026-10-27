import math
import random


def wilson_ci(successes: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    delta = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - delta), min(1.0, center + delta))


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = q * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def cluster_bootstrap_ci(values_by_cluster: dict[str, list[float]], b: int = 2000,
                         seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    keys = list(values_by_cluster.keys())
    pooled_all = [v for k in keys for v in values_by_cluster[k]]
    if not keys or not pooled_all:
        return (0.0, 0.0, 0.0)
    point = sum(pooled_all) / len(pooled_all)
    if len(keys) == 1:
        return (point, point, point)
    rng = random.Random(seed)
    m = len(keys)
    reps: list[float] = []
    for _ in range(b):
        pool = [v for _ in range(m) for v in values_by_cluster[rng.choice(keys)]]
        reps.append(sum(pool) / len(pool) if pool else point)
    reps.sort()
    return (point, _percentile(reps, alpha / 2), _percentile(reps, 1 - alpha / 2))


def paired_bootstrap_diff(before: dict[str, list[float]], after: dict[str, list[float]],
                          b: int = 2000, seed: int = 0,
                          alpha: float = 0.05) -> tuple[float, float, float]:
    if set(before) != set(after):
        raise ValueError("cluster keys must match")
    keys = list(before.keys())
    flat_before = [v for k in keys for v in before[k]]
    flat_after = [v for k in keys for v in after[k]]
    point = (sum(flat_after) / len(flat_after) - sum(flat_before) / len(flat_before)
             if flat_before and flat_after else 0.0)
    if len(keys) <= 1:
        return (point, point, point)
    rng = random.Random(seed)
    m = len(keys)
    reps: list[float] = []
    for _ in range(b):
        chosen = [rng.choice(keys) for _ in range(m)]
        pool_b = [v for k in chosen for v in before[k]]
        pool_a = [v for k in chosen for v in after[k]]
        if pool_b and pool_a:
            reps.append(sum(pool_a) / len(pool_a) - sum(pool_b) / len(pool_b))
        else:
            reps.append(point)
    reps.sort()
    return (point, _percentile(reps, alpha / 2), _percentile(reps, 1 - alpha / 2))
