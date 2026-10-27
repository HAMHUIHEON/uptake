import pytest
from forecaster.stats import wilson_ci, cluster_bootstrap_ci, paired_bootstrap_diff


@pytest.mark.parametrize("s,n,lo,hi", [
    (7, 10, 0.39677814534597083, 0.892208733443606),
    (0, 10, 0.0, 0.2775328030260577),
    (45, 100, 0.3561453790959721, 0.5475539707809277),
])
def test_wilson(s, n, lo, hi):
    a, b = wilson_ci(s, n)
    assert a == pytest.approx(lo, abs=1e-6) and b == pytest.approx(hi, abs=1e-6)
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_cluster_bootstrap_basic_properties():
    data = {"a": [1, 1, 1, 1], "b": [0, 0, 0, 0], "c": [1, 0, 1, 0], "d": [1, 1, 0, 1]}
    p, lo, hi = cluster_bootstrap_ci(data, b=2000, seed=7)
    assert p == pytest.approx(9 / 16)
    assert lo <= p <= hi and 0.0 <= lo and hi <= 1.0
    assert cluster_bootstrap_ci(data, b=2000, seed=7) == (p, lo, hi)
    assert cluster_bootstrap_ci({"only": [1, 0, 1]}, seed=1) == pytest.approx((2 / 3, 2 / 3, 2 / 3))


def test_cluster_ci_wider_than_naive_when_clusters_disagree():
    data = {f"r{i}": ([1] * 10 if i % 2 else [0] * 10) for i in range(6)}
    p, lo, hi = cluster_bootstrap_ci(data, b=3000, seed=3)
    wl, wh = wilson_ci(30, 60)
    assert (hi - lo) > (wh - wl)


def test_paired_diff():
    before = {"a": [1, 1], "b": [1, 0], "c": [1, 1]}
    after = {"a": [0, 0], "b": [0, 0], "c": [1, 0]}
    p, lo, hi = paired_bootstrap_diff(before, after, b=2000, seed=0)
    assert p == pytest.approx(1 / 6 - 5 / 6)
    assert lo <= p <= hi and hi < 0
    with pytest.raises(ValueError):
        paired_bootstrap_diff(before, {"a": [0]}, seed=0)
