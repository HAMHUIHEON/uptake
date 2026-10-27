import pytest
from forecaster.panel import build_panel


def test_panel_cycles_stance_and_keeps_context(base_receiver):
    p = build_panel(base_receiver, n=5)
    assert [r.id for r in p] == ["r-0", "r-1", "r-2", "r-3", "r-4"]
    assert [r.stance for r in p] == ["neutral", "suspicious", "trusting", "neutral", "suspicious"]
    assert all(r.state == base_receiver.state and r.power == "sender_higher" for r in p)


def test_panel_deterministic_and_validates(base_receiver):
    assert build_panel(base_receiver, n=3, seed=1) == build_panel(base_receiver, n=3, seed=1)
    with pytest.raises(ValueError):
        build_panel(base_receiver, n=0)
