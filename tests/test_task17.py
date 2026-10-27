"""Contract for SPEC_task17_calibration (offline calibration + centered board)."""
import json
import math

import pytest

from forecaster.labels import NEXT_ACTIONS
from forecaster.board import position_curve, annotate_turn
from forecaster import calibrate as C

K = len(NEXT_ACTIONS)


def _hit(actual, dist, n=3):
    return {"actual_class": actual, "action_dist": dist, "n_readings": n}


def test_base_rates_smoothed_and_normalized():
    hits = [_hit("negotiate", {}), _hit("negotiate", {}), _hit("comply", {})]
    b = C.base_rates(hits)
    assert set(b) == set(NEXT_ACTIONS)
    assert math.isclose(sum(b.values()), 1.0)
    assert math.isclose(b["negotiate"], (2 + 0.5) / (3 + K * 0.5))
    assert math.isclose(b["refuse"], 0.5 / (3 + K * 0.5))


def test_panel_prob_matches_skill_smoothing():
    p = C.panel_prob(_hit("proceed", {"proceed": 2 / 3, "ask_clarification": 1 / 3}))
    assert math.isclose(p["proceed"], (2 + 0.5) / (3 + K * 0.5))
    assert math.isclose(sum(p.values()), 1.0)


def test_blend_endpoints():
    base = {a: 1 / K for a in NEXT_ACTIONS}
    pan = C.panel_prob(_hit("proceed", {"proceed": 1.0}))
    assert C.blend(pan, base, 1.0) == pytest.approx(pan)
    assert C.blend(pan, base, 0.0) == pytest.approx(base)


def test_fit_lambda_zero_when_panel_is_wrong_and_one_when_right():
    wrong = [_hit("negotiate", {"proceed": 1.0}) for _ in range(20)]
    base = C.base_rates(wrong)
    assert C.fit_lambda(wrong, base) == 0.0
    right = [_hit("negotiate", {"negotiate": 1.0}) for _ in range(10)] + \
            [_hit("comply", {"comply": 1.0}) for _ in range(10)]
    base = C.base_rates(right)
    assert C.fit_lambda(right, base) == 1.0
    assert C.log_score([], base, 0.5) == 0.0


def test_blend_skill_zero_at_lambda_zero_and_positive_when_panel_right():
    hits = [_hit("negotiate", {"negotiate": 1.0}) for _ in range(10)] + \
           [_hit("comply", {"comply": 1.0}) for _ in range(10)]
    base = C.base_rates(hits)
    s0 = C.blend_skill(hits, base, 0.0)
    assert s0["skill_mean"] == pytest.approx(0.0) and s0["n"] == 20 and s0["lam"] == 0.0
    s1 = C.blend_skill(hits, base, 1.0)
    assert s1["skill_mean"] > 0 and s1["skill_ci"][0] > 0
    assert s1["hit_rate"] == 1.0 and s1["majority_hit_rate"] == 0.5
    e = C.blend_skill([], base, 0.5)
    assert e["n"] == 0 and tuple(e["skill_ci"]) == (0.0, 0.0)


def test_split_items_deterministic_disjoint():
    ids = [f"c{i}" for i in range(10)]
    d1, t1 = C.split_items(ids, 0.5, seed=3)
    d2, t2 = C.split_items(list(reversed(ids)), 0.5, seed=3)
    assert d1 == d2 and t1 == t2
    assert len(d1) == 5 and not set(d1) & set(t1) and set(d1) | set(t1) == set(ids)


def _eff(rel, face):
    return {"relationship": [rel, rel, rel], "face": [face, face, face]}


def test_board_norm_centers_drift():
    effs = [[_eff(0.0, -0.5) for _ in range(10)], [_eff(0.0, -0.5) for _ in range(10)]]
    norm = C.fit_board_norm(effs)
    assert norm["mu"] == pytest.approx(-0.5) and norm["n"] == 20 and norm["sd"] == pytest.approx(0.0)
    assert position_curve(effs[0])[-1] == pytest.approx(-5.0)        # old behavior unchanged
    assert position_curve(effs[0], norm=norm)[-1] == pytest.approx(0.0)
    assert set(norm["thresholds"]) == {"??", "?", "?!", "!", "!!"}
    empty = C.fit_board_norm([])
    assert empty == {"mu": 0.0, "sd": 0.0, "n": 0, "thresholds": None}


def test_annotate_with_learned_thresholds():
    norm = {"mu": 0.0, "thresholds": {"??": -2.0, "?": -1.0, "?!": -0.4, "!": 0.6, "!!": 1.2}}
    got = [annotate_turn(v, norm=norm) for v in (-2.5, -1.5, -0.5, 0.0, 0.7, 1.5)]
    assert got == ["??", "?", "?!", "=", "!", "!!"]
    assert [annotate_turn(v) for v in (-2, -1, -0.5, 0, 0.8, 1.6)] == ["??", "?", "?!", "=", "!", "!!"]


def test_cli_writes_calibration(tmp_path):
    for i in range(6):
        hits = [_hit("negotiate", {"negotiate": 2 / 3, "proceed": 1 / 3}) for _ in range(5)]
        effects = [_eff(0.3, -0.6) for _ in range(5)]
        (tmp_path / f"casino-{i}.json").write_text(json.dumps(
            {"id": f"casino-{i}", "genre": "negotiation", "hits": hits, "effects": effects}))
    (tmp_path / "report.json").write_text(json.dumps({"genre_report": {}}))
    C.main(["--out", str(tmp_path), "--seed", "0"])
    cal = json.loads((tmp_path / "calibration.json").read_text())
    neg = cal["negotiation"] if "negotiation" in cal else cal["genres"]["negotiation"]
    assert len(neg["dev_ids"]) == 3 and len(neg["test_ids"]) == 3
    assert 0.0 <= neg["lam"] <= 1.0
    assert neg["norm"]["mu"] == pytest.approx(-0.3)
    md = (tmp_path / "calibration.md").read_text()
    assert "negotiation" in md
