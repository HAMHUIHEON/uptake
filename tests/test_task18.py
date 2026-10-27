"""Contract for SPEC_task18_baselines."""
import json
import math

import pytest

from forecaster.labels import NEXT_ACTIONS
from forecaster.board import annotate_turn
from forecaster import calibrate as C
from forecaster import predict as P


def _h(cur, actual, dist=None, item="x", n=3):
    return {"cur_class": cur, "actual_class": actual, "action_dist": dist or {}, "n_readings": n, "item": item}


def test_with_context_reads_current_turn_class():
    item = {"id": "c1",
            "moves": [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
                      {"turn_index": 1, "speaker": "B", "moves": ["reject"]}],
            "hits": [{"index": 0, "actual_class": "refuse", "action_dist": {"comply": 1.0}, "n_readings": 3},
                     {"index": 5, "actual_class": "proceed", "action_dist": {"proceed": 1.0}, "n_readings": 3}]}
    out = P.with_context(item)
    from forecaster.board import actual_class
    assert out[0]["item"] == "c1" and out[0]["cur_class"] == actual_class(["propose"])
    assert out[1]["cur_class"] == "proceed"
    assert "cur_class" not in item["hits"][0]          # copies, not mutation


def test_markov_rows_and_backoff():
    hits = [_h("negotiate", "comply") for _ in range(8)] + [_h("negotiate", "refuse") for _ in range(2)]
    base = C.base_rates(hits)
    m = P.fit_markov(hits, base, beta=2.0)
    assert set(m) == set(NEXT_ACTIONS)
    for row in m.values():
        assert math.isclose(sum(row.values()), 1.0)
    assert math.isclose(m["negotiate"]["comply"], (8 + 2 * base["comply"]) / (10 + 2))
    assert m["escalate"] == pytest.approx(base)
    assert P.markov_prob(_h("negotiate", "comply"), m) == m["negotiate"]


def test_pool_normalizes_and_weights():
    u = {a: 1 / len(NEXT_ACTIONS) for a in NEXT_ACTIONS}
    peaked = {a: (0.9 if a == "comply" else 0.1 / (len(NEXT_ACTIONS) - 1)) for a in NEXT_ACTIONS}
    assert P.pool([u, peaked], [1.0, 0.0]) == pytest.approx(u)
    p = P.pool([u, peaked], [1.0, 1.0])
    assert math.isclose(sum(p.values()), 1.0) and p["comply"] == pytest.approx(0.9)


def test_fit_pool_and_linear_ignore_useless_panel():
    hits = [_h("negotiate", "comply", {"proceed": 1.0}, item=f"i{i}") for i in range(20)]
    base = C.base_rates(hits)
    m = P.fit_markov(hits, base)
    a, b = P.fit_pool(hits, m)
    assert b == 0.0 and a > 0
    assert P.fit_linear(hits, lambda h: P.markov_prob(h, m)) == 0.0


def test_compare_markov_beats_base_and_shapes():
    hits = []
    for i in range(30):
        hits.append(_h("negotiate", "comply", {"comply": 1.0}, item=f"i{i}"))
        hits.append(_h("refuse", "negotiate", {"negotiate": 1.0}, item=f"i{i}"))
    base = C.base_rates(hits)
    m = P.fit_markov(hits, base)
    models = {"base": lambda h: base, "markov": lambda h: P.markov_prob(h, m)}
    r = P.compare(hits, models, b=200, seed=0)
    assert set(r) == {"base", "markov"}
    assert r["base"]["skill_vs_base"][0] == pytest.approx(0.0)
    assert r["markov"]["skill_vs_base"][1] > 0          # CI lower > 0
    assert r["markov"]["vs_markov"][0] == pytest.approx(0.0)
    assert r["markov"]["hit_rate"] == 1.0 and r["markov"]["n"] == 60
    for k in ("log_score", "skill_vs_base", "vs_markov", "hit_rate", "n"):
        assert k in r["base"]


def _eff(rel, face):
    return {"relationship": [rel] * 3, "face": [face] * 3}


def test_z_norm_thresholds():
    effs = [[_eff(0.0, -1.0), _eff(0.0, 0.0)] * 5]
    norm = C.fit_board_norm(effs, method="z")
    assert norm["mu"] == pytest.approx(-0.5) and norm["sd"] == pytest.approx(0.5)
    t = norm["thresholds"]
    assert t == pytest.approx({"??": -1.0, "?": -0.5, "?!": -0.25, "!": 0.5, "!!": 1.0})
    flat = C.fit_board_norm([[_eff(0.0, -0.5)] * 4], method="z")
    assert flat["thresholds"] is None and annotate_turn(-3.0, norm=flat) == "="
    q = C.fit_board_norm(effs)                       # default stays quantile
    assert set(q["thresholds"]) == {"??", "?", "?!", "!", "!!"}


def test_cli_writes_predict(tmp_path):
    for i in range(6):
        moves = [{"turn_index": t, "speaker": "AB"[t % 2], "moves": ["propose"]} for t in range(6)]
        hits = [{"index": t, "actual_class": "negotiate", "action_dist": {"negotiate": 2 / 3, "proceed": 1 / 3},
                 "n_readings": 3} for t in range(5)]
        effects = [_eff(0.3, -0.6 if t % 2 else 0.0) for t in range(6)]
        (tmp_path / f"casino-{i}.json").write_text(json.dumps(
            {"id": f"casino-{i}", "genre": "negotiation", "moves": moves, "hits": hits, "effects": effects}))
    P.main(["--out", str(tmp_path), "--seeds", "0", "1"])
    d = json.loads((tmp_path / "predict.json").read_text())
    txt = json.dumps(d)
    for name in ("base", "markov", "panel_base", "panel_markov", "pool"):
        assert name in txt
    md = (tmp_path / "predict.md").read_text()
    assert "negotiation" in md and "seeds with CI lower > 0" in md
