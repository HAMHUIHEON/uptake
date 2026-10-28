"""Contract for SPEC_task20_pivot_check (synthetic data only)."""
import json
import random

import pytest

from forecaster import board as B
from forecaster import outcome as O
from forecaster.batch import spearman


def _item(cid, effects, genre="negotiation"):
    return {"id": cid, "genre": genre, "moves": [], "hits": [],
            "effects": effects, "gold": None}


def _eff(idx, speaker, value):
    return {"index": idx, "speaker": speaker, "value": value}


def _toy_items():
    # Pivot = largest |value| from index 2; mu=0 keeps signs obvious.
    return {
        "casino-1": _item("casino-1", [
            _eff(0, "A", 0.1), _eff(1, "B", 0.1),
            _eff(2, "A", 2.0), _eff(3, "B", -0.5),
        ]),
        "casino-2": _item("casino-2", [
            _eff(0, "A", 0.1), _eff(1, "B", 0.1),
            _eff(2, "B", -2.0), _eff(3, "A", 0.5),
        ]),
        "casino-3": _item("casino-3", [
            _eff(0, "A", -0.2), _eff(1, "B", 0.2),
            _eff(2, "A", 1.5), _eff(3, "B", 0.4),
        ]),
        "casino-4": _item("casino-4", [
            _eff(0, "A", 0.2), _eff(1, "B", -0.2),
            _eff(2, "B", -1.5), _eff(3, "A", -0.4),
        ]),
    }


def _toy_outcomes():
    # likeness = report BY side ABOUT opponent; pivot sign matches likeness gap.
    def rec(like_a, like_b):
        return {"A": {"points": 10, "satisfaction": 0, "likeness": like_a},
                "B": {"points": 10, "satisfaction": 0, "likeness": like_b}}

    return {1: rec(-1, 2), 2: rec(-2, 1), 3: rec(0, 1), 4: rec(-1, 2)}


def _fit_items():
    return {
        "casino-1": _item("casino-1", [_eff(0, "A", 0.5), _eff(1, "B", -0.5)]),
        "casino-2": _item("casino-2", [_eff(0, "A", -0.5), _eff(1, "B", 0.5)]),
    }


def test_pivot_rows_fields():
    rows = O.pivot_rows(_toy_items(), _toy_outcomes(), mu=0.0)
    assert [(r["dialogue"], r["side"]) for r in rows] == [
        (1, "A"), (2, "B"), (3, "A"), (4, "B")]
    by_d = {r["dialogue"]: r for r in rows}
    assert by_d[1]["pivot"] == pytest.approx(2.0)
    assert by_d[1]["opp_likeness"] == 2
    assert by_d[1]["own_likeness_gap"] == 3
    assert by_d[2]["pivot"] == pytest.approx(-2.0)
    assert by_d[2]["opp_likeness"] == -2
    assert by_d[2]["own_likeness_gap"] == -3
    assert by_d[3]["pivot"] == pytest.approx(1.5)
    assert by_d[4]["pivot"] == pytest.approx(-1.5)
    # mu centering applies to the pivot value
    rows_mu = O.pivot_rows(_toy_items(), _toy_outcomes(), mu=0.5)
    assert rows_mu[0]["pivot"] == pytest.approx(1.5)
    # list-form items work too
    assert len(O.pivot_rows(list(_toy_items().values()),
                            _toy_outcomes(), 0.0)) == 4


def test_pivot_rows_skips():
    items = dict(_toy_items())
    items["casino-5"] = _item("casino-5", [
        _eff(0, "A", 0.0), _eff(1, "B", 0.0), _eff(2, "A", 0.0)])  # no turning point
    items["casino-6"] = _item("casino-6", [
        _eff(0, "A", 0.1), _eff(1, "B", 0.1), _eff(2, "A", 3.0)])  # no outcomes
    items["yt-1"] = _item("yt-1", [
        _eff(0, "A", 0.1), _eff(1, "B", 0.1), _eff(2, "A", 3.0)])
    rows = O.pivot_rows(items, _toy_outcomes(), mu=0.0)
    assert sorted(r["dialogue"] for r in rows) == [1, 2, 3, 4]


def test_pivot_direction_share_one():
    rows = O.pivot_rows(_toy_items(), _toy_outcomes(), mu=0.0)
    d = O.pivot_direction(rows)
    assert d["agree"] == 4
    assert d["n"] == 4
    assert d["share"] == pytest.approx(1.0)
    assert len(d["ci"]) == 2
    assert d["ci"][0] <= 1.0
    assert d["ci"][1] == pytest.approx(1.0)


def test_pivot_direction_edges():
    assert O.pivot_direction([]) == {"agree": 0, "n": 0, "share": 0.0,
                                     "ci": (0.0, 1.0)}
    rows = [
        {"pivot": 0.0, "own_likeness_gap": 2},   # zero pivot excluded
        {"pivot": 1.0, "own_likeness_gap": 0},   # zero gap excluded
        {"pivot": 1.0, "own_likeness_gap": -1},  # disagree
        {"pivot": -1.0, "own_likeness_gap": -2},  # agree
    ]
    d = O.pivot_direction(rows)
    assert d["agree"] == 1
    assert d["n"] == 2
    assert d["share"] == pytest.approx(0.5)


def test_random_turn_rho_deterministic():
    items, oc = _toy_items(), _toy_outcomes()
    a = O.random_turn_rho(items, oc, 0.0, draws=200, seed=0)
    b = O.random_turn_rho(items, oc, 0.0, draws=200, seed=0)
    assert a == b
    assert isinstance(a, float)
    assert -1.0 <= a <= 1.0
    assert O.random_turn_rho({}, oc, 0.0) == 0.0
    assert O.random_turn_rho(items, {}, 0.0) == 0.0
    assert O.random_turn_rho(items, oc, 0.0, draws=0) == 0.0


def test_random_turn_rho_matches_manual_draws():
    items, oc = _toy_items(), _toy_outcomes()
    pools, opps = [], []
    for cid in sorted(items):
        eff = items[cid]["effects"]
        t = B.turning_point(eff)
        piv = next(e for e in eff if e.get("index") == t)
        x = piv["speaker"]
        pools.append([e for e in eff if e["speaker"] == x])
        opp = "B" if x == "A" else "A"
        opps.append(oc[int(cid.split("-")[1])][opp]["likeness"])
    rng = random.Random(0)
    rhos = []
    for _ in range(20):
        xs = [B._value_of(rng.choice(p)) for p in pools]
        rhos.append(spearman(xs, list(opps)))
    assert O.random_turn_rho(items, oc, 0.0, draws=20, seed=0) == pytest.approx(
        sum(rhos) / len(rhos))


def test_analyze_has_d1_d2():
    fit, ev, oc = _fit_items(), _toy_items(), _toy_outcomes()
    res = O.analyze(fit, ev, oc, b=200, seed=0)
    assert res["mu"] == pytest.approx(0.0)
    assert res["n_pivot"] == 4
    assert len(res["D1_pivot"]) == 3
    assert res["D1_pivot"][0] == pytest.approx(1.0)
    assert res["D1_random_turn_mean"] == pytest.approx(
        O.random_turn_rho(ev, oc, res["mu"]))
    assert res["D2"]["share"] == pytest.approx(1.0)
    assert res["D2"]["n"] == 4
    assert len(res["D2"]["ci"]) == 2
    for key in ("D1_pivot", "D1_random_turn_mean", "D2"):
        assert res["verdicts"][key] == "REPORTED"


def test_outcome_cli_writes_pivot_rows(tmp_path):
    fit_d, ev_d = tmp_path / "fit", tmp_path / "ev"
    for d, items in ((fit_d, _fit_items()), (ev_d, _toy_items())):
        d.mkdir(parents=True, exist_ok=True)
        for cid, item in items.items():
            (d / f"{cid}.json").write_text(json.dumps(item))
    likes = {-2: "Extremely dislike", -1: "Slightly dislike", 0: "Undecided",
             1: "Slightly like", 2: "Extremely like"}
    data = []
    for did, rec in _toy_outcomes().items():
        data.append({
            "dialogue_id": did,
            "participant_info": {
                "mturk_agent_1": {"outcomes": {
                    "points_scored": 10, "satisfaction": "Undecided",
                    "opponent_likeness": likes[rec["A"]["likeness"]]}},
                "mturk_agent_2": {"outcomes": {
                    "points_scored": 10, "satisfaction": "Undecided",
                    "opponent_likeness": likes[rec["B"]["likeness"]]}},
            },
        })
    cp = tmp_path / "casino.json"
    cp.write_text(json.dumps(data))
    O.main(["--fit", str(fit_d), "--eval", str(ev_d),
            "--casino", str(cp), "--b", "50"])
    payload = json.loads((ev_d / "outcome.json").read_text())
    assert "D1_pivot" in payload
    assert "D1_random_turn_mean" in payload
    assert "D2" in payload
    md = (ev_d / "outcome.md").read_text()
    assert "D1_pivot" in md
    assert "D1_random_turn_mean" in md
    assert "D2" in md
