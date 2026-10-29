"""Contract for task25 value model prototype (synthetic data only)."""
import json

import pytest

from forecaster import valuemodel as V


def _item(cid, moves, effects, genre="negotiation"):
    return {"id": cid, "genre": genre, "moves": moves, "hits": [],
            "effects": effects, "gold": None}


def test_features_constant():
    assert V.FEATURES == ["rec_self", "rec_other", "tagv_self", "tagv_other", "n_self"]


def test_ridge_recovers_line():
    X = [{"x": float(i)} for i in range(-5, 6)]
    y = [2.0 * float(i) + 1.0 for i in range(-5, 6)]
    m = V.ridge_fit(X, y, lam=1e-8)
    assert m["features"] == ["x"]
    assert m["means"][0] == pytest.approx(0.0)
    assert V.ridge_predict(m, {"x": 10.0}) == pytest.approx(21.0, abs=1e-3)
    assert V.ridge_predict(m, {"x": -7.0}) == pytest.approx(-13.0, abs=1e-3)


def test_ridge_list_form_and_zero_sd():
    Xl = [[float(i)] for i in range(-5, 6)]
    y = [2.0 * float(i) + 1.0 for i in range(-5, 6)]
    m = V.ridge_fit(Xl, y, lam=1e-8)
    assert V.ridge_predict(m, {"x0": 10.0}) == pytest.approx(21.0, abs=1e-3)
    assert V.ridge_predict(m, [10.0]) == pytest.approx(21.0, abs=1e-3)
    # constant column: sd 0 -> 1, no crash
    Xc = [{"x": 1.0}, {"x": 1.0}, {"x": 1.0}]
    mc = V.ridge_fit(Xc, [2.0, 2.0, 2.0], lam=1.0)
    assert mc["sds"] == [1.0]
    assert V.ridge_predict(mc, {"x": 1.0}) == pytest.approx(2.0, abs=1e-6)


def test_ridge_features_form():
    Xf = [{"rec_self": float(i), "rec_other": 0.0, "tagv_self": 0.0,
           "tagv_other": 0.0, "n_self": 1} for i in range(6)]
    yf = [2.0 * float(i) + 1.0 for i in range(6)]
    m = V.ridge_fit(Xf, yf, lam=1e-8)
    assert m["features"] == V.FEATURES
    assert len(m["w"]) == 5
    q = {"rec_self": 10.0, "rec_other": 0.0, "tagv_self": 0.0,
         "tagv_other": 0.0, "n_self": 1}
    assert V.ridge_predict(m, q) == pytest.approx(21.0, abs=1e-2)


def _prefix_item():
    effects = [
        {"index": 0, "speaker": "A", "value": 1.0},
        {"index": 1, "speaker": "B", "value": 2.0},
        {"index": 2, "speaker": "A", "value": 100.0},
        {"index": 3, "speaker": "B", "value": -100.0},
    ]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["accept"]},
        {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
        {"turn_index": 2, "speaker": "A", "moves": ["pressure"]},
        {"turn_index": 3, "speaker": "B", "moves": ["accept"]},
    ]
    return _item("casino-1", moves, effects)


def test_prefix_features_values_and_future_ignored():
    table = {"accept": 1.0, "reject": 2.0, "pressure": 50.0}
    item = _prefix_item()
    got = V.prefix_features(item, 1, "A", 0.5, table, featureset="legacy")
    assert got["rec_self"] == pytest.approx(0.5)
    assert got["rec_other"] == pytest.approx(1.5)
    assert got["tagv_self"] == pytest.approx(1.0)
    assert got["tagv_other"] == pytest.approx(2.0)
    assert got["n_self"] == 1
    gotB = V.prefix_features(item, 1, "B", 0.5, table, featureset="legacy")
    assert gotB["rec_self"] == pytest.approx(1.5)
    assert gotB["rec_other"] == pytest.approx(0.5)
    assert gotB["n_self"] == 1
    # truncating future turns changes nothing
    trunc = _item("casino-1", item["moves"][:2], item["effects"][:2])
    assert V.prefix_features(trunc, 1, "A", 0.5, table, featureset="legacy") == pytest.approx(got)
    # full prefix sees the future
    full = V.prefix_features(item, 3, "A", 0.5, table, featureset="legacy")
    assert full["rec_self"] == pytest.approx((0.5 + 99.5) / 2)
    assert full["n_self"] == 2


def test_prefix_features_none_gives_zero():
    table = {"accept": 1.0}
    item = _prefix_item()
    got = V.prefix_features(item, 0, "B", 0.5, table, featureset="legacy")
    assert got["rec_self"] == pytest.approx(0.0)
    assert got["tagv_self"] == pytest.approx(0.0)
    assert got["n_self"] == 0
    assert got["rec_other"] == pytest.approx(0.5)
    assert got["tagv_other"] == pytest.approx(1.0)
    # empty moves / unknown move -> 0
    item2 = _item("casino-2",
                  [{"turn_index": 0, "speaker": "A", "moves": []},
                   {"turn_index": 1, "speaker": "A", "moves": ["nope"]}],
                  [{"index": 0, "speaker": "A", "value": 1.0}])
    g2 = V.prefix_features(item2, 5, "A", 0.0, table, featureset="legacy")
    assert g2["tagv_self"] == pytest.approx(0.0)


def test_value_curves_length():
    table = {"accept": 1.0, "reject": -1.0}
    item = _prefix_item()
    model = {"w": [0.0] * 5, "b": 0.5, "means": [0.0] * 5,
             "sds": [1.0] * 5, "lam": 1.0, "features": list(V.FEATURES)}
    curves = V.value_curves(item, model, 0.0, table)
    assert len(curves["A"]) == 4
    assert len(curves["B"]) == 4
    assert all(v == pytest.approx(0.5) for v in curves["A"])
    empty = V.value_curves(_item("casino-9", [], []), model, 0.0, table)
    assert empty == {"A": [], "B": []}


def test_value_pivot_finds_planted_jump():
    effects = [
        {"index": 0, "speaker": "A", "value": 0.0},
        {"index": 1, "speaker": "B", "value": 0.0},
        {"index": 2, "speaker": "A", "value": 0.0},
        {"index": 3, "speaker": "B", "value": 0.0},
    ]
    item = _item("casino-1", [], effects)
    curves = {"A": [0.0, 0.1, 10.0, 10.1], "B": [0.0, 0.2, 0.3, 0.4]}
    piv = V.value_pivot(curves, item)
    assert piv["index"] == 2
    assert piv["speaker"] == "A"
    assert piv["delta"] == pytest.approx(9.9)
    assert V.value_pivot({"A": [], "B": []}, _item("x", [], [])) is None


def test_value_pivot_baseline_uses_model():
    item = _item("casino-1", [], [{"index": 0, "speaker": "A", "value": 1.0}])
    curves = {"A": [3.0], "B": [0.0]}
    p0 = V.value_pivot(curves, item)
    assert p0["delta"] == pytest.approx(3.0)
    model = {"w": [0.0] * 5, "b": 1.0, "means": [0.0] * 5,
             "sds": [1.0] * 5, "lam": 1.0, "features": list(V.FEATURES)}
    p1 = V.value_pivot(curves, item, model)
    assert p1["delta"] == pytest.approx(2.0)


def _toy_dialogues():
    items = {}
    outcomes = {}
    for i in (1, 2, 3, 4):
        eff = [{"index": 0, "speaker": "A", "value": float(i)},
               {"index": 1, "speaker": "B", "value": float(-i)},
               {"index": 2, "speaker": "A", "value": float(i)},
               {"index": 3, "speaker": "B", "value": float(-i)}]
        mv = [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
              {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
              {"turn_index": 2, "speaker": "A", "moves": ["accept"]},
              {"turn_index": 3, "speaker": "B", "moves": ["reject"]}]
        items[f"casino-{i}"] = _item(f"casino-{i}", mv, eff)
        outcomes[i] = {"A": {"points": 10, "satisfaction": 0, "likeness": -i},
                       "B": {"points": 10, "satisfaction": 0, "likeness": i}}
    return items, outcomes


def test_fit_value_and_evaluate_keys():
    items, oc = _toy_dialogues()
    table = {"accept": 1.0, "reject": -1.0}
    model = V.fit_value(items, oc, 0.0, table, lam=1.0, featureset="legacy")
    assert model["features"] == V.FEATURES
    assert len(model["w"]) == 5
    res = V.evaluate(model, items, oc, 0.0, table, b=50, seed=0)
    for key in ("V1", "V2", "V3", "V1_minus_C1"):
        assert key in res
        assert len(res[key]) == 3
    assert res["n"] == 8
    assert res["n_pivot"] == 4
    assert "w" in res and "b" in res


def test_cli_writes_value_files(tmp_path):
    items, oc = _toy_dialogues()
    fit_d, ev_d = tmp_path / "fit", tmp_path / "ev"
    fit_d.mkdir(parents=True, exist_ok=True)
    ev_d.mkdir(parents=True, exist_ok=True)
    for cid, it in items.items():
        (fit_d / f"{cid}.json").write_text(json.dumps(it))
        (ev_d / f"{cid}.json").write_text(json.dumps(it))
    likes = {-2: "Extremely dislike", -1: "Slightly dislike", 0: "Undecided",
             1: "Slightly like", 2: "Extremely like"}
    data = []
    for did, rec in oc.items():
        # clamp toy likeness into Likert range for the file loader
        la = max(-2, min(2, rec["A"]["likeness"]))
        lb = max(-2, min(2, rec["B"]["likeness"]))
        data.append({
            "dialogue_id": did,
            "participant_info": {
                "mturk_agent_1": {"outcomes": {
                    "points_scored": 10, "satisfaction": "Undecided",
                    "opponent_likeness": likes[la]}},
                "mturk_agent_2": {"outcomes": {
                    "points_scored": 10, "satisfaction": "Undecided",
                    "opponent_likeness": likes[lb]}},
            },
        })
    cp = tmp_path / "casino.json"
    cp.write_text(json.dumps(data))
    V.main(["--fit", str(fit_d), "--eval", str(ev_d),
            "--casino", str(cp), "--lam", "1.0", "--b", "20",
            "--features", "legacy"])
    payload = json.loads((ev_d / "value.json").read_text())
    for key in ("V1", "V2", "V3", "V1_minus_C1", "w", "b", "mu"):
        assert key in payload
    md = (ev_d / "value.md").read_text()
    assert "V1" in md and "V2" in md and "V3" in md
    assert "rec_self" in md
