"""Contract for task32 value residuals (synthetic data only)."""
import json

import pytest

from forecaster import outcome as O
from forecaster import scalecheck as SC
from forecaster import valuemodel as V


def _item(cid, moves, effects, genre="negotiation", scale=None):
    rec = {"id": cid, "genre": genre, "moves": moves, "hits": [],
           "effects": effects, "gold": None}
    if scale is not None:
        rec["scale"] = scale
    return rec


def _eff(idx, speaker, value, n_readings=2):
    return {"index": idx, "speaker": speaker, "value": value,
            "n_readings": n_readings, "listeners": ["X"]}


# --- turn_residual ------------------------------------------------------

def test_turn_residual_zero_when_value_matches_table():
    e = {"index": 0, "speaker": "A", "value": 1.5, "n_readings": 2}
    assert V.turn_residual(e, 0, {0: ["accept"]}, 0.0, {"accept": 1.5}) == \
        pytest.approx(0.0)
    # centered value minus multi-move mean
    e2 = {"index": 1, "speaker": "B", "value": 5.0, "n_readings": 1}
    assert V.turn_residual(e2, 1, {1: ["a", "b"]}, 1.0, {"a": 1.0, "b": 3.0}) == \
        pytest.approx(2.0)
    # untagged turn -> tagv 0
    assert V.turn_residual(e2, 9, {1: ["a"]}, 1.0, {"a": 1.0}) == \
        pytest.approx(4.0)
    assert V.turn_residual(e2, 1, {1: []}, 1.0, {"a": 1.0}) == \
        pytest.approx(4.0)


def test_turn_residual_none_without_readings_or_value():
    assert V.turn_residual({"value": 1.0, "n_readings": 0}, 0, {}, 0.0, {}) is None
    assert V.turn_residual({"value": 1.0}, 0, {}, 0.0, {}) is None
    assert V.turn_residual("nope", 0, {}, 0.0, {}) is None
    bad = {"index": 0, "n_readings": 2, "relationship": ("x", 0, 0),
           "face": (0, 0, 0)}
    assert V.turn_residual(bad, 0, {}, 0.0, {}) is None


# --- rows_for shares the helper -----------------------------------------

def _resid_toy():
    items, oc = {}, {}
    for i in (1, 2, 3):
        eff = [_eff(0, "A", float(i)),
               _eff(1, "B", float(-i)),
               _eff(2, "A", float(i + 1), n_readings=0)]
        mv = [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
              {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
              {"turn_index": 2, "speaker": "A", "moves": []}]
        items[f"casino-{i}"] = _item(f"casino-{i}", mv, eff)
        oc[i] = {"A": {"points": 10, "satisfaction": 0, "likeness": -i},
                 "B": {"points": 10, "satisfaction": 0, "likeness": i}}
    return items, oc


def test_rows_for_residuals_equal_turn_residual_mean():
    items, oc = _resid_toy()
    rows, info = SC.rows_for(items, oc, items)
    assert len(rows) == 6
    mu = info["mu"]
    table = O.move_value_table(list(items.values()), mu)
    for row in rows:
        record = items[f"casino-{row['dialogue']}"]
        by_turn = {}
        for m in record.get("moves", []) or []:
            idx = m.get("turn_index")
            if isinstance(idx, int) and idx not in by_turn:
                by_turn[idx] = list(m.get("moves", []) or [])
        terms = []
        for pos, e in enumerate(record.get("effects", []) or []):
            if e.get("speaker") != row["side"]:
                continue
            idx = e.get("index", pos)
            idx = idx if isinstance(idx, int) else pos
            r = V.turn_residual(e, idx, by_turn, mu, table)
            if r is not None:
                terms.append(r)
        if terms:
            assert row["resid"] == pytest.approx(sum(terms) / len(terms))
        else:
            assert row["resid"] is None
    # the two sides see opposite signs on this fixture
    by_key = {(r["dialogue"], r["side"]): r["resid"] for r in rows}
    assert by_key[(1, "A")] == pytest.approx(-1.0)
    assert by_key[(1, "B")] == pytest.approx(1.0)


# --- legacy regression (task25 numbers, captured pre-change) ------------

def _task25_toy():
    items, oc = {}, {}
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
        oc[i] = {"A": {"points": 10, "satisfaction": 0, "likeness": -i},
                 "B": {"points": 10, "satisfaction": 0, "likeness": i}}
    return items, oc


def test_legacy_reproduces_task25_numbers():
    assert V.FEATURES_LEGACY == ["rec_self", "rec_other", "tagv_self",
                                 "tagv_other", "n_self"]
    assert V.FEATURES_RESID == ["tagv_self", "tagv_other", "resid_self",
                                "resid_other", "n_self"]
    items, oc = _task25_toy()
    table = {"accept": 1.0, "reject": -1.0}
    got = V.prefix_features(items["casino-1"], 1, "A", 0.0, table,
                            featureset="legacy")
    assert got == pytest.approx({"rec_self": 1.0, "rec_other": -1.0,
                                 "tagv_self": 1.0, "tagv_other": -1.0,
                                 "n_self": 1})
    model = V.fit_value(items, oc, 0.0, table, lam=1.0, featureset="legacy")
    assert model["features"] == V.FEATURES_LEGACY
    assert model["featureset"] == "legacy"
    assert model["w"] == pytest.approx([1.0616648691730097, -1.0616648691730086,
                                        0.26431718061674014, -0.2643171806167401,
                                        0.0])
    assert model["b"] == pytest.approx(0.0)
    assert model["means"] == pytest.approx([0.0, 0.0, 0.0, 0.0, 2.0])
    assert model["sds"] == pytest.approx([2.7386127875258306, 2.7386127875258306,
                                          1.0, 1.0, 1.0])
    res = V.evaluate(model, items, oc, 0.0, table, b=50, seed=0)
    assert res["featureset"] == "legacy"
    assert res["V1"] == pytest.approx([1.0, 1.0, 1.0])
    assert res["V1_minus_C1"] == pytest.approx([0.0, 0.0, 0.0])
    assert res["V2"] == pytest.approx([1.0, 1.0, 1.0])
    assert res["V3"] == pytest.approx([1.0, 1.0, 1.0])
    assert res["n"] == 8
    assert res["n_dialogues"] == 4
    assert res["n_pivot"] == 4


# --- resid featureset ---------------------------------------------------

def _prefix_item():
    effects = [_eff(0, "A", 1.0), _eff(1, "B", 2.0),
               _eff(2, "A", 100.0), _eff(3, "B", -100.0, n_readings=0)]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["accept"]},
        {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
        {"turn_index": 2, "speaker": "A", "moves": ["pressure"]},
        {"turn_index": 3, "speaker": "B", "moves": ["accept"]},
    ]
    return _item("casino-1", moves, effects)


def test_resid_prefix_ignores_future_and_defaults():
    table = {"accept": 1.0, "reject": 2.0, "pressure": 50.0}
    item = _prefix_item()
    got = V.prefix_features(item, 1, "A", 0.5, table)
    assert list(got) == V.FEATURES_RESID
    assert got["tagv_self"] == pytest.approx(1.0)
    assert got["tagv_other"] == pytest.approx(2.0)
    assert got["resid_self"] == pytest.approx(-0.5)
    assert got["resid_other"] == pytest.approx(-0.5)
    assert got["n_self"] == 1
    gotB = V.prefix_features(item, 1, "B", 0.5, table, featureset="resid")
    assert gotB["resid_self"] == pytest.approx(-0.5)
    assert gotB["resid_other"] == pytest.approx(-0.5)
    assert gotB["n_self"] == 1
    trunc = _item("casino-1", item["moves"][:2], item["effects"][:2])
    assert V.prefix_features(trunc, 1, "A", 0.5, table) == pytest.approx(got)
    full = V.prefix_features(item, 3, "A", 0.5, table)
    assert full["resid_self"] == pytest.approx((-0.5 + 49.5) / 2)
    # idx3 has no readings -> excluded from the other-side mean
    assert full["resid_other"] == pytest.approx(-0.5)
    assert full["n_self"] == 2
    # empty prefix -> zeros
    none_ = V.prefix_features(item, 0, "B", 0.5, table)
    assert none_["resid_self"] == pytest.approx(0.0)
    assert none_["tagv_self"] == pytest.approx(0.0)
    assert none_["n_self"] == 0
    with pytest.raises(ValueError):
        V.prefix_features(item, 1, "A", 0.5, table, featureset="bogus")


def test_resid_fit_and_evaluate_keys():
    items, oc = {}, {}
    for i in (1, 2, 3, 4):
        eff = [_eff(0, "A", float(i)), _eff(1, "B", float(-i)),
               _eff(2, "A", float(i)), _eff(3, "B", float(-i))]
        mv = [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
              {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
              {"turn_index": 2, "speaker": "A", "moves": ["accept"]},
              {"turn_index": 3, "speaker": "B", "moves": ["reject"]}]
        items[f"casino-{i}"] = _item(f"casino-{i}", mv, eff)
        oc[i] = {"A": {"points": 10, "satisfaction": 0, "likeness": -i},
                 "B": {"points": 10, "satisfaction": 0, "likeness": i}}
    table = {"accept": 1.0, "reject": -1.0}
    model = V.fit_value(items, oc, 0.0, table, lam=1.0)
    assert model["featureset"] == "resid"
    assert model["features"] == V.FEATURES_RESID
    assert len(model["w"]) == 5
    res = V.evaluate(model, items, oc, 0.0, table, b=50, seed=0)
    assert res["featureset"] == "resid"
    for key in ("V1", "V2", "V3", "V1_minus_C1"):
        assert key in res
        assert len(res[key]) == 3
    assert res["n"] == 8
    assert res["n_pivot"] == 4


# --- CLI ----------------------------------------------------------------

def _casino_file(tmp_path, outcomes):
    likes = {-2: "Extremely dislike", -1: "Slightly dislike", 0: "Undecided",
             1: "Slightly like", 2: "Extremely like"}
    data = []
    for did, rec in outcomes.items():
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
    return str(cp)


def _write(d, records):
    d.mkdir(parents=True, exist_ok=True)
    for cid, rec in records.items():
        (d / f"{cid}.json").write_text(json.dumps(rec))


def _cli_toy(scale=None):
    items, oc = {}, {}
    for i in (1, 2, 3, 4):
        eff = [_eff(0, "A", float(i)), _eff(1, "B", float(-i)),
               _eff(2, "A", float(i)), _eff(3, "B", float(-i))]
        mv = [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
              {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
              {"turn_index": 2, "speaker": "A", "moves": ["accept"]},
              {"turn_index": 3, "speaker": "B", "moves": ["reject"]}]
        items[f"casino-{i}"] = _item(f"casino-{i}", mv, eff, scale=scale)
        oc[i] = {"A": {"points": 10, "satisfaction": 0, "likeness": -i},
                 "B": {"points": 10, "satisfaction": 0, "likeness": i}}
    return items, oc


def test_cli_both_writes_split_and_compare(tmp_path):
    items, oc = _cli_toy()
    fit_d, ev_d = tmp_path / "fit", tmp_path / "ev"
    _write(fit_d, items)
    _write(ev_d, items)
    cp = _casino_file(tmp_path, oc)
    V.main(["--fit", str(fit_d), "--eval", str(ev_d),
            "--casino", cp, "--lam", "1.0", "--b", "20"])
    for fs in ("legacy", "resid"):
        payload = json.loads((ev_d / f"value_{fs}.json").read_text())
        assert payload["featureset"] == fs
        for key in ("V1", "V2", "V3", "V1_minus_C1", "w", "b", "mu"):
            assert key in payload
        md = (ev_d / f"value_{fs}.md").read_text()
        assert "V1" in md and "V2" in md and "V3" in md
    assert "rec_self" in (ev_d / "value_legacy.md").read_text()
    assert "resid_self" in (ev_d / "value_resid.md").read_text()
    cmp_md = (ev_d / "value_compare.md").read_text()
    for token in ("V1", "V1-C1", "V2", "V3", "legacy", "resid",
                  "fit_scale", "eval_scale", str(fit_d), str(ev_d)):
        assert token in cmp_md


def test_cli_single_featureset_writes_value_files(tmp_path):
    items, oc = _cli_toy()
    fit_d, ev_d = tmp_path / "fit", tmp_path / "ev"
    _write(fit_d, items)
    _write(ev_d, items)
    cp = _casino_file(tmp_path, oc)
    V.main(["--fit", str(fit_d), "--eval", str(ev_d), "--casino", cp,
            "--b", "20", "--features", "resid"])
    payload = json.loads((ev_d / "value.json").read_text())
    assert payload["featureset"] == "resid"
    assert "resid_self" in (ev_d / "value.md").read_text()


def test_cli_refuses_scale_mismatch(tmp_path):
    fit_items, oc = _cli_toy(scale="v2")
    ev_items, _ = _cli_toy(scale="fine100")
    fit_d, ev_d = tmp_path / "fit", tmp_path / "ev"
    _write(fit_d, fit_items)
    _write(ev_d, ev_items)
    cp = _casino_file(tmp_path, oc)
    with pytest.raises(SystemExit) as ei:
        V.main(["--fit", str(fit_d), "--eval", str(ev_d), "--casino", cp,
                "--b", "20"])
    assert "scale mismatch" in str(ei.value)
    mixed, _ = _cli_toy()
    mixed["casino-1"] = _item("casino-1", mixed["casino-1"]["moves"],
                              mixed["casino-1"]["effects"], scale="fine100")
    _write(fit_d, mixed)
    with pytest.raises(SystemExit) as ei2:
        V.main(["--fit", str(fit_d), "--eval", str(ev_d), "--casino", cp,
                "--b", "20"])
    assert "scale mismatch" in str(ei2.value)
