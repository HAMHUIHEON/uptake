"""Contract for SPEC_task29_scalecheck (synthetic only, no network, no data/)."""
import json
import math
import random

import pytest

from forecaster import scalecheck as SC
from forecaster.batch import spearman


def test_rank_ties():
    assert SC.rank([10, 20, 20, 30]) == pytest.approx([1.0, 2.5, 2.5, 4.0])
    assert SC.rank([5, 5, 5]) == pytest.approx([2.0, 2.0, 2.0])
    assert SC.rank([]) == []


def test_ols_recovers_line():
    X = [[0, 0], [1, 0], [0, 1], [2, 1], [1, 2]]
    y = [1, 3, 0, 4, 1]
    coef, resid, r2 = SC.ols(X, y)
    assert coef[0] == pytest.approx(1.0, abs=1e-6)
    assert coef[1] == pytest.approx(2.0, abs=1e-6)
    assert coef[2] == pytest.approx(-1.0, abs=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)
    assert all(abs(v) < 1e-6 for v in resid)


def test_partial_no_controls_equals_spearman():
    x = [1, 2, 3, 4, 5]
    y = [2, 1, 4, 3, 5]
    assert SC.partial_spearman(x, y, []) == pytest.approx(spearman(x, y))
    x2 = [1, 1, 2, 3]
    y2 = [1, 2, 2, 3]
    assert SC.partial_spearman(x2, y2, []) == pytest.approx(spearman(x2, y2))


def test_partial_removes_confounder():
    n = 12
    c = list(range(n))
    e1 = [1, -1] * 6
    e2 = [1, 0, -1, 0] * 3
    assert sum(a * b for a, b in zip(e1, e2)) == 0
    x = [a + b for a, b in zip(c, e1)]
    y = [a + b for a, b in zip(c, e2)]
    plain = spearman(x, y)
    part = SC.partial_spearman(x, y, [c])
    assert plain > 0.5
    assert abs(part) < 0.2


def test_rank_regression_delta():
    x = list(range(10))
    y = list(range(10))
    c = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    _b, d = SC.rank_regression(x, y, c)
    assert d >= 0
    assert d > 0.1
    _b2, d2 = SC.rank_regression(x, y, x)
    assert abs(d2) < 1e-6


def _eff(idx, speaker, value, readings, n_expected=None):
    e = {"index": idx, "speaker": speaker, "value": value,
         "n_readings": len(readings), "readings": readings,
         "listeners": ["X"]}
    if n_expected is not None:
        e["n_expected"] = n_expected
    return e


def test_icc_identical_within():
    def rd(v):
        return {"rel_norm": v, "face_norm": 0.0, "rel_raw": 0, "face_raw": 0}
    rec = {"id": "casino-1", "genre": "negotiation", "scale": "v2",
           "effects": [_eff(0, "A", 1.0, [rd(1.0), rd(1.0)]),
                       _eff(1, "B", 2.0, [rd(2.0), rd(2.0)]),
                       _eff(2, "A", 3.0, [rd(3.0), rd(3.0)])]}
    out = SC.resolution({"casino-1": rec})
    assert out["icc1"] == pytest.approx(1.0)
    assert out["sb3"] == pytest.approx(1.0)
    assert out["sb5"] == pytest.approx(1.0)


def test_icc_constant():
    def rd(v):
        return {"rel_norm": v, "face_norm": 0.0, "rel_raw": 0, "face_raw": 0}
    rec = {"id": "casino-1", "genre": "negotiation", "scale": "v2",
           "effects": [_eff(0, "A", 1.0, [rd(1.0), rd(1.0)]),
                       _eff(1, "B", 1.0, [rd(1.0), rd(1.0)]),
                       _eff(2, "A", 1.0, [rd(1.0), rd(1.0)])]}
    out = SC.resolution({"casino-1": rec})
    assert out["icc1"] <= 0
    assert out["sb3"] == pytest.approx(0.0)
    assert out["sb5"] == pytest.approx(0.0)


def test_resolution_v2_handbuilt():
    rec = {"id": "casino-1", "genre": "negotiation", "scale": "v2",
           "effects": [
               {"index": 0, "speaker": "A", "listeners": ["B"],
                "n_readings": 2, "n_expected": 3, "value": 1.0,
                "readings": [{"relationship_shift": -2, "face_threat": 0},
                             {"relationship_shift": 2, "face_threat": 2}]},
               {"index": 1, "speaker": "B", "listeners": ["A"],
                "n_readings": 1, "value": 2.0,
                "readings": [{"relationship_shift": 0, "face_threat": 1}]},
           ]}
    out = SC.resolution({"casino-1": rec}, base_n=3)
    assert out["n_turns"] == 2
    assert out["distinct_turn_values"] == 2
    assert out["distinct_reading_values"] == 3
    assert out["rel_min_share"] == pytest.approx(1 / 3)
    assert out["rel_max_share"] == pytest.approx(1 / 3)
    assert out["face_min_share"] == pytest.approx(1 / 3)
    assert out["face_max_share"] == pytest.approx(1 / 3)
    assert out["parse_fail_rate"] == pytest.approx(0.5)
    assert out["z_ge2_share"] == pytest.approx(0.0)
    assert out["z_le_minus2_share"] == pytest.approx(0.0)
    assert out["z_max"] == pytest.approx(1.0)
    assert out["z_min"] == pytest.approx(-1.0)
    assert out["mean"] == pytest.approx(1.5)
    assert out["sd"] == pytest.approx(0.5)


def test_resolution_fine100_handbuilt():
    rec = {"id": "casino-2", "genre": "negotiation", "scale": "fine100",
           "effects": [
               {"index": 0, "speaker": "A", "listeners": ["B"],
                "n_readings": 2, "n_expected": 2, "value": 0.0,
                "readings": [
                    {"rel_raw": 0, "face_raw": 0,
                     "rel_norm": -2.0, "face_norm": -2.0},
                    {"rel_raw": 100, "face_raw": 100,
                     "rel_norm": 2.0, "face_norm": 2.0}]},
           ]}
    out = SC.resolution({"casino-2": rec})
    assert out["distinct_turn_values"] == 1
    assert out["distinct_reading_values"] == 2
    assert out["rel_min_share"] == pytest.approx(0.5)
    assert out["rel_max_share"] == pytest.approx(0.5)
    assert out["face_min_share"] == pytest.approx(0.5)
    assert out["face_max_share"] == pytest.approx(0.5)
    assert out["parse_fail_rate"] == pytest.approx(0.0)
    assert out["sd"] == pytest.approx(0.0)
    assert out["z_max"] == pytest.approx(0.0)


def test_board_cost(tmp_path):
    p = tmp_path / "calls.jsonl"
    lines = [
        {"model": "muse-spark-1.3-contributor", "prompt_tokens": 1000,
         "completion_tokens": 500,
         "meta": {"layer": "board", "item": "casino-1"}, "error": None},
        {"model": "muse-spark-1.3-contributor", "prompt_tokens": 2000,
         "completion_tokens": 1000,
         "meta": {"layer": "board", "item": "casino-2"}, "error": None},
        {"model": "muse-spark-1.3-contributor", "prompt_tokens": 1000,
         "completion_tokens": 500,
         "meta": {"layer": "moves", "item": "casino-1"}, "error": None},
        {"model": "muse-spark-1.3-contributor", "prompt_tokens": 1000,
         "completion_tokens": 500,
         "meta": {"layer": "board", "item": "casino-1"},
         "error": "Timeout"},
        {"model": "muse-spark-1.3-contributor", "prompt_tokens": 1000,
         "completion_tokens": 500,
         "meta": {"layer": "board", "item": "casino-99"}, "error": None},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines))
    got = SC.board_cost(str(p), ["casino-1", "casino-2"])
    assert got["calls"] == 2
    assert got["usd"] == pytest.approx(0.0006)
    assert got["usd_per_dialogue"] == pytest.approx(0.0003)
    assert got["mean_completion_tokens"] == pytest.approx(750.0)
    missing = SC.board_cost(str(tmp_path / "nope.jsonl"), ["casino-1"])
    assert missing == {"calls": 0, "usd": 0.0, "usd_per_dialogue": 0.0,
                       "mean_completion_tokens": 0.0}


def test_boot_same_resamples():
    rows = [{"dialogue": 1, "x": 1}, {"dialogue": 1, "x": 2},
            {"dialogue": 2, "x": 10}, {"dialogue": 2, "x": 20},
            {"dialogue": 3, "x": 5}, {"dialogue": 4, "x": 7}]

    def meanx(rs):
        return sum(r["x"] for r in rs) / len(rs) if rs else 0.0

    out = SC.boot(rows, {"a": meanx, "b": meanx}, b=100, seed=0)
    assert out["a"]["lo"] == out["b"]["lo"]
    assert out["a"]["hi"] == out["b"]["hi"]
    assert out["a"]["se"] > 0
    out2 = SC.boot(rows, {"a": meanx}, b=100, seed=0)
    assert out2["a"] == out["a"]


def _synthetic(n_dialogues=34, seed=1):
    rng = random.Random(seed)
    pool = ["accept", "reject", "propose", "inform", "pressure", "praise"]
    fit = {}
    ev = {}
    oc = {}
    for i in range(1, n_dialogues + 1):
        cid = f"casino-{i}"
        mv = []
        eff = []
        for t in range(4):
            sp = "A" if t % 2 == 0 else "B"
            mv.append({"turn_index": t, "speaker": sp,
                       "moves": [rng.choice(pool)],
                       "addressed_to": "", "quote": "", "failed": False})
            v = rng.uniform(-2, 2)
            eff.append({"index": t, "speaker": sp, "text": f"t{t}",
                        "listeners": ["B" if sp == "A" else "A"],
                        "value": v, "n_readings": 3, "n_expected": 3,
                        "readings": [{"rel_raw": 0, "face_raw": 0,
                                      "rel_norm": v / 2, "face_norm": v / 2}
                                     for _ in range(2)],
                        "relationship": (v / 2, v / 2, v / 2),
                        "face": (v / 2, v / 2, v / 2),
                        "action_dist": {}, "disagreement": 0.0})
        rec = {"id": cid, "genre": "negotiation", "moves": mv,
               "effects": eff, "hits": []}
        fit[cid] = rec
        ev[cid] = rec
        oc[i] = {"A": {"points": 10, "satisfaction": 0,
                       "likeness": rng.randint(-2, 2)},
                 "B": {"points": 10, "satisfaction": 0,
                       "likeness": rng.randint(-2, 2)}}
    return fit, ev, oc


def test_analyze_eval_keys_and_power():
    fit, ev, oc = _synthetic(34, seed=1)
    res = SC.analyze_eval(ev, oc, fit, b=50, b_r2=20, seed=0,
                          same_scale=True)
    for key in ("n_dialogues", "n_rows", "stats", "verdicts", "power",
                "info"):
        assert key in res
    for key in ("C1", "C6_tag", "C6diff", "R1", "R2", "R3_beta",
                "R3_dr2", "R4"):
        assert key in res["stats"]
        assert key in res["verdicts"]
    for key in ("mu", "sd", "share_moves", "n_rows"):
        assert key in res["info"]
    assert res["n_dialogues"] >= 30
    for key in ("C6diff", "R1"):
        n80 = res["power"][key]["n80"]
        assert abs(n80["0.03"] / 4 - n80["0.06"]) <= 1.0


def _casino_file(tmp_path, outcomes):
    likes = {-2: "Extremely dislike", -1: "Slightly dislike",
             0: "Undecided", 1: "Slightly like", 2: "Extremely like"}
    data = []
    for did, rec in outcomes.items():
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
    return str(cp)


def _write_records(d, records):
    d.mkdir(parents=True, exist_ok=True)
    for cid, rec in records.items():
        (d / f"{cid}.json").write_text(json.dumps(rec))


def _tiny_records(rng):
    out = {}
    for i in (1, 2, 3):
        cid = f"casino-{i}"
        mv = [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
              {"turn_index": 1, "speaker": "B", "moves": ["reject"]}]
        eff = [
            {"index": 0, "speaker": "A", "text": "hi",
             "listeners": ["B"], "value": float(rng.uniform(-1, 1)),
             "n_readings": 2, "n_expected": 2,
             "readings": [{"relationship_shift": 1, "face_threat": 0},
                          {"relationship_shift": 0, "face_threat": 1}]},
            {"index": 1, "speaker": "B", "text": "yo",
             "listeners": ["A"], "value": float(rng.uniform(-1, 1)),
             "n_readings": 2, "n_expected": 2,
             "readings": [{"relationship_shift": -1, "face_threat": 1},
                          {"relationship_shift": 1, "face_threat": 0}]},
        ]
        out[cid] = {"id": cid, "genre": "negotiation", "moves": mv,
                    "effects": eff, "hits": []}
    oc = {1: {"A": {"points": 10, "satisfaction": 0, "likeness": 1},
              "B": {"points": 10, "satisfaction": 0, "likeness": -1}},
          2: {"A": {"points": 10, "satisfaction": 0, "likeness": 0},
              "B": {"points": 10, "satisfaction": 0, "likeness": 1}},
          3: {"A": {"points": 10, "satisfaction": 0, "likeness": -1},
              "B": {"points": 10, "satisfaction": 0, "likeness": 0}}}
    return out, oc


def test_cli_residual_smoke(tmp_path, capsys):
    rng = random.Random(2)
    records, oc = _tiny_records(rng)
    fit_d = tmp_path / "fit"
    ev_d = tmp_path / "ev"
    _write_records(fit_d, records)
    _write_records(ev_d, records)
    cp = _casino_file(tmp_path, oc)
    out = str(tmp_path / "res")
    SC.main(["residual", "--fit", str(fit_d), "--casino", cp,
             "--eval", str(ev_d), "--out", out, "--b", "20",
             "--b-r2", "10"])
    payload = json.loads((tmp_path / "res.json").read_text())
    assert "stats" in payload and "power" in payload
    md = (tmp_path / "res.md").read_text()
    assert "C1" in md and "Power" in md
    printed = capsys.readouterr().out
    assert "C1" in printed


def test_cli_pilot_smoke(tmp_path, capsys):
    rng = random.Random(3)
    records, oc = _tiny_records(rng)
    fit_d = tmp_path / "fit"
    c0_d = tmp_path / "c0"
    c1_d = tmp_path / "c1"
    _write_records(fit_d, records)
    _write_records(c0_d, records)
    _write_records(c1_d, records)
    cp = _casino_file(tmp_path, oc)
    ids_f = tmp_path / "ids.txt"
    ids_f.write_text("casino-1\ncasino-2\ncasino-3\n")
    out = str(tmp_path / "pilot")
    SC.main(["pilot", "--fit", str(fit_d), "--casino", cp,
             "--ids-file", str(ids_f), "--cond", f"P0={c0_d}",
             "--cond", f"P1={c1_d}", "--base", "P0",
             "--out", out, "--b", "20"])
    payload = json.loads((tmp_path / "pilot.json").read_text())
    assert "conditions" in payload
    assert set(payload["conditions"]) == {"P0", "P1"}
    md = (tmp_path / "pilot.md").read_text()
    assert "P0" in md and "P1" in md
    printed = capsys.readouterr().out
    assert "P0" in printed and "P1" in printed
