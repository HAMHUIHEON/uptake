"""Contract for SPEC_task34_pairwise (synthetic only, fake clients)."""
import json

import pytest

from forecaster import board as B
from forecaster import outcome as O
from forecaster import pairwise as P
from forecaster import scalecheck as SC
from forecaster.trace import CallLog, LoggingClient
from forecaster.transcribe import Turn
from forecaster.usage import Meter


def _record(cid, texts, genre="negotiation", speakers=("A", "B")):
    effects = [{"index": i, "speaker": speakers[i % len(speakers)],
                "text": t, "listeners": ["X"], "n_readings": 1,
                "value": 0.0}
               for i, t in enumerate(texts)]
    return {"id": cid, "genre": genre, "model": "src-model",
            "moves": [{"turn_index": i, "speaker": speakers[i % len(speakers)],
                       "moves": ["inform"], "addressed_to": "",
                       "quote": "", "failed": False}
                      for i in range(len(texts))],
            "effects": effects, "board": effects, "hits": [],
            "gold": [["inform"]] * len(texts)}


def _write_src(tmp_path, records):
    src = tmp_path / "src"
    src.mkdir()
    for cid, rec in records.items():
        (src / f"{cid}.json").write_text(json.dumps(rec))
    return src


# --- Bradley-Terry ---

def test_bt_recovers_planted_ordering():
    rows = ([(("d", 0), ("d", 1), "a")] * 10
            + [(("d", 1), ("d", 2), "a")] * 10
            + [(("d", 0), ("d", 2), "a")] * 10)
    ratings, counts = P.fit_bradley_terry(rows)
    assert ratings[("d", 0)] > ratings[("d", 1)] > ratings[("d", 2)]
    assert counts == {("d", 0): 20, ("d", 1): 20, ("d", 2): 20}


def test_bt_centred_and_ties():
    rows = [(("d", 0), ("d", 1), "tie")] * 10
    ratings, _ = P.fit_bradley_terry(rows)
    assert ratings[("d", 0)] == pytest.approx(ratings[("d", 1)], abs=1e-9)
    rows2 = ([(("d", 0), ("d", 1), "a")] * 7
             + [(("d", 0), ("d", 1), "tie")] * 2
             + [(("d", 0), ("d", 1), "b")] * 1)
    ratings2, _ = P.fit_bradley_terry(rows2)
    assert ratings2[("d", 0)] > ratings2[("d", 1)]
    assert sum(ratings2.values()) == pytest.approx(0.0, abs=1e-9)
    assert sum(ratings.values()) == pytest.approx(0.0, abs=1e-9)


def test_bt_single_vote_connected_by_prior():
    ratings, counts = P.fit_bradley_terry([(("d", 0), ("d", 1), "a")])
    assert ratings[("d", 0)] > ratings[("d", 1)]
    assert all(v == v and abs(v) != float("inf")  # finite
               for v in ratings.values())
    assert counts == {("d", 0): 1, ("d", 1): 1}
    assert P.fit_bradley_terry([]) == ({}, {})


# --- pairing ---

def test_pairing_coverage_no_within_dialogue_balanced():
    records = {f"casino-{i}": _record(f"casino-{i}", [f"t{j}" for j in range(3)])
               for i in range(4)}
    pool, _ = P.collect_pool(records)
    assert len(pool) == 12
    pairs = P.make_pairs(pool, k=3, seed=0)
    assert pairs
    from collections import Counter
    appear = Counter()
    genre_of = {e["dialogue"]: e["genre"] for e in pool}
    for p in pairs:
        assert p["a"]["dialogue"] != p["b"]["dialogue"]
        assert genre_of[p["a"]["dialogue"]] == genre_of[p["b"]["dialogue"]]
        appear[(p["a"]["dialogue"], p["a"]["turn"])] += 1
        appear[(p["b"]["dialogue"], p["b"]["turn"])] += 1
    for e in pool:
        assert appear[(e["dialogue"], e["turn"])] >= 3
    share_a_first = sum(1 for p in pairs if p["first"] == "a") / len(pairs)
    assert 0.3 < share_a_first < 0.7
    assert P.make_pairs(pool, k=3, seed=0) == pairs
    assert P.make_pairs(pool, k=3, seed=1) != pairs


def test_pairing_genre_isolation_and_singletons():
    recs = {"casino-1": _record("casino-1", ["a", "b"], genre="g1"),
            "casino-2": _record("casino-2", ["c", "d"], genre="g1"),
            "casino-3": _record("casino-3", ["e", "f"], genre="g2")}
    pool, _ = P.collect_pool(recs)
    pairs = P.make_pairs(pool, k=4, seed=0)
    assert pairs
    by_id = {e["dialogue"]: e["genre"] for e in pool}
    for p in pairs:
        assert by_id[p["a"]["dialogue"]] == by_id[p["b"]["dialogue"]]
        assert "casino-3" not in (p["a"]["dialogue"], p["b"]["dialogue"])


# --- votes ---

def test_derandomize_both_orders():
    assert P.derandomize("first", "a") == "a"
    assert P.derandomize("second", "a") == "b"
    assert P.derandomize("first", "b") == "b"
    assert P.derandomize("second", "b") == "a"
    assert P.derandomize("tie", "a") == "tie"
    assert P.derandomize("tie", "b") == "tie"
    assert P.derandomize(None, "a") is None
    assert P.derandomize("bogus", "a") is None


def test_parse_pairwise():
    assert P.parse_pairwise('{"better": "first", "why": "ok"}') == \
        {"better": "first", "why": "ok"}
    assert P.parse_pairwise('-note {"better":"SECOND","why":"x"} tail') == \
        {"better": "second", "why": "x"}
    assert P.parse_pairwise('{"better": "Tie", "why": "x"}')["better"] == "tie"
    assert P.parse_pairwise("garbage no json") is None
    assert P.parse_pairwise('{"better": "maybe", "why": "x"}') is None
    assert P.parse_pairwise('{"why": "x"}') is None
    assert P.parse_pairwise(None) is None


def test_panel_stances_match_board():
    panel = P.pairwise_panel(3)
    assert [r.stance for r in panel] == ["neutral", "trusting", "suspicious"]
    for r in panel:
        assert r.relationship == "participant in this conversation"
        assert r.power == "equal"
        assert r.channel == "conversation"


def test_prompt_windows_match_board():
    turns = [Turn("A" if i % 2 == 0 else "B", f"msg{i}") for i in range(8)]
    ctx = {"casino-1": {"turns": turns,
                        "pos_of": {i: i for i in range(8)},
                        "genre": "negotiation"}}
    pair = {"id": 0,
            "a": {"dialogue": "casino-1", "turn": 7, "speaker": "B",
                  "listener": "A"},
            "b": {"dialogue": "casino-1", "turn": 1, "speaker": "B",
                  "listener": "A"},
            "first": "a"}
    rec = P.pairwise_panel(0)[0]
    msgs = P.build_pair_messages(pair, ctx, rec)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "stance: neutral" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert "Snippet 1 (listener: A)" in user
    assert "MESSAGE TO JUDGE B: msg7" in user
    assert "MESSAGE TO JUDGE B: msg1" in user
    assert '"better": "first" | "second" | "tie"' in user
    board_user = B.build_listener_messages(turns, 7, "A", rec, 6)[1]["content"]
    assert P.snippet_lines(turns, 7) == board_user.splitlines()[:-1]
    assert P.snippet_lines(turns, 1) == ["A: msg0"]
    assert P.snippet_lines(turns, 0) == []


# --- run with fake clients ---

class _Fixed:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def complete(self, messages, model, temperature):
        self.calls += 1
        return self.raw


def _src3():
    return {f"casino-{i}": _record(f"casino-{i}", ["hello", "hi back"])
            for i in range(1, 4)}


def test_run_fake_client_outputs(tmp_path):
    records = _src3()
    pool, contexts = P.collect_pool(records)
    pairs = P.make_pairs(pool, k=2, seed=0)
    client = LoggingClient(_Fixed('{"better": "first", "why": "warmer"}'),
                           CallLog(tmp_path / "calls.jsonl"), "t")
    raw, stats = P.judge_all(client, pairs, contexts, workers=4)
    assert stats["budget_stop"] is False
    assert stats["n_attempted_votes"] == len(pairs) * 3
    assert all(v == "first" for per in raw.values() for v in per.values())
    out = tmp_path / "out"
    run = P.write_run_outputs(out, pairs, raw, stats, {"seed": 0}, {}, 0.0)
    assert run["n_pairs"] == len(pairs)
    assert run["n_votes"] == run["n_expected"] == len(pairs) * 3
    assert run["parse_fail_rate"] == pytest.approx(0.0)
    assert run["position_bias_share"] == pytest.approx(1.0)
    stored = [json.loads(line) for line in
              (out / "pairs.jsonl").read_text().splitlines()]
    assert len(stored) == len(pairs)
    for line in stored:
        assert set(("id", "a", "b", "first", "votes")) <= set(line)
        for stance in ("neutral", "trusting", "suspicious"):
            assert line["votes"][stance] == line["first"]
    calls = (tmp_path / "calls.jsonl").read_text().splitlines()
    assert len(calls) == len(pairs) * 3
    first_call = json.loads(calls[0])
    assert first_call["meta"]["layer"] == "pairwise"
    assert set(("pair", "stance", "dialogues")) <= set(first_call["meta"])
    priced = SC.board_cost(str(tmp_path / "calls.jsonl"),
                           ["casino-1", "casino-2", "casino-3"])
    assert priced["calls"] == len(pairs) * 3


def test_run_garbage_votes_null(tmp_path):
    records = _src3()
    pool, contexts = P.collect_pool(records)
    pairs = P.make_pairs(pool, k=2, seed=0)
    raw, stats = P.judge_all(_Fixed("garbage no json"), pairs, contexts,
                             workers=2)
    assert stats["n_calls"] == 2 * len(pairs) * 3  # one retry each
    run = P.write_run_outputs(tmp_path, pairs, raw, stats, {}, {}, 0.0)
    assert run["n_votes"] == 0
    assert run["parse_fail_rate"] == pytest.approx(1.0)
    assert run["position_bias_share"] == pytest.approx(0.0)
    rows = P.pair_votes(P.load_pairs(tmp_path))
    assert rows == []


def test_run_budget_stop_clean(tmp_path):
    records = _src3()
    pool, contexts = P.collect_pool(records)
    pairs = P.make_pairs(pool, k=2, seed=0)
    raw, stats = P.judge_all(_Fixed('{"better":"tie","why":"x"}'), pairs,
                             contexts, meter=Meter(budget_usd=0.0))
    assert stats["budget_stop"] is True
    assert stats["n_attempted_votes"] == 0
    run = P.write_run_outputs(tmp_path, pairs, raw, stats, {}, {}, 0.0)
    assert run["budget_stop"] is True
    assert run["n_votes"] == 0
    assert len(P.load_pairs(tmp_path)) == len(pairs)


# --- fit ---

def _pairs_file(tmp_path, pairs):
    d = tmp_path / "pairs"
    d.mkdir()
    (d / "pairs.jsonl").write_text(
        "\n".join(json.dumps(p) for p in pairs))
    return d


def test_fit_cli_end_to_end(tmp_path):
    src = _write_src(tmp_path, _src3())
    pool, _ = P.collect_pool({k: json.loads((src / f"{k}.json").read_text())
                              for k in ("casino-1", "casino-2", "casino-3")})
    pairs = P.make_pairs(pool, k=2, seed=0)
    for p in pairs:
        p["votes"] = {"neutral": "a", "trusting": "a", "suspicious": "tie"}
    pairs_d = _pairs_file(tmp_path, pairs)
    out = tmp_path / "out"
    P.main(["fit", "--pairs", str(pairs_d), "--src", str(src),
            "--out", str(out), "--half-seed", "1"])
    ratings = json.loads((out / "ratings.json").read_text())
    assert ratings["n_turns_rated"] == len(pool)
    by_key = {(r["dialogue"], r["turn"]): r for r in ratings["ratings"]}
    assert set(by_key) == {(e["dialogue"], e["turn"]) for e in pool}
    assert sum(r["rating"] for r in ratings["ratings"]) == pytest.approx(
        0.0, abs=1e-9)
    for row in ratings["reception_pw"]:
        keys = [k for k, r in by_key.items()
                if k[0] == row["dialogue"] and r["speaker"] == row["side"]]
        assert keys
        assert row["reception_pw"] == pytest.approx(
            sum(by_key[k]["rating"] for k in keys) / len(keys))
        assert row["n_turns"] == len(keys)
    rel = json.loads((out / "reliability.json").read_text())
    assert set(("rho", "n_turns")) <= set(rel)
    assert rel["n_turns"] >= 2
    assert -1.0 <= rel["rho"] <= 1.0
    emitted = {p.stem: json.loads(p.read_text())
               for p in (out / "records").glob("casino-*.json")}
    assert set(emitted) == {"casino-1", "casino-2", "casino-3"}
    for cid, rec in emitted.items():
        assert rec["scale"] == "pairwise"
        assert rec["moves"]  # moves preserved for tag baselines
        assert rec["id"] == cid
        for e in rec["effects"]:
            assert set(("index", "speaker", "text", "value",
                        "n_readings")) <= set(e)
            assert e["value"] == pytest.approx(
                by_key[(cid, e["index"])]["rating"])
    got = O.reception(emitted["casino-1"], 0.0)
    assert set(got) == {"A", "B"}
    res = SC.resolution(emitted)
    assert res["n_turns"] == len(pool)


def test_emit_records_unrated_zero(tmp_path):
    src = _write_src(tmp_path, {"casino-1": _record("casino-1", ["a", "b"])})
    pairs = [{"id": 0,
              "a": {"dialogue": "casino-1", "turn": 0, "speaker": "A",
                    "listener": "B"},
              "b": {"dialogue": "casino-2", "turn": 0, "speaker": "A",
                    "listener": "B"},
              "first": "a",
              "votes": {"neutral": "a", "trusting": None,
                        "suspicious": "tie"}}]
    rows = P.pair_votes(pairs)
    assert len(rows) == 2
    ratings, counts = P.fit_bradley_terry(rows)
    records = {p.stem: json.loads(p.read_text())
               for p in src.glob("casino-*.json")}
    written = P.emit_records(records, ratings, counts, tmp_path / "rec")
    assert written == ["casino-1"]
    rec = json.loads((tmp_path / "rec" / "casino-1.json").read_text())
    assert rec["scale"] == "pairwise"
    by_idx = {e["index"]: e for e in rec["effects"]}
    assert by_idx[0]["n_readings"] == 2
    assert by_idx[0]["value"] == pytest.approx(ratings[("casino-1", 0)])
    assert by_idx[1]["n_readings"] == 0
    assert by_idx[1]["value"] == pytest.approx(0.0)


def test_split_half_deterministic():
    pairs = []
    for i in range(8):
        pairs.append({"id": i,
                      "a": {"dialogue": "casino-1", "turn": 0,
                            "speaker": "A", "listener": "B"},
                      "b": {"dialogue": "casino-2", "turn": i % 2,
                            "speaker": "A", "listener": "B"},
                      "first": "a",
                      "votes": {"neutral": "a", "trusting": "b",
                                "suspicious": "tie"}})
    r1 = P.split_half_reliability(pairs, half_seed=1)
    assert r1 == P.split_half_reliability(pairs, half_seed=1)
    assert 1 <= r1["n_turns"] <= 3
    assert r1["n_votes_half1"] + r1["n_votes_half2"] == 8 * 3


# --- fine100 split-half ---

def _fine100_record(cid, values, stances=("neutral", "trusting", "suspicious")):
    effects = []
    for i, v in enumerate(values):
        readings = [{"stance": s, "rel_norm": float(v), "face_norm": 0.0}
                    for s in stances]
        effects.append({"index": i, "speaker": "A", "text": f"t{i}",
                        "n_readings": len(readings), "readings": readings})
    return {"id": cid, "genre": "negotiation", "moves": [], "effects": effects}


def test_fine100_split_half_toy():
    records = {"casino-1": _fine100_record("casino-1", [0, 1, 2, 3, 4])}
    got = P.fine100_split_half(records)
    assert got["nt_vs_s"] == {"rho": pytest.approx(1.0), "n_turns": 5}
    assert got["ns_vs_t"] == {"rho": pytest.approx(1.0), "n_turns": 5}
    assert got["mean_rho"] == pytest.approx(1.0)


def test_fine100_split_half_missing_stance(tmp_path):
    rec = _fine100_record("casino-1", [0, 1, 2, 3])
    for e in rec["effects"][:2]:
        e["readings"] = [r for r in e["readings"]
                         if r["stance"] != "suspicious"]
    src = _write_src(tmp_path, {"casino-1": rec})
    out_f = tmp_path / "rel.json"
    P.main(["reliability-fine100", "--src", str(src), "--out", str(out_f)])
    got = json.loads(out_f.read_text())
    assert got["nt_vs_s"]["n_turns"] == 2
    assert got["ns_vs_t"]["n_turns"] == 4
    assert got["mean_rho"] == pytest.approx(
        (got["nt_vs_s"]["rho"] + got["ns_vs_t"]["rho"]) / 2)
    assert P.fine100_split_half({})["mean_rho"] == pytest.approx(0.0)


# --- board_cost prices pairwise calls ---

def test_board_cost_pairwise(tmp_path):
    p = tmp_path / "calls.jsonl"
    lines = [
        {"model": "muse-spark-1.3-contributor", "prompt_tokens": 1000,
         "completion_tokens": 500,
         "meta": {"layer": "pairwise", "pair": 0,
                  "dialogues": ["casino-1", "casino-2"]}, "error": None},
        {"model": "muse-spark-1.3-contributor", "prompt_tokens": 1000,
         "completion_tokens": 500,
         "meta": {"layer": "pairwise", "pair": 1,
                  "dialogues": ["casino-9", "casino-8"]}, "error": None},
        {"model": "muse-spark-1.3-contributor", "prompt_tokens": 1000,
         "completion_tokens": 500,
         "meta": {"layer": "moves", "item": "casino-1"}, "error": None},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines))
    got = SC.board_cost(str(p), ["casino-1", "casino-2"])
    assert got["calls"] == 1
    assert got["usd"] == pytest.approx(0.0002)
    assert got["usd_per_dialogue"] == pytest.approx(0.0001)
    both = SC.board_cost(str(p), None)
    assert both["calls"] == 2
    board_only = SC.board_cost(str(p), None, layers=("board",))
    assert board_only["calls"] == 0
