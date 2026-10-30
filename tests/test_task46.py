"""Contract for SPEC_task46_perturb (synthetic only, fake clients)."""
import json

import pytest

from forecaster import board as B
from forecaster import pairwise as P
from forecaster import perturb as PT
from forecaster.trace import CallLog, LoggingClient


def _eff(idx, speaker, value, text):
    return {"index": idx, "speaker": speaker, "text": text,
            "value": value, "n_readings": 3, "listeners": ["X"]}


def _record(cid, effects, genre="negotiation", scale="pairwise"):
    moves = [{"turn_index": e["index"], "speaker": e["speaker"],
              "moves": ["inform"], "addressed_to": "",
              "quote": "", "failed": False} for e in effects]
    return {"id": cid, "genre": genre, "scale": scale, "model": "src",
            "moves": moves, "effects": effects, "board": effects,
            "hits": [], "gold": None}


class _Fixed:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def complete(self, messages, model, temperature):
        self.calls += 1
        return self.raw


class _Queued:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, messages, model, temperature):
        self.calls.append(messages)
        if not self.outputs:
            return ""
        return self.outputs.pop(0)


# --- reuse ---------------------------------------------------------------

def test_reuses_pairwise_helpers():
    assert PT.PW.parse_pairwise is P.parse_pairwise
    assert PT.PW.pairwise_panel is P.pairwise_panel
    assert PT.PW.derandomize is P.derandomize
    assert PT.PW.STANCES == P.STANCES
    assert PT.board._call_board is B._call_board


def test_rewrite_prompt_auditable():
    assert isinstance(PT.REWRITE_PROMPT, str)
    assert "RUDE" in PT.REWRITE_PROMPT
    assert "POLITE" in PT.REWRITE_PROMPT
    assert "offer" in PT.REWRITE_PROMPT
    assert "numbers" in PT.REWRITE_PROMPT
    assert "one short paragraph" in PT.REWRITE_PROMPT.lower()
    msgs = PT.build_rewrite_messages("hello world test offer 5 deal", "RUDE", "A")
    assert msgs[0]["content"] == PT.REWRITE_PROMPT
    assert "RUDE" in msgs[1]["content"]


# --- select ---------------------------------------------------------------

def _select_fixture(tmp_path):
    rec_dir = tmp_path / "records"
    rec_dir.mkdir()
    long = "this is a sufficiently long turn with many words for testing"
    effects = [
        _eff(0, "A", 0.0, "hi"),  # short: skipped
        _eff(1, "A", 0.1, f"{long} one"),
        _eff(2, "B", 0.5, f"{long} two"),
        _eff(3, "A", 1.0, f"{long} three"),
        _eff(4, "B", 2.0, f"{long} four"),
        _eff(5, "A", 5.0, f"{long} five"),
    ]
    (rec_dir / "casino-1.json").write_text(
        json.dumps(_record("casino-1", effects)))
    table_p = tmp_path / "table.json"
    table_p.write_text(json.dumps({"scale": "pairwise", "mu": 0.0, "sd": 1.0,
                                   "table": {}, "sd_resid": 1.0}))
    ids_p = tmp_path / "ids.txt"
    ids_p.write_text("casino-1\n")
    return rec_dir, table_p, ids_p


def test_select_picks_median_abs_z(tmp_path):
    rec_dir, table_p, ids_p = _select_fixture(tmp_path)
    out = tmp_path / "out"
    PT.main(["select", "--records", str(rec_dir), "--table", str(table_p),
             "--ids-file", str(ids_p), "--n", "20", "--out", str(out)])
    targets = json.loads((out / "targets.json").read_text())
    assert len(targets) == 1
    t = targets[0]
    assert t["dialogue"] == "casino-1"
    # eligible |z|: 0.1,0.5,1.0,2.0,5.0 -> median 1.0 at turn 3
    assert t["turn"] == 3
    assert t["z"] == pytest.approx(1.0)
    assert set(("dialogue", "turn", "speaker", "text", "z")) <= set(t)


def test_select_tie_earlier_and_first_n(tmp_path):
    rec_dir = tmp_path / "records"
    rec_dir.mkdir()
    long = "another long turn with plenty of words for the tie test"
    # |z| all 1.0 -> sorted by turn, median of 3 is turn 1
    effects = [_eff(i, "A" if i % 2 == 0 else "B", 1.0 if i % 2 == 0 else -1.0,
                    f"{long} {i}") for i in range(3)]
    (rec_dir / "casino-1.json").write_text(
        json.dumps(_record("casino-1", effects)))
    (rec_dir / "casino-2.json").write_text(
        json.dumps(_record("casino-2", effects)))
    table_p = tmp_path / "table.json"
    table_p.write_text(json.dumps({"mu": 0.0, "sd": 1.0}))
    ids_p = tmp_path / "ids.txt"
    ids_p.write_text("casino-1\ncasino-2\n")
    out = tmp_path / "out"
    PT.main(["select", "--records", str(rec_dir), "--table", str(table_p),
             "--ids-file", str(ids_p), "--n", "1", "--out", str(out)])
    targets = json.loads((out / "targets.json").read_text())
    assert [t["dialogue"] for t in targets] == ["casino-1"]
    assert targets[0]["turn"] == 1


# --- write ----------------------------------------------------------------

def test_write_stores_both_variants(tmp_path):
    targets = [{"dialogue": "casino-1", "turn": 2, "speaker": "A",
                "text": "I offer five for your three, please consider it",
                "z": 0.1}]
    out = tmp_path / "out"
    out.mkdir()
    fake = _Queued(["rude rewrite here", "polite rewrite here"])
    client = LoggingClient(fake, CallLog(out / "calls.jsonl"), "t")
    variants = PT.write_variants(targets, out, client)
    assert len(variants) == 1
    v = variants[0]
    assert v["orig"] == targets[0]["text"]
    assert v["rude"] == "rude rewrite here"
    assert v["polite"] == "polite rewrite here"
    stored = json.loads((out / "variants.json").read_text())
    assert stored == variants
    calls = (out / "calls.jsonl").read_text().splitlines()
    assert len(calls) == 2
    metas = [json.loads(line)["meta"] for line in calls]
    assert [m["style"] for m in metas] == ["RUDE", "POLITE"]
    assert all(m["layer"] == "perturb-write" for m in metas)


def test_write_handles_null_after_retry(tmp_path):
    targets = [{"dialogue": "casino-1", "turn": 2, "speaker": "A",
                "text": "I offer five for your three, please consider it",
                "z": 0.1}]
    out = tmp_path / "out"
    out.mkdir()
    # rude fails twice (empty), polite succeeds
    fake = _Queued(["", "  ", "polite rewrite here"])
    client = LoggingClient(fake, CallLog(out / "calls.jsonl"), "t")
    variants = PT.write_variants(targets, out, client)
    assert variants[0]["rude"] is None
    assert variants[0]["polite"] == "polite rewrite here"
    calls = (out / "calls.jsonl").read_text().splitlines()
    assert len(calls) == 3


# --- vote -----------------------------------------------------------------

def _vote_fixture():
    long = "this turn has more than enough words for context testing"
    effects = [_eff(0, "A", 0.2, f"{long} zero"),
               _eff(1, "B", -0.1, f"{long} one"),
               _eff(2, "A", 0.3, f"{long} two target here")]
    rec = _record("casino-1", effects)
    variants = [{"dialogue": "casino-1", "turn": 2,
                 "orig": "I offer five for three please",
                 "rude": "Give me five for three now",
                 "polite": "Thanks, could I please offer five for three?"}]
    return variants, {"casino-1": rec}


def test_vote_builds_presentations_and_records_position(tmp_path):
    variants, records = _vote_fixture()
    out = tmp_path / "out"
    out.mkdir()
    client = LoggingClient(
        _Fixed('{"better": "first", "why": "warmer"}'),
        CallLog(out / "calls.jsonl"), "t")
    run, _pres = PT.run_vote(variants, records, out, client, workers=2)
    assert run["n_presentations"] == 6
    assert run["n_expected"] == 18
    assert run["n_votes"] == 18
    assert run["parse_fail_rate"] == pytest.approx(0.0)
    assert run["position_bias_share"] == pytest.approx(1.0)
    lines = [json.loads(line) for line in
             (out / "votes.jsonl").read_text().splitlines()]
    assert len(lines) == 6
    by_pair: dict[str, list] = {}
    for line in lines:
        assert set(("id", "dialogue", "turn", "pair", "a", "b", "first",
                    "votes", "raw")) <= set(line)
        assert line["first"] in ("a", "b")
        assert set(line["votes"]) == {"neutral", "trusting", "suspicious"}
        assert set(line["raw"]) == {"neutral", "trusting", "suspicious"}
        # all raw first -> derandomized equals presentation order
        for s in ("neutral", "trusting", "suspicious"):
            assert line["raw"][s] == "first"
            assert line["votes"][s] == line["first"]
        by_pair.setdefault(line["pair"], []).append(line["first"])
    assert set(by_pair) == {"orig_rude", "orig_polite", "rude_polite"}
    for firsts in by_pair.values():
        assert sorted(firsts) == ["a", "b"]
    calls = (out / "calls.jsonl").read_text().splitlines()
    assert len(calls) == 18


def test_vote_skips_null_variants(tmp_path):
    variants, records = _vote_fixture()
    variants = list(variants) + [{"dialogue": "casino-1", "turn": 1,
                                  "orig": "x", "rude": None,
                                  "polite": "y"}]
    pres, _info = PT.build_presentations(variants, records)
    assert len(pres) == 6
    assert all(p["turn"] == 2 for p in pres)


def test_vote_prompt_shows_shared_context_once():
    rec = P.pairwise_panel(0)[0]
    msgs = PT.build_perturb_messages(["A: hello there friend",
                                      "B: hi back to you"],
                                     "A", "orig text", "rude text",
                                     rec, "B")
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert "Candidate A (A): orig text" in user
    assert "Candidate B (A): rude text" in user
    # shared context appears once, not twice
    assert user.count("A: hello there friend") == 1
    assert '"better": "first" | "second" | "tie"' in user
    assert P.parse_pairwise('{"better": "tie", "why": "x"}') is not None


# --- report ---------------------------------------------------------------

def _report_fixture(tmp_path, n_correct=17, n_total=20):
    variants = []
    for i in range(n_total):
        did = f"casino-{i + 1}"
        orig = f"Original offer text for {did} with numbers 5 and ask"
        if i == 0:
            orig = "x" * 200
        variants.append({"dialogue": did, "turn": 2, "orig": orig,
                         "rude": f"Rude blunt refusal for {did} numbers 5",
                         "polite": f"Polite thanks hedge for {did} numbers 5"})
    lines = []
    pid = 0
    for i in range(n_total):
        did = f"casino-{i + 1}"
        correct = i < n_correct
        for pair_id in PT.PAIR_IDS:
            a_var, b_var = PT.CANONICAL[pair_id]
            pred = PT.PREDICTED[pair_id]
            winner = pred if correct else (
                b_var if pred == a_var else a_var)
            wab = "a" if winner == a_var else "b"
            for first in ("a", "b"):
                raw_vote = "first" if wab == first else "second"
                lines.append({
                    "id": pid, "dialogue": did, "turn": 2,
                    "pair": pair_id, "a": a_var, "b": b_var,
                    "first": first,
                    "votes": {s: wab for s in
                              ("neutral", "trusting", "suspicious")},
                    "raw": {s: raw_vote for s in
                            ("neutral", "trusting", "suspicious")},
                })
                pid += 1
    var_p = tmp_path / "variants.json"
    votes_p = tmp_path / "votes.jsonl"
    var_p.write_text(json.dumps(variants))
    votes_p.write_text("\n".join(json.dumps(line) for line in lines))
    return var_p, votes_p


def test_report_known_answer(tmp_path):
    var_p, votes_p = _report_fixture(tmp_path, n_correct=17, n_total=20)
    out_md = tmp_path / "report.md"
    PT.main(["report", "--votes", str(votes_p), "--variants", str(var_p),
             "--out", str(out_md)])
    # recompute via the pure helper for exact assertions
    votes = [json.loads(line) for line in votes_p.read_text().splitlines()]
    variants = json.loads(var_p.read_text())
    m = PT.compute_report_metrics(votes, variants)
    assert m["n_targets"] == 20
    assert m["ordering_correct"] == 17
    assert m["ordering_accuracy"] == pytest.approx(0.85)
    assert m["per_pair_correct"] == {"orig_rude": 17, "orig_polite": 17,
                                     "rude_polite": 17}
    assert m["mean_margin"] == pytest.approx(0.7)
    assert m["position_bias"] == pytest.approx(0.5)
    assert m["verdict"] == "directionally sensitive"
    md = out_md.read_text()
    assert "directionally sensitive" in md
    assert "0.850" in md
    assert "17/20" in md
    rows = [line for line in md.splitlines() if line.startswith("| casino-")]
    assert len(rows) == 20
    assert "x" * 120 in md
    assert "x" * 121 not in md


def test_report_insensitive_below_threshold(tmp_path):
    var_p, votes_p = _report_fixture(tmp_path, n_correct=5, n_total=20)
    votes = [json.loads(line) for line in votes_p.read_text().splitlines()]
    variants = json.loads(var_p.read_text())
    m = PT.compute_report_metrics(votes, variants)
    assert m["ordering_accuracy"] == pytest.approx(0.25)
    assert m["verdict"] == "insensitive at single-turn edits"
