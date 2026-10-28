"""Contract for SPEC_task19_transfer_outcome (synthetic data only)."""
import json

import pytest

from forecaster import batch as BA
from forecaster import calibrate as C
from forecaster import casino as CAS
from forecaster import outcome as O
from forecaster import predict as P
from forecaster.board import actual_class
from forecaster.labels import NEXT_ACTIONS


def _item(cid, moves, hits, effects=None, genre="negotiation"):
    return {"id": cid, "genre": genre, "moves": moves, "hits": hits,
            "effects": effects or [], "gold": None}


def test_cur_class_mapping_assumption():
    assert actual_class(["propose"]) == "negotiate"
    assert actual_class(["reject"]) == "refuse"


def _markov_items(n, start=0):
    items = {}
    for i in range(n):
        cid = f"casino-{start + i}"
        moves = [
            {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
            {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
            {"turn_index": 2, "speaker": "A", "moves": ["propose"]},
        ]
        hits = [
            {"index": 0, "actual_class": "comply",
             "action_dist": {"proceed": 1.0}, "n_readings": 3},
            {"index": 1, "actual_class": "negotiate",
             "action_dist": {"proceed": 1.0}, "n_readings": 3},
        ]
        items[cid] = _item(cid, moves, hits)
    return items


def _noise_items(n, start=0, per=6):
    items = {}
    for i in range(n):
        cid = f"casino-{start + i}"
        moves = [{"turn_index": t, "speaker": "AB"[t % 2],
                  "moves": ["propose" if t % 2 == 0 else "reject"]}
                 for t in range(per + 1)]
        hits = [{"index": t, "actual_class": NEXT_ACTIONS[(i + t) % len(NEXT_ACTIONS)],
                 "action_dist": {"proceed": 1.0}, "n_readings": 3}
                for t in range(per)]
        items[cid] = _item(cid, moves, hits)
    return items


def test_transfer_markov_informative_b2_supported():
    fit = _markov_items(10, 0)
    ev = _markov_items(10, 100)
    ev["yt-1"] = _item("yt-1", [], [], genre="youtube")  # filtered by genre
    res = P.transfer(fit, ev, b=200, seed=0)
    assert res["n_fit_items"] == 10
    assert res["n_eval_items"] == 10
    assert res["n_eval_hits"] == 20
    assert set(res["models"]) == {"raw_panel", "base", "markov", "panel_base",
                                  "panel_markov", "pool"}
    assert set(res["fitted"]) == {"lam_base", "lam_markov", "a", "b"}
    assert set(res["contrasts"]) == {"B1_panel_base_vs_raw", "B2_markov_vs_base",
                                     "B3_pool_vs_markov", "B4_panel_base_vs_markov"}
    b2 = res["contrasts"]["B2_markov_vs_base"]
    assert b2["supported"] is True
    assert b2["diff"][1] > 0
    assert res["models"]["markov"]["hit_rate"] == pytest.approx(1.0)


def test_transfer_panel_noise_b3_not_supported():
    fit = _noise_items(20, 0)
    ev = _noise_items(20, 100)
    res = P.transfer(fit, ev, b=200, seed=0)
    assert res["fitted"]["b"] == 0.0  # useless panel gets zero weight
    b3 = res["contrasts"]["B3_pool_vs_markov"]
    assert b3["supported"] is False
    assert b3["diff"][1] <= 0


def _write_items(d, items):
    d.mkdir(parents=True, exist_ok=True)
    for cid, item in items.items():
        (d / f"{cid}.json").write_text(json.dumps(item))


def test_transfer_cli_writes_files(tmp_path):
    fit_d, ev_d = tmp_path / "fit", tmp_path / "eval"
    _write_items(fit_d, _noise_items(6, 0))
    _write_items(ev_d, _noise_items(6, 100))
    P.transfer_main(["--fit", str(fit_d), "--eval", str(ev_d), "--b", "50"])
    payload = json.loads((ev_d / "transfer.json").read_text())
    assert set(payload["models"]) == {"raw_panel", "base", "markov", "panel_base",
                                      "panel_markov", "pool"}
    assert "B3_pool_vs_markov" in payload["contrasts"]
    md = (ev_d / "transfer.md").read_text()
    assert "pool" in md and "B3_pool_vs_markov" in md
    assert ("SUPPORTED" in md and "NOT SUPPORTED" in md) or "SUPPORTED" in md


def test_is_skipped_covers_task19_files():
    for name in ("transfer.json", "outcome.json", "predict.json",
                 "commentary.json", "commentary_2.json"):
        assert C._is_skipped(name) is True
    assert C._is_skipped("casino-1.json") is False


def test_dialogue_id_of():
    assert O.dialogue_id_of("casino-123") == 123
    assert O.dialogue_id_of("casino-0") == 0
    assert O.dialogue_id_of("yt-1") is None
    assert O.dialogue_id_of("casino-") is None
    assert O.dialogue_id_of("casino-abc") is None
    assert O.dialogue_id_of("") is None
    assert O.dialogue_id_of(None) is None


def _toy_casino_entry(did, sat, like, points_a=10, points_b=20, annotated=True):
    entry = {
        "dialogue_id": did,
        "participant_info": {
            "mturk_agent_1": {"outcomes": {"points_scored": points_a,
                                           "satisfaction": sat,
                                           "opponent_likeness": like}},
            "mturk_agent_2": {"outcomes": {"points_scored": points_b,
                                           "satisfaction": sat,
                                           "opponent_likeness": like}},
        },
        "chat_logs": [{"id": "mturk_agent_1", "text": "hi"},
                      {"id": "mturk_agent_2", "text": "yo"}],
    }
    entry["annotations"] = [["hi", "small-talk"]] if annotated else []
    return entry


def test_outcome_likert_mapping_all_five(tmp_path):
    sats = ["Extremely dissatisfied", "Slightly dissatisfied", "Undecided",
            "Slightly satisfied", "Extremely satisfied"]
    likes = ["Extremely dislike", "Slightly dislike", "Undecided",
             "Slightly like", "Extremely like"]
    data = [_toy_casino_entry(i + 1, s, like) for i, (s, like) in enumerate(zip(sats, likes))]
    data.append(_toy_casino_entry(99, "Super happy", "Undecided"))  # unknown -> skip
    cp = tmp_path / "casino.json"
    cp.write_text(json.dumps(data))
    out = O.load_outcomes(cp)
    assert 99 not in out
    for i, want in enumerate([-2, -1, 0, 1, 2]):
        rec = out[i + 1]
        assert rec["A"] == {"points": 10, "satisfaction": want, "likeness": want}
        assert rec["B"] == {"points": 20, "satisfaction": want, "likeness": want}


def _four_turn_item():
    return {
        "id": "casino-1", "genre": "negotiation",
        "moves": [
            {"turn_index": 0, "speaker": "A", "moves": ["empathize", "propose"]},
            {"turn_index": 1, "speaker": "B", "moves": ["pressure"]},
            {"turn_index": 2, "speaker": "A", "moves": ["inform"]},
            {"turn_index": 3, "speaker": "B", "moves": ["praise", "reject"]},
        ],
        "effects": [
            {"speaker": "A", "value": 1.0},
            {"speaker": "A", "value": 2.0},
            {"speaker": "B", "value": -1.0},
            {"speaker": "B", "value": -3.0},
        ],
        "hits": [], "gold": None,
    }


def test_reception_sign_and_side_split():
    item = _four_turn_item()
    rec = O.reception(item, mu=0.5)
    assert rec["A"] == pytest.approx(1.0)   # mean(1-.5, 2-.5)
    assert rec["B"] == pytest.approx(-2.5)  # mean(-1-.5, -3-.5)
    assert rec["A"] > 0 > rec["B"]
    assert O.dyad_value(item, mu=0.5) == pytest.approx(-0.75)
    assert O.reception(_item("casino-9", [], []), mu=0.0) == {"A": None, "B": None}
    assert O.dyad_value(_item("casino-9", [], []), mu=0.0) is None


def test_flagged_tag_turns():
    item = _four_turn_item()
    flags = O.flagged(item, mu=0.5, sd=1.0)
    assert flags == {"A": False, "B": True}  # B has (-3-.5) <= -2
    assert O.flagged(item, mu=0.5, sd=0.0) == {"A": False, "B": False}
    tags = O.tag_score(item)
    assert tags["A"] == pytest.approx(0.5)   # 1/2 warm, 0 hostile
    assert tags["B"] == pytest.approx(-0.5)  # 1/2 warm, 2/2 hostile
    assert O.turn_count(item) == {"A": 2, "B": 2}
    assert O.tag_score(_item("casino-9", [], [])) == {"A": None, "B": None}


def test_boot_spearman_monotone():
    rows = [{"dialogue": i, "x": i, "y": 2 * i} for i in range(5)]
    pt, lo, hi = O.boot_spearman(rows, "x", "y", b=50, seed=0)
    assert pt == pytest.approx(1.0)
    assert lo <= pt <= hi
    assert O.boot_spearman(rows[:2], "x", "y") == [0.0, 0.0, 0.0]


def test_boot_spearman_diff_and_mean_diff_sign():
    rows = [{"dialogue": i, "good": float(i), "bad": 0.0, "y": float(i)}
            for i in range(6)]
    d = O.boot_spearman_diff(rows, "good", "bad", "y", b=50, seed=0)
    assert d[0] == pytest.approx(1.0)  # 1.0 - 0.0
    mrows = [{"dialogue": i, "flagged": i < 3, "y": 2.0 if i < 3 else 0.0}
             for i in range(6)]
    m = O.boot_mean_diff(mrows, "flagged", "y", b=50, seed=0)
    assert m[0] == pytest.approx(2.0)
    assert m[0] > 0


def _mono_outcome_data():
    fit, ev, oc = {}, {}, {}
    for i in range(1, 4):
        eff = [{"speaker": "A", "value": 0.5}, {"speaker": "B", "value": -0.5}]
        fit[f"casino-{i}"] = _item(f"casino-{i}", [], [], eff)
    for i in range(1, 7):
        eff = [{"speaker": "A", "value": float(i)}, {"speaker": "A", "value": float(i)},
               {"speaker": "B", "value": float(i)}, {"speaker": "B", "value": float(i)}]
        mv = [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
              {"turn_index": 1, "speaker": "B", "moves": ["propose"]}]
        ev[f"casino-{i}"] = _item(f"casino-{i}", mv, [], eff)
        oc[i] = {"A": {"points": i, "satisfaction": i, "likeness": i},
                 "B": {"points": i, "satisfaction": i, "likeness": i}}
    return fit, ev, oc


def test_analyze_monotone_c1_supported():
    fit, ev, oc = _mono_outcome_data()
    res = O.analyze(fit, ev, oc, b=200, seed=0)
    assert res["mu"] == pytest.approx(0.0)
    assert res["sd"] == pytest.approx(0.5)
    assert res["n_dialogues"] == 6
    assert res["n_participants"] == 12
    assert res["n_flagged"] == 0
    for key in ("C1", "C2", "C3", "C4_tag", "C4_turns", "C4_board_minus_tag", "C5"):
        assert len(res[key]) == 3
    assert res["C1"][0] == pytest.approx(1.0)
    assert res["verdicts"]["C1"] == "SUPPORTED"
    assert res["verdicts"]["C2"] == "SUPPORTED"
    assert res["verdicts"]["C3"] == "REPORTED"
    assert res["verdicts"]["C5"] == "REPORTED"


def test_outcome_cli_writes_files(tmp_path):
    fit, ev, _ = _mono_outcome_data()
    fit_d, ev_d, ev2_d = tmp_path / "fit", tmp_path / "ev", tmp_path / "ev2"
    _write_items(fit_d, fit)
    _write_items(ev_d, ev)
    _write_items(ev2_d, dict(list(ev.items())[:4]))
    likes = ["Extremely dislike", "Slightly dislike", "Undecided",
             "Slightly like", "Extremely like"]
    sats = ["Extremely dissatisfied", "Slightly dissatisfied", "Undecided",
            "Slightly satisfied", "Extremely satisfied"]
    data = [_toy_casino_entry(i + 1, sats[i % 5], likes[i % 5]) for i in range(6)]
    cp = tmp_path / "casino.json"
    cp.write_text(json.dumps(data))
    O.main(["--fit", str(fit_d), "--eval", str(ev_d), str(ev2_d),
            "--casino", str(cp), "--b", "50"])
    for d in (ev_d, ev2_d):
        payload = json.loads((d / "outcome.json").read_text())
        assert "C1" in payload and "verdicts" in payload
        md = (d / "outcome.md").read_text()
        assert "C1" in md and "C4_board_minus_tag" in md


def _mixed_casino_file(tmp_path):
    data = [
        {"dialogue_id": 1,
         "chat_logs": [{"id": "mturk_agent_1", "text": "hi"},
                       {"id": "mturk_agent_2", "text": "yo"}],
         "annotations": [["hi", "small-talk"], ["yo", "small-talk"]]},
        {"dialogue_id": 2,
         "chat_logs": [{"id": "mturk_agent_1", "text": "hey"},
                       {"id": "mturk_agent_2", "text": "Accept-Deal"}],
         "annotations": []},
        {"dialogue_id": 3,
         "chat_logs": [{"id": "mturk_agent_1", "text": "a"}],
         "annotations": []},
    ]
    cp = tmp_path / "casino.json"
    cp.write_text(json.dumps(data))
    return cp


def test_load_casino_unannotated(tmp_path):
    cp = _mixed_casino_file(tmp_path)
    assert [d["dialogue_id"] for d in CAS.load_casino(cp)] == [1]  # default unchanged
    res = CAS.load_casino(cp, annotated=False)
    assert sorted(d["dialogue_id"] for d in res) == [2, 3]
    assert all(d["gold"] is None for d in res)
    d2 = next(d for d in res if d["dialogue_id"] == 2)
    assert [(t.speaker, t.text) for t in d2["turns"]] == [("A", "hey")]  # DEAL cut
    assert [d["dialogue_id"] for d in
            CAS.load_casino(cp, annotated=False, exclude_ids=[2])] == [3]


def test_manifest_unannotated(tmp_path, capsys):
    cp = _mixed_casino_file(tmp_path)
    BA.manifest_main(["--casino", str(cp), "--n", "10", "--unannotated"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert sorted(i["id"] for i in lines) == ["casino-2", "casino-3"]
    assert all(i["gold"] is False for i in lines)
    assert all(i["genre"] == "negotiation" and i["kind"] == "casino" for i in lines)
    BA.manifest_main(["--casino", str(cp), "--n", "10"])
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [i["id"] for i in lines] == ["casino-1"]
    assert all("gold" not in i for i in lines)  # default unchanged


class _FakeClient:
    def complete(self, messages, model, temperature, reasoning_effort=None):
        text = json.dumps(messages)
        if "relationship_shift" in text:
            return ('{"next_action":"proceed","relationship_shift":0,'
                    '"face_threat":0,"understood":"ok"}')
        return '{"moves":["smalltalk"],"addressed_to":"all","quote":""}'


def test_run_batch_none_gold(tmp_path):
    cp = _mixed_casino_file(tmp_path)
    items = [
        {"id": "casino-1", "genre": "negotiation", "kind": "casino",
         "dialogue_id": 1, "public": True},
        {"id": "casino-2", "genre": "negotiation", "kind": "casino",
         "dialogue_id": 2, "public": True, "gold": False},
    ]
    out = tmp_path / "out"
    summaries = BA.run_batch(items, _FakeClient(), out, casino_path=cp,
                             workers=1, n=1, k=1, inner_workers=1)
    by_id = {s["id"]: s for s in summaries}
    assert by_id["casino-2"]["gold"] is None
    assert by_id["casino-1"]["gold"] is not None
    record = json.loads((out / "casino-2.json").read_text())
    assert record["gold"] is None
    rep = BA.genre_report(summaries, b=50, seed=0)
    assert rep["negotiation"]["n_conv"] == 2
    assert "micro_f1" in rep["negotiation"]  # F1 over gold items only


def _real_shape_casino_data():
    return [
        {"dialogue_id": 0,
         "participant_info": {
             "mturk_agent_1": {"outcomes": {"points_scored": 19,
                                            "satisfaction": "Slightly satisfied",
                                            "opponent_likeness": "Slightly like"}},
             "mturk_agent_2": {"outcomes": {"points_scored": 18,
                                            "satisfaction": "Extremely satisfied",
                                            "opponent_likeness": "Extremely like"}}},
         "chat_logs": [{"id": "mturk_agent_1", "text": "hi"},
                       {"id": "mturk_agent_2", "text": "yo"}],
         "annotations": [["hi", "small-talk"]]},
        {"dialogue_id": 1,
         "participant_info": {
             "mturk_agent_1": {"outcomes": {"points_scored": 15,
                                            "satisfaction": "Undecided",
                                            "opponent_likeness": "Undecided"}},
             "mturk_agent_2": {"outcomes": {"points_scored": 20,
                                            "satisfaction": "Extremely dissatisfied",
                                            "opponent_likeness": "Extremely dislike"}}},
         "chat_logs": [{"id": "mturk_agent_1", "text": "hey"},
                       {"id": "mturk_agent_2", "text": "yo"}],
         "annotations": [["hey", "small-talk"]]},
    ]


def test_load_outcomes_real_shape(tmp_path):
    cp = tmp_path / "casino.json"
    cp.write_text(json.dumps(_real_shape_casino_data()))
    assert O.load_outcomes(cp)[0] == {
        "A": {"points": 19, "satisfaction": 1, "likeness": 1},
        "B": {"points": 18, "satisfaction": 2, "likeness": 2},
    }


def test_analyze_real_shape_e2e(tmp_path):
    cp = tmp_path / "casino.json"
    cp.write_text(json.dumps(_real_shape_casino_data()))
    outcomes = O.load_outcomes(cp)
    assert len(outcomes) == 2

    def _ev_item(cid, val):
        return {
            "id": cid, "genre": "negotiation",
            "moves": [
                {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
                {"turn_index": 1, "speaker": "B", "moves": ["propose"]},
            ],
            "effects": [{"speaker": "A", "value": val},
                        {"speaker": "B", "value": -val}],
            "hits": [], "gold": None,
        }

    fit = {"casino-0": _ev_item("casino-0", 0.5),
           "casino-1": _ev_item("casino-1", -0.5)}
    ev = {"casino-0": _ev_item("casino-0", 1.0),
          "casino-1": _ev_item("casino-1", 2.0)}
    res = O.analyze(fit, ev, outcomes, b=50, seed=0)
    assert res["n_participants"] > 0
    assert res["n_dialogues"] == 2
    rows = O.participant_rows(ev, outcomes, res["mu"], res["sd"])
    assert len(rows) > 0
