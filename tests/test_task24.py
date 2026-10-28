"""Contract for task24 tag-value baseline (synthetic data only)."""
import pytest

from forecaster import outcome as O


def _item(cid, moves, effects, genre="negotiation"):
    return {"id": cid, "genre": genre, "moves": moves, "hits": [],
            "effects": effects, "gold": None}


def test_move_value_table_means():
    fit = {
        "casino-1": _item("casino-1",
                          [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
                           {"turn_index": 1, "speaker": "B",
                            "moves": ["reject", "pressure"]}],
                          [{"index": 0, "speaker": "A", "value": 1.0},
                           {"index": 1, "speaker": "B", "value": -1.0}]),
        "casino-2": _item("casino-2",
                          [{"turn_index": 0, "speaker": "A", "moves": ["accept"]}],
                          [{"index": 0, "speaker": "A", "value": 3.0}]),
    }
    table = O.move_value_table(fit, mu=0.0)
    assert table["accept"] == pytest.approx(2.0)
    assert table["reject"] == pytest.approx(-1.0)
    assert table["pressure"] == pytest.approx(-1.0)
    # mu centering
    table_mu = O.move_value_table(fit, mu=0.5)
    assert table_mu["accept"] == pytest.approx(1.5)
    # list form works too
    table_list = O.move_value_table(list(fit.values()), mu=0.0)
    assert table_list == pytest.approx(table)


def test_move_value_table_first_entry_and_missing():
    fit = {
        "casino-1": _item("casino-1",
                          [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
                           {"turn_index": 0, "speaker": "A", "moves": ["reject"]},
                           {"turn_index": 1, "speaker": "B", "moves": []}],
                          [{"index": 0, "speaker": "A", "value": 2.0},
                           {"index": 1, "speaker": "B", "value": 5.0},
                           {"index": 9, "speaker": "B", "value": 7.0}]),
    }
    table = O.move_value_table(fit, mu=0.0)
    assert table["accept"] == pytest.approx(2.0)
    assert "reject" not in table
    # empty-moves and unmatched turns contribute to nothing else
    assert set(table) == {"accept"}
    # effects without index fall back to position
    fit2 = {"casino-2": _item("casino-2",
                              [{"turn_index": 0, "speaker": "A",
                                "moves": ["accept"]},
                               {"turn_index": 1, "speaker": "B",
                                "moves": ["reject"]}],
                              [{"speaker": "A", "value": 1.0},
                               {"speaker": "B", "value": -2.0}])}
    table2 = O.move_value_table(fit2, mu=0.0)
    assert table2["accept"] == pytest.approx(1.0)
    assert table2["reject"] == pytest.approx(-2.0)


def test_tag_value_means_and_none():
    table = {"accept": 2.0, "reject": -1.0}
    item = _item("casino-1",
                 [{"turn_index": 0, "speaker": "A",
                   "moves": ["accept", "reject"]},
                  {"turn_index": 1, "speaker": "A", "moves": []},
                  {"turn_index": 2, "speaker": "B", "moves": ["unknown-move"]}],
                 [])
    got = O.tag_value(item, table)
    assert got["A"] == pytest.approx(0.25)  # mean(0.5, 0.0)
    assert got["B"] == pytest.approx(0.0)  # unknown move -> 0
    # None when side has no entries
    only_a = _item("casino-2",
                   [{"turn_index": 0, "speaker": "A", "moves": ["accept"]}],
                   [])
    got2 = O.tag_value(only_a, table)
    assert got2["A"] == pytest.approx(2.0)
    assert got2["B"] is None
    assert O.tag_value(_item("casino-3", [], []), table) == {"A": None, "B": None}


def test_participant_rows_tag_value_field():
    items = {
        "casino-1": _item("casino-1",
                          [{"turn_index": 0, "speaker": "A",
                            "moves": ["accept"]},
                           {"turn_index": 1, "speaker": "B",
                            "moves": ["reject"]}],
                          [{"index": 0, "speaker": "A", "value": 1.0},
                           {"index": 1, "speaker": "B", "value": -1.0}]),
    }
    outcomes = {1: {"A": {"points": 10, "satisfaction": 0, "likeness": 1},
                    "B": {"points": 10, "satisfaction": 0, "likeness": -1}}}
    # default unchanged: no tag_value key
    rows = O.participant_rows(items, outcomes, 0.0, 1.0)
    assert rows and all("tag_value" not in r for r in rows)
    # with table: field present and correct
    table = {"accept": 2.0, "reject": -1.0}
    rows2 = O.participant_rows(items, outcomes, 0.0, 1.0, table=table)
    by_side = {r["side"]: r for r in rows2}
    assert by_side["A"]["tag_value"] == pytest.approx(2.0)
    assert by_side["B"]["tag_value"] == pytest.approx(-1.0)


def test_analyze_contains_c6_keys():
    fit, ev, oc = {}, {}, {}
    for i in range(1, 4):
        eff = [{"speaker": "A", "value": 0.5}, {"speaker": "B", "value": -0.5}]
        mv = [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
              {"turn_index": 1, "speaker": "B", "moves": ["reject"]}]
        fit[f"casino-{i}"] = _item(f"casino-{i}", mv, eff)
    for i in range(1, 7):
        eff = [{"speaker": "A", "value": float(i)}, {"speaker": "A", "value": float(i)},
               {"speaker": "B", "value": float(i)}, {"speaker": "B", "value": float(i)}]
        mv = [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
              {"turn_index": 1, "speaker": "B", "moves": ["reject"]}]
        ev[f"casino-{i}"] = _item(f"casino-{i}", mv, eff)
        oc[i] = {"A": {"points": i, "satisfaction": i, "likeness": i},
                 "B": {"points": i, "satisfaction": i, "likeness": i}}
    res = O.analyze(fit, ev, oc, b=50, seed=0)
    assert "C6_tag_value" in res
    assert "C6_board_minus_tag_value" in res
    assert len(res["C6_tag_value"]) == 3
    assert len(res["C6_board_minus_tag_value"]) == 3
    assert res["verdicts"]["C6_tag_value"] in ("SUPPORTED", "NOT SUPPORTED")
    assert res["verdicts"]["C6_board_minus_tag_value"] in ("SUPPORTED", "NOT SUPPORTED")


def test_outcome_cli_writes_c6_rows(tmp_path):
    import json

    fit_d, ev_d = tmp_path / "fit", tmp_path / "ev"
    fit_d.mkdir(parents=True, exist_ok=True)
    ev_d.mkdir(parents=True, exist_ok=True)
    mv = [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
          {"turn_index": 1, "speaker": "B", "moves": ["reject"]}]
    for i in range(1, 4):
        eff = [{"speaker": "A", "value": 0.5}, {"speaker": "B", "value": -0.5}]
        (fit_d / f"casino-{i}.json").write_text(
            json.dumps(_item(f"casino-{i}", mv, eff)))
        (ev_d / f"casino-{i}.json").write_text(
            json.dumps(_item(f"casino-{i}", mv, eff)))
    data = []
    for did in (1, 2, 3):
        data.append({
            "dialogue_id": did,
            "participant_info": {
                "mturk_agent_1": {"outcomes": {
                    "points_scored": 10, "satisfaction": "Undecided",
                    "opponent_likeness": "Undecided"}},
                "mturk_agent_2": {"outcomes": {
                    "points_scored": 10, "satisfaction": "Undecided",
                    "opponent_likeness": "Undecided"}},
            },
        })
    cp = tmp_path / "casino.json"
    cp.write_text(json.dumps(data))
    O.main(["--fit", str(fit_d), "--eval", str(ev_d),
            "--casino", str(cp), "--b", "20"])
    md = (ev_d / "outcome.md").read_text()
    assert "C6_tag_value" in md
    assert "C6_board_minus_tag_value" in md
