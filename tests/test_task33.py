"""Contract for SPEC_task33_residual_marks (synthetic data only)."""
import json

import pytest

from forecaster import calibrate as C
from forecaster import commentary as CM
from forecaster import outcome as O
from forecaster import valuemodel as V
from forecaster.viewer import build_view, render_html


def _eff(idx, speaker, value, n_readings=2, text=""):
    return {"index": idx, "speaker": speaker, "text": text or f"t{idx}",
            "value": value, "disagreement": 0.0,
            "n_readings": n_readings, "listeners": ["X"]}


def _item(cid, effects, moves, genre="negotiation"):
    return {"id": cid, "genre": genre, "moves": moves,
            "effects": effects, "hits": []}


# --- mark_for with z_resid -------------------------------------------------

def test_mark_for_residual_thresholds():
    assert CM.mark_for(0.0, False, 2.0) == "!!"
    assert CM.mark_for(0.0, False, 2.5) == "!!"
    assert CM.mark_for(0.0, False, 1.5) == "!"
    assert CM.mark_for(0.0, False, 1.0) == "!"
    assert CM.mark_for(0.0, False, 0.99) is None
    assert CM.mark_for(0.0, False, -0.99) is None
    assert CM.mark_for(0.0, False, -1.0) == "?"
    assert CM.mark_for(0.0, False, -1.5) == "?"
    assert CM.mark_for(0.0, False, -2.0) == "??"
    assert CM.mark_for(0.0, False, -2.5) == "??"
    assert CM.mark_for(0.0, True, 0.5) == "!?"
    assert CM.mark_for(0.0, True, -0.5) == "!?"
    assert CM.mark_for(0.0, False, 0.0) is None
    # z_resid decides; z is ignored once z_resid is given
    assert CM.mark_for(9.0, False, 0.0) is None
    assert CM.mark_for(0.0, False, 9.0) == "!!"
    assert CM.mark_for(1.5, True, 0.2) == "!?"
    assert CM.mark_for(0.0, False, "bogus") is None
    # legacy path unchanged
    assert CM.mark_for(2.5, False) == "!"
    assert CM.mark_for(-2.5, False) == "?"
    assert CM.mark_for(0.0, True) == "!?"


# --- planted turn: "!!" with residual norm, "!" with legacy norm -----------

def _planted():
    effects = [_eff(0, "A", 0.1), _eff(1, "B", 1.5)]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["inform"]},
        {"turn_index": 1, "speaker": "B", "moves": ["propose"]},
    ]
    return _item("r1", effects, moves)


def test_planted_turn_double_vs_single_mark():
    item = _planted()
    legacy = {"mu": 0.0, "sd": 1.0}
    got = CM.pick_turns(item, legacy, None, k=3)
    assert [t["mark"] for t in got] == ["!"]
    assert [t["index"] for t in got] == [1]
    assert "z_resid" not in got[0]
    assert "resid" not in got[0]
    assert "tagv" not in got[0]

    resid_norm = {"mu": 0.0, "sd": 1.0,
                  "table": {"inform": 0.0, "propose": 0.0},
                  "sd_resid": 0.5}
    tv = {e["index"]: e for e in CM.turn_values(item, resid_norm)}
    assert tv[1]["value"] == pytest.approx(1.5)
    assert tv[1]["z"] == pytest.approx(1.5)
    assert tv[1]["tagv"] == pytest.approx(0.0)
    assert tv[1]["resid"] == pytest.approx(1.5)
    assert tv[1]["z_resid"] == pytest.approx(3.0)
    got2 = CM.pick_turns(item, resid_norm, None, k=3)
    assert [t["index"] for t in got2] == [1]
    assert [t["mark"] for t in got2] == ["!!"]
    assert got2[0]["label_ko"] == "대묘수"
    assert got2[0]["label_en"] == "brilliant move"


def test_high_value_matching_table_gets_no_mark():
    item = _item("r2", [_eff(0, "A", 3.0)],
                 [{"turn_index": 0, "speaker": "A", "moves": ["hot"]}])
    resid_norm = {"mu": 0.0, "sd": 1.0,
                  "table": {"hot": 3.0}, "sd_resid": 0.5}
    tv = CM.turn_values(item, resid_norm)
    assert tv[0]["z"] == pytest.approx(3.0)
    assert tv[0]["tagv"] == pytest.approx(3.0)
    assert tv[0]["resid"] == pytest.approx(0.0)
    assert tv[0]["z_resid"] == pytest.approx(0.0)
    assert CM.pick_turns(item, resid_norm, None, k=3) == []
    # the same turn is remarkable on the absolute scale
    got = CM.pick_turns(item, {"mu": 0.0, "sd": 1.0}, None, k=3)
    assert [t["mark"] for t in got] == ["!"]


def test_pick_turns_ranks_by_abs_z_resid():
    effects = [_eff(0, "A", 2.0), _eff(1, "B", 1.2)]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["m1"]},
        {"turn_index": 1, "speaker": "B", "moves": ["m2"]},
    ]
    item = _item("r3", effects, moves)
    resid_norm = {"mu": 0.0, "sd": 1.0,
                  "table": {"m1": 0.5, "m2": -1.0}, "sd_resid": 1.0}
    tv = {e["index"]: e for e in CM.turn_values(item, resid_norm)}
    assert tv[0]["z_resid"] == pytest.approx(1.5)
    assert tv[1]["z_resid"] == pytest.approx(2.2)
    got = CM.pick_turns(item, resid_norm, None, k=1)
    assert [(t["index"], t["mark"]) for t in got] == [(1, "!!")]
    # legacy norm ranks by |z|: turn 0 (|z|=2.0) beats turn 1 (|z|=1.2)
    got_legacy = CM.pick_turns(item, {"mu": 0.0, "sd": 1.0}, None, k=1)
    assert [(t["index"], t["mark"]) for t in got_legacy] == [(0, "!")]


# --- residual_norm ----------------------------------------------------------

def _fit_items():
    return {
        "f1": _item("f1",
                    [_eff(0, "A", 2.0),
                     _eff(1, "B", 0.0),
                     _eff(2, "A", 99.0, n_readings=0)],
                    [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
                     {"turn_index": 1, "speaker": "B", "moves": ["accept"]},
                     {"turn_index": 2, "speaker": "A", "moves": ["accept"]}]),
    }


def test_residual_norm_toy():
    items = _fit_items()
    table = {"accept": 1.0}
    got = C.residual_norm(items, 0.0, table)
    assert set(got) == {"sd_resid", "n"}
    # residuals [1.0, -1.0]; the n_readings=0 turn is skipped
    assert got["n"] == 2
    assert got["sd_resid"] == pytest.approx(1.0)
    # same terms through the shared helper
    by_turn = {0: ["accept"], 1: ["accept"], 2: ["accept"]}
    terms = []
    for pos, e in enumerate(items["f1"]["effects"]):
        r = V.turn_residual(e, e["index"], by_turn, 0.0, table)
        if r is not None:
            terms.append(r)
    assert terms == pytest.approx([1.0, -1.0])
    # list input matches dict input; mu is honoured
    assert C.residual_norm(list(items.values()), 0.0, table) == got
    assert C.residual_norm(items, 1.0, {"accept": 0.0}) == got


def test_residual_norm_degenerate():
    flat = {"f1": _item("f1",
                        [_eff(0, "A", 2.0), _eff(1, "B", 2.0)],
                        [{"turn_index": 0, "speaker": "A", "moves": ["a"]},
                         {"turn_index": 1, "speaker": "B", "moves": ["a"]}])}
    got = C.residual_norm(flat, 0.0, {"a": 1.0})
    assert got["n"] == 2
    assert got["sd_resid"] == pytest.approx(1.0)
    assert C.residual_norm({}, 0.0, {}) == {"sd_resid": 1.0, "n": 0}
    assert C.residual_norm([], 0.0, {}) == {"sd_resid": 1.0, "n": 0}
    unread = {"f1": _item("f1",
                          [_eff(0, "A", 2.0, n_readings=0)],
                          [{"turn_index": 0, "speaker": "A",
                            "moves": ["a"]}])}
    assert C.residual_norm(unread, 0.0, {"a": 1.0}) == {"sd_resid": 1.0,
                                                      "n": 0}


# --- commentary.main writes the residual norm --------------------------------

def test_commentary_main_writes_residual_norm(tmp_path):
    fit = tmp_path / "fit"
    fit.mkdir()
    rec = _item("f1",
                [_eff(0, "A", 2.0)],
                [{"turn_index": 0, "speaker": "A", "moves": ["accept"]}])
    rec["scale"] = "v2"
    (fit / "f1.json").write_text(json.dumps(rec))
    item = _item("i1",
                 [_eff(0, "A", 2.0, text="a"), _eff(1, "B", 5.0, text="b")],
                 [{"turn_index": 0, "speaker": "A", "moves": ["accept"]},
                  {"turn_index": 1, "speaker": "B", "moves": ["propose"]}])
    item["scale"] = "v2"
    item_p = tmp_path / "item.json"
    item_p.write_text(json.dumps(item))
    out_p = tmp_path / "o.json"
    CM.main(["--item", str(item_p), "--fit", str(fit),
             "--out", str(out_p)])
    with open(out_p, encoding="utf-8") as f:
        result = json.load(f)
    norm = result["norm"]
    assert norm["mu"] == pytest.approx(2.0)
    assert norm["sd"] == pytest.approx(0.0)
    assert norm["scale"] == "v2"
    assert norm["table"] == {"accept": pytest.approx(0.0)}
    assert set(norm["table"]) <= {"accept"}
    assert norm["sd_resid"] == pytest.approx(1.0)
    assert norm["sd_resid"] == pytest.approx(
        C.residual_norm({"f1": rec}, 2.0, {"accept": 0.0})["sd_resid"])
    assert norm["table"] == O.move_value_table({"f1": rec}, 2.0)
    json.dumps(norm)  # JSON-serializable
    by_index = {t["index"]: t for t in result["turns"]}
    assert by_index[1]["mark"] == "!!"
    assert by_index[1]["z"] == pytest.approx(0.0)
    assert by_index[1]["tagv"] == pytest.approx(0.0)
    assert by_index[1]["resid"] == pytest.approx(3.0)
    assert by_index[1]["z_resid"] == pytest.approx(3.0)
    for t in result["turns"]:
        assert {"z", "z_resid", "tagv", "resid"} <= set(t)


# --- viewer renders "!!" and "??" --------------------------------------------

def _blob_data(html):
    tag = '<script type="application/json" id="data">'
    start = html.index(tag) + len(tag)
    return json.loads(html[start:html.index("</script>", start)])


def test_viewer_renders_double_marks():
    item = _item("casino-7",
                 [_eff(0, "A", 3.0, text="up"),
                  _eff(1, "B", -3.0, text="down")],
                 [{"turn_index": 0, "speaker": "A", "moves": ["up"]},
                  {"turn_index": 1, "speaker": "B", "moves": ["down"]}])
    norm = {"mu": 0.0, "sd": 1.0,
            "table": {"up": 0.0, "down": 0.0}, "sd_resid": 1.0}
    com = CM.comment(item, norm, None, k=3, client=None)
    assert com["norm"]["table"] == {"up": 0.0, "down": 0.0}
    assert com["norm"]["sd_resid"] == 1.0
    assert [(t["index"], t["mark"]) for t in com["turns"]] == \
        [(0, "!!"), (1, "??")]
    assert com["turns"][1]["label_ko"] == "대실수"
    assert com["turns"][1]["label_en"] == "blunder"
    view = build_view(item, com)
    assert [t["mark"] for t in view["turns"]] == ["!!", "??"]
    assert view["turns"][0]["resid"] == pytest.approx(3.0)
    assert view["turns"][0]["z_resid"] == pytest.approx(3.0)
    assert view["turns"][1]["resid"] == pytest.approx(-3.0)
    assert view["turns"][1]["z_resid"] == pytest.approx(-3.0)
    assert view["score"]["counts"] == {"!": 1, "?": 1, "!?": 0}
    html = render_html(view)
    assert 'data-mark="!!"' in html
    assert 'data-mark="??"' in html
    assert '.badge[data-mark="!!"]' in html
    assert '.badge[data-mark="??"]' in html
    for mark in ("!!", "!", "!?", "?", "??"):
        assert (f'<span class="badge" data-mark="{mark}">{mark}</span>'
                in html)
    assert "z_resid" in html
    data = _blob_data(html)
    assert data["turns"][0]["z_resid"] == pytest.approx(3.0)
    assert data["turns"][1]["z_resid"] == pytest.approx(-3.0)
    # legacy commentary leaves viewer turns without residual keys
    legacy_view = build_view(
        item, CM.comment(item, {"mu": 0.0, "sd": 1.0}, None, k=3))
    assert [t["mark"] for t in legacy_view["turns"]] == ["!", "?"]
    assert all("resid" not in t and "z_resid" not in t
               for t in legacy_view["turns"])
