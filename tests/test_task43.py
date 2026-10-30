"""Contract for SPEC_task43_pairwise_default (synthetic data only)."""
import json
import math

import pytest

from forecaster import batch as BA
from forecaster import commentary as CM
from forecaster import valuemodel as V
from forecaster import viewer2 as V2


def _eff(idx, speaker, value, text="", n_readings=3):
    return {"index": idx, "speaker": speaker, "text": text or f"t{idx}",
            "value": value, "disagreement": 0.0,
            "n_readings": n_readings, "listeners": ["X"]}


def _item(cid, effects, moves, scale="pairwise", genre="negotiation"):
    return {"id": cid, "genre": genre, "scale": scale,
            "moves": moves, "effects": effects, "hits": []}


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


# --- commentary on a pairwise record + pairwise table -----------------------

def _pw_table():
    return {"scale": "pairwise", "mu": 0.0, "sd": 0.6,
            "table": {"propose": 0.1, "counter": -0.1}, "sd_resid": 0.5}


def _pw_item():
    return _item("pw-1",
                 [_eff(0, "A", 0.9, text="a"),
                  _eff(1, "B", -0.8, text="b"),
                  _eff(2, "A", 0.2, text="c")],
                 [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
                  {"turn_index": 1, "speaker": "B", "moves": ["counter"]},
                  {"turn_index": 2, "speaker": "A", "moves": ["propose"]}])


def test_commentary_pairwise_table_dir(tmp_path):
    fit = tmp_path / "fit"
    fit.mkdir()
    _write(fit / "table.json", _pw_table())
    item_p = tmp_path / "item.json"
    _write(item_p, _pw_item())
    out_p = tmp_path / "o.json"
    CM.main(["--item", str(item_p), "--fit", str(fit),
             "--out", str(out_p)])
    result = json.loads(out_p.read_text(encoding="utf-8"))
    assert result["norm"]["scale"] == "pairwise"
    assert "scale_note" not in result
    assert result["standing_scale"] == "pairwise→0-100 via z"
    by_index = {t["index"]: t for t in result["turns"]}
    assert set(by_index) == {0, 1}
    assert by_index[0]["resid"] == pytest.approx(0.8)
    assert by_index[0]["z_resid"] == pytest.approx(1.6)
    assert by_index[0]["mark"] == "!"
    assert by_index[1]["resid"] == pytest.approx(-0.7)
    assert by_index[1]["z_resid"] == pytest.approx(-1.4)
    assert by_index[1]["mark"] == "?"
    for t in result["turns"]:
        assert {"z", "z_resid", "tagv", "resid"} <= set(t)
        assert 0.0 <= t["standing_before"] <= 100.0
        assert 0.0 <= t["standing_after"] <= 100.0
        assert "잔고" in t["line_ko"]
    assert by_index[0]["standing_after"] == pytest.approx(50 + 25 * 1.5)


def test_commentary_pairwise_records_fit(tmp_path):
    fit = tmp_path / "fit"
    fit.mkdir()
    _write(fit / "f1.json", _item(
        "f1", [_eff(0, "A", 0.5), _eff(1, "B", -0.5)],
        [{"turn_index": 0, "speaker": "A", "moves": ["inform"]},
         {"turn_index": 1, "speaker": "B", "moves": ["inform"]}],
        scale="pairwise"))
    _write(fit / "f2.json", _item(
        "f2", [_eff(0, "A", 0.3)],
        [{"turn_index": 0, "speaker": "A", "moves": ["inform"]}],
        scale="pairwise"))
    item_p = tmp_path / "item.json"
    _write(item_p, _pw_item())
    out_p = tmp_path / "o.json"
    CM.main(["--item", str(item_p), "--fit", str(fit),
             "--out", str(out_p)])
    result = json.loads(out_p.read_text(encoding="utf-8"))
    assert result["norm"]["scale"] == "pairwise"
    assert result["norm"]["mu"] == pytest.approx(0.1)
    assert result["norm"]["sd"] == pytest.approx(0.432, abs=0.01)
    assert "scale_note" not in result
    assert result["standing_scale"] == "pairwise→0-100 via z"


def test_pairwise_standing_clip():
    assert CM.pairwise_standing(0.0, 0.0, 0.6) == pytest.approx(50.0)
    assert CM.pairwise_standing(0.6, 0.0, 0.6) == pytest.approx(75.0)
    assert CM.pairwise_standing(5.0, 0.0, 0.6) == pytest.approx(100.0)
    assert CM.pairwise_standing(-5.0, 0.0, 0.6) == pytest.approx(0.0)
    assert CM.pairwise_standing(0.6, 0.0, 0.0) == pytest.approx(50.0)


# --- mismatched scales ------------------------------------------------------

def _fine100_fit(tmp_path):
    fit = tmp_path / "fit100"
    fit.mkdir()
    _write(fit / "f1.json", _item(
        "f1", [_eff(0, "A", 60.0), _eff(1, "B", 40.0)],
        [{"turn_index": 0, "speaker": "A", "moves": ["inform"]},
         {"turn_index": 1, "speaker": "B", "moves": ["inform"]}],
        scale="fine100"))
    return fit


def test_commentary_mismatched_scales_error(tmp_path):
    fit = _fine100_fit(tmp_path)
    item_p = tmp_path / "item.json"
    _write(item_p, _pw_item())  # pairwise item vs 0–100 fit
    with pytest.raises(SystemExit) as exc:
        CM.main(["--item", str(item_p), "--fit", str(fit),
                 "--out", str(tmp_path / "o.json")])
    assert "scale mismatch" in str(exc.value)
    assert "--same-scale" in str(exc.value)

    tab = tmp_path / "tab"
    tab.mkdir()
    _write(tab / "table.json", _pw_table())
    item2_p = tmp_path / "item2.json"
    _write(item2_p, _item(
        "i2", [_eff(0, "A", 60.0, text="a"), _eff(1, "B", 40.0, text="b")],
        [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
         {"turn_index": 1, "speaker": "B", "moves": ["counter"]}],
        scale="fine100"))
    with pytest.raises(SystemExit):
        CM.main(["--item", str(item2_p), "--fit", str(tab),
                 "--out", str(tmp_path / "o2.json")])


def test_commentary_same_scale_rank_path(tmp_path):
    tab = tmp_path / "tab"
    tab.mkdir()
    _write(tab / "table.json", {"scale": "fine100", "mu": 50.0, "sd": 25.0,
                                "table": {"m0": 10.0, "m1": 50.0,
                                          "m2": 90.0},
                                "sd_resid": 5.0})
    item = _item("r1",
                 [_eff(0, "A", 0.5, text="a"),
                  _eff(1, "B", -0.3, text="b"),
                  _eff(2, "A", 0.1, text="c")],
                 [{"turn_index": 0, "speaker": "A", "moves": ["m0"]},
                  {"turn_index": 1, "speaker": "B", "moves": ["m1"]},
                  {"turn_index": 2, "speaker": "A", "moves": ["m2"]}])
    item_p = tmp_path / "item.json"
    _write(item_p, item)
    out_p = tmp_path / "o.json"
    CM.main(["--item", str(item_p), "--fit", str(tab), "--same-scale",
             "--out", str(out_p)])
    result = json.loads(out_p.read_text(encoding="utf-8"))
    assert "rank" in result["scale_note"]
    assert result["standing_scale"] == "pairwise→0-100 via z"
    tv = {e["index"]: e for e in CM.turn_values(item, result["norm"])}
    assert tv[0]["resid"] == pytest.approx(2 / 3)
    assert tv[0]["z_resid"] == pytest.approx(1.4142, abs=1e-3)
    assert tv[1]["z_resid"] == pytest.approx(-0.7071, abs=1e-3)
    assert tv[2]["z_resid"] == pytest.approx(-0.7071, abs=1e-3)
    for e in tv.values():
        assert math.isfinite(e["z_resid"])
    for t in result["turns"]:
        assert 0.0 <= t["standing_before"] <= 100.0
        assert 0.0 <= t["standing_after"] <= 100.0


def test_commentary_same_scale_reverse_direction(tmp_path):
    tab = tmp_path / "tab"
    tab.mkdir()
    _write(tab / "table.json", _pw_table())
    item_p = tmp_path / "item.json"
    _write(item_p, _item(
        "i2", [_eff(0, "A", 60.0, text="a"), _eff(1, "B", 40.0, text="b")],
        [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
         {"turn_index": 1, "speaker": "B", "moves": ["counter"]}],
        scale="fine100"))
    out_p = tmp_path / "o.json"
    CM.main(["--item", str(item_p), "--fit", str(tab), "--same-scale",
             "--out", str(out_p)])
    result = json.loads(out_p.read_text(encoding="utf-8"))
    assert "rank" in result["scale_note"]
    assert "standing_scale" not in result


# --- valuemodel fit with --scale pairwise -----------------------------------

def _casino(path):
    def _agent(points, sat, like):
        return {"outcomes": {"points_scored": points, "satisfaction": sat,
                             "opponent_likeness": like}}

    path.write_text(json.dumps([
        {"dialogue_id": 1,
         "participant_info": {
             "mturk_agent_1": _agent(5, "Slightly satisfied", "Slightly like"),
             "mturk_agent_2": _agent(3, "Undecided", "Undecided")}},
        {"dialogue_id": 2,
         "participant_info": {
             "mturk_agent_1": _agent(4, "Undecided", "Slightly dislike"),
             "mturk_agent_2": _agent(6, "Slightly satisfied",
                                     "Extremely like")}}]),
        encoding="utf-8")


def test_valuemodel_fit_pairwise_writes_table(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write(src / "casino-1.json", _item(
        "casino-1", [_eff(0, "A", 0.5), _eff(1, "B", -0.2)],
        [{"turn_index": 0, "speaker": "A", "moves": ["inform"]},
         {"turn_index": 1, "speaker": "B", "moves": ["propose"]}],
        scale="pairwise"))
    _write(src / "casino-2.json", _item(
        "casino-2", [_eff(0, "A", -0.4), _eff(1, "B", 0.6)],
        [{"turn_index": 0, "speaker": "A", "moves": ["inform"]},
         {"turn_index": 1, "speaker": "B", "moves": ["propose"]}],
        scale="pairwise"))
    casino_p = tmp_path / "casino.json"
    _casino(casino_p)
    out = tmp_path / "out"
    V.main(["fit", "--src", str(src), "--out", str(out),
            "--scale", "pairwise", "--casino", str(casino_p)])
    payload = json.loads((out / "table.json").read_text(encoding="utf-8"))
    assert payload["scale"] == "pairwise"
    assert payload["mu"] == pytest.approx(0.125)
    assert payload["table"]["inform"] == pytest.approx(-0.075)
    assert payload["table"]["propose"] == pytest.approx(0.075)
    assert payload["move_stats"]["inform"]["mean"] == pytest.approx(-0.075)
    assert payload["move_stats"]["inform"]["sd"] > 0
    assert payload["move_stats"]["inform"]["n"] == 2
    assert payload["sd_resid"] > 0
    assert len(payload["model"]["w"]) == 5
    assert payload["model"]["featureset"] == "resid"
    # commentary accepts the emitted table
    assert CM._load_fit_table(out)["scale"] == "pairwise"


def test_valuemodel_fit_pairwise_rejects_other_scales(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    _write(src / "casino-1.json", _item(
        "casino-1", [_eff(0, "A", 60.0)],
        [{"turn_index": 0, "speaker": "A", "moves": ["inform"]}],
        scale="fine100"))
    with pytest.raises(SystemExit) as exc:
        V.main(["fit", "--src", str(src), "--out", str(tmp_path / "out"),
                "--scale", "pairwise"])
    assert "scale mismatch" in str(exc.value)


# --- viewer2 on the pairwise scale ------------------------------------------

def _pw_norm():
    return {"mu": 0.0, "sd": 0.6, "scale": "pairwise",
            "table": {"propose": 0.0, "counter": 0.0}, "sd_resid": 0.5}


def test_viewer2_pairwise_badge_and_hover():
    a = _eff(0, "A", 0.6, text="a good move")
    a["relationship"] = [2.0, 2.0, 2.0]  # stale absolute readings: ignored
    b = _eff(1, "B", -0.6, text="b replies")
    b["relationship"] = [-2.0, -2.0, -2.0]
    item = _item("casino-7", [a, b],
                 [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
                  {"turn_index": 1, "speaker": "B", "moves": ["counter"]}])
    com = CM.comment(item, _pw_norm(), None, k=3)
    view = V2.build_view2(item, com)
    assert view["scale"] == "pairwise"
    # tide runs on z_resid exactly as before
    assert view["momentum"] == pytest.approx([1.2, 1.86])
    # side standings use the 0–100 conversion of the BT rating
    assert view["standings"] == [
        {"A": pytest.approx(75.0), "B": pytest.approx(50.0)},
        {"A": pytest.approx(75.0), "B": pytest.approx(25.0)},
    ]
    assert view["turns"][0]["rating"] == pytest.approx(0.6)
    assert view["turns"][0]["rating_label"] == "쌍대 평점"
    html = V2.render_html(view)
    assert "읽기 척도: 쌍대 비교(BT 평점)" in html
    assert "쌍대 평점" in html


def test_viewer2_zero_to_100_keeps_old_badge():
    e = _eff(0, "A", 2.0, text="a")
    e["relationship"] = [1.0, 1.0, 1.0]
    item = _item("casino-7", [e],
                 [{"turn_index": 0, "speaker": "A", "moves": ["propose"]}],
                 scale="fine100")
    com = CM.comment(item, {"mu": 0.0, "sd": 1.0, "scale": "fine100",
                            "table": {"propose": 0.0}, "sd_resid": 1.0},
                     None, k=3)
    view = V2.build_view2(item, com)
    html = V2.render_html(view)
    assert "panel 0–100" in html
    assert "쌍대 평점" not in html
    assert "rating" not in view["turns"][0]


# --- batch keeps its reader; pairwise is a separate run ---------------------

def test_batch_reader_pairwise_errors():
    with pytest.raises(SystemExit):
        BA.batch_main(["--manifest", "m", "--out", "o",
                       "--reader", "pairwise"])
    assert BA.build_arg_parser().parse_args(
        ["--manifest", "m", "--out", "o"]).reader == "dist5"
