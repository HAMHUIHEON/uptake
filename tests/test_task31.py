"""Contract for SPEC_task31_default_fine100 (synthetic only, fake clients)."""
import json

import pytest

from forecaster import batch as BA
from forecaster import board as B
from forecaster import commentary as CM
from forecaster import outcome as OC
from forecaster.viewer import build_view, render_html


class _Fake:
    def complete(self, messages, model, temperature, reasoning_effort=None):
        text = json.dumps(messages)
        if "relationship_after" in text:
            return ('{"next_action":"proceed","relationship_after":75,'
                    '"face_effect":50,"understood":"ok"}')
        if "relationship_shift" in text:
            return ('{"next_action":"proceed","relationship_shift":1,'
                    '"face_threat":0,"understood":"ok"}')
        return '{"moves":["smalltalk"],"addressed_to":"all","quote":""}'


def _turns_path(tmp_path):
    tp = tmp_path / "t.jsonl"
    tp.write_text(
        '{"speaker":"A","text":"yo","chunk":0}\n'
        '{"speaker":"B","text":"sup","chunk":0}\n')
    return str(tp)


def _run_record(tmp_path, scale):
    item = {"id": "y1", "genre": "youtube", "kind": "turns",
            "path": _turns_path(tmp_path), "public": False}
    out = tmp_path / ("out_" + scale)
    BA.run_item(_Fake(), item, {}, out, n=1, k=1,
                inner_workers=1, scale=scale)
    with open(out / "y1.json", encoding="utf-8") as f:
        return json.load(f)


def test_batch_scale_cli_default():
    p = BA.build_arg_parser()
    assert p.parse_args(["--manifest", "m", "--out", "o"]).scale == "fine100"
    assert p.parse_args(["--manifest", "m", "--out", "o",
                         "--scale", "v2"]).scale == "v2"
    assert p.parse_args(["--manifest", "m", "--out", "o",
                         "--scale", "seven"]).scale == "seven"


@pytest.mark.parametrize("scale", ["fine100", "v2"])
def test_batch_item_scales_have_downstream_keys(tmp_path, scale):
    data = _run_record(tmp_path, scale)
    assert data["scale"] == scale
    assert len(data["effects"]) == 2
    need = {"index", "speaker", "text", "listeners", "relationship",
            "face", "action_dist", "disagreement", "n_readings", "readings"}
    for e in data["effects"]:
        assert need <= set(e)
        assert e["n_readings"] > 0
        assert e["action_dist"]
    hits = B.forecast_hits(data["effects"], data["moves"])
    assert len(hits) >= 1
    assert all({"predicted", "hit", "p_actual"} <= set(h) for h in hits)
    assert len(B.position_curve(data["effects"])) == 2
    assert len(CM.turn_values(data, {"mu": 0.0, "sd": 1.0})) == 2
    rec = OC.reception(data, 0.0)
    assert rec["A"] is not None and rec["B"] is not None


def test_run_item_default_is_v2(tmp_path):
    item = {"id": "y1", "genre": "youtube", "kind": "turns",
            "path": _turns_path(tmp_path), "public": False}
    out = tmp_path / "out"
    BA.run_item(_Fake(), item, {}, out, n=1, k=1, inner_workers=1)
    with open(out / "y1.json", encoding="utf-8") as f:
        data = json.load(f)
    assert data["scale"] == "v2"


def _rec(cid, scale="v2", with_scale=True, value=0.5):
    d = {
        "id": cid, "genre": "negotiation",
        "moves": [
            {"turn_index": 0, "speaker": "A", "moves": ["propose"],
             "addressed_to": "", "quote": "", "failed": False},
            {"turn_index": 1, "speaker": "B", "moves": ["accept"],
             "addressed_to": "", "quote": "", "failed": False},
        ],
        "effects": [
            {"index": 0, "speaker": "A", "text": "a",
             "value": value, "disagreement": 0.0},
            {"index": 1, "speaker": "B", "text": "b",
             "value": -value, "disagreement": 0.0},
        ],
        "hits": [],
    }
    if with_scale:
        d["scale"] = scale
    return d


def _fit_dir(tmp_path, scale="v2", with_scale=True):
    fit = tmp_path / "fit"
    fit.mkdir()
    (fit / "f1.json").write_text(json.dumps(_rec("f1", scale, with_scale)))
    return fit


def test_commentary_main_rejects_mismatch(tmp_path):
    fit = _fit_dir(tmp_path, "v2")
    item_p = tmp_path / "item.json"
    item_p.write_text(json.dumps(_rec("i1", "fine100")))
    with pytest.raises(SystemExit) as ei:
        CM.main(["--item", str(item_p), "--fit", str(fit),
                 "--out", str(tmp_path / "o.json")])
    assert "scale mismatch: item=fine100 fit=v2" in str(ei.value)


def test_commentary_main_rejects_mismatch_missing_fit_scale(tmp_path):
    fit = _fit_dir(tmp_path, with_scale=False)
    item_p = tmp_path / "item.json"
    item_p.write_text(json.dumps(_rec("i1", "fine100")))
    with pytest.raises(SystemExit) as ei:
        CM.main(["--item", str(item_p), "--fit", str(fit),
                 "--out", str(tmp_path / "o.json")])
    assert "scale mismatch: item=fine100 fit=v2" in str(ei.value)


@pytest.mark.parametrize("scale", ["fine100", "v2"])
def test_commentary_main_accepts_matching(tmp_path, scale):
    fit = _fit_dir(tmp_path, scale)
    item_p = tmp_path / "item.json"
    item_p.write_text(json.dumps(_rec("i1", scale)))
    out_p = tmp_path / "o.json"
    CM.main(["--item", str(item_p), "--fit", str(fit),
             "--out", str(out_p)])
    with open(out_p, encoding="utf-8") as f:
        result = json.load(f)
    assert result["norm"]["scale"] == scale


def _viewer_item(scale="fine100", with_scale=True):
    effects = []
    moves = []
    speakers = ["A", "B", "A", "B"]
    for i, sp in enumerate(speakers):
        effects.append({
            "index": i, "speaker": sp, "text": f"text{i}",
            "relationship": [1.0, 1.0, 1.0], "face": [0.0, 0.0, 0.0],
            "listeners": ["B" if sp == "A" else "A"],
            "action_dist": {"proceed": 1.0}, "disagreement": 0.0,
            "n_readings": 2,
            "readings": [
                {"turn": i, "listener": "B", "member": f"B@{i}-0",
                 "stance": "neutral", "sample": 0,
                 "next_action": "proceed",
                 "rel_raw": 75, "face_raw": 50,
                 "rel_norm": 1.0, "face_norm": 0.0},
                {"turn": i, "listener": "B", "member": f"B@{i}-1",
                 "stance": "neutral", "sample": 1,
                 "next_action": "proceed",
                 "rel_raw": 75, "face_raw": 50,
                 "rel_norm": 1.0, "face_norm": 0.0},
            ],
        })
        moves.append({"turn_index": i, "speaker": sp, "moves": ["propose"]})
    item = {"id": "casino-7", "genre": "negotiation",
            "moves": moves, "effects": effects, "hits": []}
    if with_scale:
        item["scale"] = scale
    return item


def _viewer_commentary():
    return {"norm": {"mu": 0.0, "sd": 1.0, "scale": "fine100"},
            "curves": {"A": [1.0, 1.0, 2.0, 2.0],
                       "B": [0.0, 1.0, 1.0, 2.0],
                       "balance": [1.0, 0.0, 1.0, 0.0]},
            "turns": [{"index": 2, "speaker": "A", "mark": "!",
                       "label_ko": "묘수", "label_en": "strong move",
                       "line_ko": "ko", "line_en": "en",
                       "reference": {"id": "r", "author": "a",
                                     "title": "t", "year": 2000,
                                     "venue": "v"}}]}


def _blob_data(html):
    tag = '<script type="application/json" id="data">'
    start = html.index(tag) + len(tag)
    return json.loads(html[start:html.index("</script>", start)])


def test_viewer_fine100_label_and_raw():
    view = build_view(_viewer_item("fine100"), _viewer_commentary())
    assert view["scale"] == "fine100"
    assert view["turns"][0]["raw_rel"] == pytest.approx(75.0)
    assert view["turns"][0]["raw_face"] == pytest.approx(50.0)
    assert view["turns"][2]["mark"] == "!"
    html = render_html(view)
    assert "panel 0–100" in html
    assert "75.0" in html


def test_viewer_v2_label_no_raw():
    view = build_view(_viewer_item(with_scale=False),
                      _viewer_commentary())
    assert view["scale"] == "v2"
    html = render_html(view)
    assert "panel −2..+2" in html
    data = _blob_data(html)
    assert all(t["raw_rel"] is None and t["raw_face"] is None
               for t in data["turns"])
