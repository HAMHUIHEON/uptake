"""Contract for SPEC_task26_viewer_stakes (synthetic data only)."""
import json
import re

from forecaster import commentary as CM
from forecaster.viewer import build_view, render_html


def _eff(idx, speaker, value, text=""):
    return {"index": idx, "speaker": speaker,
            "text": text or f"text{idx}",
            "value": value, "disagreement": 0.0}


def _item(cid="casino-7"):
    effects = [
        _eff(0, "A", 0.2),
        _eff(1, "B", 0.3),
        _eff(2, "A", 1.5),
        _eff(3, "B", -1.6),
    ]
    return {
        "id": cid, "genre": "negotiation",
        "moves": [
            {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
            {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
            {"turn_index": 2, "speaker": "A", "moves": ["counter"]},
            {"turn_index": 3, "speaker": "B", "moves": ["accept"]},
        ],
        "effects": effects, "hits": [],
    }


def _markov():
    return {
        "proceed": {"negotiate": 0.5, "proceed": 0.01},
        "negotiate": {"refuse": 0.01},
        "refuse": {"proceed": 0.5},
    }


def _commentary(item):
    return CM.comment(item, {"mu": 0.0, "sd": 1.0}, _markov(),
                      k=3, client=None)


def _offer(a_food=1, a_water=0, a_fire=3):
    return {
        "A": {"Food": a_food, "Water": a_water, "Firewood": a_fire},
        "B": {"Food": 3 - a_food, "Water": 3 - a_water,
              "Firewood": 3 - a_fire},
    }


def _stakes(cid="casino-7"):
    return {
        cid: {
            "offers": {"2": _offer()},
            "track": [
                {"A": None, "B": None, "offer_turn": None},
                {"A": None, "B": None, "offer_turn": None},
                {"A": 19, "B": 18, "offer_turn": 2},
                {"A": 19, "B": 18, "offer_turn": 2},
            ],
        }
    }


def _blob_data(html):
    tag = '<script type="application/json" id="data">'
    assert html.count(tag) == 1
    start = html.index(tag) + len(tag)
    return json.loads(html[start:html.index("</script>", start)])


def test_stakes_present_in_view():
    item = _item()
    view = build_view(item, _commentary(item), stakes=_stakes())
    assert view["stakes"] == {
        "track": _stakes()["casino-7"]["track"],
        "offers": {"2": _offer()},
    }
    data = _blob_data(render_html(view))
    assert data["stakes"]["track"][2] == {"A": 19, "B": 18, "offer_turn": 2}
    assert data["stakes"]["offers"] == {"2": _offer()}


def test_stakes_absent_in_view():
    item = _item()
    com = _commentary(item)
    assert build_view(item, com)["stakes"] is None
    assert build_view(item, com, stakes=None)["stakes"] is None
    assert build_view(item, com, stakes=_stakes("casino-9"))["stakes"] is None
    assert build_view(item, com, stakes={})["stakes"] is None
    data = _blob_data(render_html(build_view(item, com)))
    assert data["stakes"] is None


def test_second_chart_only_when_stakes():
    item = _item()
    com = _commentary(item)
    html = render_html(build_view(item, com, stakes=_stakes()))
    assert '<svg id="stakes-chart"' in html
    assert "판돈: 테이블 위 점수 / Stake: points on the table" in html
    assert ("points each side would get if the latest offer were accepted "
            "(CaSiNo point values)") in html
    plain = render_html(build_view(item, com))
    assert '<svg id="stakes-chart"' not in plain
    assert "판돈: 테이블 위 점수 / Stake: points on the table" not in plain
    assert "points each side would get if the latest offer were accepted" not in plain


def test_stakes_chart_scale_and_main_yaxis_ticks():
    html = render_html(build_view(_item(), _commentary(_item()),
                                  stakes=_stakes()))
    # stakes chart: fixed 0-36 scale with ticks 0, 12, 24, 36
    assert "sticks=[0,12,24,36]" in html
    assert "(v/36)" in html
    # main chart: min / 0 / max labels, 1 decimal
    assert "yticks=[mn,0,mx]" in html
    assert "Number(yticks[yi]).toFixed(1)" in html


def test_offer_chip_and_score_text():
    html = render_html(build_view(_item(), _commentary(_item()),
                                  stakes=_stakes()))
    assert '"offer A "' in html
    assert '"on-table points at the end: A "' in html


def test_no_external_resources_with_stakes():
    view = build_view(_item(), _commentary(_item()), stakes=_stakes())
    html = render_html(view)
    assert "<script src" not in html
    assert '<link rel="stylesheet"' not in html
    assert re.findall(r"https?://\S+", html) == []
    assert "http" not in html
    assert "innerHTML" not in html


def test_cli_stakes(tmp_path):
    from forecaster.viewer import main

    item, com = _item(), _commentary(_item())
    item_p = tmp_path / "item.json"
    com_p = tmp_path / "commentary.json"
    stakes_p = tmp_path / "stakes.json"
    item_p.write_text(json.dumps(item))
    com_p.write_text(json.dumps(com))
    stakes_p.write_text(json.dumps(_stakes()))
    out_p = tmp_path / "board.html"
    main(["--item", str(item_p), "--commentary", str(com_p),
          "--out", str(out_p), "--stakes", str(stakes_p)])
    html = out_p.read_text()
    assert '<svg id="stakes-chart"' in html
    assert _blob_data(html)["stakes"]["track"][3]["offer_turn"] == 2
    # without --stakes there is no second chart
    out2 = tmp_path / "plain.html"
    main(["--item", str(item_p), "--commentary", str(com_p),
          "--out", str(out2)])
    html2 = out2.read_text()
    assert '<svg id="stakes-chart"' not in html2
    assert _blob_data(html2)["stakes"] is None
