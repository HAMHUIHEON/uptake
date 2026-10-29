"""Contract for SPEC_task40_viewer_v2_polish (synthetic data only)."""
import json
import os
import re

from forecaster import commentary as CM
from forecaster import viewer2 as V2


def _eff(idx, speaker, value, rel=0.0, text="", **kw):
    e = {"index": idx, "speaker": speaker, "text": text or f"t{idx}",
         "value": value, "disagreement": 0.0, "n_readings": 2,
         "relationship": [rel, rel, rel], "face": [0.0, 0.0, 0.0]}
    e.update(kw)
    return e


def _item(cid, effects, moves, scale="fine100"):
    return {"id": cid, "genre": "negotiation", "scale": scale,
            "moves": moves, "effects": effects, "hits": []}


def _resid_norm(table, sd_resid=1.0):
    return {"mu": 0.0, "sd": 1.0, "scale": "fine100",
            "table": dict(table), "sd_resid": sd_resid}


def _blob_data(html):
    tag = '<script type="application/json" id="data">'
    start = html.index(tag) + len(tag)
    return json.loads(html[start:html.index("</script>", start)])


def _hero_toy():
    # z_resid per turn: A +0.5, B -1.0, A +2.0 (top speaker A).
    # M = [0.5, 1.275, 2.70125]; |dM| = [0.5, 0.775, 1.42625].
    effects = [
        _eff(0, "A", 0.5, text="first small move"),
        _eff(1, "B", -1.0, text="second reply here"),
        _eff(2, "A", 2.0, text="third decisive strike wins the exchange"),
    ]
    moves = [{"turn_index": i, "speaker": sp, "moves": ["m"]}
             for i, sp in enumerate(["A", "B", "A"])]
    return _item("hero-1", effects, moves)


# --- hero moment ------------------------------------------------------

def test_hero_swings_ranking():
    import pytest

    got = V2.hero_swings([0.5, 2.0, 2.1])
    assert [r["pos"] for r in got] == [1, 0, 2]
    assert [r["delta"] for r in got] == pytest.approx([1.5, 0.5, 0.1])
    assert V2.hero_swings([], 3) == []
    assert V2.hero_swings([1.0], 3) == [{"pos": 0, "delta": 1.0}]


def test_hero_pick_banner_and_runners():
    item = _hero_toy()
    com = CM.comment(item, _resid_norm({"m": 0.0}), None, k=3)
    view = V2.build_view2(item, com)
    assert view["momentum"][2] == view["momentum"][2]  # sanity
    top = view["hero"]["top"]
    assert top["pos"] == 2 and top["index"] == 2
    assert top["delta"] == abs(top["delta"]) > 1.4
    assert top["text80"] == "third decisive strike wins the exchange"
    assert top["line"]  # turn 2 is marked (!!), so the line is attached
    runners = view["hero"]["runners"]
    assert [r["pos"] for r in runners] == [1, 0]
    html = V2.render_html(view)
    assert "판이 갈린 수 · #2 A" in html
    assert "third decisive strike wins the exchange" in html
    assert top["line"] in html
    assert "그 장면 보기" in html
    assert 'id="heroBtn"' in html
    assert "#1 B" in html and "#0 A" in html  # runners listed small
    # JS wiring: scrub-to-hero button, pulse, reached highlight
    assert "heroBtn" in V2._JS and "pulseT" in V2._JS
    assert '"reached"' in V2._JS


def test_hero_empty_record_has_no_banner():
    item = _item("empty", [], [])
    view = V2.build_view2(item, CM.comment(
        item, {"mu": 0.0, "sd": 1.0}, None, k=3))
    assert view["hero"] == {"top": None, "runners": []}
    assert 'id="hero"' not in V2.render_html(view)


# --- confidence band ---------------------------------------------------

def test_spread_prefers_rel_spread_then_disagreement():
    e = _eff(0, "A", 1.0, text="x")
    e["readings"] = [{"rel_spread": 10.0}, {"rel_spread": 20.0}]
    e["disagreement"] = 0.7
    assert V2.spread_of_effect(e) == (15.0, "rel_spread")
    e2 = _eff(1, "B", 1.0, text="y")
    e2["disagreement"] = 0.5
    assert V2.spread_of_effect(e2) == (0.5, "disagreement")
    e3 = {"index": 2, "speaker": "A", "text": "z", "value": 1.0}
    assert V2.spread_of_effect(e3) == (None, None)
    assert V2.spread_of_effect(None) == (None, None)


def test_band_present_only_with_spread_or_disagreement():
    moves = [{"turn_index": 0, "speaker": "A", "moves": ["m"]},
             {"turn_index": 1, "speaker": "B", "moves": ["m"]}]
    norm = _resid_norm({"m": 0.0})

    dist = _item("d5", [
        _eff(0, "A", 1.0, text="a",
             readings=[{"rel_spread": 18.0}, {"rel_spread": 18.0}]),
        _eff(1, "B", 0.5, text="b",
             readings=[{"rel_spread": 12.0}]),
    ], moves, scale="dist5")
    v_dist = V2.build_view2(dist, CM.comment(dist, norm, None, k=3))
    assert v_dist["spread_src"] == "rel_spread"
    assert v_dist["spread"] == [18.0, 12.0]

    dis = _item("dg", [
        _eff(0, "A", 1.0, text="a", disagreement=0.42),
        _eff(1, "B", 0.5, text="b", disagreement=0.10),
    ], moves)
    v_dis = V2.build_view2(dis, CM.comment(dis, norm, None, k=3))
    assert v_dis["spread_src"] == "disagreement"
    assert v_dis["spread"] == [0.42, 0.10]

    bare = _item("bare", [
        {"index": 0, "speaker": "A", "text": "a", "value": 1.0},
        {"index": 1, "speaker": "B", "text": "b", "value": 0.5},
    ], moves)
    v_bare = V2.build_view2(bare, CM.comment(bare, norm, None, k=3))
    assert v_bare["spread_src"] is None
    assert v_bare["spread"] == [None, None]

    # JS draws the band only when spread data exists; hover shows 확신도.
    assert "conf-band" in V2._JS and "bandD" in V2._JS
    assert "spreadPx" in V2._JS and "data.spread_src" in V2._JS
    assert "확신도: 심사위원 분포 퍼짐" in V2._JS
    assert "확신도: 심사위원 의견 불일치" in V2._JS
    assert ".conf-band" in V2._CSS


# --- square fixes ------------------------------------------------------

def test_square_smooth_trail_and_fixed_fonts():
    assert "polyline" not in V2._JS
    assert 'setAttribute("d",smoothD(pts))' in V2._JS
    assert '" 잔고 →"' in V2._JS and '" 잔고 ↑"' in V2._JS
    assert re.search(r"\.axis-tx\{[^}]*font-size:12px", V2._CSS)
    assert re.search(r"\.sq-corner\{[^}]*font-size:13px", V2._CSS)
    assert re.search(r"#settle\{[^}]*font-size:14px", V2._CSS)


# --- audio path --------------------------------------------------------

def _write_pair(tmp_path):
    item = _hero_toy()
    com = CM.comment(item, _resid_norm({"m": 0.0}), None, k=3)
    item_p = tmp_path / "item.json"
    com_p = tmp_path / "commentary.json"
    item_p.write_text(json.dumps(item))
    com_p.write_text(json.dumps(com))
    return item_p, com_p


def test_audio_copy_next_to_html_by_default(tmp_path):
    from forecaster.viewer2 import main

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    clip = src_dir / "clip.wav"
    clip.write_bytes(b"RIFF-fake-audio")
    item_p, com_p = _write_pair(tmp_path)
    out_p = tmp_path / "out" / "tide.html"
    main(["--item", str(item_p), "--commentary", str(com_p),
          "--out", str(out_p), "--audio", str(clip)])
    copied = out_p.parent / "tide.audio.wav"
    assert copied.is_file()
    assert copied.read_bytes() == b"RIFF-fake-audio"
    html = out_p.read_text()
    data = _blob_data(html)
    assert data["audio"]["src"] == "tide.audio.wav"
    assert str(clip) not in html  # never the raw CLI string


def test_audio_no_copy_uses_relpath(tmp_path):
    from forecaster.viewer2 import main

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    clip = src_dir / "clip.wav"
    clip.write_bytes(b"RIFF-fake-audio")
    item_p, com_p = _write_pair(tmp_path)
    out_p = tmp_path / "out" / "tide.html"
    main(["--item", str(item_p), "--commentary", str(com_p),
          "--out", str(out_p), "--audio", str(clip), "--no-copy-audio"])
    assert not (out_p.parent / "tide.audio.wav").exists()
    html = out_p.read_text()
    data = _blob_data(html)
    assert data["audio"]["src"] == os.path.relpath(str(clip),
                                                  str(out_p.parent))


def test_audio_missing_file_falls_back_to_basename(tmp_path):
    assert V2.resolve_audio_src(None, tmp_path / "t.html") is None
    assert V2.resolve_audio_src("nope.wav", tmp_path / "t.html") == "nope.wav"


# --- speaker legend ----------------------------------------------------

def test_speaker_legend_for_three_speakers():
    effects = [_eff(0, "A", 1.0, text="a"),
               _eff(1, "B", 0.5, text="b"),
               _eff(2, "C", -1.0, text="c")]
    moves = [{"turn_index": i, "speaker": sp, "moves": ["m"]}
             for i, sp in enumerate("ABC")]
    item = _item("multi", effects, moves)
    com = CM.comment(item, _resid_norm({"m": 0.0}), None, k=3)
    names = {"A": "영희", "B": "철수", "C": "민수"}
    html = V2.render_html(V2.build_view2(item, com, names=names))
    for disp in ("영희", "철수", "민수"):
        assert disp in html
    for col in V2.SPEAKER_COLORS[:3]:
        assert col in html
    assert '<span class="sw"' in html
    # two speakers: no speaker swatches, mark legend still there
    two = V2.render_html(V2.build_view2(
        _hero_toy(), CM.comment(_hero_toy(), _resid_norm({"m": 0.0}),
                                None, k=3)))
    assert '<span class="sw"' not in two
    assert "묘수" in two


def test_dots_autohide_only_past_40_turns():
    assert "auto-hide" in V2._JS and "N>40" in V2._JS
    assert "#tide:hover .dot.auto-hide" in V2._CSS


# --- motion ------------------------------------------------------------

def test_motion_eases_and_respects_reduced_motion():
    assert "1/0.3" in V2._JS  # 300 ms per turn
    assert "prefers-reduced-motion" in V2._JS
    assert "prefers-reduced-motion" in V2._CSS
    assert "easeOutBack" in V2._JS and "markBorn" in V2._JS


# --- genre-aware references --------------------------------------------

_ANY_IDS = {"grice_maxims", "sacks_turn_taking", "schegloff_repair",
            "goffman_facework"}


def test_reference_genres_in_file():
    refs = CM.load_references()
    by_id = {r["id"]: r for r in refs}
    assert _ANY_IDS <= set(by_id)
    for rid, ref in by_id.items():
        want = ["any"] if rid in _ANY_IDS else ["negotiation"]
        assert ref["genres"] == want, rid


def test_choose_reference_filters_by_genre():
    refs = CM.load_references()
    meeting = {"mark": "!", "moves": ["propose"], "genre": "meeting"}
    got = CM.choose_reference(meeting, refs, ())
    assert got["id"] == "goffman_facework"
    assert got["genres"] == ["any"]
    # explicit genre parameter works the same
    got2 = CM.choose_reference({"mark": "!", "moves": ["propose"]},
                               refs, (), genre="meeting")
    assert got2["id"] == "goffman_facework"
    # default (no genre) keeps the negotiation behaviour
    plain = CM.choose_reference({"mark": "!", "moves": ["propose"]},
                                refs, ())
    assert plain["id"] == "fisher_ury_interests"
    # refs without genres default to negotiation
    bare_refs = [{"id": "x", "marks": ["!"], "moves": ["propose"]}]
    assert CM.choose_reference({"mark": "!", "moves": ["propose"]},
                               bare_refs, ())["id"] == "x"


def test_comment_uses_item_genre():
    refs = CM.load_references()
    effects = [_eff(0, "A", 2.0, text="a"), _eff(1, "B", 1.5, text="b")]
    moves = [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
             {"turn_index": 1, "speaker": "B", "moves": ["counter"]}]
    item = _item("m1", effects, moves)
    item["genre"] = "meeting"
    res = CM.comment(item, _resid_norm({"propose": 0.0, "counter": 0.0}),
                     None, k=3, refs=refs)
    assert res["turns"]
    by_id = {r["id"]: r for r in refs}
    for t in res["turns"]:
        assert by_id[t["reference"]["id"]]["genres"] == ["any"]
