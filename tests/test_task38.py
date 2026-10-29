"""Contract for SPEC_task38_viewer_v2_tide (synthetic data only)."""
import json

import pytest

from forecaster import commentary as CM
from forecaster import viewer2 as V2


def _eff(idx, speaker, value, rel=0.0, text="", start=None, end=None):
    e = {"index": idx, "speaker": speaker, "text": text or f"t{idx}",
         "value": value, "disagreement": 0.0, "n_readings": 2,
         "relationship": [rel, rel, rel], "face": [0.0, 0.0, 0.0]}
    if start is not None and end is not None:
        e["start"] = start
        e["end"] = end
    return e


def _item(cid, effects, moves, scale="fine100"):
    return {"id": cid, "genre": "negotiation", "scale": scale,
            "moves": moves, "effects": effects, "hits": []}


def _resid_norm(table, sd_resid=1.0):
    return {"mu": 0.0, "sd": 1.0, "scale": "fine100",
            "table": dict(table), "sd_resid": sd_resid}


def _toy_two_speaker():
    effects = [
        _eff(0, "A", 2.0, rel=1.0, text="a good move"),
        _eff(1, "B", 1.0, rel=-0.4, text="b replies"),
    ]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
        {"turn_index": 1, "speaker": "B", "moves": ["counter"]},
    ]
    return _item("casino-7", effects, moves)


def _blob_data(html):
    tag = '<script type="application/json" id="data">'
    start = html.index(tag) + len(tag)
    return json.loads(html[start:html.index("</script>", start)])


# --- momentum ------------------------------------------------------------

def test_momentum_series_values_and_decay():
    item = _toy_two_speaker()
    norm = _resid_norm({"propose": 0.0, "counter": 0.0})
    tv = CM.turn_values(item, norm)
    assert [e["z_resid"] for e in tv] == pytest.approx([2.0, 1.0])
    got = V2.momentum_series(tv, ["A", "B"])
    # M0 = 2.0; M1 = 0.55*2.0 - 1.0 = 0.1
    assert got == pytest.approx([2.0, 0.1])


def test_momentum_direction_a_good_move_lowers_boundary():
    mid = 150.0
    assert V2.tide_y(2.0, mid) > mid  # down, into B's base
    assert V2.tide_y(-2.0, mid) < mid  # up, into A's base
    assert V2.tide_y(0.0, mid) == pytest.approx(mid)
    assert V2.tide_y(99.0, mid) == pytest.approx(V2.tide_y(3.0, mid))
    assert V2.tide_y(-99.0, mid) == pytest.approx(V2.tide_y(-3.0, mid))


def test_momentum_falls_back_to_z_without_resid():
    effects = [_eff(0, "A", 1.5), _eff(1, "B", -0.5)]
    moves = [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
             {"turn_index": 1, "speaker": "B", "moves": ["reject"]}]
    item = _item("x", effects, moves)
    tv = CM.turn_values(item, {"mu": 0.0, "sd": 1.0})
    assert all("z_resid" not in e for e in tv)
    got = V2.momentum_series(tv, ["A", "B"])
    assert got == pytest.approx([1.5, 0.55 * 1.5 + 0.5])


# --- multi-party bands ----------------------------------------------------

def test_band_width_follows_received_z():
    effects, moves = [], []
    for i, (sp, v) in enumerate([("A", 2.0), ("B", 0.0), ("C", -2.0),
                                 ("A", 2.0), ("B", 0.0), ("C", -2.0)]):
        effects.append(_eff(i, sp, v))
        moves.append({"turn_index": i, "speaker": sp, "moves": ["m"]})
    item = _item("multi", effects, moves)
    tv = CM.turn_values(item, _resid_norm({"m": 0.0}))
    bands = V2.band_series(tv, ["A", "B", "C"])
    assert len(bands) == 6
    for row in bands:
        assert sum(row.values()) == pytest.approx(1.0)
        assert set(row) == {"A", "B", "C"}
    last = bands[-1]
    assert last["A"] > last["B"] > last["C"]
    assert last["C"] >= 0.05  # floored by max(0.2, ...)


def test_build_view2_picks_bands_for_three_speakers():
    effects = [_eff(0, "A", 1.0), _eff(1, "B", 0.5), _eff(2, "C", -1.0)]
    moves = [{"turn_index": i, "speaker": sp, "moves": ["m"]}
             for i, sp in enumerate("ABC")]
    item = _item("multi", effects, moves)
    com = CM.comment(item, _resid_norm({"m": 0.0}), None, k=3)
    view = V2.build_view2(item, com)
    assert view["mode"] == "bands"
    assert view["bands"] is not None and len(view["bands"]) == 3
    view2 = V2.build_view2(_toy_two_speaker(),
                           CM.comment(_toy_two_speaker(),
                                      _resid_norm({"propose": 0.0,
                                                   "counter": 0.0}),
                                      None, k=3))
    assert view2["mode"] == "tide"
    assert view2["bands"] is None


# --- standings ------------------------------------------------------------

def test_standings_per_side_carry_forward():
    item = _toy_two_speaker()
    com = CM.comment(item, _resid_norm({"propose": 0.0, "counter": 0.0}),
                     None, k=3)
    view = V2.build_view2(item, com)
    assert view["standings"] == [
        {"A": pytest.approx(75.0), "B": pytest.approx(50.0)},
        {"A": pytest.approx(75.0), "B": pytest.approx(40.0)},
    ]
    assert view["square"]["xs"] == pytest.approx([75.0, 75.0])
    assert view["square"]["ys"] == pytest.approx([50.0, 40.0])
    assert view["square"]["lo"] == 20.0
    assert view["square"]["hi"] == 80.0
    assert view["settlement"] == "A 잔고 50 → 75 · B 잔고 50 → 40"


def test_standing_of_scale():
    assert CM.standing_of(0.0) == pytest.approx(50.0)
    assert CM.standing_of(1.0) == pytest.approx(75.0)
    assert CM.standing_of(-2.0) == pytest.approx(0.0)
    assert CM.rel_mean_of({"relationship": [1.0, 0.0, 2.0]}) == \
        pytest.approx(1.0)
    assert CM.rel_mean_of({"value": 2.0}) == pytest.approx(2.0)


# --- guess mode -----------------------------------------------------------

def test_guess_class_thresholds():
    assert V2.guess_class(1.0) == "!"
    assert V2.guess_class(2.5) == "!"
    assert V2.guess_class(0.99) == "="
    assert V2.guess_class(0.0) == "="
    assert V2.guess_class(-0.99) == "="
    assert V2.guess_class(-1.0) == "?"
    assert V2.guess_class(-2.5) == "?"
    assert V2.score_guess("!", 1.5) == 1
    assert V2.score_guess("=", 1.5) == 0
    assert V2.score_guess("=", 0.2) == 1
    assert V2.score_guess("?", -1.2) == 1


def test_emitted_data_table_carries_guess_key():
    item = _toy_two_speaker()
    com = CM.comment(item, _resid_norm({"propose": 0.0, "counter": 0.0}),
                     None, k=3)
    view = V2.build_view2(item, com)
    assert [t["guess"] for t in view["turns"]] == ["!", "!"]
    assert view["turns"][0]["z_resid"] == pytest.approx(2.0)
    item2 = _item("g", [_eff(0, "A", -1.5)],
                  [{"turn_index": 0, "speaker": "A", "moves": ["m"]}])
    com2 = CM.comment(item2, _resid_norm({"m": 0.0}), None, k=3)
    assert V2.build_view2(item2, com2)["turns"][0]["guess"] == "?"


# --- HTML -----------------------------------------------------------------

def test_html_controls_guess_square_and_marks():
    item = _toy_two_speaker()
    com = CM.comment(item, _resid_norm({"propose": 0.0, "counter": 0.0}),
                     None, k=3)
    html = V2.render_html(V2.build_view2(item, com))
    assert "<script src" not in html
    assert "http" not in html
    # replay controls
    assert 'id="play"' in html
    assert 'id="scrub"' in html
    assert 'data-speed="2"' in html and 'data-speed="4"' in html
    # guess mode
    assert "다음 수 맞히기" in html
    assert 'data-g="!"' in html and 'data-g="="' in html
    assert 'data-g="?"' in html
    assert "나 0 / 해설자 0" in html
    # square with zoomed axes + corner labels
    assert 'id="square"' in html
    assert "20–80" in html
    for corner in ("둘 다 잘됨", "둘 다 잃음", "한쪽만"):
        assert corner in html
    # five mark labels
    for ko in ("묘수", "호수(좋은 수)", "특이수", "실수", "악수"):
        assert ko in html
    for en in ("brilliant", "good", "interesting", "mistake", "blunder"):
        assert en in html
    data = _blob_data(html)
    assert len(data["momentum"]) == 2
    assert data["settlement"].startswith("A 잔고 50 →")
    assert data["audio"] is None


def test_html_audio_and_outcomes(tmp_path):
    effects = [_eff(0, "A", 2.0, rel=1.0, start=0.0, end=3.0),
               _eff(1, "B", 1.0, rel=-0.4, start=3.0, end=6.0)]
    moves = [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
             {"turn_index": 1, "speaker": "B", "moves": ["counter"]}]
    item = _item("casino-7", effects, moves)
    com = CM.comment(item, _resid_norm({"propose": 0.0, "counter": 0.0}),
                     None, k=3)
    outcomes = {"A": {"points": 10, "satisfaction": 1, "likeness": 2},
                "B": {"points": 8, "satisfaction": 0, "likeness": -1}}
    view = V2.build_view2(item, com, outcomes=outcomes,
                          audio_src="clip.wav")
    assert view["audio"] == {"src": "clip.wav", "linked": True}
    assert view["turns"][0]["start"] == pytest.approx(0.0)
    assert view["turns"][1]["end"] == pytest.approx(6.0)
    html = V2.render_html(view)
    assert '<audio id="aud"' in html and "clip.wav" in html
    assert "actual likeness" in html
    # unlinked when no timings
    plain = V2.build_view2(_toy_two_speaker(), com, audio_src="clip.wav")
    assert plain["audio"] == {"src": "clip.wav", "linked": False}


def test_cli_writes_html(tmp_path):
    from forecaster.viewer2 import main

    item = _toy_two_speaker()
    com = CM.comment(item, _resid_norm({"propose": 0.0, "counter": 0.0}),
                     None, k=3)
    item_p = tmp_path / "item.json"
    com_p = tmp_path / "commentary.json"
    item_p.write_text(json.dumps(item))
    com_p.write_text(json.dumps(com))
    casino_p = tmp_path / "casino.json"
    casino_p.write_text(json.dumps([{
        "dialogue_id": 7,
        "participant_info": {
            "mturk_agent_1": {"outcomes": {
                "points_scored": 10, "satisfaction": "Slightly satisfied",
                "opponent_likeness": "Extremely like"}},
            "mturk_agent_2": {"outcomes": {
                "points_scored": 8, "satisfaction": "Undecided",
                "opponent_likeness": "Slightly dislike"}},
        }}]))
    out_p = tmp_path / "tide.html"
    main(["--item", str(item_p), "--commentary", str(com_p),
          "--out", str(out_p), "--casino", str(casino_p),
          "--audio", "clip.wav"])
    html = out_p.read_text()
    data = _blob_data(html)
    assert data["outcomes"]["A"]["points"] == 10
    assert data["audio"]["src"] == "clip.wav"
    assert "tide board" in html


def test_empty_record_renders():
    item = _item("empty", [], [])
    view = V2.build_view2(item, CM.comment(
        item, {"mu": 0.0, "sd": 1.0}, None, k=3))
    assert view["mode"] == "tide"
    assert view["settlement"] == ""
    html = V2.render_html(view)
    assert 'id="tide"' in html


# --- commentary wording ---------------------------------------------------

def test_mark_labels_per_mark():
    assert CM.LABEL_KO == {"!!": "묘수", "!": "호수(좋은 수)",
                           "!?": "특이수", "?": "실수", "??": "악수"}
    assert CM.LABEL_EN == {"!!": "brilliant", "!": "good",
                           "!?": "interesting", "?": "mistake",
                           "??": "blunder"}


def _wording_turn():
    return {"index": 3, "speaker": "B", "moves": ["counter"],
            "mark": "?", "label_ko": CM.LABEL_KO["?"],
            "label_en": CM.LABEL_EN["?"],
            "standing_before": 53.0, "standing_after": 45.0}


def _wording_ref():
    return {"id": "r", "author": "An Author", "title": "A Title",
            "year": 2000, "venue": "V",
            "idea_en": "an idea", "idea_ko": "아이디어"}


def test_template_line_uses_standings_and_speaker_label():
    out = CM.template_line(_wording_turn(), _wording_ref())
    assert "잔고 53 → 45" in out["ko"]
    assert "standing 53 → 45" in out["en"]
    assert "B" in out["ko"] and "B" in out["en"]
    assert "흑" not in out["ko"] and "흑" not in out["en"]
    assert "백" not in out["ko"] and "백" not in out["en"]
    assert "형세" not in out["ko"]


def test_template_line_display_name_when_provided():
    turn = _wording_turn()
    turn["display_name"] = "영희"
    out = CM.template_line(turn, _wording_ref())
    assert "영희" in out["ko"] and "영희" in out["en"]


def test_pick_turns_carries_standings():
    item = _toy_two_speaker()
    got = CM.pick_turns(item, _resid_norm({"propose": 0.0, "counter": 0.0}),
                        None, k=3)
    assert got
    for t in got:
        assert "standing_before" in t and "standing_after" in t
    first = next(t for t in got if t["index"] == 0)
    assert first["standing_before"] == pytest.approx(50.0)
    assert first["standing_after"] == pytest.approx(75.0)


class _FakeClient:
    def __init__(self, raw):
        self.raw = raw
        self.seen = []

    def complete(self, messages, model, temperature, **kw):
        self.seen.append((messages, model, temperature))
        return self.raw


def test_llm_prompt_uses_standings():
    turn, ref = _wording_turn(), _wording_ref()
    client = _FakeClient(json.dumps({"ko": "짧은 해설", "en": "short"}))
    out = CM.llm_line(client, turn, ref, "m")
    assert out["source"] == "llm"
    system, user = (m["content"] for m in client.seen[0][0])
    assert "0-100" in system and "0-100" in user
    assert "standing 53 -> 45" in user
    assert "position +53" not in user and "position +45" not in user
