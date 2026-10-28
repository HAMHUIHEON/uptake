"""Contract for SPEC_task22_viewer (synthetic data only)."""
import json
import re

from forecaster import commentary as CM
from forecaster.transcribe import Turn
from forecaster.viewer import build_view, render_html


def _eff(idx, speaker, value, text=""):
    return {"index": idx, "speaker": speaker,
            "text": text or f"text{idx}",
            "value": value, "disagreement": 0.0}


def _item(cid="casino-7", texts=None):
    effects = [
        _eff(0, "A", 0.2, (texts or {}).get(0, "")),
        _eff(1, "B", 0.3, (texts or {}).get(1, "")),
        _eff(2, "A", 1.5, (texts or {}).get(2, "")),
        _eff(3, "B", -1.6, (texts or {}).get(3, "")),
    ]
    return {
        "id": cid, "genre": "negotiation",
        "moves": [
            {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
            {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
            {"turn_index": 2, "speaker": "A", "moves": ["inform"]},
            {"turn_index": 3, "speaker": "B", "moves": ["inform"]},
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


def _blob_data(html):
    tag = '<script type="application/json" id="data">'
    assert html.count(tag) == 1
    start = html.index(tag) + len(tag)
    return json.loads(html[start:html.index("</script>", start)])


def test_no_external_resources_with_youtube():
    view = build_view(_item(), _commentary(_item()), youtube_id="dQw4w9WgXcQ",
                      turns=[Turn("A", "hi", 0)])
    html = render_html(view)
    assert "<script src" not in html
    assert '<link rel="stylesheet"' not in html
    urls = re.findall(r"https?://\S+", html)
    assert urls, "expected the youtube embed url"
    assert all("youtube-nocookie" in u for u in urls)
    assert "<iframe" in html


def test_no_external_resources_without_youtube():
    view = build_view(_item(), _commentary(_item()))
    html = render_html(view)
    assert "<script src" not in html
    assert '<link rel="stylesheet"' not in html
    assert re.findall(r"https?://\S+", html) == []
    assert "http" not in html


def test_script_payload_escaped_but_survives():
    payload = "a</script><b>x"
    item = _item(texts={0: payload})
    view = build_view(item, _commentary(item))
    html = render_html(view)
    assert payload not in html
    assert "</script><b>x" not in html
    data = _blob_data(html)
    assert data["turns"][0]["text"] == payload


def test_view_pivots_reference_titles_and_score():
    item = _item()
    com = _commentary(item)
    view = build_view(item, com)
    assert view["curves"] == com["curves"]
    assert view["pivots"] == com["turns"]
    assert len(view["pivots"]) == 3
    for piv in view["pivots"]:
        assert piv["reference"]["title"]
    # value/z come from turn_values; mark from the same-index pivot
    tv = {e["index"]: e for e in CM.turn_values(item, com["norm"])}
    marks = {p["index"]: p["mark"] for p in com["turns"]}
    for turn in view["turns"]:
        assert turn["value"] == tv[turn["index"]]["value"]
        assert turn["z"] == tv[turn["index"]]["z"]
        assert turn["mark"] == marks.get(turn["index"])
    # score counts match pivots
    expect = {"!": 0, "?": 0, "!?": 0}
    for piv in view["pivots"]:
        expect[piv["mark"]] += 1
    assert view["score"]["counts"] == expect
    for side in ("A", "B"):
        for mark in ("!", "?", "!?"):
            assert view["score"][side]["counts"][mark] >= 0
    summed = {m: view["score"]["A"]["counts"][m] + view["score"]["B"]["counts"][m]
              for m in ("!", "?", "!?")}
    assert summed == expect
    assert view["score"]["A"]["final"] == com["curves"]["A"][-1]
    assert view["score"]["B"]["final"] == com["curves"]["B"][-1]
    assert view["score"]["A"]["n_turns"] == 2
    assert view["score"]["B"]["n_turns"] == 2


def test_t_two_chunk_formula():
    item = _item()
    com = _commentary(item)
    turns = [Turn("A", "aa", 0), Turn("B", "bbbb", 0),
             Turn("A", "ccc", 1), Turn("B", "d", 1)]
    view = build_view(item, com, youtube_id="dQw4w9WgXcQ", turns=turns)
    # chunk 0: total 6 -> 0, floor(540*2/6)=180
    # chunk 1: total 4 -> 540, 540+floor(540*3/4)=945
    assert [t["t"] for t in view["turns"]] == [0, 180, 540, 945]
    html = render_html(view)
    assert "?start=" in html
    # otherwise null
    assert all(t["t"] is None
               for t in build_view(item, com)["turns"])
    assert all(t["t"] is None
               for t in build_view(item, com, turns=turns)["turns"])
    assert all(t["t"] is None
               for t in build_view(item, com, youtube_id="dQw4w9WgXcQ")["turns"])


def test_no_iframe_when_youtube_none():
    view = build_view(_item(), _commentary(_item()))
    assert view["youtube_id"] is None
    assert "<iframe" not in render_html(view)


def test_validity_outcomes_dataset_render(tmp_path):
    item = _item()
    com = _commentary(item)
    outcomes = {"A": {"points": 10, "satisfaction": 1, "likeness": 2},
                "B": {"points": 8, "satisfaction": 0, "likeness": -1}}
    stats = {"C1": [0.4, 0.1, 0.6], "C2": [0.2, -0.1, 0.5],
             "n_dialogues": 30, "n_participants": 60}
    view = build_view(item, com, outcomes=outcomes, outcome_stats=stats)
    assert view["outcomes"] == outcomes
    assert view["validity"]["C1"] == [0.4, 0.1, 0.6]
    assert view["validity"]["C2"] == [0.2, -0.1, 0.5]
    assert view["validity"]["n"] == 30
    assert view["dataset"]["id"] == "chawla_casino_2021"
    html = render_html(view)
    assert "model reading of moves, not a judgment of people" in html
    assert "reported by participants after the negotiation" in html
    assert "held-out dialogues the curve tracks how much" in html
    # non-casino items carry no dataset credit
    plain = build_view(_item("yt-1"), _commentary(_item("yt-1")))
    assert plain["dataset"] is None


def test_cli_writes_html(tmp_path):
    from forecaster.viewer import main

    item, com = _item(), _commentary(_item())
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
    outcome_p = tmp_path / "outcome.json"
    outcome_p.write_text(json.dumps(
        {"C1": [0.4, 0.1, 0.6], "C2": [0.2, -0.1, 0.5],
         "n_dialogues": 30}))
    turns_p = tmp_path / "turns.jsonl"
    turns_p.write_text(
        '{"speaker": "A", "text": "aa", "chunk": 0}\n'
        '{"speaker": "B", "text": "bbbb", "chunk": 0}\n'
        '{"speaker": "A", "text": "ccc", "chunk": 1}\n'
        '{"speaker": "B", "text": "d", "chunk": 1}\n')
    out_p = tmp_path / "board.html"
    main(["--item", str(item_p), "--commentary", str(com_p),
          "--out", str(out_p), "--casino", str(casino_p),
          "--outcome", str(outcome_p), "--youtube-id", "dQw4w9WgXcQ",
          "--turns", str(turns_p)])
    html = out_p.read_text()
    assert "<iframe" in html
    data = _blob_data(html)
    assert [t["t"] for t in data["turns"]] == [0, 180, 540, 945]
    assert data["outcomes"]["A"]["points"] == 10
    assert data["validity"]["n"] == 30
