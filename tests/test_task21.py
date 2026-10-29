"""Contract for SPEC_task21_commentary (synthetic data only, fake clients)."""
import json

import pytest

from forecaster import commentary as CM


def _eff(idx, speaker, value, text="", dis=0.0):
    return {"index": idx, "speaker": speaker, "text": text,
            "value": value, "disagreement": dis}


def _item(cid, effects, moves, genre="negotiation"):
    return {"id": cid, "genre": genre, "moves": moves,
            "effects": effects, "hits": []}


def test_references_schema_unique():
    refs = CM.load_references()
    assert len(refs) > 0
    ids = []
    for r in refs:
        for key in ("id", "author", "title", "year", "idea_en",
                    "idea_ko", "moves", "marks"):
            assert key in r, f"missing {key} in {r.get('id')}"
        assert isinstance(r["moves"], list) and r["moves"]
        assert isinstance(r["marks"], list) and r["marks"]
        ids.append(r["id"])
    assert len(ids) == len(set(ids))
    assert "austin_uptake" in ids


def test_mark_for_thresholds():
    assert CM.mark_for(1.0, False) == "!"
    assert CM.mark_for(2.5, False) == "!"
    assert CM.mark_for(0.99, False) is None
    assert CM.mark_for(-1.0, False) == "?"
    assert CM.mark_for(-2.5, False) == "?"
    assert CM.mark_for(-0.99, False) is None
    assert CM.mark_for(0.0, True) == "!?"
    assert CM.mark_for(0.5, True) == "!?"
    assert CM.mark_for(-0.5, True) == "!?"
    assert CM.mark_for(0.0, False) is None
    assert CM.mark_for(1.5, True) == "!"
    assert CM.mark_for(-1.5, True) == "?"


def test_side_curves_carry_forward_balance():
    effects = [
        _eff(0, "A", 1.0),
        _eff(1, "B", 2.0),
        _eff(2, "A", -0.5),
        _eff(3, "X", 10.0),
    ]
    item = _item("t1", effects, [])
    c = CM.side_curves(item, {"mu": 0.0, "sd": 1.0})
    assert c["A"] == pytest.approx([1.0, 1.0, 0.5, 0.5])
    assert c["B"] == pytest.approx([0.0, 2.0, 2.0, 2.0])
    assert c["balance"] == pytest.approx([1.0, -1.0, -1.5, -1.5])
    # mu centering applies
    c2 = CM.side_curves(item, {"mu": 0.5, "sd": 1.0})
    assert c2["A"] == pytest.approx([0.5, 0.5, -0.5, -0.5])
    assert c2["B"] == pytest.approx([0.0, 1.5, 1.5, 1.5])
    assert c2["balance"] == pytest.approx([0.5, -1.0, -2.0, -2.0])


def _markov_toy():
    effects = [
        _eff(0, "A", 0.2, "t0"),
        _eff(1, "B", 0.3, "t1"),
        _eff(2, "A", 1.5, "t2"),
        _eff(3, "B", 0.1, "t3"),
    ]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
        {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
        {"turn_index": 2, "speaker": "A", "moves": ["inform"]},
        {"turn_index": 3, "speaker": "B", "moves": ["inform"]},
    ]
    markov = {
        "proceed": {"negotiate": 0.5, "proceed": 0.01},
        "negotiate": {"refuse": 0.01},
        "refuse": {"proceed": 0.5},
    }
    return _item("m1", effects, moves), {"mu": 0.0, "sd": 1.0}, markov


def test_pick_turns_order_k_extra():
    item, norm, markov = _markov_toy()
    # candidates: 1="!?"(0.3), 2="!"(1.5), 3="!?"(0.1)
    got = CM.pick_turns(item, norm, markov, k=3)
    assert [t["index"] for t in got] == [1, 2, 3]
    assert [t["mark"] for t in got] == ["!?", "!", "!?"]
    assert got[0]["label_ko"] == "특이수"
    assert got[1]["label_ko"] == "호수(좋은 수)"
    assert got[1]["label_en"] == "good"
    assert got[0]["label_en"] == "interesting"
    # k=1 picks top |z| (index 2) plus the first "!?" (index 1)
    got1 = CM.pick_turns(item, norm, markov, k=1)
    assert [t["index"] for t in got1] == [1, 2]
    # k=2 picks 2 and 1 by |z|, extra adds 3
    got2 = CM.pick_turns(item, norm, markov, k=2)
    assert [t["index"] for t in got2] == [1, 2, 3]
    # sorted by index, curve fields present
    for t in got:
        assert "curve_before" in t and "curve_after" in t
    # speaker A turn 2: A curve before=0.2, after=1.7
    t2 = next(t for t in got if t["index"] == 2)
    assert t2["curve_before"] == pytest.approx(0.2)
    assert t2["curve_after"] == pytest.approx(1.7)


def test_pick_turns_tie_by_index():
    effects = [_eff(0, "A", 1.5), _eff(1, "B", -1.5)]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
        {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
    ]
    item = _item("tie", effects, moves)
    got = CM.pick_turns(item, {"mu": 0.0, "sd": 1.0}, None, k=1)
    assert [t["index"] for t in got] == [0]


def test_choose_reference_overlap_used_fallback():
    refs = CM.load_references()
    # larger overlap wins: concede+counter matches cialdini (2) over single-overlap refs
    turn = {"mark": "!", "moves": ["concede", "counter"]}
    assert CM.choose_reference(turn, refs, ()).__class__ is dict
    assert CM.choose_reference(turn, refs, ())["id"] == "cialdini_reciprocity"
    # avoids used on ties: single propose, first table hit is fisher
    turn1 = {"mark": "!", "moves": ["propose"]}
    first = CM.choose_reference(turn1, refs, ())
    assert first["id"] == "fisher_ury_interests"
    second = CM.choose_reference(turn1, refs, (first["id"],))
    assert second["id"] != first["id"]
    assert second["id"] == "schelling_focal"
    # fallback to austin_uptake when no move overlap
    turn_none = {"mark": "!", "moves": ["no_such_move_xyz"]}
    assert CM.choose_reference(turn_none, refs, ())["id"] == "austin_uptake"
    turn_q = {"mark": "?", "moves": ["no_such_move_xyz"]}
    assert CM.choose_reference(turn_q, refs, ())["id"] == "austin_uptake"


class _FakeClient:
    def __init__(self, raw):
        self.raw = raw
        self.seen = []

    def complete(self, messages, model, temperature, **kw):
        self.seen.append((messages, model, temperature))
        return self.raw


def _turn_ref():
    turn = {"index": 5, "speaker": "A", "moves": ["propose"],
            "mark": "!", "label_ko": "호수(좋은 수)", "label_en": "good",
            "curve_before": 0.0, "curve_after": 1.5}
    ref = {"id": "fisher_ury_interests", "author": "Roger Fisher; William Ury",
           "title": "Getting to Yes: Negotiating Agreement Without Giving In",
           "year": 1981, "venue": "Houghton Mifflin",
           "idea_en": "Focus on interests, not positions.",
           "idea_ko": "입장이 아니라 이해에 집중."}
    return turn, ref


def test_llm_line_valid():
    turn, ref = _turn_ref()
    raw = json.dumps({"ko": "짧은 해설 한 줄", "en": "a short line"})
    out = CM.llm_line(_FakeClient(raw), turn, ref, "m")
    assert out["source"] == "llm"
    assert out["ko"] == "짧은 해설 한 줄"
    assert out["en"] == "a short line"


def test_llm_line_year_fallback():
    turn, ref = _turn_ref()
    raw = json.dumps({"ko": "해설 (1999) 인용", "en": "a line"})
    out = CM.llm_line(_FakeClient(raw), turn, ref, "m")
    assert out["source"] == "template_fallback"
    assert ref["idea_ko"] in out["ko"]


def test_llm_line_quoted_title_fallback():
    turn, ref = _turn_ref()
    raw = json.dumps({"ko": "해설 『Fake Book』 인용", "en": "a line"})
    out = CM.llm_line(_FakeClient(raw), turn, ref, "m")
    assert out["source"] == "template_fallback"


def test_llm_line_bad_json_fallback():
    turn, ref = _turn_ref()
    out = CM.llm_line(_FakeClient("not json at all"), turn, ref, "m")
    assert out["source"] == "template_fallback"
    assert ref["idea_en"] in out["en"]


def test_comment_template_only_known_refs():
    item, norm, markov = _markov_toy()
    refs = CM.load_references()
    ids = {r["id"] for r in refs}
    res = CM.comment(item, norm, markov, k=3, client=None)
    assert res["id"] == "m1"
    assert res["norm"] == {"mu": 0.0, "sd": 1.0}
    assert set(res["curves"]) == {"A", "B", "balance"}
    assert len(res["turns"]) == 3
    for t in res["turns"]:
        assert t["reference"]["id"] in ids
        assert set(t["reference"]) == {"id", "author", "title", "year", "venue"}
        assert t["line_ko"] and t["line_en"]
        assert t["source"] == "template"
