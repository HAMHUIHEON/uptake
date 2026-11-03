"""Task77: end-of-conversation standing read for one speaker."""
import hashlib
import json
import re

import pytest

from forecaster import multilayer as M
from forecaster.usage import BudgetExceeded

HI, LO = [0, 0, 0, .2, .8], [.8, .2, 0, 0, 0]
TURNS = [{"speaker": "A", "text": "I need the refund by Friday."},
         {"speaker": "B", "text": "Sure, I will send it today."}]
GOAL_SENT = ("Judge the outcome stake against the goal given for this speaker, not against "
             "what they seemed to ask for in a single turn.")


class Stub:
    def __init__(self, fn):
        self.fn, self.seen = fn, []

    def complete(self, m, **k):
        self.seen.append(m)
        out = self.fn(m)
        return out if isinstance(out, str) else json.dumps(out)


def test_prompts_differ_only_by_goal_sentence():
    p0, p1 = M.end_read_prompt(False), M.end_read_prompt(True)
    assert GOAL_SENT in p1 and GOAL_SENT not in p0
    assert p1.replace(GOAL_SENT + " ", "") == p0
    for p in (p0, p1):
        low = p.lower()
        assert "whole finished conversation" in low and "one named speaker" in low
        for s in M.END_STAKES:
            assert s in low
        assert M.BINS_NOTE in p
        assert "verbatim" in low and "original language" in low and "json only" in low
    assert M.END_STAKES == ("outcome", "relationship", "face", "understanding")


def test_goal_line_only_when_given():
    s = Stub(lambda m: {"stakes": {}})
    r = M.Reader(s, "m")
    M.read_end(r, TURNS, "A")
    u0 = s.seen[-1][1]["content"]
    assert "Read for speaker: A" in u0 and "Goal at the start:" not in u0
    assert "Conversation:" in u0 and "A: I need the refund by Friday." in u0
    assert "B: Sure, I will send it today." in u0
    assert s.seen[-1][0]["content"] == M.end_read_prompt(False)
    M.read_end(r, TURNS, "A", goal="Get the refund")
    u1 = s.seen[-1][1]["content"]
    assert "Read for speaker: A" in u1 and "Goal at the start: Get the refund" in u1
    assert s.seen[-1][0]["content"] == M.end_read_prompt(True)


def test_score_parses_like_read_stakes_and_null_stays_none():
    def fn(m):
        return {"stakes": {"outcome": {"score": HI, "quote": "I will send it today"},
                           "relationship": {"score": None, "quote": ""},
                           "face": {"score": LO, "quote": ""},
                           "understanding": {"score": None, "quote": ""}}}
    out = M.read_end(M.Reader(Stub(fn), "m"), TURNS, "A", goal="Get the refund")
    assert out["speaker"] == "A" and out["goal"] == "Get the refund"
    assert out["stakes"]["outcome"] == {**M.bins_to_score(HI), "quote": "I will send it today"}
    assert out["stakes"]["face"] == {**M.bins_to_score(LO), "quote": ""}
    assert out["stakes"]["relationship"] == {"score": None, "spread": None, "quote": ""}
    assert out["stakes"]["understanding"] == {"score": None, "spread": None, "quote": ""}


def test_non_verbatim_quote_emptied_score_stays():
    def fn(m):
        return {"stakes": {s: {"score": HI, "quote": "made up words"} for s in M.END_STAKES}}
    out = M.read_end(M.Reader(Stub(fn), "m"), TURNS, "A")
    exp = M.bins_to_score(HI)
    assert out["stakes"]["outcome"]["quote"] == ""
    assert out["stakes"]["outcome"]["score"] == exp["score"]
    assert out["stakes"]["outcome"]["spread"] == exp["spread"]


def test_malformed_json_returns_none_without_raising():
    out = M.read_end(M.Reader(Stub(lambda m: "not json {{"), "m"), TURNS, "A")
    assert all(v == {"score": None, "spread": None, "quote": ""} for v in out["stakes"].values())
    for bad in ({"nope": 1}, {"stakes": "garbage"}, {"stakes": {"outcome": "garbage"}},
                {"stakes": {"outcome": {"score": [1, 2], "quote": None}}}):
        out = M.read_end(M.Reader(Stub(lambda m, b=bad: b), "m"), TURNS, "A")
        assert out["stakes"]["outcome"]["score"] is None
        assert out["stakes"]["outcome"]["spread"] is None
        assert out["stakes"]["outcome"]["quote"] == ""


def test_existing_prompts_unchanged():
    blob = "\n".join(M.wanted_actual_prompt(c) for c in M.STAKE_ITEMS)
    assert hashlib.sha256(blob.encode()).hexdigest() == "b9a1db98c33ef020de3dd4d9ba85fedc9a1d5a7191a338cc34e8e2b22de9ba4f"


GOAL_SENT2 = ("Judge the outcome stake by comparing the settled list against the goal "
              "given for this speaker.")


def test_end_read2_prompts_differ_only_by_goal_sentence():
    p0, p1 = M.end_read2_prompt(False), M.end_read2_prompt(True)
    assert GOAL_SENT2 in p1 and GOAL_SENT2 not in p0
    assert p1.replace(GOAL_SENT2 + " ", "") == p0
    for p in (p0, p1):
        low = p.lower()
        assert "whole finished conversation" in low and "one named speaker" in low
        for s in M.END_STAKES:
            assert s in low
        assert "0 means far worse off" in low and "50 means no change" in low
        assert "100 means fully got what they wanted" in low
        assert M.BINS_NOTE in p
        assert "verbatim" in low and "original language" in low and "json only" in low
        assert "do not judge" in low and "do not judge the person" in low
        assert "settled" in low and "speaker_gets" in p and "others_get" in p
        assert "an empty list is correct when nothing was settled" in low
        assert "never infer an amount" in low
        assert "use null for an amount" in low
        assert "then score the four stakes using that list" in low
        assert low.index("first list what the conversation settled") < low.index("for each of the four stakes")
        assert '"settled"' in p and '"stakes"' in p
        for bad in ("points", "satisfaction", "labels", "deal record", "case name"):
            assert bad not in low
        for good in ("hats", "balls", "books", "rugs", "hat", "ball", "book", "rug"):
            assert re.search(r"\b" + re.escape(good) + r"\b", low) is None


def test_read_end2_same_user_message_as_read_end():
    s1 = Stub(lambda m: {"stakes": {}})
    M.read_end(M.Reader(s1, "m"), TURNS, "A", goal="Get the refund")
    s2 = Stub(lambda m: {"settled": [], "stakes": {}})
    M.read_end2(M.Reader(s2, "m"), TURNS, "A", goal="Get the refund")
    assert s2.seen[-1][1]["content"] == s1.seen[-1][1]["content"]
    assert "Goal at the start: Get the refund" in s2.seen[-1][1]["content"]
    assert s2.seen[-1][0]["content"] == M.end_read2_prompt(True)
    s1 = Stub(lambda m: {"stakes": {}})
    M.read_end(M.Reader(s1, "m"), TURNS, "A")
    s2 = Stub(lambda m: {"settled": [], "stakes": {}})
    M.read_end2(M.Reader(s2, "m"), TURNS, "A")
    assert s2.seen[-1][1]["content"] == s1.seen[-1][1]["content"]
    assert "Goal at the start:" not in s2.seen[-1][1]["content"]
    assert s2.seen[-1][0]["content"] == M.end_read2_prompt(False)


def test_read_end2_settled_parses_numbers_strings_nulls():
    def fn(m):
        return {"settled": [{"item": "a", "speaker_gets": 50, "others_get": "30"},
                            {"item": "b", "speaker_gets": None, "others_get": "3.5"},
                            {"item": "c", "speaker_gets": "0", "others_get": 0.0},
                            {"item": "d", "speaker_gets": " 12 ", "others_get": ""}],
                "stakes": {}}
    out = M.read_end2(M.Reader(Stub(fn), "m"), TURNS, "A")
    assert out["settled"] == [{"item": "a", "speaker_gets": 50.0, "others_get": 30.0},
                              {"item": "b", "speaker_gets": None, "others_get": 3.5},
                              {"item": "c", "speaker_gets": 0.0, "others_get": 0.0},
                              {"item": "d", "speaker_gets": 12.0, "others_get": None}]


def test_read_end2_malformed_rows_dropped_and_missing_empty():
    def fn(m):
        return {"settled": [{"item": "ok", "speaker_gets": 1, "others_get": 2},
                            "garbage", None, 123, ["x"],
                            {"no_item": 1}, {"item": 123}, {"item": None},
                            {"item": "bad-amounts", "speaker_gets": {"x": 1}, "others_get": [1]},
                            {"item": "also-ok", "speaker_gets": "nonsense", "others_get": True}],
                "stakes": {}}
    out = M.read_end2(M.Reader(Stub(fn), "m"), TURNS, "A")
    assert out["settled"] == [{"item": "ok", "speaker_gets": 1.0, "others_get": 2.0},
                              {"item": "bad-amounts", "speaker_gets": None, "others_get": None},
                              {"item": "also-ok", "speaker_gets": None, "others_get": None}]
    for bad in ({"stakes": {}}, {"settled": None, "stakes": {}}, {"settled": "garbage"},
                {"settled": {}, "stakes": {}}, "not json {{"):
        out = M.read_end2(M.Reader(Stub(lambda m, b=bad: b), "m"), TURNS, "A")
        assert out["settled"] == []


def test_read_end2_scores_parse_like_read_end():
    def fn(m):
        return {"settled": [], "stakes": {"outcome": {"score": HI, "quote": "I will send it today"},
                                          "relationship": {"score": None, "quote": ""},
                                          "face": {"score": LO, "quote": ""},
                                          "understanding": {"score": None, "quote": ""}}}
    out2 = M.read_end2(M.Reader(Stub(fn), "m"), TURNS, "A", goal="Get the refund")
    out1 = M.read_end(M.Reader(Stub(fn), "m"), TURNS, "A", goal="Get the refund")
    assert out2["stakes"] == out1["stakes"]
    assert out2["stakes"]["outcome"] == {**M.bins_to_score(HI), "quote": "I will send it today"}
    assert out2["stakes"]["face"] == {**M.bins_to_score(LO), "quote": ""}
    assert out2["stakes"]["relationship"] == {"score": None, "spread": None, "quote": ""}

    def fn_bad(m):
        return {"settled": [], "stakes": {s: {"score": HI, "quote": "made up words"} for s in M.END_STAKES}}
    out2 = M.read_end2(M.Reader(Stub(fn_bad), "m"), TURNS, "A")
    exp = M.bins_to_score(HI)
    assert out2["stakes"]["outcome"]["quote"] == ""
    assert out2["stakes"]["outcome"]["score"] == exp["score"]
    assert out2["stakes"]["outcome"]["spread"] == exp["spread"]

    out2 = M.read_end2(M.Reader(Stub(lambda m: "not json {{"), "m"), TURNS, "A")
    assert all(v == {"score": None, "spread": None, "quote": ""} for v in out2["stakes"].values())
    assert out2["settled"] == []


def test_budget_exceeded_propagates_from_both():
    def boom(m):
        raise BudgetExceeded("budget spent")
    with pytest.raises(BudgetExceeded):
        M.read_end(M.Reader(Stub(boom), "m"), TURNS, "A")
    with pytest.raises(BudgetExceeded):
        M.read_end2(M.Reader(Stub(boom), "m"), TURNS, "A")
    with pytest.raises(BudgetExceeded):
        M.read_end(M.Reader(Stub(boom), "m"), TURNS, "A", goal="Get the refund")
    with pytest.raises(BudgetExceeded):
        M.read_end2(M.Reader(Stub(boom), "m"), TURNS, "A", goal="Get the refund")


def test_other_errors_keep_none_behaviour():
    def boom(m):
        raise RuntimeError("transient")
    out1 = M.read_end(M.Reader(Stub(boom), "m"), TURNS, "A")
    assert all(v == {"score": None, "spread": None, "quote": ""} for v in out1["stakes"].values())
    out2 = M.read_end2(M.Reader(Stub(boom), "m"), TURNS, "A")
    assert all(v == {"score": None, "spread": None, "quote": ""} for v in out2["stakes"].values())
    assert out2["settled"] == []


def test_end_read_prompt_unchanged():
    assert hashlib.sha256(M.end_read_prompt(False).encode()).hexdigest() == "0bda399ceebb8be1190323d2c2ddb64adcf869122db29cf0730bccfdf0e016b7"
    assert hashlib.sha256(M.end_read_prompt(True).encode()).hexdigest() == "8b90fc3feb34201cc56909ca36a06609f7e2e762e5b00abeac57c121abd320f9"
