"""C5: alternatives are scene-written wordings with same-panel paired votes.

Model path only (stub client, no network): two model-written rewrites per
decisive turn, each scored against the original by one panel in both
presentation orders (ties count half), n>=8 with a Wilson interval,
method "paired", and a Laplace-smoothed point estimate so finite votes
never assert 0% or 100%.
"""
import json

import pytest

from forecaster import report as R
from forecaster.usage import Meter


TURNS = [
    {"speaker": "A", "text": "We have a problem up here."},
    {"speaker": "B", "text": "Say again your situation."},
    {"speaker": "A", "text": "We need priority, we are low on fuel."},
]

ORIGINAL = "We need priority, we are low on fuel."

REWRITES = [
    "Declaring minimum fuel, request priority landing now, will confirm short final.",
    "We are critically low on fuel and need immediate vectors to land; please confirm.",
]


class StubClient:
    """Generation returns REWRITES; votes follow vote_plan in call order."""

    def __init__(self, vote_plan, rewrites=None):
        self.vote_plan = list(vote_plan)
        self.rewrites = REWRITES if rewrites is None else rewrites
        self.calls = []

    def complete(self, messages, model=None, temperature=0.0, **kwargs):
        system = messages[0]["content"] if messages else ""
        self.calls.append(system)
        if "two different ways" in system:
            return json.dumps({"alternatives": list(self.rewrites)})
        if self.vote_plan:
            pick = self.vote_plan.pop(0)
        else:
            pick = "Tie"
        return json.dumps({"clearer": pick})


def _mixed_plan():
    # Per alternative, both orders x five stances: 6 wins, 2 ties, 2 losses.
    # Order 0: first=original so alt wins are "B"; order 1 is flipped.
    one = (["B", "B", "B", "Tie", "A"]
           + ["A", "A", "A", "Tie", "B"])
    return one + list(one)


def _unanimous_plan(winner):
    # winner="alt": every vote prefers the alternative in both orders.
    if winner == "alt":
        one = ["B"] * 5 + ["A"] * 5
    else:
        one = ["A"] * 5 + ["B"] * 5
    return one + list(one)


def _build(client, tmp_path, meter=None):
    cache_dir = tmp_path / "cache"
    return R.build_alternatives(
        ORIGINAL, [], {}, client=client, model="stub-model",
        meter=meter, cache_dir=cache_dir, turns=TURNS,
        decisive_turn=2, speaker="A",
        intended="appeal_need: ask for priority or help with an urgent need",
        original=ORIGINAL)


def test_paired_votes_with_interval_and_model_texts(tmp_path):
    client = StubClient(_mixed_plan())
    alts = _build(client, tmp_path, Meter(budget_usd=5.0))
    assert len(alts) >= 2
    for alt in alts:
        assert alt["method"] == "paired"
        assert alt["n_compared"] >= 8
        assert 0.0 < alt["p_better"] < 1.0
        assert alt["ci_low"] < alt["ci_high"]
        assert alt["ci_low"] <= alt["p_better"] <= alt["ci_high"]
        assert alt["ci_low"] <= alt["p_empirical"] <= alt["ci_high"]
    # (6 wins + 2 x half ties + 1) / (10 + 2) = 8/12.
    assert alts[0]["p_better"] == pytest.approx(0.667, abs=0.001)
    assert alts[0]["p_empirical"] == pytest.approx(0.7)
    texts = [a["text"] for a in alts]
    assert texts == REWRITES
    for sentinel in ("To be explicit:", "this is urgent, and I need action now"):
        assert all(sentinel not in t for t in texts)


def test_unanimous_votes_never_assert_100(tmp_path):
    client = StubClient(_unanimous_plan("alt"))
    alts = _build(client, tmp_path)
    for alt in alts:
        assert alt["method"] == "paired"
        assert alt["n_compared"] >= 8
        assert 0.0 < alt["p_better"] < 1.0
        assert alt["p_empirical"] == pytest.approx(1.0)
        # Laplace (10 + 1) / (10 + 2).
        assert alt["p_better"] == pytest.approx(11 / 12, abs=0.001)


def test_unanimous_against_never_asserts_0(tmp_path):
    client = StubClient(_unanimous_plan("orig"))
    alts = _build(client, tmp_path)
    for alt in alts:
        assert 0.0 < alt["p_better"] < 1.0
        assert alt["p_empirical"] == pytest.approx(0.0)
        assert alt["p_better"] == pytest.approx(1 / 12, abs=0.001)


def test_all_ties_land_near_half(tmp_path):
    client = StubClient(["Tie"] * 20)
    alts = _build(client, tmp_path)
    for alt in alts:
        assert alt["p_empirical"] == pytest.approx(0.5)
        assert alt["p_better"] == pytest.approx(0.5, abs=0.001)


def test_same_inputs_are_cached(tmp_path):
    client = StubClient(_mixed_plan())
    first = _build(client, tmp_path)
    n_calls = len(client.calls)
    assert n_calls > 0
    second = _build(client, tmp_path)
    assert len(client.calls) == n_calls
    assert [a["text"] for a in second] == [a["text"] for a in first]
    assert [a["p_better"] for a in second] == [a["p_better"] for a in first]


def test_budget_stop_falls_back_without_crashing(tmp_path):
    client = StubClient(_mixed_plan())
    alts = _build(client, tmp_path, Meter(budget_usd=0.0))
    assert len(alts) >= 2
    assert all(a["method"] == "offline-proxy" for a in alts)
    assert len(client.calls) == 0


def test_bad_rewrites_fall_back_to_templates_but_still_pair(tmp_path):
    client = StubClient(_mixed_plan(), rewrites=["same", "same"])
    alts = _build(client, tmp_path)
    assert all(a["method"] == "paired" for a in alts)
    assert any("To be explicit:" in a["text"] for a in alts)


# --- the criterion: one item from the decisive turn -----------------------------

USE = {"deliverable", "warmth", "competence", "facts"}
ITEM_GRADES = {"deliverable": "measured", "warmth": "measured", "competence": "measured",
               "facts": "measured", "trust": "reference", "commitment": "reference",
               "status": "reference", "time": "ungraded"}
SAID = [{"speaker": s, "text": f"Plain words of turn {i}."} for i, s in enumerate("ABAB")]
INTENDED = "Asked the other side to name a date."


def _item(cls, wanted, actual):
    return {"class": cls, "wanted": {"score": wanted, "spread": 5.0},
            "actual": {"score": actual, "spread": 5.0}, "reply_turn": 3, "quote": "not yet"}


def _reading(**items):
    from forecaster.multilayer import ITEM_CLASS
    return {"turn": 2, "speaker": "A",
            "items": {k: _item(ITEM_CLASS[k], *v) for k, v in items.items()}}


def _judged():
    return {"turn": 2, "why": "It closed the offer.", "opened": INTENDED,
            "received": "Gave no date.", "source": "model estimate (m, rep 0)"}


def _decisive(reading):
    return R.build_decisive("t1", SAID, {}, {}, {}, [], uptake=[reading], use=USE,
                            judged=_judged(), grades=ITEM_GRADES)


def test_criterion_takes_the_largest_wanted_minus_actual_among_measured():
    # By wanted x (100 - actual) the first item is the larger; by the difference
    # the second one is. A bigger reference gap does not count while a measured
    # item broke; an item that did not break does not count at all.
    reading = _reading(deliverable=(90.0, 50.0), facts=(60.0, 10.0),
                       trust=(95.0, 5.0), warmth=(90.0, 80.0))
    got = R.pick_criterion(reading, USE, ITEM_GRADES, 2)
    assert (got["item"], got["grade"], got["wanted"], got["actual"]) == ("facts", "measured", 60.0, 10.0)
    assert "#2" in got["source"] and "measured broken" in got["source"]
    assert set(got) == {"item", "grade", "wanted", "actual", "source"}


def test_criterion_tie_goes_to_the_first_item():
    reading = _reading(deliverable=(80.0, 20.0), facts=(70.0, 10.0))
    assert R.pick_criterion(reading, USE, ITEM_GRADES, 2)["item"] == "deliverable"


def test_criterion_widens_to_reference_only_when_no_measured_item_broke():
    reading = _reading(facts=(60.0, 90.0), trust=(70.0, 30.0), status=(90.0, 20.0))
    got = R.pick_criterion(reading, USE, ITEM_GRADES, 2)
    assert (got["item"], got["grade"], got["wanted"], got["actual"]) == ("status", "reference", 90.0, 20.0)
    assert "reference-grade" in got["source"]


def test_criterion_without_broken_item_says_so():
    for reading in (_reading(facts=(60.0, 90.0), deliverable=(20.0, 10.0)), None):
        got = R.pick_criterion(reading, USE, ITEM_GRADES, 2)
        assert got["item"] is None and got["grade"] is None
        assert got["wanted"] is None and got["actual"] is None
        assert got["source"].startswith("no broken item")


def test_ungraded_item_is_not_widened_to():
    got = R.pick_criterion(_reading(time=(90.0, 10.0)), USE, ITEM_GRADES, 2)
    assert got["item"] is None


def test_decisive_follows_the_measured_criterion():
    d = _decisive(_reading(deliverable=(90.0, 50.0), facts=(60.0, 10.0)))
    assert (d["item"], d["grade"], d["wanted"], d["actual"]) == ("facts", "measured", 60.0, 10.0)
    assert d["read"] == "Gave no date. (reading: facts landed 10/100)"
    assert d["criterion"]["item"] == "facts"


def test_decisive_marks_a_reference_criterion():
    d = _decisive(_reading(trust=(70.0, 30.0)))
    assert (d["item"], d["grade"], d["wanted"], d["actual"]) == ("trust", "reference", 70.0, 30.0)
    assert d["read"] == "Gave no date. (reading: trust landed 30/100, reference grade)"


def test_decisive_without_broken_item_says_so():
    d = _decisive(_reading(facts=(60.0, 90.0)))
    assert "item" not in d and "wanted" not in d and "actual" not in d
    assert d["read"] == "Gave no date. (reading: no broken item of measured or reference grade)"
    assert d["intended"] == INTENDED


def test_decisive_rule_path_uses_the_criterion_item():
    from forecaster.multilayer import build_alerts_items
    # The severity gap of deliverable is the larger, so it picks the turn;
    # the criterion is still the largest difference among broken items.
    grades = {"status": "measured", "items": ITEM_GRADES, "cuts": [10, 20, 30, 40], "causes": {}}
    reading = _reading(deliverable=(90.0, 50.0), facts=(60.0, 10.0))
    build_alerts_items([reading], {"stakes_weight": 90.0}, grades)
    d = R.build_decisive("t1", SAID, {}, {}, {}, [], uptake=[reading], use=USE, grades=ITEM_GRADES)
    assert d["turn"] == 2 and d["item"] == "facts"
    assert d["intended"].startswith("facts: reached for 60/100")
    assert d["read"].startswith("facts: the wanted effect landed 10/100")
    assert "facts: largest wanted minus actual" in d["intended_source"]


def test_alternatives_carry_and_vote_on_the_criterion(tmp_path):
    criterion = R.pick_criterion(_reading(facts=(60.0, 10.0)), USE, ITEM_GRADES, 2)
    seen = []

    class Client(StubClient):
        def complete(self, messages, model=None, temperature=0.0, **kwargs):
            seen.append(messages[1]["content"])
            return super().complete(messages, model=model, temperature=temperature, **kwargs)

    alts = R.build_alternatives(ORIGINAL, [], {}, client=Client(_mixed_plan()), model="stub-model",
                                cache_dir=tmp_path / "cache", turns=TURNS, decisive_turn=2,
                                speaker="A", intended=INTENDED, original=ORIGINAL,
                                criterion=criterion)
    assert all(a["criterion"] == criterion for a in alts)
    assert len(seen) == 1 + 20
    assert all("Criterion: the effect A wanted on facts (" in user for user in seen)


def test_without_broken_item_prompts_use_the_intended_sentence(tmp_path):
    seen = []

    class Client(StubClient):
        def complete(self, messages, model=None, temperature=0.0, **kwargs):
            seen.append(messages[1]["content"])
            return super().complete(messages, model=model, temperature=temperature, **kwargs)

    alts = R.build_alternatives(ORIGINAL, [], {}, client=Client(_mixed_plan()), model="stub-model",
                                cache_dir=tmp_path / "cache", turns=TURNS, decisive_turn=2,
                                speaker="A", intended=INTENDED, original=ORIGINAL)
    assert alts[0]["criterion"]["item"] is None
    assert all(f"Criterion: the effect A wanted, as stated in this sentence: {INTENDED}" in u
               for u in seen)


def test_vote_and_rewrite_caches_split_by_criterion(tmp_path):
    client = StubClient([])
    measured = R.pick_criterion(_reading(facts=(60.0, 10.0)), USE, ITEM_GRADES, 2)
    other = R.pick_criterion(_reading(deliverable=(60.0, 10.0)), USE, ITEM_GRADES, 2)

    def run(criterion):
        return R.build_alternatives(ORIGINAL, [], {}, client=client, model="stub-model",
                                    cache_dir=tmp_path / "cache", turns=TURNS, decisive_turn=2,
                                    speaker="A", intended=INTENDED, original=ORIGINAL,
                                    criterion=criterion)

    run(measured)
    n_calls = len(client.calls)
    assert n_calls == 1 + 20
    run(measured)
    assert len(client.calls) == n_calls
    run(other)
    assert len(client.calls) == 2 * n_calls


def _stub_case(reading, tmp_path, monkeypatch):
    """A synthetic case on disk whose decisive turn carries this reading."""
    import forecaster.multilayer as ML
    hs = tmp_path / "highstakes"
    hs.mkdir(parents=True)
    with open(hs / "casino-t1_turns.jsonl", "w", encoding="utf-8") as f:
        for t in SAID:
            f.write(json.dumps(t) + "\n")
    with open(hs / "casino-t1_case.json", "w", encoding="utf-8") as f:
        json.dump({"title": "Synthetic case", "outcome": "Synthetic outcome.",
                   "source_url": "https://example.com/r", "outcome_turn": 3}, f)
    grades = {"status": "measured", "items": ITEM_GRADES, "cuts": [10, 20, 30, 40], "causes": {}}
    real = ML.build_multilayer

    def fake(turns, **kwargs):
        got = real(turns, grades=grades)
        got["uptake"][2]["items"].update(reading["items"])
        got["alerts"] = ML.build_alerts_items(got["uptake"], {"stakes_weight": 90.0}, grades)
        return got

    monkeypatch.setattr(ML, "build_multilayer", fake)


def _section_six(reading, tmp_path, monkeypatch):
    """Section 6 of the page and the report, from a stubbed reading."""
    _stub_case(reading, tmp_path, monkeypatch)
    report = R.build_report("t1", data_root=str(tmp_path))
    page = R.render_html(report)
    return page.split('id="alternatives"')[1].split("</section>")[0], report


def test_report_shows_a_measured_criterion(tmp_path, monkeypatch):
    section, r = _section_six(_reading(facts=(60.0, 10.0)), tmp_path, monkeypatch)
    assert r["alternatives_control"]["criterion"] == r["alternatives"][0]["criterion"]
    assert r["alternatives_control"]["criterion"]["item"] == "facts"
    assert r["decisive"]["item"] == "facts" and r["decisive"]["grade"] == "measured"
    assert "Wordings are compared on facts (measured grade): wanted 60/100, landed 10/100." in section


def test_report_shows_a_reference_criterion(tmp_path, monkeypatch):
    section, r = _section_six(_reading(facts=(60.0, 90.0), trust=(70.0, 30.0)), tmp_path, monkeypatch)
    assert r["alternatives_control"]["criterion"]["grade"] == "reference"
    assert (r["decisive"]["item"], r["decisive"]["grade"]) == ("trust", "reference")
    assert "reference grade" in r["decisive"]["read"]
    assert "Wordings are compared on trust (reference grade): wanted 70/100, landed 30/100." in section


def test_report_shows_no_broken_item(tmp_path, monkeypatch):
    section, r = _section_six(_reading(facts=(60.0, 90.0)), tmp_path, monkeypatch)
    assert r["alternatives_control"]["criterion"]["item"] is None
    assert "item" not in r["decisive"]
    assert "Wordings are compared on the speaker's intended sentence" in section
    assert "no broken item of measured or reference grade on turn #2" in section
    assert "None" not in section


def test_limit_states_the_real_criterion_not_clarity(tmp_path, monkeypatch):
    import forecaster.reader as RD
    _stub_case(_reading(facts=(60.0, 10.0)), tmp_path, monkeypatch)
    monkeypatch.setattr(RD, "MuseClient", lambda meter=None: StubClient([]))
    r = R.build_report("t1", data_root=str(tmp_path), model="m", cache_dir=tmp_path / "cache")
    limits = " ".join(r["reliability"]["limits"])
    assert "clarity" not in limits
    assert "paired votes on which wording better produces the effect A wanted on facts (" in limits
