"""Contract for the one-call conversation judgment and its use in the report.

Synthetic conversations and a fake client only (no network, no private data).
"""
import json

import pytest

from forecaster import judgment as J
from forecaster import report as R
from forecaster.usage import BudgetExceeded

TURNS = [
    {"speaker": "Ana", "text": "Please send the signed form today."},
    {"speaker": "Ben", "text": "The form is with legal, I cannot send it."},
    {"speaker": "Ana", "text": "Then  tell me who has it."},
    {"speaker": "Ben", "text": "Someone over there, I think."},
]
SCENES = [{"id": "S1", "발화": "a line nobody here said at all", "상황_한줄": "two envoys part",
           "교훈_en": "Say what you heard."}]
IDS = {"S1"}


def _answer(**over):
    data = {
        "divides": [{"turn": 2, "why": "The request became a demand for a name.",
                     "wanted": "a name to chase", "happened": "only a vague pointer"}],
        "decisive_turn": 2, "decisive_why": "Nothing firm was on offer after it.",
        "turns": [
            {"turn": 0, "opened": "Asked for the form.", "response_turn": 1, "received": "Passed it to legal."},
            {"turn": 1, "opened": "Declined and named legal.", "response_turn": 2, "received": "Pressed for a name."},
            {"turn": 2, "opened": "Narrowed the ask to a name.", "response_turn": 3, "received": "Gave no name."},
            {"turn": 3, "opened": "Closed vaguely.", "response_turn": None, "received": None},
        ],
        "speakers": {
            "Ana": {"goal": "Ana wanted the signed form sent today.",
                    "shifts": [{"turn": 2, "from": "asking", "to": "pressing", "quote": "tell me who has it"}]},
            "Ben": {"goal": "Ben wanted to pass the request along.", "shifts": []},
        },
        "scene": {"scene_id": "S1", "why": "Both end on what was not confirmed."},
    }
    data.update(over)
    return data


class FakeClient:
    """Answers the judgment prompt with ``answer``; other prompts with an empty object."""

    def __init__(self, answer=None, fail=False):
        self.answer, self.fail, self.calls, self.prompts = answer, fail, 0, []

    def complete(self, messages, model=None, temperature=None, **kwargs):
        if messages[0]["content"] != J.JUDGE_PROMPT:
            return "{}"
        self.calls += 1
        self.prompts.append("\n".join(m["content"] for m in messages))
        if self.fail:
            raise RuntimeError("service down")
        return json.dumps(self.answer if self.answer is not None else _answer())


def _judge(client, tmp_path, turns=TURNS, rep=0, monkey=None):
    return J.judge_conversation(turns, [], [], {}, client, "m", None, tmp_path, rep=rep)


@pytest.fixture(autouse=True)
def _small_catalog(monkeypatch):
    monkeypatch.setattr(J, "load_scenes", lambda: list(SCENES))
    monkeypatch.setattr(R, "_load_scene_list", lambda: list(SCENES))


# --- input ----------------------------------------------------------------------

def test_prompt_holds_the_turns_values_and_conditions_and_no_record(tmp_path):
    item = {"class": "outcome", "wanted": {"score": 80.0, "spread": 5.0}, "actual": {"score": 20.0, "spread": 5.0},
            "reply_turn": 1, "quote": "cited words"}
    uptake = [{"turn": 0, "speaker": "Ana", "items": {"deliverable": item},
               "layers": {"severity": 12.5}, "title": "SECRET-TITLE"}]
    alerts = [{"turn": 0, "level": "warning", "reason": "SECRET-REASON"}]
    conditions = {"relationship": "coworkers", "power": 10.0, "channel": "chat", "stakes_weight": 40.0,
                  "evidence": "SECRET-EVIDENCE", "culture": "SECRET-CULTURE"}
    client = FakeClient()
    turns = [{**t, "id": "SECRET-ID", "note": "SECRET-NOTE", "outcome": "SECRET-OUTCOME"} for t in TURNS]
    J.judge_conversation(turns, uptake, alerts, conditions, client, "m", None, tmp_path)
    prompt = client.prompts[0]
    assert "SECRET" not in prompt and "cited words" not in prompt
    for shown in ("Ana: Please send the signed form today.", "deliverable w80 a20", '"severity": 12.5',
                  "alert warning", '"relationship": "coworkers"', '"power": 10.0', '"channel": "chat"',
                  '"stakes_weight": 40.0', "S1: two envoys part / Say what you heard."):
        assert shown in prompt, shown


def test_report_passes_no_record_field_to_the_judgment(tmp_path, monkeypatch):
    _write_case(tmp_path)
    client = _use_client(monkeypatch, FakeClient())
    R.build_report("t1", data_root=str(tmp_path), model="m", cache_dir=tmp_path / "cache")
    prompt = client.prompts[0]
    for hidden in ("t1", "Hidden title words", "Hidden outcome words", "Hidden note words"):
        assert hidden not in prompt, hidden


def test_speaker_who_first_talks_after_turn_thirteen_is_in_the_prompt(tmp_path):
    turns = [{"speaker": "A" if i % 2 else "B", "text": f"Routine line {i}."} for i in range(14)]
    turns.append({"speaker": "Late", "text": "I only join at the end."})
    client = FakeClient()
    _judge(client, tmp_path, turns)
    assert "#14 Late: I only join at the end." in client.prompts[0]
    assert "- Late: turns #14" in client.prompts[0]


def test_goal_path_sees_every_speakers_turn(tmp_path):
    turns = [{"speaker": "A" if i % 2 else "B", "text": f"Routine line {i}."} for i in range(14)]
    turns.append({"speaker": "Late", "text": "I only join at the end."})
    seen = []

    class Goals:
        def complete(self, messages, model=None, temperature=None):
            seen.append(messages[1]["content"])
            return json.dumps({"goals": {"Late": "Late wanted to join the plan."}})

    got = R.model_speaker_goals(turns, {"Late": "from own turns"}, Goals(), "m", None, tmp_path)
    assert "Late: I only join at the end." in seen[0]
    assert got == {"Late": "Late wanted to join the plan."}


def test_scene_recounting_this_same_event_is_not_offered():
    turns = [{"speaker": "A", "text": "the gate is closed so we hold here until the second truck comes"}]
    same = {"id": "X", "발화": "the gate is closed so we hold here until the second", "교훈_en": "x"}
    other = {"id": "Y", "발화": "the gate is closed today", "교훈_en": "y"}
    assert [s["id"] for s in J.offered_scenes(turns, [same, other])] == ["Y"]


# --- cache and repeats ----------------------------------------------------------

def test_same_rep_uses_the_cache_and_new_rep_asks_again(tmp_path):
    client = FakeClient()
    first = _judge(client, tmp_path, rep=0)
    assert client.calls == 1
    again = _judge(client, tmp_path, rep=0)
    assert client.calls == 1 and again == first
    repeat = _judge(client, tmp_path, rep=1)
    assert client.calls == 2 and repeat["rep"] == 1
    _judge(client, tmp_path, rep=1)
    assert client.calls == 2


# --- checks ---------------------------------------------------------------------

def test_valid_answer_passes_whole():
    got = J.validate(_answer(), TURNS, IDS)
    assert got["dropped"] == []
    assert [d["turn"] for d in got["divides"]] == [2]
    assert got["decisive"]["turn"] == 2
    assert got["speakers"]["Ana"]["shifts"][0]["evidence_quote"] == "tell me who has it"
    assert got["speakers"]["Ben"]["shifts"] == []
    assert got["scene"]["scene_id"] == "S1"


@pytest.mark.parametrize("turn", [99, -1, "2", True, None])
def test_turn_out_of_range_is_dropped(turn):
    answer = _answer(divides=[{"turn": turn, "why": "w", "wanted": "a", "happened": "b"}], decisive_turn=None)
    assert J.validate(answer, TURNS, IDS)["divides"] is None


def test_reply_must_be_later_and_by_another_speaker():
    for bad in (0, 2, 99):  # backward, the speaker's own later turn, out of range
        answer = _answer(turns=[{"turn": 0 if bad != 0 else 1, "opened": "Asked.", "response_turn": bad,
                                 "received": "Took it."}],
                         divides=[], decisive_turn=None)
        got = J.validate(answer, TURNS, IDS)
        assert got["turns"][0]["response_turn"] is None
        assert got["turns"][0]["received"] is None
        assert any("reply" in d for d in got["dropped"])


def test_divide_without_a_valid_reply_is_dropped():
    turns = _answer()["turns"]
    turns[2]["response_turn"] = 2
    got = J.validate(_answer(turns=turns), TURNS, IDS)
    assert got["divides"] is None and got["decisive"] is None


def test_quote_not_in_that_turn_is_dropped():
    shifts = [{"turn": 2, "from": "a", "to": "b", "quote": "words nobody said"},
              {"turn": 0, "from": "a", "to": "b", "quote": "tell me who has it"},
              {"turn": 2, "from": "a", "to": "b", "quote": "tell me who has it"}]
    answer = _answer(speakers={"Ana": {"goal": "Ana wanted the form.", "shifts": shifts}})
    got = J.validate(answer, TURNS, IDS)
    assert len(got["speakers"]["Ana"]["shifts"]) == 1
    assert got["speakers"]["Ana"]["shifts"][0]["turn"] == 2


def test_shift_by_another_speaker_is_dropped():
    shifts = [{"turn": 1, "from": "a", "to": "b", "quote": "The form is with legal"}]
    got = J.validate(_answer(speakers={"Ana": {"goal": "Ana wanted the form.", "shifts": shifts}}), TURNS, IDS)
    assert got["speakers"]["Ana"]["shifts"] is None


def test_unknown_scene_id_is_dropped_not_read_as_no_fit():
    got = J.validate(_answer(scene={"scene_id": "NOPE", "why": "It fits."}), TURNS, IDS)
    assert got["scene"] is None and "scene: 'NOPE'" in got["dropped"]


def test_decisive_outside_the_kept_divides_is_dropped():
    got = J.validate(_answer(decisive_turn=0), TURNS, IDS)
    assert got["decisive"] is None and got["divides"]


def test_label_shaped_goal_and_invented_score_are_dropped():
    speakers = {"Ana": {"goal": "appeal_need: ask for priority", "shifts": []},
                "Ben": {"goal": "Ben landed the deflection 85/100.", "shifts": []}}
    got = J.validate(_answer(speakers=speakers), TURNS, IDS)
    assert got["speakers"]["Ana"]["goal"] is None and got["speakers"]["Ben"]["goal"] is None


def test_empty_lists_stay_empty_and_nothing_fills_them():
    got = J.validate(_answer(divides=[], decisive_turn=None), TURNS, IDS)
    assert got["divides"] == [] and got["decisive"] is None


# --- failure --------------------------------------------------------------------

def test_failed_call_and_no_client_are_code_fallback(tmp_path):
    down = _judge(FakeClient(fail=True), tmp_path)
    assert down["source"] == "code fallback: model call failed (RuntimeError)"
    assert down["divides"] is None and down["speakers"] == {}
    assert _judge(None, tmp_path)["source"] == "code fallback: no model for this run"


def test_reply_with_nothing_valid_is_code_fallback(tmp_path):
    got = _judge(FakeClient(answer={"unrelated": 1}), tmp_path)
    assert got["source"] == "code fallback: model reply had no valid field"


def test_budget_stop_is_not_swallowed(tmp_path):
    class Spent:
        def complete(self, *a, **k):
            raise BudgetExceeded("over")

    with pytest.raises(BudgetExceeded):
        _judge(Spent(), tmp_path)


# --- report ---------------------------------------------------------------------

def _write_case(root, case_id="t1"):
    hs = root / "highstakes"
    hs.mkdir(parents=True, exist_ok=True)
    with open(hs / f"casino-{case_id}_turns.jsonl", "w", encoding="utf-8") as f:
        for t in TURNS:
            f.write(json.dumps(t) + "\n")
    with open(hs / f"casino-{case_id}_case.json", "w", encoding="utf-8") as f:
        json.dump({"title": "Hidden title words", "outcome": "Hidden outcome words",
                   "note": "Hidden note words", "source_url": "https://example.com/r", "outcome_turn": 3}, f)


def _item(cls, wanted, actual):
    return {"class": cls, "wanted": {"score": wanted, "spread": 5.0}, "actual": {"score": actual, "spread": 5.0},
            "reply_turn": None, "quote": ""}


def _use_client(monkeypatch, client, readings=True):
    import forecaster.multilayer as ML
    import forecaster.reader as RD
    monkeypatch.setattr(RD, "MuseClient", lambda meter=None: client)
    if readings:
        grades = {"status": "measured", "items": {"deliverable": "measured"}, "causes": {}, "cuts": [10, 20, 30, 40]}
        real = ML.build_multilayer

        def fake(turns, **kwargs):
            got = real(turns, grades=grades)
            got["uptake"][2]["items"]["deliverable"] = _item("outcome", 80.0, 20.0)
            got["uptake"][2]["broken"] = ["deliverable"]
            got["conditions"] = {"relationship": "coworkers", "stakes_weight": 90.0, "source": "model estimate"}
            got["alerts"] = ML.build_alerts_items(got["uptake"], got["conditions"], grades)
            got["standing"] = ML.build_standing_items(got["uptake"], use={"deliverable"})
            return got

        monkeypatch.setattr(ML, "build_multilayer", fake)
    return client


def _built(tmp_path, monkeypatch, answer=None, fail=False, rep=0):
    _write_case(tmp_path)
    _use_client(monkeypatch, FakeClient(answer=answer, fail=fail))
    return R.build_report("t1", data_root=str(tmp_path), model="m", cache_dir=tmp_path / "cache", judgment_rep=rep)


def test_report_fields_come_from_the_judgment(tmp_path, monkeypatch):
    r = _built(tmp_path, monkeypatch)
    assert {k: r["judgment"][k] for k in ("model", "rep", "source")} == {"model": "m", "rep": 0, "source": "Muse"}
    assert r["decisive"]["turn"] == 2
    assert r["decisive"]["intended"] == "Narrowed the ask to a name."
    assert r["decisive"]["intended_source"].startswith("model estimate")
    assert r["turning_points_source"] == "model judgment"
    tp = r["turning_points"][0]
    assert (tp["turn"], tp["speaker"], tp["received_by"], tp["response_turn"]) == (2, "Ana", "Ben", 3)
    assert tp["opened"] == "Narrowed the ask to a name." and tp["received_as"] == "Gave no name."
    assert tp["source"].startswith("model estimate") and tp["intentional"] == "unclear"
    ana = r["speakers"][0]
    assert ana["initial_goal"] == "Ana wanted the signed form sent today."
    assert ana["goal_source"].startswith("model estimate")
    assert ana["shifts"][0]["evidence_quote"] == "tell me who has it"
    assert r["speakers"][1]["shifts"] == []
    assert any("may be influenced by what it knows" in x for x in r["reliability"]["limits"])


def test_report_values_come_from_the_reading_not_the_model(tmp_path, monkeypatch):
    answer = _answer()
    answer["divides"][0].update(wanted_score=11, actual_score=99, severity=77, item="trust")
    answer.update(wanted=11, actual=99)
    r = _built(tmp_path, monkeypatch, answer=answer)
    tp = r["turning_points"][0]
    assert (tp["item"], tp["wanted"], tp["actual"]) == ("deliverable", 80.0, 20.0)
    assert "deliverable wanted 80/100, landed 20/100" in tp["wanted_vs_happened"]
    assert (r["decisive"]["item"], r["decisive"]["wanted"], r["decisive"]["actual"]) == ("deliverable", 80.0, 20.0)
    assert "landed 20/100" in r["decisive"]["read"]
    assert "99" not in json.dumps(r["turning_points"]) and "77" not in json.dumps(r["turning_points"])


def test_judgment_changes_no_reading_or_alert(tmp_path, monkeypatch):
    judged = _built(tmp_path, monkeypatch)
    down = _built(tmp_path / "other", monkeypatch, fail=True)
    assert judged["uptake"] == down["uptake"]
    assert judged["alerts"] == down["alerts"]
    assert judged["standing"] == down["standing"]


def test_perspective_is_copied_from_the_catalog(tmp_path, monkeypatch):
    answer = _answer(scene={"scene_id": "S1", "why": "Both end unconfirmed.", "line": "An invented line."})
    catalog = [{**SCENES[0], "원문출처": ["https://example.com/scene"], "작품": "w", "작가": "a"}]
    monkeypatch.setattr(R, "_load_scene_list", lambda: catalog)
    r = _built(tmp_path, monkeypatch, answer=answer)
    p = r["perspective"]
    assert (p["line"], p["source"], p["scene_id"]) == ("Say what you heard.", "https://example.com/scene", "S1")
    assert p["why"] == "Both end unconfirmed."
    page = R.render_html(r)
    assert "An invented line." not in page and '<a href="https://example.com/scene">' in page


def test_no_fitting_scene_is_shown_as_such(tmp_path, monkeypatch):
    r = _built(tmp_path, monkeypatch, answer=_answer(scene={"scene_id": None, "why": "None is about a handoff."}))
    p = r["perspective"]
    assert p["scene_id"] is None
    assert p["line"] == "No scene in the catalog fits this conversation."
    assert p["source"] == "model estimate (m, rep 0): no fitting scene among 1"
    assert "no fitting scene among 1" in R.render_html(r) and 'href="model' not in R.render_html(r)


def test_zero_turning_points_from_the_model_are_reported_not_filled(tmp_path, monkeypatch):
    r = _built(tmp_path, monkeypatch, answer=_answer(divides=[], decisive_turn=None))
    assert r["turning_points"] == [] and r["turning_points_source"] == "model judgment"
    assert r["decisive"]["source"].startswith("code fallback")


def test_failure_falls_back_with_marks(tmp_path, monkeypatch):
    r = _built(tmp_path, monkeypatch, fail=True)
    why = "code fallback: model call failed (RuntimeError)"
    assert r["judgment"]["source"] == why
    assert r["judgment"]["fields"]["turning_points"] == why
    assert r["decisive"]["source"] == why and why in r["decisive"]["intended_source"]
    assert r["perspective"]["source"].startswith(why) and r["perspective"]["line"]
    assert all(why in sp["goal_source"] and sp["shifts_source"] == why for sp in r["speakers"])
    assert r["turning_points_source"] == "per-item readings"
    assert not any("may be influenced" in x for x in r["reliability"]["limits"])
    page = R.render_html(r)
    assert f"Source: {why}" in page and "Judgment: code fallback" in page


def test_one_bad_field_falls_back_alone(tmp_path, monkeypatch):
    r = _built(tmp_path, monkeypatch, answer=_answer(scene={"scene_id": "NOPE", "why": "It fits."}))
    assert r["judgment"]["source"] == "Muse"
    assert r["decisive"]["turn"] == 2
    assert r["perspective"]["basis"].startswith("code fallback")
    assert "scene: 'NOPE'" in r["judgment"]["dropped"]


def test_html_marks_judged_fields_as_estimates(tmp_path, monkeypatch):
    page = R.render_html(_built(tmp_path, monkeypatch))
    goals = page.split('id="goals"')[1].split("</section>")[0]
    assert "Basis: model estimate" in goals and "Turn #2 (model estimate)" in goals
    points = page.split('id="turning-points"')[1].split("</section>")[0]
    assert "Basis: model estimate" in points
    assert "Turn by turn" in points and "Pressed for a name." in points
    assert "Basis: model estimate" in page.split('id="intent"')[1].split("</section>")[0]
    assert "Why (model estimate)" in page.split('id="perspective"')[1].split("</section>")[0]
    assert "Judgment: Muse (model m, rep 0)" in page


def test_rep_reaches_the_report_and_the_cli_key(tmp_path, monkeypatch):
    assert _built(tmp_path, monkeypatch, rep=1)["judgment"]["rep"] == 1
    assert R.cache_key("c", [], "m", None, 0) != R.cache_key("c", [], "m", None, 1)


def test_no_path_says_intentional_yes(tmp_path, monkeypatch):
    judged = _built(tmp_path, monkeypatch)
    assert {tp["intentional"] for tp in judged["turning_points"]} == {"unclear"}
    down = _built(tmp_path / "other", monkeypatch, fail=True)
    assert {tp["intentional"] for tp in down["turning_points"]} == {"unclear"}
    moves = {0: {"moves": ["appeal_need"]}, 1: {"moves": ["inform"]}, 2: {"moves": ["pressure"]}}
    effects = {0: {"action_dist": {"proceed": 1.0}}, 2: {"action_dist": {"proceed": 1.0}}}
    _, scored = R.build_alerts(TURNS, moves, effects, 3)
    panel = R.build_turning_points(TURNS, moves, effects, scored)
    assert panel and {tp["intentional"] for tp in panel} == {"unclear"}
    steering = {0: {"moves": ["propose"]}, 1: {"moves": ["accept"]}, 2: {"moves": ["propose"]}}
    assert {tp["intentional"] for tp in R.build_turning_points(TURNS, steering, {}, [])} == {"unclear"}
    plain = [{"speaker": "A", "text": "I offer a trade."}, {"speaker": "B", "text": "I agree."},
             {"speaker": "A", "text": "Another offer."}, {"speaker": "B", "text": "No."}]
    assert {tp["intentional"] for tp in R.build_turning_points(plain, {}, {}, [])} == {"unclear"}


def test_grade_limit_states_the_measured_values_and_only_when_measured():
    grade = {"grade": "reference", "repeat": "3/6 same turn", "swap": "3/6 kept", "mask": "4/6 kept", "divides": "0.60"}
    line = R.judgment_grade_limit(grade)
    assert "reference" in line and "3/6 same turn" in line and "4/6 kept" in line
    assert R.judgment_grade_limit({}) == "" and R.judgment_grade_limit(None) == ""


def test_top_severity_line_comes_from_values_with_ties_to_the_earlier_turn():
    uptake = [{"turn": 0, "layers": {"severity": 36.4}}, {"turn": 32, "layers": {"severity": 35.8}},
              {"turn": 5, "layers": None}, {"turn": 9, "layers": {"severity": 35.8}}, {"turn": 3, "layers": {"severity": 1.0}}]
    assert R.top_severity_line(uptake) == "#0 (36.4), #9 (35.8), #32 (35.8)"
    assert R.top_severity_line([]) == ""

