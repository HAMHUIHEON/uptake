"""C5b: negative controls for the alternative judgment (stub client).

Same-panel paired votes judge the original against itself (~0.5) and
against a deliberately worse wording (<0.5), blocking a judge that
always prefers the new wording. Model path uses a content-aware stub
(no network); offline path checks the disclosed fallbacks.
"""
import json

from forecaster import report as R


TURNS = [
    {"speaker": "A", "text": "We have a problem up here."},
    {"speaker": "B", "text": "Say again your situation."},
    {"speaker": "A", "text": "We need priority, we are low on fuel."},
]

ORIGINAL = "We need priority, we are low on fuel."

WORSE = ("We might be getting a bit low on fuel, just mentioning it, "
         "no rush.")


MIDDLE = "We need priority, we are low on fuel, can you sequence us?"


class ControlStub:
    """Gen returns WORSE / MIDDLE; votes tie identical pairs, pick the middle
    wording over the original, else pick the original."""

    def __init__(self, worse=WORSE, middle=MIDDLE):
        self.worse = worse
        self.middle = middle
        self.calls = []
        self.users = []

    def complete(self, messages, model=None, temperature=0.0, **kwargs):
        system = messages[0]["content"] if messages else ""
        user = messages[1]["content"] if len(messages) > 1 else ""
        self.calls.append(system)
        self.users.append(user)
        if "WORSE way" in system:
            return json.dumps({"worse": self.worse})
        if "SLIGHTLY better way" in system:
            return json.dumps({"middle": self.middle})
        first = second = None
        for line in user.splitlines():
            if line.startswith("A: "):
                first = line[3:]
            elif line.startswith("B: "):
                second = line[3:]
        if first == second:
            return json.dumps({"clearer": "Tie"})
        if MIDDLE in (first, second):
            return json.dumps({"clearer": "A" if first == MIDDLE else "B"})
        pick = "A" if first == ORIGINAL else "B"
        return json.dumps({"clearer": pick})


def _build(client, tmp_path, **kwargs):
    cache_dir = tmp_path / "cache"
    return R.build_alternative_controls(
        ORIGINAL, [], {}, client=client, model="stub-model",
        meter=None, cache_dir=cache_dir, turns=TURNS,
        decisive_turn=2, speaker="A",
        intended="appeal_need: ask for priority or help with an urgent need",
        original=ORIGINAL, **kwargs)


def test_identical_lands_at_half_with_paired_votes(tmp_path):
    controls = _build(ControlStub(), tmp_path)
    same = controls["identical"]
    assert same["method"] == "paired"
    assert same["n_compared"] >= 8
    assert same["n"] == same["n_compared"]
    assert same["p_better"] == 0.5
    assert same["p_empirical"] == 0.5


def test_worse_is_model_written_and_below_half(tmp_path):
    controls = _build(ControlStub(), tmp_path)
    worse = controls["worse"]
    assert worse["method"] == "paired"
    assert worse["n_compared"] >= 8
    assert worse["text"] == WORSE
    assert worse["text"] != ORIGINAL
    assert worse["p_better"] < 0.5
    assert worse["p_empirical"] == 0.0


def test_same_inputs_are_cached(tmp_path):
    client = ControlStub()
    first = _build(client, tmp_path)
    n_calls = len(client.calls)
    assert n_calls > 0
    second = _build(client, tmp_path)
    assert len(client.calls) == n_calls
    assert second == first


def test_bad_worse_rewrite_falls_back_but_still_pairs(tmp_path):
    controls = _build(ControlStub(worse=ORIGINAL), tmp_path)
    worse = controls["worse"]
    assert worse["method"] == "paired"
    assert worse["text"] != ORIGINAL
    assert "no rush at all" in worse["text"]
    assert worse["p_better"] < 0.5


def test_offline_controls_are_disclosed(tmp_path):
    controls = R.build_alternative_controls(ORIGINAL, [], {})
    same = controls["identical"]
    assert same["method"] == "offline-identity"
    assert same["p_better"] == 0.5
    assert same["n"] == 0
    worse = controls["worse"]
    assert worse["method"] == "offline-proxy"
    assert worse["text"] != ORIGINAL
    assert ORIGINAL[:20] in worse["text"]
    assert 0.0 < worse["p_better"] < 1.0


# --- middle wording and the shared criterion ---------------------------------

CRITERION = {"item": "deliverable", "grade": "measured", "wanted": 80.0, "actual": 20.0,
             "source": "largest wanted minus actual on turn #2, among measured broken items"}


def test_middle_gets_the_same_panel_orders_and_smoothing(tmp_path):
    controls = _build(ControlStub(), tmp_path, criterion=CRITERION)
    middle = controls["middle"]
    assert middle["text"] == MIDDLE
    assert middle["method"] == "paired"
    assert middle["n_compared"] == 10 and middle["n"] == 10
    assert middle["p_empirical"] == 1.0
    assert middle["p_better"] == round(11 / 12, 3)
    assert middle["ci_low"] < middle["ci_high"]


def test_every_control_vote_is_on_the_one_criterion(tmp_path):
    client = ControlStub()
    controls = _build(client, tmp_path, criterion=CRITERION)
    assert controls["criterion"] == CRITERION
    # Two wordings written; identical pairs share one vote across orders.
    assert len(client.users) == 2 + 5 + 10 + 10
    for user in client.users:
        assert "Criterion: the effect A wanted on deliverable (" in user


def test_control_prompts_name_the_criterion_not_urgency():
    for prompt in (R.COMPARE_PROMPT, R.REWRITE_PROMPT, R.WORSE_PROMPT, R.MIDDLE_PROMPT):
        assert "criterion" in prompt
        for gone in ("urgent", "busy listener", "action now", "routine"):
            assert gone not in prompt.lower()
    for prompt in (R.WORSE_PROMPT, R.MIDDLE_PROMPT):
        assert "Do not add any topic that is not already in the original." in prompt


def test_controls_cache_splits_by_criterion(tmp_path):
    client = ControlStub()
    _build(client, tmp_path, criterion=CRITERION)
    n_calls = len(client.calls)
    _build(client, tmp_path, criterion=CRITERION)
    assert len(client.calls) == n_calls
    other = {**CRITERION, "item": "facts", "wanted": 70.0, "actual": 30.0}
    _build(client, tmp_path, criterion=other)
    assert len(client.calls) == 2 * n_calls


def test_no_middle_sentence_scores_nothing(tmp_path):
    controls = _build(ControlStub(middle=ORIGINAL), tmp_path)
    middle = controls["middle"]
    assert middle["text"] == ""
    assert middle["p_better"] is None and middle["n"] == 0
    assert middle["method"] == "unavailable"


def test_offline_middle_scores_nothing():
    middle = R.build_alternative_controls(ORIGINAL, [], {})["middle"]
    assert middle["p_better"] is None and middle["method"] == "unavailable"


def _paired(p):
    return {"p_better": p, "method": "paired"}


def test_ordering_states_the_values_and_whether_they_hold():
    alts = [_paired(0.8), _paired(0.6)]
    held = R.alternative_ordering(alts, {"worse": _paired(0.1), "middle": _paired(0.5)})
    assert held == {"worse": 0.1, "middle": 0.5, "alternatives_mean": 0.7, "holds": True}
    assert R.alternative_ordering(alts, {"worse": _paired(0.1), "middle": _paired(0.75)})["holds"] is False
    assert R.alternative_ordering(alts, {"worse": _paired(0.5), "middle": _paired(0.5)})["holds"] is False


def test_ordering_is_not_claimed_without_paired_votes():
    alts = [{"p_better": 0.6, "method": "offline-proxy"}, _paired(0.8)]
    controls = {"worse": _paired(0.1), "middle": _paired(0.5)}
    assert R.alternative_ordering(alts, controls)["holds"] is None
    missing = {"worse": _paired(0.1), "middle": {"p_better": None, "method": "unavailable"}}
    got = R.alternative_ordering([_paired(0.6), _paired(0.8)], missing)
    assert got["holds"] is None and got["middle"] is None


def _write_case(root, case_id="t1"):
    hs = root / "highstakes"
    hs.mkdir(parents=True, exist_ok=True)
    turns = [
        {"speaker": "A", "text": ORIGINAL},
        {"speaker": "B", "text": "Roger, continue as cleared."},
    ]
    with open(hs / f"casino-{case_id}_turns.jsonl", "w",
              encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    with open(hs / f"casino-{case_id}_case.json", "w",
              encoding="utf-8") as f:
        json.dump({"title": "Synthetic fuel case",
                   "outcome": "Synthetic outcome.",
                   "source_url": "https://example.com/record",
                   "outcome_turn": 1}, f)
    panel_dir = hs / "cases" / f"casino-{case_id}"
    panel_dir.mkdir(parents=True, exist_ok=True)
    with open(panel_dir / "panel_record.json", "w", encoding="utf-8") as f:
        json.dump({"id": f"casino-{case_id}", "genre": "negotiation",
                   "moves": [
                       {"turn_index": 0, "speaker": "A",
                        "moves": ["inform", "appeal_need"],
                        "quote": "need priority"},
                       {"turn_index": 1, "speaker": "B",
                        "moves": ["accept"], "quote": "roger"},
                   ],
                   "effects": [
                       {"index": 0, "speaker": "A",
                        "action_dist": {"proceed": 1.0},
                        "disagreement": 0.5,
                        "readings": [
                            {"next_action": "proceed"},
                            {"next_action": "proceed"},
                        ]},
                       {"index": 1, "speaker": "B",
                        "action_dist": {"proceed": 1.0},
                        "disagreement": 0.0,
                        "readings": [
                            {"next_action": "proceed"},
                            {"next_action": "proceed"},
                        ]},
                   ]}, f)


def test_report_carries_controls_in_json_and_html(tmp_path):
    _write_case(tmp_path)
    report = R.build_report("t1", data_root=str(tmp_path))
    controls = report["alternatives_control"]
    assert set(controls) == {"identical", "worse", "middle", "criterion", "ordering"}
    assert controls["identical"]["p_better"] == 0.5
    assert controls["worse"]["text"]
    assert controls["ordering"]["holds"] is None
    html = R.render_html(report)
    assert "Negative controls (same panel)" in html
    assert 'id="alternatives"' in html
    section = html.split('id="alternatives"')[1].split("</section>")[0]
    assert "Middle wording: not scored in this run" in section
    assert "None" not in section
