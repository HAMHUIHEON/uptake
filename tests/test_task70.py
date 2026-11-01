"""Task70: per-turn four-stake reading wired into the report.

Stub/offline only (no network): alerts come only from missed, partly
or cost gaps with the stake set; the report carries uptake for every
turn except the last with no forecast/lead output; standing sections
stay per-stake; --max-turns caps the trial; README and HTML lead with
the product sentence.
"""
import html
import json
from pathlib import Path

from forecaster import report as R
from forecaster import v2 as V2


def _entry(wanted="priority now", subs=(), happened="stayed the course",
           gap="missed", reply_turn=1, quote="continue as cleared"):
    return {"wanted": wanted, "sub_stakes": list(subs),
            "happened": happened, "gap": gap,
            "reply_turn": reply_turn, "quote": quote}


def _uptake(gaps_by_turn):
    out = []
    for t, gaps in enumerate(gaps_by_turn):
        axes = {}
        for axis in V2.STAKES:
            gap = gaps.get(axis, "unclear") if isinstance(gaps, dict) \
                else gaps
            axes[axis] = _entry(gap=gap, subs=V2.STAKE_TREE[axis][:1])
        out.append({"turn": t, "speaker": "A" if t % 2 == 0 else "B",
                    "axes": axes})
    return out


def test_axis_alerts_only_from_missed_partly_cost():
    uptake = _uptake([
        {"outcome": "met", "relationship": "not_relevant",
         "face": "unclear", "understanding": "missed"},
        {"outcome": "partly", "relationship": "cost",
         "face": "met", "understanding": "unclear"},
    ])
    alerts = R.build_axis_alerts(uptake)
    by_key = {(a["turn"], a["axis"]): a for a in alerts}
    assert set(by_key) == {(0, "understanding"), (1, "outcome"),
                           (1, "relationship")}
    assert by_key[(0, "understanding")]["level"] == "warning"
    assert by_key[(1, "outcome")]["level"] == "caution"
    assert by_key[(1, "relationship")]["level"] == "critical"
    for a in alerts:
        assert a["axis"] in V2.STAKES
        assert a["level"] in ("advisory", "caution", "warning",
                             "critical")
        assert set(("turn", "level", "axis", "reason",
                    "resolved")) <= set(a)
        assert "basis" not in a and "lead_turns" not in a


def test_axis_alert_resolved_on_later_met():
    uptake = _uptake([{"outcome": "missed"}, {"outcome": "met"}])
    alerts = R.build_axis_alerts(uptake)
    assert len(alerts) == 1
    assert alerts[0]["resolved"] is True
    uptake = _uptake([{"outcome": "missed"}, {"outcome": "unclear"}])
    assert R.build_axis_alerts(uptake)[0]["resolved"] is False


def test_stake_tallies_stay_per_stake():
    uptake = _uptake([{"outcome": "missed", "face": "met"},
                      {"outcome": "cost", "face": "partly"}])
    tally = R.stake_tally(uptake)
    assert set(tally) == set(V2.STAKES)
    assert tally["outcome"] == {"met": 0, "partly": 0, "missed": 1,
                               "cost": 1, "turns": [0, 1]}
    assert tally["face"]["met"] == 1 and tally["face"]["partly"] == 1
    assert tally["relationship"] == {"met": 0, "partly": 0, "missed": 0,
                                    "cost": 0, "turns": []}
    by_sp = R.speaker_stake_tally(uptake)
    assert by_sp["A"]["outcome"]["missed"] == 1
    assert by_sp["B"]["outcome"]["cost"] == 1
    assert by_sp["A"]["face"]["met"] == 1
    assert by_sp["B"]["face"]["partly"] == 1


def _write_case(root, case_id="t70"):
    hs = root / "highstakes"
    hs.mkdir(parents=True, exist_ok=True)
    turns = [
        {"speaker": "A", "text": "We need priority, we are low on fuel."},
        {"speaker": "B", "text": "Roger, continue as cleared."},
        {"speaker": "A", "text": "Confirming we continue."},
        {"speaker": "B", "text": "Cleared to land."},
    ]
    with open(hs / f"casino-{case_id}_turns.jsonl", "w",
              encoding="utf-8") as f:
        for t in turns:
            f.write(json.dumps(t) + "\n")
    with open(hs / f"casino-{case_id}_case.json", "w",
              encoding="utf-8") as f:
        json.dump({"title": "Synthetic", "outcome": "Synthetic.",
                   "source_url": "https://example.com/r",
                   "outcome_turn": 3}, f)
    return root


def _run(case_id, root, out, *extra):
    R.main(["--case", case_id, "--out", str(out), "--budget", "5.0",
            "--data-root", str(root), *extra])


def test_report_uptake_shape_and_no_forecast_output(tmp_path):
    _write_case(tmp_path)
    out = tmp_path / "out"
    _run("t70", tmp_path, out)
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    assert "forecast_check" not in r
    # Every turn except the last, each item with the contract shape.
    assert [u["turn"] for u in r["uptake"]] == [0, 1, 2]
    from forecaster.multilayer import ITEMS
    for u in r["uptake"]:
        assert set(u["items"]) == set(ITEMS)
        for name in ITEMS:
            assert set(("class", "wanted", "actual", "reply_turn", "quote")) <= set(u["items"][name])
        assert set(u["expected"]) == set(ITEMS) and set(u["layers"]["classes"]) == set(V2.STAKES)
        assert set(u["scales"]) == {"next_action", "relationship", "face"}
    assert r["conditions"]["source"] == "not read"
    # Alerts come from readings, with the class set, no lead keys.
    for a in r["alerts"]:
        assert a["axis"] in V2.STAKES and a["items"]
        assert "basis" not in a and "lead_turns" not in a
    blob = json.dumps(r["alerts"]) + json.dumps(r["reliability"])
    for banned in ("forecast", "predict", "in advance", "before_reply",
                   "after_reply", "lead_turns", "could have been known"):
        assert banned not in blob.lower(), banned


def test_report_html_has_product_sentence_and_per_stake_standing(tmp_path):
    _write_case(tmp_path)
    out = tmp_path / "out"
    _run("t70", tmp_path, out)
    page = (out / "report.html").read_text(encoding="utf-8")
    assert html.escape(R.PRODUCT_SENTENCE) in page
    for section in ("turning-points", "balance"):
        body = page.split(f'<section id="{section}">')[1].split(
            "</section>")[0]
        assert "stake" in body.lower()
        assert ("never merged into one number" in body.lower()
                or "no judged stake gaps" in body.lower())


def test_max_turns_caps_uptake_cli(tmp_path, capsys):
    _write_case(tmp_path)
    out = tmp_path / "out"
    _run("t70", tmp_path, out, "--max-turns", "2")
    with open(out / "report.json", encoding="utf-8") as f:
        r = json.load(f)
    assert [u["turn"] for u in r["uptake"]] == [0, 1]
    lines = capsys.readouterr().out.strip().splitlines()
    assert lines[-1].startswith("SPEND=")
    # A different cap is a different report, not a cache hit.
    out2 = tmp_path / "out2"
    _run("t70", tmp_path, out2, "--max-turns", "1")
    with open(out2 / "report.json", encoding="utf-8") as f:
        r2 = json.load(f)
    assert [u["turn"] for u in r2["uptake"]] == [0]


def test_readme_first_paragraph_is_product_sentence():
    readme = Path(__file__).resolve().parent.parent / "README.md"
    lines = readme.read_text(encoding="utf-8").splitlines()
    assert lines[0].strip() == "# Uptake"
    assert lines[2].strip().startswith(R.PRODUCT_SENTENCE)
