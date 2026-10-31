"""Contract for SPEC_task51_scenes_eval (synthetic only, fake clients)."""
import ast
import json
import sys
from pathlib import Path

import pytest

from forecaster import board as B
from forecaster import pairwise as P
from forecaster import scenes_eval as SE
from forecaster.stats import wilson_ci
from forecaster.trace import CallLog, LoggingClient
from forecaster.usage import Meter


class _Fixed:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def complete(self, messages, model, temperature):
        self.calls += 1
        return self.raw


class _Queued:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, messages, model, temperature):
        self.calls.append(messages)
        if not self.outputs:
            return ""
        return self.outputs.pop(0)


# --- reuse + packaging -----------------------------------------------------

def test_reuses_pairwise_stances_and_wrapper():
    assert SE.PW is P
    assert SE.PW.STANCES == P.STANCES == ("neutral", "trusting", "suspicious")
    assert SE.PW.pairwise_panel is P.pairwise_panel
    assert SE.board._call_board is B._call_board


def test_stdlib_only_direct_imports():
    src = Path(SE.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    stdlib = set(sys.stdlib_module_names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in stdlib, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                continue  # sibling module (pairwise/board/stats/trace/usage)
            mod = (node.module or "").split(".")[0]
            assert mod in stdlib, node.module


def test_entry_point_registered():
    root = Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert "uptake-scenes-eval = \"forecaster.scenes_eval:main\"" in text


# --- prepare ---------------------------------------------------------------

def _prepare_rows():
    return [
        {"id": "s02", "계열": "문학", "작품": "리어 왕",
         "발화": "Nothing, my lord.", "말한사람": "코딜리어",
         "받은사람": "리어 왕", "맥락": {"관계": "부녀", "권력거리": "큼",
                                     "채널": "대면"},
         "해석": "사랑이 없다는 모욕으로 읽었다", "근거": "1막 1장",
         "오해발생": True,
         "입력": {"상황": "왕이 딸들에게 사랑을 고백하라 명했다",
                  "상황_가림": "노인이 자식들에게 사랑을 고백하라 명했다"}},
        {"id": "s01", "입력": {"상황": "짧은 상황",
                              "맥락": {"관계": "동료", "권력거리": "작음",
                                       "채널": "문서"}}},
        {"id": "nodic-입력"},  # no 입력 -> dropped
        {"id": "blank", "입력": {"상황": "   "}},  # blank 상황 -> dropped
        "NOT JSON",  # bad line -> dropped
        "",  # blank line ignored entirely
    ]


def test_prepare_filters_and_copies_fields(tmp_path):
    src = tmp_path / "corpus.jsonl"
    src.write_text("\n".join(r if isinstance(r, str) else json.dumps(r)
                             for r in _prepare_rows()))
    out = tmp_path / "out"
    SE.main(["prepare", "--src", str(src), "--out", str(out)])
    scenes = json.loads((out / "scenes.json").read_text())
    assert [s["id"] for s in scenes] == ["s01", "s02"]
    full = scenes[1]
    assert full["계열"] == "문학"
    assert full["작품"] == "리어 왕"
    assert full["상황"] == "왕이 딸들에게 사랑을 고백하라 명했다"
    assert full["상황_가림"] == "노인이 자식들에게 사랑을 고백하라 명했다"
    assert full["발화"] == "Nothing, my lord."
    assert full["말한사람"] == "코딜리어"
    assert full["받은사람"] == "리어 왕"
    assert full["맥락"] == {"관계": "부녀", "권력거리": "큼", "채널": "대면"}
    assert full["해석"] == "사랑이 없다는 모욕으로 읽었다"
    assert full["근거"] == "1막 1장"
    assert full["오해발생"] is True
    minimal = scenes[0]
    assert minimal["상황"] == "짧은 상황"
    assert minimal["발화"] == ""
    assert minimal["맥락"] == {"관계": "동료", "권력거리": "작음",
                              "채널": "문서"}
    assert minimal["오해발생"] is False


def test_prepare_counts(tmp_path):
    src = tmp_path / "corpus.jsonl"
    src.write_text("\n".join(r if isinstance(r, str) else json.dumps(r)
                             for r in _prepare_rows()))
    counts = SE.prepare_corpus(src, tmp_path / "out")
    assert counts["n_total"] == 5
    assert counts["n_kept"] == 2
    assert counts["n_dropped"] == 3
    assert counts["n_bad_json"] == 1
    assert counts["per_계열"] == {"문학": 1, "(none)": 1}


# --- read ------------------------------------------------------------------

def _read_scenes():
    return [
        {"id": "s1", "계열": "문학", "작품": "리어 왕",
         "상황": "왕이 사랑 고백을 명했다", "상황_가림": "노인이 고백을 명했다",
         "발화": "Nothing, my lord.", "발화_가림": "아무것도, 어르신.",
         "말한사람": "코딜리어", "받은사람": "리어 왕",
         "맥락": {"관계": "부녀", "권력거리": "큼", "채널": "대면"},
         "해석": "모욕으로 읽었다", "근거": "1막", "오해발생": True},
        {"id": "s2", "계열": "역사", "작품": "X", "상황": "두 번째 상황",
         "상황_가림": "", "발화": "두 번째 발화", "발화_가림": "",
         "말한사람": "A", "받은사람": "B",
         "맥락": {"관계": "동맹", "권력거리": "작음", "채널": "서신"},
         "해석": "수용으로 읽었다", "근거": "사서", "오해발생": False},
    ]


def test_read_builds_three_stances_times_two_versions(tmp_path):
    scenes = _read_scenes()
    out = tmp_path / "out"
    out.mkdir()
    fake = _Queued(['{"reception": "나쁘게", "reading": "모욕으로 읽었다"}'] * 20)
    client = LoggingClient(fake, CallLog(out / "calls.jsonl"), "t")
    run_r, preds_r = SE.read_scenes(scenes, "real", out, client, workers=2)
    run_m, preds_m = SE.read_scenes(scenes, "masked", out, client, workers=2)
    assert len(preds_r) == len(preds_m) == 2 * 3
    for preds, version in ((preds_r, "real"), (preds_m, "masked")):
        assert [p["stance"] for p in preds] == [
            "neutral", "trusting", "suspicious"] * 2
        assert {p["version"] for p in preds} == {version}
        assert all(p["reception"] == "나쁘게" for p in preds)
        assert all(p["reading"] == "모욕으로 읽었다" for p in preds)
    assert run_r["n_preds"] == 6 and run_r["parse_fail_rate"] == pytest.approx(0.0)
    assert run_m["version"] == "masked"
    assert len(json.loads((out / "pred_real.jsonl").read_text().splitlines()[0])) >= 5
    assert len((out / "pred_masked.jsonl").read_text().splitlines()) == 6
    assert json.loads((out / "run_real.json").read_text())["n_expected"] == 6
    # real uses 상황/발화, masked uses 상황_가림/발화_가림
    users = [c[1]["content"] for c in fake.calls]
    assert any("왕이 사랑 고백을 명했다" in u for u in users)
    assert any("노인이 고백을 명했다" in u for u in users)
    assert any("아무것도, 어르신." in u for u in users)
    # s2 has no masked text -> masked falls back to the real text
    assert any("두 번째 상황" in u for u in users)
    # stance personas come from the pairwise panel
    metas = [json.loads(line)["meta"]
             for line in (out / "calls.jsonl").read_text().splitlines()]
    assert len(metas) == 12
    assert {m["stance"] for m in metas} == set(P.STANCES)
    assert all(m["layer"] == "scenes-read" for m in metas)


def test_read_prompt_carries_scene_context():
    rec = P.pairwise_panel(0)[0]
    msgs = SE.build_scene_messages(_read_scenes()[0], rec, "real")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert "stance: neutral" in msgs[0]["content"]
    assert "reception" in msgs[0]["content"]
    assert SE.READER_CONTRACT in msgs[1]["content"]
    user = msgs[1]["content"]
    assert "왕이 사랑 고백을 명했다" in user
    assert "Nothing, my lord." in user
    assert "부녀" in user and "큼" in user and "대면" in user


def test_parse_scene_reading():
    assert SE.parse_scene_reading('{"reception": "잘", "reading": "ok"}') == \
        {"reception": "잘", "reading": "ok"}
    assert SE.parse_scene_reading('x {"reception":"보통","reading":"r"} y') == \
        {"reception": "보통", "reading": "r"}
    assert SE.parse_scene_reading('{"reception": "좋게", "reading": "r"}') is None
    assert SE.parse_scene_reading('{"reading": "r"}') is None
    assert SE.parse_scene_reading("garbage no json") is None
    assert SE.parse_scene_reading(None) is None


def test_read_garbage_votes_null_with_retry(tmp_path):
    run, preds = SE.read_scenes(_read_scenes(), "real", tmp_path,
                                _Fixed("garbage no json"), workers=2)
    assert run["n_calls"] == 2 * 6
    assert run["n_preds"] == 0
    assert run["parse_fail_rate"] == pytest.approx(1.0)
    assert all(p["reception"] is None and p["reading"] is None for p in preds)


def test_read_budget_stop_clean(tmp_path):
    run, preds = SE.read_scenes(
        _read_scenes(), "real", tmp_path,
        _Fixed('{"reception": "잘", "reading": "r"}'),
        meter=Meter(budget_usd=0.0))
    assert run["budget_stop"] is True
    assert run["n_attempted"] == 0
    assert len(preds) == 6  # null placeholders keep the file aligned


def test_read_rejects_bad_version(tmp_path):
    with pytest.raises(ValueError):
        SE.read_scenes(_read_scenes(), "bogus", tmp_path, _Fixed("{}"))


# --- judge -----------------------------------------------------------------

def test_judge_prompt_is_module_constant():
    assert isinstance(SE.JUDGE_PROMPT, str)
    assert "match" in SE.JUDGE_PROMPT and "partial" in SE.JUDGE_PROMPT
    assert "miss" in SE.JUDGE_PROMPT


def test_parse_judge():
    assert SE.parse_judge('{"verdict": "match", "why": "same"}') == \
        {"verdict": "match", "why": "same"}
    assert SE.parse_judge('x {"verdict":"Partial","why":"w"}')["verdict"] == "partial"
    assert SE.parse_judge('{"verdict": "MISS", "why": "w"}')["verdict"] == "miss"
    assert SE.parse_judge('{"verdict": "maybe", "why": "w"}') is None
    assert SE.parse_judge('{"why": "w"}') is None
    assert SE.parse_judge("garbage no json") is None
    assert SE.parse_judge(None) is None


def test_judge_sees_only_truth_and_prediction():
    scene = _read_scenes()[0]
    pred = {"scene_id": "s1", "stance": "neutral", "version": "real",
            "reception": "나쁘게", "reading": "모욕으로 읽었다"}
    truth = {"해석": scene["해석"], "근거": scene["근거"]}
    msgs = SE.build_judge_messages(pred, truth)
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == SE.JUDGE_PROMPT
    user = msgs[1]["content"]
    assert "모욕으로 읽었다" in user  # 해석 + prediction
    assert "1막" in user  # 근거
    assert "나쁘게" in user  # predicted reception
    for forbidden in ("왕이 사랑 고백을 명했다", "Nothing, my lord.",
                      "리어 왕", "부녀", "코딜리어", "reception must be"):
        assert forbidden not in user
        assert forbidden not in msgs[0]["content"]


def test_judge_run_writes_versioned_file(tmp_path):
    preds = [{"scene_id": "s1", "stance": s, "version": "real",
              "reception": "나쁘게", "reading": "모욕으로 읽었다"}
             for s in P.STANCES]
    truths = {"s1": {"해석": "모욕으로 읽었다", "근거": "1막"}}
    out = tmp_path / "out"
    out.mkdir()
    client = LoggingClient(_Fixed('{"verdict": "match", "why": "same"}'),
                           CallLog(out / "calls.jsonl"), "t")
    judgments, stats = SE.judge_preds(preds, truths, out, client, workers=2)
    assert stats["version"] == "real"
    assert stats["out"] == str(out / "judge_real.jsonl")
    assert stats["n_judged"] == 3
    assert [j["verdict"] for j in judgments] == ["match"] * 3
    lines = (out / "judge_real.jsonl").read_text().splitlines()
    assert len(lines) == 3
    assert all(json.loads(line)["stance"] in P.STANCES for line in lines)
    # explicit file path is written exactly
    SE.judge_preds(preds, truths, out / "custom.jsonl",
                   _Fixed('{"verdict": "miss", "why": "x"}'))
    assert len((out / "custom.jsonl").read_text().splitlines()) == 3


# --- report ----------------------------------------------------------------

def _report_fixture(tmp_path):
    scenes = [
        {"id": "s1", "계열": "문학", "작품": "리어 왕",
         "발화": "x" * 100, "오해발생": True},
        {"id": "s2", "계열": "문학", "작품": "햄릿",
         "발화": "짧은 발화", "오해발생": False},
        {"id": "s3", "계열": "역사", "작품": "사기",
         "발화": "세 번째 발화", "오해발생": True},
        {"id": "s4", "계열": "역사", "작품": "삼국지",
         "발화": "네 번째 발화", "오해발생": True},
        {"id": "s5", "계열": "고전", "작품": "논어",
         "발화": "다섯 번째 발화", "오해발생": False},
    ]
    # verdicts per scene per version in stance order
    verdicts = {
        ("s1", "real"): ["match", "match", "partial"],
        ("s1", "masked"): ["match", "miss", "miss"],
        ("s2", "real"): ["miss", "miss", "miss"],
        ("s2", "masked"): ["partial", "partial", "miss"],
        ("s3", "real"): ["match", "match", "match"],
        ("s3", "masked"): ["miss", "miss", "miss"],
        ("s4", "real"): ["partial", "miss", "miss"],
        ("s4", "masked"): ["match", "match", "match"],
        ("s5", "real"): ["match", "partial", "miss"],
        ("s5", "masked"): ["partial", "partial", "partial"],
    }
    receptions = {
        ("s1", "real"): ["나쁘게"] * 3,
        ("s2", "real"): ["잘"] * 3,
        ("s3", "real"): ["보통"] * 3,
        ("s4", "real"): ["잘", "보통", "나쁘게"],  # tie -> 보통
        ("s5", "real"): [None, None, None],  # no prediction
    }
    preds, judges = [], []
    for (sid, version), vs in verdicts.items():
        for stance, verdict in zip(P.STANCES, vs):
            rec = (receptions.get((sid, version)) or [None, None, None])[
                P.STANCES.index(stance)] if version == "real" else "보통"
            reading = ("y" * 120) if (sid == "s1" and stance == "neutral"
                                      and version == "real") else f"읽기-{sid}"
            preds.append({"scene_id": sid, "stance": stance, "version": version,
                          "reception": rec, "reading": reading})
            judges.append({"scene_id": sid, "stance": stance, "version": version,
                           "verdict": verdict, "why": "w"})
    base = tmp_path / "r"
    base.mkdir()
    (base / "scenes.json").write_text(json.dumps(scenes))
    (base / "pred_real.jsonl").write_text("\n".join(
        json.dumps(p) for p in preds if p["version"] == "real"))
    (base / "pred_masked.jsonl").write_text("\n".join(
        json.dumps(p) for p in preds if p["version"] == "masked"))
    (base / "judge_real.jsonl").write_text("\n".join(
        json.dumps(j) for j in judges if j["version"] == "real"))
    (base / "judge_masked.jsonl").write_text("\n".join(
        json.dumps(j) for j in judges if j["version"] == "masked"))
    return base


def test_report_known_answer_and_wilson(tmp_path):
    base = _report_fixture(tmp_path)
    out_md = base / "report.md"
    SE.main(["report", "--out", str(out_md),
             "--scenes", str(base / "scenes.json"),
             "--pred-real", str(base / "pred_real.jsonl"),
             "--pred-masked", str(base / "pred_masked.jsonl"),
             "--judge-real", str(base / "judge_real.jsonl"),
             "--judge-masked", str(base / "judge_masked.jsonl")])
    m = SE.generate_report(str(base / "scenes.json"),
                           str(base / "pred_real.jsonl"),
                           str(base / "pred_masked.jsonl"),
                           str(base / "judge_real.jsonl"),
                           str(base / "judge_masked.jsonl"), str(out_md))
    assert m["n_scenes"] == 5
    assert (m["k1"], m["k2"]) == (3, 3)
    assert m["m1"] == pytest.approx(0.6)
    assert m["m2"] == pytest.approx(0.6)
    assert m["m3"] == pytest.approx(0.0)
    assert m["m1_ci"] == list(wilson_ci(3, 5))
    assert m["m2_ci"] == list(wilson_ci(3, 5))
    lo1, hi1 = wilson_ci(3, 5)
    assert lo1 < 0.6 < hi1  # small-n interval is wide, as expected
    assert m["verdict"] == "reads famous scenes"
    assert m["per_계열"]["문학"]["m1"] == pytest.approx(0.5)
    assert m["per_계열"]["고전"]["m1"] == pytest.approx(1.0)
    assert m["crosstab"] == {"잘": {"true": 0, "false": 1},
                            "보통": {"true": 2, "false": 0},
                            "나쁘게": {"true": 1, "false": 0}}
    assert m["n_no_pred"] == 1
    md = out_md.read_text()
    assert "M1=0.600" in md and "M2=0.600" in md and "M3=0.000" in md
    assert f"{lo1:.3f}" in md  # Wilson interval printed
    assert "verdict: reads famous scenes" in md
    assert "문학" in md and "역사" in md and "고전" in md
    assert "scenes without a real-version prediction: 1" in md
    rows = [line for line in md.splitlines() if line.startswith("| s")]
    assert len(rows) == 5
    assert "x" * 80 in md and "x" * 81 not in md  # 발화 ≤ 80
    assert "y" * 100 in md and "y" * 101 not in md  # reading ≤ 100
    assert "partial 2/3" in md  # per-version majority judgment shown


def test_report_defaults_to_colocated_files(tmp_path):
    base = _report_fixture(tmp_path)
    SE.main(["report", "--out", str(base / "report2.md")])
    assert "reads famous scenes" in (base / "report2.md").read_text()


def test_amendment12_verdict_bands():
    assert SE.amendment12_verdict(0.60, 0.15) == "reads famous scenes"
    assert SE.amendment12_verdict(0.80, 0.16) == "partly from memory"
    assert SE.amendment12_verdict(0.44, 0.0) == "not supported"
    assert SE.amendment12_verdict(0.50, 0.0) == "inconclusive"


def test_majority_helpers():
    assert SE.scene_success(["match", "miss", "miss"]) is False
    assert SE.scene_success(["partial", "partial", None]) is True
    assert SE.majority_verdict(["match", "partial", "miss"]) == "partial"
    assert SE.majority_verdict([None, None, "match"]) == "miss"
    assert SE.majority_reception(["잘", "보통", "나쁘게"]) == "보통"
    assert SE.majority_reception([None, None, None]) == ""
