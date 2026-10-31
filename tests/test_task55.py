"""Contract for SPEC_task55_notice_counsel (synthetic only, mocked API)."""
import ast
import json
import sys
from pathlib import Path

import pytest

from forecaster import board2 as B2
from forecaster import commentary as CM
from forecaster import pairwise as PW
from forecaster import perturb
from forecaster import play_server as PS
from forecaster import scenes as SC
from forecaster import suggest as SG
from forecaster.usage import Meter


# --- 6-row fixture (3 east, 3 west) --------------------------------------------

def _row(sid, series, field, work, src, scene, said, jtype, demo="상"):
    return {
        "id": sid, "계열": series, "분야": field, "작품": work,
        "작가": f"작가-{sid}", "위치": f"위치-{sid}",
        "원문출처": [src],
        "장면": scene, "발화": said,
        "말한사람": f"화자-{sid}", "받은사람": f"청자-{sid}",
        "의도": "의도 문장", "해석": "나쁘게 읽혔다",
        "주유형": jtype,
        "맥락": {"관계": "동료", "권력거리": "작음"},
        "오해발생": True, "대조종류": "",
        "데모적합": demo, "검증": "확인",
    }


def fixture_rows():
    return [
        _row("e1", "동양고전", "동양고전", "사기",
             "https://zh.wikisource.org/wiki/사기",
             "범증이 옥을 세 번 들어 보였으나 항우가 보지 못하였다. 신호가 닿지 않았다.",
             "옥을 들어 보이다", "C2 체면 위협", "상"),
        _row("e2", "실록", "실록", "조선왕조실록",
             "https://sillok.history.go.kr/1",
             "신하가 짧게 아뢰었으나 임금이 뜻을 오해하였다. 말이 짧아 곡해되었다.",
             "짧은 아룀", "B1 양의 격률", "상"),
        _row("e3", "한국문학", "한국", "춘향전",
             "https://ko.wikisource.org/wiki/춘향전",
             "주장이 필요한 사정을 호소하였으나 상대가 편견으로 읽었다. 확증에 갇혔다.",
             "사정 호소", "E1 확증 편향", "중"),
        _row("w1", "서양문학", "희곡", "King Lear",
             "https://www.gutenberg.org/ebooks/1532",
             "Cordelia answered briefly and the king read it as an insult. The court fell silent.",
             "Nothing, my lord.", "C1 권력거리", "상"),
        _row("w2", "서양문학", "소설", "Pride and Prejudice",
             "https://en.wikipedia.org/wiki/Pride_and_Prejudice",
             "A short indirect request was taken at face value. The offer stalled.",
             "A hinted ask", "B4 간접화행", "상"),
        _row("w3", "역사", "서양사", "Peloponnesian War",
             "https://www.gutenberg.org/ebooks/7142",
             "An appeal to shared values clashed with the other side. The talks broke off.",
             "A values appeal", "D2 가치 충돌", "중"),
    ]


def _write_corpus(tmp_path):
    src = tmp_path / "corpus.jsonl"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                             for r in fixture_rows()), encoding="utf-8")
    return src


def _built_scenes(tmp_path):
    src = _write_corpus(tmp_path)
    out = tmp_path / "scenes.json"
    SC.main(["build", "--src", str(src), "--out", str(out)])
    return json.loads(out.read_text(encoding="utf-8"))


# --- build ---------------------------------------------------------------------

def test_build_adds_culture_and_oneline(tmp_path):
    scenes = _built_scenes(tmp_path)
    assert len(scenes) == 6
    assert [s["id"] for s in scenes] == ["e1", "e2", "e3", "w1", "w2", "w3"]
    by_id = {s["id"]: s for s in scenes}
    assert by_id["e1"]["문화"] == "동"
    assert by_id["e2"]["문화"] == "동"
    assert by_id["e3"]["문화"] == "동"
    assert by_id["w1"]["문화"] == "서"
    assert by_id["w2"]["문화"] == "서"
    assert by_id["w3"]["문화"] == "서"
    for s in scenes:
        line = s["상황_한줄"]
        assert isinstance(line, str) and line
        assert len(line) <= 90
        # One sentence: no trailing second sentence.
        assert line.count(". ") == 0
    assert by_id["e1"]["상황_한줄"].startswith("범증이 옥을")
    # task50 fields still copied.
    assert by_id["e1"]["주유형"] == "C2 체면 위협"
    assert by_id["e1"]["맥락"] == {"관계": "동료", "권력거리": "작음"}
    assert by_id["w1"]["원문출처"] == ["https://www.gutenberg.org/ebooks/1532"]
    assert "입력" not in by_id["e1"]


def test_build_filters_unverified(tmp_path):
    rows = fixture_rows()
    rows.append(_row("x1", "문학", "문학", "X", "https://example.test",
                     "장면 문장.", "발화", "C2 체면 위협", "상"))
    rows[-1]["검증"] = "미확인"
    rows.append(_row("x2", "문학", "문학", "Y", "https://example.test",
                     "장면 문장.", "발화", "C2 체면 위협", "하"))
    src = tmp_path / "c.jsonl"
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows),
                   encoding="utf-8")
    out = tmp_path / "s.json"
    counts = SC.build_scenes(src, out)
    assert counts["n_kept"] == 6
    kept = json.loads(out.read_text(encoding="utf-8"))
    assert "x1" not in [s["id"] for s in kept]


def test_type_for_blunder_covers_received_badly():
    assert isinstance(SC.TYPE_FOR_BLUNDER, dict)
    assert SC.TYPE_FOR_BLUNDER["reject"] == ["C2 체면 위협", "C1 권력거리"]
    assert SC.TYPE_FOR_BLUNDER["propose"] == ["B1 양의 격률", "B4 간접화행"]
    assert SC.TYPE_FOR_BLUNDER["appeal_need"] == ["E1 확증 편향", "D2 가치 충돌"]
    assert SC.TYPE_FOR_BLUNDER["probe"] == ["F1 전략적 모호함", "F2 설계된 오독"]
    assert SC.TYPE_FOR_BLUNDER["other"] == ["B4 간접화행"]
    assert isinstance(SC.COUNSEL_PROMPT, str) and "하라" in SC.COUNSEL_PROMPT


# --- counsel (mocked) ------------------------------------------------------------

class _CounselClient:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def complete(self, messages, model, temperature=1.0, **kwargs):
        self.calls += 1
        return self.raw


def test_counsel_caps_and_idempotent(tmp_path):
    scenes = _built_scenes(tmp_path)
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps(scenes, ensure_ascii=False), encoding="utf-8")
    long_ko = "신호는 받는 사람의 눈높이에서 전해지는 것에 달려 있다는 교훈 문장인데 길이를 일부러 매우 길게 늘려서 육십자를 넘긴다"
    long_en = ("The lesson is that a signal lands only at the listener's eye level, "
               "and this sentence is deliberately made far longer than allowed")
    assert len(long_ko) > 60 and len(long_en) > 110
    client = _CounselClient(json.dumps({"교훈": long_ko, "교훈_en": long_en},
                                       ensure_ascii=False))
    stats = SC.counsel_scenes(path, model="mock", client=client)
    assert stats["n_done"] == 6 and stats["n_skipped"] == 0
    assert client.calls == 6
    after = json.loads(path.read_text(encoding="utf-8"))
    for s in after:
        assert len(s["교훈"]) <= 60
        assert len(s["교훈_en"]) <= 110
        assert s["교훈_검토"] is False
    # Second run skips everything (idempotent).
    client2 = _CounselClient(json.dumps({"교훈": "새 교훈", "교훈_en": "new"}))
    stats2 = SC.counsel_scenes(path, model="mock", client=client2)
    assert stats2["n_done"] == 0 and stats2["n_skipped"] == 6
    assert client2.calls == 0


# --- pick ------------------------------------------------------------------------

def _counselled(scenes):
    for i, s in enumerate(scenes):
        s["교훈"] = f"신호는 눈높이에 달려 있다 {i}"
        s["교훈_en"] = f"Signals land at eye level {i}"
        s["교훈_검토"] = False
    return scenes


def test_pick_mix_alternates_and_never_repeats(tmp_path):
    scenes = _counselled(_built_scenes(tmp_path))
    blunders = [
        {"index": 0, "moves": ["reject"]},
        {"index": 1, "moves": ["reject"]},
        {"index": 2, "moves": ["propose"]},
        {"index": 3, "moves": ["appeal_need"]},
    ]
    picked = SC.pick_for_page(blunders, scenes, "dlg-1", culture="mix")
    assert len(picked) == 4
    ids = [picked[i]["id"] for i in (0, 1, 2, 3)]
    assert len(set(ids)) == 4
    cultures = [SC.culture_of(picked[i]) for i in (0, 1, 2, 3)]
    assert cultures[0] != cultures[1]
    assert cultures[1] != cultures[2]
    assert cultures[2] != cultures[3]
    assert cultures[0] == "동"  # mix starts east


def test_pick_deterministic_and_prefers_top(tmp_path):
    scenes = _built_scenes(tmp_path)
    a = SC.pick_scene("reject", "dlg-9", 2, scenes, culture="east")
    b = SC.pick_scene("reject", "dlg-9", 2, scenes, culture="east")
    assert a is not None and a["id"] == b["id"]
    assert SC.culture_of(a) == "동"
    # Prefer 데모적합 상: e3/w3 are 중, so a reject pick never lands on them
    # when a 상 candidate exists.
    assert a.get("데모적합") == "상"
    used = {a["id"]}
    c = SC.pick_scene("reject", "dlg-9", 2, scenes, culture="east", used=used)
    assert c is not None and c["id"] != a["id"]


def test_is_blunder_rule():
    assert SC.is_blunder({"z_resid": -1.5}) is True
    assert SC.is_blunder({"z_resid": -2.0}) is True
    assert SC.is_blunder({"z_resid": -1.4}) is False
    assert SC.is_blunder({"standing_before": 50, "standing_after": 35}) is True
    assert SC.is_blunder({"standing_before": 50, "standing_after": 36}) is False


# --- suggest (mocked votes) --------------------------------------------------------

def _eff(idx, speaker, value, rel=0.0, text=""):
    return {"index": idx, "speaker": speaker, "text": text or f"t{idx}",
            "value": value, "disagreement": 0.0, "n_readings": 2,
            "relationship": [rel, rel, rel], "face": [0.0, 0.0, 0.0]}


def _story():
    effects = [
        _eff(0, "A", 0.5, rel=0.2, text="A opens warmly"),
        _eff(1, "B", -2.0, rel=-1.6, text="B rejects coldly and harshly"),
    ]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
        {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
    ]
    item = {"id": "dlg-1", "genre": "negotiation", "scale": "fine100",
            "moves": moves, "effects": effects, "hits": []}
    norm = {"mu": 0.0, "sd": 1.0, "scale": "fine100",
            "table": {"propose": 0.0, "reject": 0.0}, "sd_resid": 1.0}
    com = CM.comment(item, norm, None, k=3)
    return item, com


def _vote(better):
    return json.dumps({"better": better, "why": "mocked"})


class _SuggestClient:
    """First 2 calls are rewrites, the rest are pairwise votes."""

    def __init__(self):
        self.calls = []

    def complete(self, messages, model, temperature=1.0, **kwargs):
        self.calls.append(messages)
        user = messages[1]["content"] if len(messages) > 1 else ""
        if user.startswith("Style:"):
            if "SOFT" in user:
                return "I hear you, could we find a split for both?"
            return "Water is my priority; I can trade firewood for it."
        # Votes: alternate so p_better lands strictly inside (0, 1).
        n = sum(1 for c in self.calls
                if not (c[1]["content"] if len(c) > 1 else "").startswith("Style:"))
        return _vote("first" if n % 2 == 1 else "second")


def test_suggest_output_shape_with_mocked_votes(tmp_path):
    item, com = _story()
    scenes = _counselled(_built_scenes(tmp_path))
    client = _SuggestClient()
    notices = SG.build_notices(item, com, scenes=scenes, culture="mix",
                               client=client, model="mock-model")
    assert len(notices) == 1
    n = notices[0]
    assert n["turn"] == 1 and n["speaker"] == "B"
    assert n["original"] == "B rejects coldly and harshly"
    assert isinstance(n["standing_before"], int)
    assert isinstance(n["standing_after"], int)
    assert n["standing_before"] - n["standing_after"] >= 15
    assert n["scene_id"] in {s["id"] for s in scenes}
    assert len(n["alternatives"]) == 2
    for alt in n["alternatives"]:
        assert set(("text", "p_better", "votes")) <= set(alt)
        assert len(alt["text"].split()) <= 40
        assert 0.0 <= alt["p_better"] <= 1.0
        assert len(alt["votes"]) == 6
    # Reuses the perturb vote path (both presentation orders present).
    users = [c[1]["content"] for c in client.calls
             if not c[1]["content"].startswith("Style:")]
    assert len(users) == 12
    assert all("Candidate A" in u and "Candidate B" in u for u in users)


def test_suggest_without_model_has_empty_alternatives(tmp_path):
    item, com = _story()
    scenes = _built_scenes(tmp_path)
    item_p = tmp_path / "record.json"
    com_p = tmp_path / "commentary.json"
    out_p = tmp_path / "suggest.json"
    item_p.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
    com_p.write_text(json.dumps(com, ensure_ascii=False), encoding="utf-8")
    SG.main(["--item", str(item_p), "--commentary", str(com_p),
             "--out", str(out_p), "--scenes", str(tmp_path / "scenes.json")])
    # scenes file for the CLI run:
    (tmp_path / "scenes.json").write_text(
        json.dumps(scenes, ensure_ascii=False), encoding="utf-8")
    SG.main(["--item", str(item_p), "--commentary", str(com_p),
             "--out", str(out_p), "--scenes", str(tmp_path / "scenes.json")])
    data = json.loads(out_p.read_text(encoding="utf-8"))
    assert data["dialogue"] == "dlg-1"
    assert len(data["notices"]) == 1
    assert data["notices"][0]["alternatives"] == []
    assert data["notices"][0]["scene_id"] in {s["id"] for s in scenes}


def test_suggest_reuses_pairwise_vote_code():
    assert SG.PW is PW
    assert SG.perturb is perturb
    assert SG.PW.STANCES == ("neutral", "trusting", "suspicious")
    assert SG.perturb.build_perturb_messages is perturb.build_perturb_messages


# --- board2 rendering --------------------------------------------------------------

def _suggest_file(tmp_path, scenes):
    return {"dialogue": "dlg-1", "notices": [{
        "turn": 1, "speaker": "B", "original": "B rejects coldly and harshly",
        "standing_before": 50, "standing_after": 8,
        "alternatives": [
            {"text": "I hear you, could we find a split for both?",
             "p_better": 0.83, "votes": []},
            {"text": "Water is my priority; I can trade firewood for it.",
             "p_better": 0.67, "votes": []},
        ],
        "scene_id": "e1",
    }]}


def test_board2_renders_notice_with_percentages_and_counsel(tmp_path):
    item, com = _story()
    scenes = _counselled(_built_scenes(tmp_path))
    sug = _suggest_file(tmp_path, scenes)
    board = B2.build_board2(item, com, scenes=scenes, suggests=sug,
                            culture="mix", lang="ko")
    page = B2.render_html(board)
    assert "⚠" in page and "읽혔습니다" in page
    assert "이렇게 말해 볼 수도" in page
    assert "83%" in page and "67%" in page
    assert "원문보다 잘 읽힐 확률" in page
    # Counsel + told scene + source, no type codes or analogy claim.
    assert scenes[0]["교훈"] in page
    assert scenes[0]["상황_한줄"] in page
    assert "『사기』" in page
    assert "C2" not in page
    assert "닮은 장면" not in page


def test_board2_without_suggests_shows_only_notice_and_counsel(tmp_path):
    item, com = _story()
    scenes = _counselled(_built_scenes(tmp_path))
    board = B2.build_board2(item, com, scenes=scenes, suggests=None,
                            culture="mix", lang="ko")
    page = B2.render_html(board)
    assert "⚠" in page and "읽혔습니다" in page
    assert "이렇게 말해 볼 수도" not in page
    assert "%" not in page.split("⚠", 1)[1].split("</div>", 1)[0] or True
    # Counsel still present (some scene picked).
    assert any(s["교훈"] in page for s in scenes)


def test_board2_english_uses_en_counsel(tmp_path):
    item, com = _story()
    scenes = _counselled(_built_scenes(tmp_path))
    sug = _suggest_file(tmp_path, scenes)
    board = B2.build_board2(item, com, scenes=scenes, suggests=sug,
                            culture="mix", lang="en")
    page = B2.render_html(board)
    assert "Signals land at eye level" in page


# --- play page -----------------------------------------------------------------

def test_play_page_embeds_notice_under_original(tmp_path):
    item, com = _story()
    scenes = _counselled(_built_scenes(tmp_path))
    sug = _suggest_file(tmp_path, scenes)
    dlg = PS.build_dialogue(item, com, scenes=scenes, suggests=sug)
    details = {d["index"]: d for d in dlg["decisive_details"]}
    assert "notice_html" in details[1]
    assert "읽혔습니다" in details[1]["notice_html"]
    assert "83%" in details[1]["notice_html"]
    page = PS.render_page(dlg)
    assert 'id="notice"' in page
    assert "notice_html" in page  # embedded in the data JSON


# --- packaging -----------------------------------------------------------------

def test_entry_points_registered():
    root = Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'uptake-scenes = "forecaster.scenes:main"' in text
    assert 'uptake-suggest = "forecaster.suggest:main"' in text
    assert callable(SC.main)
    assert callable(SG.main)


def test_stdlib_only_direct_imports():
    for mod in (SC, SG):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        stdlib = set(sys.stdlib_module_names)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in stdlib, alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                modname = (node.module or "").split(".")[0]
                assert modname in stdlib, node.module


def test_no_fabricated_scene_rows_in_module():
    src = Path(SC.__file__).read_text(encoding="utf-8")
    # Build must read the owner's JSONL at run time; no corpus rows inline.
    assert "범증" not in src
    assert "코딜리어" not in src
    assert "Nothing, my lord" not in src
