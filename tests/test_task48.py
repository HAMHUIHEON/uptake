"""Contract for SPEC_task48_board (synthetic data only)."""
import json
from pathlib import Path

from forecaster import board2 as B2
from forecaster import commentary as CM
from forecaster import viewer2 as V2


def _eff(idx, speaker, value, rel=0.0, text="", **kw):
    e = {"index": idx, "speaker": speaker, "text": text or f"t{idx}",
         "value": value, "disagreement": 0.0, "n_readings": 2,
         "relationship": [rel, rel, rel], "face": [0.0, 0.0, 0.0]}
    e.update(kw)
    return e


def _item(cid, effects, moves, scale="fine100"):
    return {"id": cid, "genre": "negotiation", "scale": scale,
            "moves": moves, "effects": effects, "hits": []}


def _resid_norm(table, sd_resid=1.0):
    return {"mu": 0.0, "sd": 1.0, "scale": "fine100",
            "table": dict(table), "sd_resid": sd_resid}


def _story_fixture():
    # z_resid == value (zero table); marks: A0 "?", B1 "!!", B2 "?".
    # A standings 50 -> 21, B standings 50 -> 14.
    effects = [
        _eff(0, "A", -1.0, rel=-1.16, text="A opens with a demand"),
        _eff(1, "B", 2.0, rel=0.4,
             text="No. That will not be possible, we need more time"),
        _eff(2, "B", -1.2, rel=-1.44, text="B rejects again coldly"),
        _eff(3, "A", 0.2, rel=-1.16, text="A proposes a tweak"),
    ]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
        {"turn_index": 1, "speaker": "B", "moves": ["reject"]},
        {"turn_index": 2, "speaker": "B", "moves": ["reject"]},
        {"turn_index": 3, "speaker": "A", "moves": ["counter"]},
    ]
    item = _item("casino-878", effects, moves)
    norm = _resid_norm({"propose": 0.0, "reject": 0.0, "counter": 0.0})
    return item, CM.comment(item, norm, None, k=3)


def _casino_fixture():
    return [
        {"dialogue_id": 878,
         "participant_info": {
             "mturk_agent_1": {"outcomes": {
                 "points_scored": 17, "satisfaction": "Slightly satisfied",
                 "opponent_likeness": "Slightly like"}},
             "mturk_agent_2": {"outcomes": {
                 "points_scored": 12, "satisfaction": "Undecided",
                 "opponent_likeness": "Extremely like"}},
         }},
    ]


def _run_main(tmp_path, item, com, casino=None, extra=()):
    item_p = tmp_path / "record.json"
    com_p = tmp_path / "commentary.json"
    out_p = tmp_path / "board.html"
    item_p.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
    com_p.write_text(json.dumps(com, ensure_ascii=False), encoding="utf-8")
    argv = ["--item", str(item_p), "--commentary", str(com_p),
           "--out", str(out_p)]
    if casino is not None:
        cas_p = tmp_path / "casino.json"
        cas_p.write_text(json.dumps(casino, ensure_ascii=False),
                         encoding="utf-8")
        argv += ["--casino", str(cas_p)]
    argv += list(extra)
    B2.main(argv)
    return out_p.read_text(encoding="utf-8")


# --- full page with outcomes ------------------------------------------------

def test_full_page_sections(tmp_path):
    item, com = _story_fixture()
    assert len(com["turns"]) == 3
    html = _run_main(tmp_path, item, com, _casino_fixture())

    # 1. verdict = task47 caption + the one-line note
    caption = V2.build_view2(item, com)["caption"]
    assert caption and "#1" in caption
    assert caption in html
    assert "관계 점수는 상대 눈에 비친 점수입니다 (50 = 보통)" in html

    # 2. score cards with correct start/end + sign words + Korean outcomes
    assert "A · 50 → 21" in html
    assert "B · 50 → 14" in html
    assert "↓ 잃음" in html
    assert "점수 17, 상대를 약간 좋아함" in html
    assert "점수 12, 상대를 매우 좋아함" in html
    assert "Extremely like" not in html
    assert "Slightly like" not in html
    assert "결과 없음" not in html

    # 3. simplified tide + legend
    assert 'height="220"' in html
    assert "빨강 A · 파랑 B · 선이 내려오면 A 우세" in html
    assert html.count('class="flag"') == 3
    assert ">#0<" in html and ">#1<" in html and ">#2<" in html
    assert "band" not in html.lower()
    assert html.count("<title>") == 1  # no hover tooltips in the svg

    # 4. decisive cards: quote + commentary line + Korean move kind
    # task48b: header uses turn index, not moment ordinal
    assert '#1 B — "No. That will not be possible' in html
    assert '#0 A — "A opens with a demand' in html
    assert "제안" in html and "거절" in html
    assert "A의 관계 점수가 50 → 21로 떨어졌다" in html
    assert "잔고" not in html

    # 5. lesson quote block from the first decisive turn's reference
    refs = {r["id"]: r for r in CM.load_references()}
    idea = refs[com["turns"][0]["reference"]["id"]]["idea_ko"]
    assert "<blockquote>" in html
    assert idea in html

    # 6. footer: disclaimer + dashboard link
    assert "모델이 발언을 읽은 것이지 사람을 판정한 것이 아닙니다" in html
    assert "dashboard_casino-878.html" in html
    assert "<a " in html

    # page frame: mobile viewport, narrow column, light theme
    assert ('<meta name="viewport" '
            'content="width=device-width, initial-scale=1">') in html
    assert "max-width:720px" in html
    assert "#0d1117" not in html
    assert "<audio" not in html


def test_without_casino_cards_omit_outcome_line(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    assert "결과 없음" not in html
    assert 'class="outcome"' not in html
    assert "상대를" not in html
    assert "A · 50 → 21" in html
    assert "B · 50 → 14" in html


def test_names_and_audio(tmp_path):
    item, com = _story_fixture()
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF----WAVE")
    html = _run_main(tmp_path, item, com, _casino_fixture(),
                     extra=["--names", '{"A": "Alice", "B": "Bob"}',
                            "--audio", str(audio)])
    assert "Alice · 50 → 21" in html
    assert "빨강 Alice · 파랑 Bob · 선이 내려오면 Alice 우세" in html
    assert '#0 Alice — "' in html
    assert "<audio" in html
    assert "board.audio.wav" in html
    assert str(audio) not in html
    assert (tmp_path / "board.audio.wav").is_file()


# --- caps, scales, helpers ---------------------------------------------------

def test_flags_and_cards_capped_at_three(tmp_path):
    effects = [_eff(i, "A" if i % 2 == 0 else "B", 0.5 * (i % 3),
                    rel=0.1, text=f"turn text number {i}")
               for i in range(6)]
    moves = [{"turn_index": i, "speaker": "A" if i % 2 == 0 else "B",
              "moves": ["reject"]} for i in range(6)]
    item = _item("cap-3", effects, moves)
    com = {"norm": {"mu": 0.0, "sd": 1.0},
           "turns": [
               {"index": i, "speaker": "A" if i % 2 == 0 else "B",
                "text": f"turn text number {i}", "moves": ["reject"],
                "mark": "?", "line_ko": f"{i}수 뒤 잔고 50 → 8. 평범한 수.",
                "reference": {"id": "fisher_ury_interests"}}
               for i in range(5)
           ]}
    html = _run_main(tmp_path, item, com)
    assert html.count('class="flag"') == 3
    assert html.count('class="moment"') == 3
    assert ">#1<" in html and ">#2<" in html and ">#5<" in html
    assert ">#3<" not in html and ">#4<" not in html
    assert "B의 관계 점수가 50 → 53으로 올랐다" in html
    assert "53 → 53으로 그대로였다" in html
    assert "잔고" not in html
    assert "입장이 아니라 이해(관심사)에 집중하고" in html


def test_pairwise_standings_match_viewer2(tmp_path):
    effects = [
        {"index": 0, "speaker": "A", "text": "a",
         "value": 0.6, "n_readings": 3, "listeners": ["X"]},
        {"index": 1, "speaker": "B", "text": "b",
         "value": 0.6, "n_readings": 3, "listeners": ["X"]},
        {"index": 2, "speaker": "A", "text": "c",
         "value": -0.6, "n_readings": 3, "listeners": ["X"]},
    ]
    moves = [
        {"turn_index": 0, "speaker": "A", "moves": ["propose"]},
        {"turn_index": 1, "speaker": "B", "moves": ["counter"]},
        {"turn_index": 2, "speaker": "A", "moves": ["propose"]},
    ]
    item = _item("pw-1", effects, moves, scale="pairwise")
    norm = {"mu": 0.0, "sd": 0.6, "scale": "pairwise",
            "table": {"propose": 0.0, "counter": 0.0}, "sd_resid": 0.5}
    com = CM.comment(item, norm, None, k=3)
    html = _run_main(tmp_path, item, com)
    trail = V2.build_view2(item, com)["standings"][-1]
    end_a = CM.standing_round(trail["A"])
    end_b = CM.standing_round(trail["B"])
    assert f"A · 50 → {end_a}" in html
    assert f"B · 50 → {end_b}" in html
    assert end_a == 27 and end_b == 73
    assert "↑ 얻음" in html and "↓ 잃음" in html


def test_delta_words_and_outcome_mapping():
    assert B2.delta_word(21) == ("↓", "잃음")
    assert B2.delta_word(50) == ("→", "지킴")
    assert B2.delta_word(73) == ("↑", "얻음")
    assert B2.likeness_ko("Extremely like") == "매우 좋아함"
    assert B2.likeness_ko("Slightly like") == "약간 좋아함"
    assert B2.likeness_ko("Undecided") == "미정"
    assert B2.likeness_ko("Slightly dislike") == "약간 싫어함"
    assert B2.likeness_ko("Extremely dislike") == "매우 싫어함"
    assert B2.likeness_ko(1) == "약간 좋아함"
    assert B2.likeness_ko(2) == "매우 좋아함"
    assert B2.satisfaction_ko("Extremely satisfied") == "매우 만족함"
    assert B2.satisfaction_ko("Slightly satisfied") == "약간 만족함"
    assert B2.satisfaction_ko("Undecided") == "미정"
    assert B2.satisfaction_ko(-1) == "약간 불만족"
    assert (B2.outcome_line({"points": 17, "likeness": 1})
            == "점수 17, 상대를 약간 좋아함")
    assert "미정" in B2.outcome_line({"points": 3, "likeness": "Undecided"})
    assert B2.outcome_line(None) is None


def test_reuses_viewer2_and_commentary_helpers():
    assert B2.MOVE_KO is V2.MOVE_KO
    assert B2.build_view2 is V2.build_view2
    assert B2.tide_y is V2.tide_y
    assert B2.resolve_audio_src is V2.resolve_audio_src
    assert B2.standing_round is CM.standing_round
    assert B2.load_references is CM.load_references
    assert B2.tide_caption is V2.tide_caption
    assert B2.hero_swings is V2.hero_swings
    assert B2.SPEAKER_COLORS is V2.SPEAKER_COLORS


def test_pyproject_entry_point():
    root = Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'uptake-board2 = "forecaster.board2:main"' in text
    assert callable(B2.main)


# --- task48b polish ------------------------------------------------------------

def test_tide_fills_present():
    svg = B2.tide_svg([0.5, -1.0, 1.2], [])
    assert 'class="tide-a"' in svg
    assert 'class="tide-b"' in svg
    assert V2.SPEAKER_COLORS[0] in svg
    assert V2.SPEAKER_COLORS[1] in svg
    assert svg.count("<path") >= 3  # two fills + boundary line
    assert "band" not in svg.lower()
    assert "<title>" not in svg


def test_decisive_diversity_includes_reversal():
    # M = [2.5, 4.5, 6.3, 4.8] -> deltas +2.5, +2.0, +1.8, -1.5.
    # Naive top-3 by |dM| would be 0,1,2 (all A, all +); diverse must
    # include 3 (opposite sign to the first big swing, other speaker).
    view = {
        "turns": [
            {"index": i, "speaker": "A" if i < 3 else "B",
             "display": "A" if i < 3 else "B",
             "text": f"neutral text {i}",
             "moves": ["propose"] if i < 3 else ["reject"],
             "line_ko": None}
            for i in range(4)
        ],
        "momentum": [2.5, 4.5, 6.3, 4.8],
        "colors": {"A": V2.SPEAKER_COLORS[0],
                   "B": V2.SPEAKER_COLORS[1]},
        "standings": [{"A": 50.0, "B": 50.0} for _ in range(4)],
    }
    com = {
        "turns": [
            {"index": i, "speaker": "A" if i < 3 else "B",
             "text": f"neutral text {i}",
             "moves": ["propose"] if i < 3 else ["reject"],
             "line_ko": f"{i}수 뒤 잔고 50 → 60. 이론 문장.",
             "reference": {"id": "fisher_ury_interests"}}
            for i in range(4)
        ]
    }
    dec = B2.decisive_of(com, view, {})
    assert len(dec) == 3
    idxs = {d["index"] for d in dec}
    assert 3 in idxs
    assert 0 in idxs
    assert len({d["speaker"] for d in dec}) > 1


def test_header_uses_turn_index(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    assert '#0 A — "' in html
    assert '#1 B — "' in html
    assert '#2 B — "' in html
    assert '#3 A — "' not in html
    assert '#3 B — "' not in html


def test_no_raw_move_label_in_card_body(tmp_path):
    import re
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    stripped = re.sub(r'title="[^"]*"', "", html)
    for raw in ("reject", "propose", "counter"):
        assert re.search(r"\b" + re.escape(raw) + r"\b", stripped) is None
    assert "거절" in stripped and "제안" in stripped
    assert 'title="reject"' in html
    assert 'title="propose"' in html
    refs = {r["id"]: r for r in CM.load_references()}
    idea0 = refs[com["turns"][0]["reference"]["id"]]["idea_ko"]
    assert html.count(idea0) == 1
    idea1 = refs[com["turns"][1]["reference"]["id"]]["idea_ko"]
    assert idea1 not in html


def test_scene_card_hook(tmp_path):
    assert callable(B2.scene_card_html)
    assert B2.scene_card_html({}) == ""
    item, com = _story_fixture()
    board = B2.build_board2(item, com)
    orig = B2.scene_card_html
    try:
        B2.scene_card_html = lambda turn: "<p>SCENE-MARK</p>"
        html = B2.render_html(board)
    finally:
        B2.scene_card_html = orig
    assert html.count("SCENE-MARK") == 3


# --- task48c board nits ------------------------------------------------------------

def _line_bodies(html):
    import re
    return re.findall(r'<p class="line">(.*?)</p>', html, re.S)


def test_card_body_plain_language(tmp_path):
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    bodies = _line_bodies(html)
    assert len(bodies) == 3
    for b in bodies:
        assert "수 " not in b
        assert "(??)" not in b
        assert "??" not in b
        assert "(?)" not in b and "(!!)" not in b
        assert "잔고" not in b
        assert "의 관계 점수가" in b
        assert ("올랐다" in b or "떨어졌다" in b
                or "그대로였다" in b)
    assert "A의 관계 점수가 50 → 21로 떨어졌다" in html
    assert "B의 관계 점수가 50 → 60으로 올랐다" in html
    assert "B의 관계 점수가 60 → 14로 떨어졌다" in html


def test_card_body_rewrites_dashboard_wording():
    # dashboard wording straight from the spec report
    com = {"turns": [
        {"index": 1, "speaker": "B", "text": "pleading text",
         "moves": ["reject", "appeal_need"], "mark": "??",
         "line_ko": "1수 B 악수(??) — 거절, 사정 호소 뒤 잔고 50 → 8. 이론 문장.",
         "reference": {"id": "fisher_ury_interests"}},
    ]}
    dec = B2.decisive_of(com, {}, {})
    assert len(dec) == 1
    line = dec[0]["line"]
    assert "수 " not in line
    assert "(??)" not in line
    assert "??" not in line
    assert "거절" in line and "사정 호소" in line
    assert "B의 관계 점수가" in line


def test_flag_labels_match_card_turn_numbers(tmp_path):
    import re
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    flags = re.findall(r'<g class="flag">.*?<text[^>]*>(.*?)</text>', html)
    cards = re.findall(r'<p class="quote">#(\S+) ', html)
    assert len(flags) == 3 == len(cards)
    assert all(f.startswith("#") for f in flags)
    assert flags == [f"#{c}" for c in cards]


def test_lesson_has_source_span(tmp_path):
    import re
    item, com = _story_fixture()
    html = _run_main(tmp_path, item, com)
    m = re.search(r"<blockquote>(.*?)</blockquote>", html, re.S)
    assert m, "lesson block missing"
    bq = m.group(1)
    refs = {r["id"]: r for r in CM.load_references()}
    ref = refs[com["turns"][0]["reference"]["id"]]
    assert ref["idea_ko"] in bq  # sentence kept
    assert "<span" in bq and "</span>" in bq
    assert "—" in bq
    short_title = ref["title"].split(":")[0].strip()
    assert short_title in bq
    for part in ref["author"].split(";"):
        token = re.sub(r"\(.*?\)", "", part).strip().split()[-1]
        assert token in bq
