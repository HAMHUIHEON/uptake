"""Contract for SPEC_task53_rewrite_play (synthetic data only, mocked API)."""
import http.server
import io
import json
import types
from pathlib import Path

import pytest

from forecaster import commentary as CM
from forecaster import pairwise as PW
from forecaster import perturb
from forecaster import play_server as PS
from forecaster.usage import BudgetExceeded, Meter


def _eff(idx, speaker, value, rel=0.0, text="", dis=0.0):
    return {"index": idx, "speaker": speaker, "text": text or f"t{idx}",
            "value": value, "disagreement": dis, "n_readings": 2,
            "relationship": [rel, rel, rel], "face": [0.0, 0.0, 0.0]}


def _item(cid, effects, moves, scale="fine100"):
    return {"id": cid, "genre": "negotiation", "scale": scale,
            "moves": moves, "effects": effects, "hits": []}


def _resid_norm(table, sd_resid=1.0):
    return {"mu": 0.0, "sd": 1.0, "scale": "fine100",
            "table": dict(table), "sd_resid": sd_resid}


def _story_fixture():
    # Standings (50 + 25·rel, half up): A 50→63, B 50→53, B 53→20, A 63→55.
    # Marks with a zero table: 1.2 → "!", -1.5 → "?", so decisive == [0, 2].
    effects = [
        _eff(0, "A", 1.2, rel=0.5, text="A opens warmly"),
        _eff(1, "B", -0.3, rel=0.1, text="B hedges a little"),
        _eff(2, "B", -1.5, rel=-1.2, text="B rejects coldly"),
        _eff(3, "A", 0.4, rel=0.2, text="A tweaks the offer"),
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


def _vote(better):
    return json.dumps({"better": better, "why": "mocked"})


class FakeClient:
    """Queued pairwise replies; records every call for reuse checks."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, messages, model, temperature=1.0, **kwargs):
        self.calls.append({"messages": messages, "model": model,
                           "temperature": temperature})
        if self.outputs:
            return self.outputs.pop(0)
        return _vote("tie")


class _FakeSocket:
    """In-memory stand-in for the handler's socket (no sockets needed)."""

    def __init__(self, raw: bytes):
        self._r = io.BytesIO(raw)
        self._w = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        return self._r if "r" in mode else self._w

    def sendall(self, data):
        self._w.write(data)

    def close(self):
        pass


def _handle(state, raw: bytes):
    sock = _FakeSocket(raw)
    PS._Handler(sock, ("127.0.0.1", 12345),
                types.SimpleNamespace(state=state))
    head, _, body = sock._w.getvalue().partition(b"\r\n\r\n")
    lines = head.split(b"\r\n")
    return int(lines[0].split(b" ")[1]), body


def _get(state, path):
    raw = (f"GET {path} HTTP/1.1\r\nHost: x\r\n"
           f"Connection: close\r\n\r\n").encode("utf-8")
    status, body = _handle(state, raw)
    return status, body.decode("utf-8")


def _get_json(state, path):
    status, text = _get(state, path)
    try:
        return status, json.loads(text)
    except ValueError:
        return status, {}


def _post(state, path, obj):
    data = json.dumps(obj).encode("utf-8")
    raw = (f"POST {path} HTTP/1.1\r\nHost: x\r\n"
           f"Content-Type: application/json\r\n"
           f"Content-Length: {len(data)}\r\n"
           f"Connection: close\r\n\r\n").encode("utf-8") + data
    status, body = _handle(state, raw)
    try:
        return status, json.loads(body.decode("utf-8"))
    except ValueError:
        return status, {}


def _write_dirs(tmp_path, item, com):
    rec = tmp_path / "records"
    cdir = tmp_path / "commentary"
    rec.mkdir()
    cdir.mkdir()
    (rec / f"{item['id']}.json").write_text(
        json.dumps(item, ensure_ascii=False), encoding="utf-8")
    (cdir / f"{item['id']}.json").write_text(
        json.dumps(com, ensure_ascii=False), encoding="utf-8")
    return rec, cdir


def _state(tmp_path, outputs, budget=5.0, log=None):
    item, com = _story_fixture()
    rec, cdir = _write_dirs(tmp_path, item, com)
    return PS.PlayState(records_dir=rec, commentary_dir=cdir,
                        model="mock-model", meter=Meter(budget_usd=budget),
                        log_path=log, client=FakeClient(outputs)), item, com


# --- dialogue payload ------------------------------------------------------


def test_dialogue_payload_shape():
    item, com = _story_fixture()
    dlg = PS.build_dialogue(item, com)
    assert dlg["id"] == "casino-878"
    assert len(dlg["turns"]) == 4
    for t in dlg["turns"]:
        assert {"index", "speaker", "text",
                "score_before", "score_after"} <= set(t)
    assert [(t["score_before"], t["score_after"]) for t in dlg["turns"]] == [
        (50, 63), (50, 53), (53, 20), (63, 55)]
    assert dlg["decisive"] == [0, 2]
    assert dlg["decisive_turns"] == [0, 2]
    assert len(dlg["decisive"]) <= 3
    assert len(dlg["standings"]) == 4
    assert dlg["standings"][2] == {"index": 2, "speaker": "B",
                                   "before": 53, "after": 20}
    assert len(dlg["decisive_details"]) == 2
    line = dlg["decisive_details"][1]["line"]
    assert "이 말에서" in line and "의 관계 점수가" in line and "→" in line


def test_dialogue_fallback_decisive_without_commentary():
    item, _ = _story_fixture()
    dlg = PS.build_dialogue(item, {})
    # |Δ| = 13/3/33/8 → top 3 are indices 2, 0, 3, shown chronologically.
    assert dlg["decisive"] == [0, 2, 3]
    assert len(dlg["decisive_details"]) == 3


def test_decisive_line_matches_spec_example():
    assert (PS._decisive_line("B", 50, 8)
            == "이 말에서 B의 관계 점수가 50 → 8로 떨어졌다")
    assert "올랐다" in PS._decisive_line("A", 50, 63)
    assert "그대로였다" in PS._decisive_line("A", 55, 55)


def test_margin_word_and_validate_text():
    assert PS.margin_word(0.5) == "부드럽다"
    assert PS.margin_word(0.16) == "부드럽다"
    assert PS.margin_word(0.15) == "비슷하다"
    assert PS.margin_word(0.0) == "비슷하다"
    assert PS.margin_word(-0.15) == "비슷하다"
    assert PS.margin_word(-1.0) == "거칠다"
    assert PS.margin_word(None) == "비슷하다"
    assert PS.validate_text("hello") == (True, "")
    assert PS.validate_text("x" * 400)[0] is True
    assert PS.validate_text("")[0] is False
    assert PS.validate_text("   ")[0] is False
    assert PS.validate_text("x" * 401)[0] is False
    assert PS.validate_text(None)[0] is False


# --- pairwise judging (mocked) ---------------------------------------------


def test_compare_six_votes_mixed():
    item, _ = _story_fixture()
    # orig_first: first/second/tie → orig/player/tie;
    # player_first: first/second/tie → player/orig/tie.
    client = FakeClient([_vote(b) for b in
                         ("first", "second", "tie") * 2])
    out = PS.compare_texts(client, item, 2, "부드럽게 다시 쓰기",
                           model="mock-model")
    assert 0.0 <= out["p_better"] <= 1.0
    assert out["p_better"] == pytest.approx(0.5)
    assert out["margin"] == pytest.approx(0.0)
    assert out["model"] == "mock-model"
    assert out["cost_usd"] == pytest.approx(0.0)
    assert len(out["votes"]) == 6
    assert sorted(v["stance"] for v in out["votes"]) == sorted(
        list(PW.STANCES) * 2)
    assert sorted(v["winner"] for v in out["votes"]) == [
        "orig", "orig", "player", "player", "tie", "tie"]
    assert len(client.calls) == 6


def test_compare_all_player_wins():
    item, _ = _story_fixture()
    client = FakeClient([_vote("second")] * 3 + [_vote("first")] * 3)
    out = PS.compare_texts(client, item, 2, "정말 부드럽게", model="m")
    assert out["p_better"] == pytest.approx(1.0)
    assert out["margin"] == pytest.approx(1.0)
    assert all(v["winner"] == "player" for v in out["votes"])


def test_compare_unknown_turn_raises():
    item, _ = _story_fixture()
    with pytest.raises(ValueError):
        PS.compare_texts(FakeClient([]), item, 99, "x")
    with pytest.raises(ValueError):
        PS.compare_texts(FakeClient([]), item, "2", "x")


def test_compare_budget_exceeded_raises():
    item, _ = _story_fixture()
    meter = Meter(budget_usd=0.0)
    with pytest.raises(BudgetExceeded):
        PS.compare_texts(FakeClient([_vote("tie")] * 6), item, 2, "x",
                         meter=meter)


def test_reuses_pairwise_vote_code():
    assert PS.PW is PW
    assert PS.perturb is perturb
    assert PS.PW.STANCES == ("neutral", "trusting", "suspicious")
    assert PS.perturb.build_perturb_messages is perturb.build_perturb_messages
    assert PS.PW.parse_pairwise is PW.parse_pairwise
    item, _ = _story_fixture()
    client = FakeClient([_vote("tie")] * 6)
    PS.compare_texts(client, item, 2, "다시 쓰기")
    user_texts = [m["messages"][1]["content"] for m in client.calls]
    assert len(user_texts) == 6
    assert all("Candidate A" in t and "Candidate B" in t for t in user_texts)
    # Both presentation orders are used.
    assert any("Candidate A (B): B rejects coldly" in t for t in user_texts)
    assert any("Candidate A (B): 다시 쓰기" in t for t in user_texts)


# --- server routes (mocked API, in-memory handler driver) --------------------


def test_server_dialogue_route(tmp_path):
    state, _, _ = _state(tmp_path, [])
    status, dlg = _get_json(state, "/api/dialogue?id=casino-878")
    assert status == 200
    assert dlg["id"] == "casino-878"
    assert len(dlg["turns"]) == 4
    assert dlg["decisive"] == [0, 2]
    assert len(dlg["standings"]) == 4
    assert _get_json(state, "/api/dialogue")[0] == 400
    assert _get_json(state, "/api/dialogue?id=nope")[0] == 404
    assert _get(state, "/nope")[0] == 404


def test_server_compare_route(tmp_path):
    state, _, _ = _state(tmp_path, [_vote("tie")] * 12)
    url = "/api/compare"
    status, out = _post(state, url, {"id": "casino-878", "turn": 2,
                                     "text": "부드럽게 다시 쓰기"})
    assert status == 200
    assert 0.0 <= out["p_better"] <= 1.0
    assert len(out["votes"]) == 6
    assert {"p_better", "votes", "margin",
            "cost_usd", "model"} <= set(out)
    assert out["model"] == "mock-model"
    # 400-char boundary is accepted.
    status, _ = _post(state, url, {"id": "casino-878", "turn": 2,
                                   "text": "x" * 400})
    assert status == 200
    assert _post(state, url, {"id": "casino-878", "turn": 2,
                              "text": ""})[0] == 400
    assert _post(state, url, {"id": "casino-878", "turn": 2,
                              "text": "   "})[0] == 400
    assert _post(state, url, {"id": "casino-878", "turn": 2,
                              "text": "x" * 401})[0] == 400
    assert _post(state, url, {"id": "casino-878", "turn": 99,
                              "text": "x"})[0] == 400
    assert _post(state, url, {"id": "nope", "turn": 2, "text": "x"})[0] == 404


def test_server_budget_402(tmp_path):
    state, _, _ = _state(tmp_path, [_vote("tie")] * 6, budget=0.0)
    status, out = _post(state, "/api/compare",
                        {"id": "casino-878", "turn": 2, "text": "x"})
    assert status == 402
    assert out.get("error") == "budget_exceeded"
    assert out.get("message")


def test_server_logs_comparison(tmp_path):
    log = tmp_path / "play.jsonl"
    state, _, _ = _state(tmp_path, [_vote("first")] * 6, log=log)
    status, _ = _post(state, "/api/compare",
                      {"id": "casino-878", "turn": 2, "text": "기록되는 시도"})
    assert status == 200
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert {"ts", "id", "turn", "text", "votes", "cost"} <= set(rec)
    assert rec["id"] == "casino-878" and rec["turn"] == 2
    assert rec["text"] == "기록되는 시도"
    assert len(rec["votes"]) == 6
    assert "META_API_KEY" not in lines[0] and "api_key" not in lines[0]


def test_server_binds_loopback_only(tmp_path, monkeypatch):
    import forecaster.reader as R

    item, com = _story_fixture()
    rec, cdir = _write_dirs(tmp_path, item, com)
    captured = {}

    class FakeServer:
        def __init__(self, addr, handler):
            captured["addr"] = addr
            captured["handler"] = handler

        def serve_forever(self):
            raise KeyboardInterrupt

    class FakeMuse:
        def __init__(self, meter=None):
            self.meter = meter

    monkeypatch.setattr(http.server, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(R, "MuseClient", FakeMuse)
    PS.main(["--records", str(rec), "--commentary-dir", str(cdir),
             "--port", "8765", "--budget", "1.0"])
    assert captured["addr"] == ("127.0.0.1", 8765)
    assert captured["handler"] is PS._Handler


# --- page ------------------------------------------------------------------


def _page_html():
    item, com = _story_fixture()
    return PS.render_page(PS.build_dialogue(item, com))


def test_page_contains_prompt_result_and_share_templates():
    page = _page_html()
    assert "같은 뜻을 당신이라면 어떻게 말할까요?" in page
    assert "자에게 물어보기" in page
    assert "읽는 중… (약 5초)" in page
    assert "당신의 말이 원문보다 잘 받아들여질 확률" in page
    assert "원문보다 부드럽다" in page
    assert "원문보다 비슷하다" in page
    assert "원문보다 거칠다" in page
    assert "다시 써 보기" in page
    assert "수를" in page and "%짜리 말로 고쳤다 — uptake" in page
    assert "중립" in page and "우호" in page and "의심" in page
    assert "이 말에서" in page and "의 관계 점수가" in page
    assert ("모델이 발언을 읽은 것이지 사람을 판정한 것이 아닙니다" in page)
    assert "한 번 묻는 비용 약" in page
    assert 'id="tabs"' in page
    assert ('<meta name="viewport" '
            'content="width=device-width, initial-scale=1">') in page
    assert "max-width:480px" in page
    assert "#0d1117" not in page


def test_server_root_serves_page(tmp_path):
    state, _, _ = _state(tmp_path, [])
    status, page = _get(state, "/?id=casino-878")
    assert status == 200
    assert "casino-878" in page
    assert "같은 뜻을 당신이라면 어떻게 말할까요?" in page
    assert "당신의 말이 원문보다 잘 받아들여질 확률" in page
    assert "%짜리 말로 고쳤다 — uptake" in page
    # No id → first dialogue.
    assert _get(state, "/")[0] == 200
    assert _get(state, "/?id=nope")[0] == 404


def test_static_export_embeds_api_url(tmp_path):
    item, com = _story_fixture()
    item_p = tmp_path / "record.json"
    com_p = tmp_path / "commentary.json"
    out_p = tmp_path / "play.html"
    item_p.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
    com_p.write_text(json.dumps(com, ensure_ascii=False), encoding="utf-8")
    PS.play_main(["--item", str(item_p), "--commentary", str(com_p),
                  "--out", str(out_p),
                  "--api-url", "http://example.test:9999"])
    page = out_p.read_text(encoding="utf-8")
    assert "http://example.test:9999" in page
    assert "같은 뜻을 당신이라면 어떻게 말할까요?" in page
    assert "당신의 말이 원문보다 잘 받아들여질 확률" in page
    assert "%짜리 말로 고쳤다 — uptake" in page
    assert "한 번 묻는 비용 약" in page


def test_pyproject_entry_points():
    root = Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'uptake-play-server = "forecaster.play_server:main"' in text
    assert 'uptake-play = "forecaster.play_server:play_main"' in text
    assert callable(PS.main)
    assert callable(PS.play_main)
