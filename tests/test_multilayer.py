"""Multilayer reading: scores, blanks, no look-ahead, layers above items."""
import json
from forecaster import multilayer as M

T = [{"speaker": "A", "text": "We need priority, please."}, {"speaker": "B", "text": "Turn left, heading 250."},
     {"speaker": "A", "text": "FUTURE-MARKER roger."}, {"speaker": "B", "text": "ok"}, {"speaker": "A", "text": "end"}]
HI, LO = [0, 0, 0, .2, .8], [.8, .2, 0, 0, 0]

class Stub:
    def __init__(self, fn): self.fn, self.seen = fn, []
    def complete(self, m, **k):
        self.seen.append(m)
        return json.dumps(self.fn(m))

def stake_reply(_m):
    return {"items": {i: {"wanted": HI, "actual": LO, "reply_turn": 1, "quote": "turn left heading 250"} for i in M.ITEMS}, "emotion_intensity": 40}

def test_bins_score_and_spread():
    s = M.bins_to_score([0, 0, 1, 0, 0]); assert s == {"score": 50.0, "spread": 0.0}
    assert M.bins_to_score(None) is None and M.bins_to_score([1, 2]) is None and M.bins_to_score([0, 0, 0, 0, 0]) is None

def test_reading_before_reply_never_sees_later_turns():
    r = M.Reader(Stub(lambda m: {"items": {}}), "m")
    M.read_expected(r, T, 1, {})
    for call in r.client.seen:
        assert "FUTURE-MARKER" not in json.dumps(call, ensure_ascii=False)
    assert all("#2" not in l for l in M.lines_before(T, 1))

def test_bad_quote_is_dropped_and_score_kept():
    d = stake_reply(None); d["items"]["money"]["quote"] = "made up"
    r = M.Reader(Stub(lambda m: d), "m")
    out = M.read_stakes(r, T, 0)["items"]
    assert out["money"]["quote"] == "" and out["money"]["actual"]["score"] < 30
    assert out["time"]["actual"]["score"] < 30

def test_null_stays_blank_not_zero():
    r = M.Reader(Stub(lambda m: {"items": {i: {"wanted": None, "actual": None} for i in M.ITEMS}}), "m")
    out = M.read_stakes(r, T, 0)["items"]
    assert out["trust"]["wanted"] is None and M.broken_items(out) == []

def test_cause_only_when_broken():
    calls = []
    def fn(m):
        sys = m[0]["content"]
        calls.append(sys[:20])
        return {"items": {}} if "wish" in sys or "Stake class" in sys else {}
    M.read_turn(M.Reader(Stub(fn), "m"), T, 0, {})
    assert not any(c.startswith("One turn") for c in calls)

def test_cause_keeps_only_known_values():
    def fn(m):
        text = str(m)
        if "Group: amount" in text: return {"causes": {"amount": 80, "bogus": 50}}
        if "Group: emotion" in text: return {"causes": {"face": 300}}
        if "Group: power" in text: return {"causes": {}}
        if "misreading codes" in text: return {"types": {"B4": 70, "Z9": 90}}
        return {"kind": {"accident": 1, "conflict": 3}, "stages": {"said_to_received": 20}}
    c = M.read_cause(M.Reader(Stub(fn), "m"), T, 0)
    assert c["say"] == {"amount": 80.0} and c["mind"] == {} and c["world"] == {} and c["types"] == {"B4": 70.0}
    assert c["kind"]["conflict"] == 75.0 and c["stages"]["meant_to_said"] is None

def test_cause_asks_each_group_separately():
    seen = []
    M.read_cause(M.Reader(Stub(lambda m: seen.append(str(m)) or {}), "m"), T, 0)
    assert len(seen) == 5 and sum("irony" in x and "channel_timing" in x for x in seen) == 0

def test_repeat_readings_do_not_share_cache(tmp_path):
    s = Stub(lambda m: {"items": {}})
    M.Reader(s, "m", cache_dir=tmp_path, rep=0).ask("s", "u"); M.Reader(s, "m", cache_dir=tmp_path, rep=0).ask("s", "u")
    assert len(s.seen) == 1
    M.Reader(s, "m", cache_dir=tmp_path, rep=1).ask("s", "u"); assert len(s.seen) == 2

def test_layers_weight_by_wanted_and_stakes_raise_severity():
    r = M.read_stakes(M.Reader(Stub(stake_reply), "m"), T, 0)
    low, high = M.combine(r, 10), M.combine(r, 100)
    assert low["classes"]["outcome"] < 20 and high["severity"] > low["severity"] and low["miss"] == high["miss"]

def test_blanks_stay_out_of_layers():
    r = {"items": {i: {"class": M.ITEM_CLASS[i], "wanted": None, "actual": None} for i in M.ITEMS}}
    c = M.combine(r); assert c["classes"]["face"] is None and c["confidence"] is None and c["miss"] == 0

def test_features_need_no_model():
    f = M.code_features(T, 0); assert f["words"] == 4 and f["reply_words"] == 4 and f["is_question"] is False and f["speakers"] == 2

def test_failed_reply_is_counted_not_blank():
    class Bad:
        n = 0
        def complete(self, m, **k):
            Bad.n += 1
            return "not json"
    r = M.Reader(Bad(), "m"); assert r.ask("s", "u") is None and r.failures == 1 and Bad.n == 2

def test_recovery_uses_same_speaker_and_class():
    e = lambda c, a: {"class": c, "wanted": None, "actual": {"score": a, "spread": 1} if a else None}
    rs = [{"turn": 0, "speaker": "A", "broken": ["money"], "items": {"money": e("outcome", 20)}},
          {"turn": 1, "speaker": "B", "broken": [], "items": {"money": e("outcome", 99)}},
          {"turn": 2, "speaker": "A", "broken": [], "items": {"time": e("outcome", 80), "trust": e("relationship", 95)}}]
    assert M.recovery(rs) == {"0:money": 80}

def test_expected_reading_does_not_get_stakes_or_evidence():
    r = M.Reader(Stub(lambda m: {"items": {}}), "m")
    M.read_expected(r, T, 1, {"stakes_weight": 100, "evidence": "FUTURE-MARKER", "power": 10})
    blob = json.dumps(r.client.seen, ensure_ascii=False)
    assert "FUTURE-MARKER" not in blob and "stakes_weight" not in blob and "power" in blob

def _r(turn, sp, wanted, actual):
    e = {"class": "outcome", "wanted": {"score": wanted, "spread": 10}, "actual": {"score": actual, "spread": 10}, "reply_turn": None, "quote": ""}
    items = {i: {"class": M.ITEM_CLASS[i], "wanted": None, "actual": None, "reply_turn": None, "quote": ""} for i in M.ITEMS}
    items["commitment"] = e
    return {"turn": turn, "speaker": sp, "items": items, "broken": M.broken_items(items), "expected": {}, "cause": None}

def _cells(**values):
    items = M.blank_reading(T, 0)["items"]
    for name, (wanted, actual) in values.items():
        items[name] = {**items[name], "wanted": wanted and {"score": wanted, "spread": 10},
                       "actual": actual is not None and {"score": actual, "spread": 10} or None}
    return {"items": items}

USE = {"deliverable", "warmth", "competence", "facts"}

def test_turn_effect_adds_signed_and_marks_on_the_board_cuts():
    for values, value, mark in (
            ({"deliverable": (100.0, 37.6)}, -0.248, "="),
            ({"deliverable": (100.0, 37.5)}, -0.25, "?!"),
            ({"deliverable": (100.0, 25.0), "facts": (100.0, 37.6)}, -0.748, "?!"),
            ({"deliverable": (100.0, 25.0), "facts": (100.0, 37.5)}, -0.75, "?"),
            ({"deliverable": (100.0, 12.6), "facts": (100.0, 12.5)}, -1.498, "?"),
            ({"deliverable": (100.0, 12.5), "facts": (100.0, 12.5)}, -1.5, "??"),
            ({"warmth": (100.0, 68.7), "competence": (100.0, 68.7)}, 0.748, "="),
            ({"warmth": (100.0, 68.7), "competence": (100.0, 68.8)}, 0.75, "!"),
            ({"warmth": (100.0, 87.4), "competence": (100.0, 87.5)}, 1.498, "!"),
            ({"warmth": (100.0, 87.5), "competence": (100.0, 87.5)}, 1.5, "!!"),
            ({"facts": (80.0, 50.0)}, 0.0, "=")):
        got = M.turn_effect(_cells(**values), USE)
        assert (got["value"], got["mark"]) == (value, mark), values

def test_turn_effect_leaves_reference_items_out():
    use = M.measured_items({"status": "measured", "items": {"deliverable": "measured", "commitment": "reference", "time": "ungraded"}})
    got = M.turn_effect(_cells(deliverable=(50.0, 25.0), commitment=(100.0, 0.0), time=(100.0, 0.0)), use)
    assert got == {"value": -0.25, "mark": "?!", "items": {"deliverable": -0.25}}
    only = M.turn_effect(_cells(commitment=(100.0, 0.0)), use)
    assert only == {"value": None, "mark": None, "items": {}}

def test_turn_effect_blank_turn_gets_no_value_and_no_mark():
    assert M.turn_effect(M.blank_reading(T, 0), USE) == {"value": None, "mark": None, "items": {}}
    half = M.turn_effect(_cells(deliverable=(80.0, None), facts=(None, 20.0)), USE)
    assert half == {"value": None, "mark": None, "items": {}}

def test_turn_effect_shows_where_the_value_came_from():
    got = M.turn_effect(_cells(deliverable=(57.0, 24.0), facts=(79.0, 22.6), warmth=(None, None)), USE)
    assert got["items"] == {"deliverable": -0.2964, "facts": -0.43292}
    assert got["value"] == -0.72932 and got["mark"] == "?!"

def test_standing_signs_and_blanks():
    s = M.build_standing_items([_r(0, "A", 80, 90), _r(1, "B", 80, 10), _r(2, "A", 80, 90)])
    assert s["end"]["A"]["outcome"]["score"] > 0 and s["end"]["B"]["outcome"]["score"] < 0
    assert s["end"]["A"]["face"]["judged"] == 0 and s["end"]["A"]["face"]["mean"] is None
    assert s["splits"]["outcome"]["turn"] in (0, 1, 2) and s["splits"]["face"] is None

def test_alert_level_rises_with_stakes_and_resolves():
    g = {"status": "provisional", "items": {}, "cuts": [10, 20, 30, 40]}
    low = M.build_alerts_items([_r(0, "A", 80, 10), _r(1, "B", 50, 50)], {"stakes_weight": 30}, g)
    high = M.build_alerts_items([_r(0, "A", 80, 10), _r(1, "A", 50, 80)], {"stakes_weight": 100}, g)
    assert M.LEVELS.index(high[0]["level"]) > M.LEVELS.index(low[0]["level"]) and high[0]["resolved"] and not low[0]["resolved"]
    assert high[0]["items"] == ["commitment"]

def test_alerts_only_use_measured_items():
    g = {"status": "measured", "items": {"commitment": "reference"}, "cuts": [1, 2, 3, 4]}
    assert M.build_alerts_items([_r(0, "A", 80, 10)], {"stakes_weight": 100}, g) == []

def test_no_client_gives_blanks_not_values():
    m = M.build_multilayer(T)
    assert len(m["uptake"]) == 4 and m["alerts"] == [] and m["uptake"][0]["items"]["money"]["wanted"] is None and m["conditions"]["source"] == "not read"
    assert m["conditions"]["stakes_source"] == "not read" and m["conditions"]["stakes_weight_model"] is None

BASIS = "The board report lists a total loss of the aircraft."
PROVISIONAL = {"status": "provisional", "items": {}, "cuts": [10.0, 20.0, 30.0, 40.0]}

def _whole_reply(m):
    system = m[0]["content"]
    if "describe its conditions" in system:
        return {"relationship": "crew and tower", "power": -20, "channel": "radio", "translated": False,
                "culture": None, "stakes_weight": 20, "evidence": "Turn left, heading 250."}
    return {"items": {}} if "heard only" in system else stake_reply(m)

def _build(stakes_record=None, client=None, cache_dir=None, tmp=None):
    return M.build_multilayer(T, client=client or Stub(_whole_reply), model="m", cache_dir=cache_dir or tmp, max_turns=2,
                              record={"title": "t", "note": "n"}, grades=PROVISIONAL, stakes_record=stakes_record)

def test_stakes_record_sets_weight_severity_and_source(tmp_path):
    est, rec = _build(tmp=tmp_path / "a"), _build({"weight": 100, "basis": f"  {BASIS} "}, tmp=tmp_path / "b")
    assert rec["conditions"]["stakes_weight"] == 100.0 and rec["conditions"]["stakes_weight_model"] == 20.0
    assert rec["conditions"]["stakes_source"] == f"record: {BASIS}" and rec["conditions"]["source"] == "model estimate"
    for got, weight in ((est, 20.0), (rec, 100.0)):
        assert got["uptake"][0]["layers"] == M.combine(got["uptake"][0], weight, M.measured_items(PROVISIONAL))
    assert (rec["alerts"][0]["severity"], est["alerts"][0]["severity"]) == (59.2, 11.8)
    assert M.LEVELS.index(rec["alerts"][0]["level"]) > M.LEVELS.index(est["alerts"][0]["level"])

def test_stakes_record_zero_weight_is_a_value():
    got = M.apply_stakes_record({"stakes_weight": 90.0, "source": "model estimate"}, {"weight": 0, "basis": BASIS})
    assert got["stakes_weight"] == 0.0 and got["stakes_source"] == f"record: {BASIS}"

def test_missing_or_bad_stakes_record_stays_model_estimate():
    for bad in (None, {}, "80", [80, BASIS], {"weight": 80}, {"basis": BASIS}, {"weight": 100.5, "basis": BASIS},
                {"weight": -1, "basis": BASIS}, {"weight": "80", "basis": BASIS}, {"weight": True, "basis": BASIS},
                {"weight": float("nan"), "basis": BASIS}, {"weight": 80, "basis": ""}, {"weight": 80, "basis": "   "},
                {"weight": 80, "basis": None}, {"weight": 80, "basis": 7}):
        got = M.apply_stakes_record({"stakes_weight": 35.0, "source": "model estimate"}, bad)
        assert got == {"stakes_weight": 35.0, "source": "model estimate",
                       "stakes_weight_model": 35.0, "stakes_source": "model estimate"}, bad

def test_no_stakes_record_changes_no_value_only_adds_the_source(tmp_path):
    got = _build(tmp=tmp_path)
    conditions = M.read_context(M.Reader(Stub(_whole_reply), "m"), T, {"title": "t", "note": "n"})
    assert got["conditions"] == {**conditions, "stakes_weight_model": 20.0, "stakes_source": "model estimate"}
    assert got["alerts"] == M.build_alerts_items(got["uptake"], conditions, PROVISIONAL)

def test_stakes_record_stays_out_of_every_prompt_and_needs_no_new_reading(tmp_path):
    plain, recorded = Stub(_whole_reply), Stub(_whole_reply)
    _build(client=plain, tmp=tmp_path / "a")
    _build({"weight": 73, "basis": BASIS}, client=recorded, tmp=tmp_path / "b")
    blob = lambda s: sorted(json.dumps(c, ensure_ascii=False) for c in s.seen)
    assert blob(plain) == blob(recorded) and any("describe its conditions" in c for c in blob(recorded))
    assert BASIS not in "".join(blob(recorded)) and "stakes_record" not in "".join(blob(recorded))
    again = Stub(_whole_reply)  # the readings already cached without a record serve the recorded run
    got = _build({"weight": 73, "basis": BASIS}, client=again, tmp=tmp_path / "a")
    assert again.seen == [] and got["conditions"]["stakes_weight"] == 73.0

def test_render_keeps_dash_for_blank_and_shows_items():
    import html
    from forecaster.report import render_readings
    r = _r(0, "A", 80, 10); r["layers"] = M.combine(r, 50); r["cause"] = {"say": {"amount": 80}, "mind": {}, "world": {}, "kind": {"accident": 100}, "stages": {"said_to_received": 20}}
    h = render_readings([r], {"source": "model estimate", "power": 10}, html.escape)
    assert "commitment: wanted 80, expected –, landed 10" in h and "say: amount" in h and "said_to_received" in h and "<td>–</td>" in h and "power: 10" in h
