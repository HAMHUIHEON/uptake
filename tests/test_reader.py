import json
from forecaster.reader import build_messages, parse_reading, read_panel
from forecaster.panel import build_panel

MSG = "Can you get the report to me by Friday? No pressure."


def test_prompt_contains_receiver_not_intent(base_receiver):
    msgs = build_messages(MSG, base_receiver)
    text = json.dumps(msgs, ensure_ascii=False)
    assert MSG in text and base_receiver.state in text and "slack" in text
    for word in ["speech_act", "next_action", "tone", "trigger", "gloss", "ask_clarification", "probe"]:
        assert word in text
    assert "sender's intent:" not in text.lower() and "intended" not in text.lower()


def test_parse_plain_fenced_and_prose():
    raw = '{"speech_act":"Order","tone":"-1.6","next_action":"Comply","trigger":"by Friday","gloss":"a hard deadline"}'
    r = parse_reading(raw, "r-0", 0, MSG)
    assert (r.speech_act, r.tone, r.next_action, r.trigger) == ("command", -2, "comply", "by Friday")
    fenced = "Sure.\n```json\n" + raw.replace("Order", "ask") + "\n```"
    assert parse_reading(fenced, "r-0", 1, MSG).speech_act == "request"
    prose = "Here you go: " + raw + " hope that helps"
    assert parse_reading(prose, "r-0", 2, MSG).sample == 2


def test_parse_rejects_and_cleans():
    assert parse_reading("not json", "r", 0, MSG) is None
    assert parse_reading('{"speech_act":"yell","tone":0,"next_action":"comply","trigger":"","gloss":""}', "r", 0, MSG) is None
    r = parse_reading('{"speech_act":"request","tone":9,"next_action":"clarify","trigger":"by Monday","gloss":"x"}', "r", 0, MSG)
    assert r.tone == 2 and r.next_action == "ask_clarification" and r.trigger == ""


def test_read_panel_retries_once_then_skips(base_receiver, fake_client_cls):
    ok = '{"speech_act":"request","tone":0,"next_action":"comply","trigger":"by Friday","gloss":"g"}'
    panel = build_panel(base_receiver, n=2)
    # receiver 0: sample0 ok, sample1 bad->ok ; receiver 1: sample0 bad->bad (skipped), sample1 ok
    client = fake_client_cls([ok, "bad", ok, "bad", "bad", ok])
    out = read_panel(client, MSG, panel, k=2, model="m", temperature=0.7)
    assert len(client.calls) == 6
    assert [(r.receiver_id, r.sample) for r in out] == [("r-0", 0), ("r-0", 1), ("r-1", 1)]
    assert client.calls[0]["model"] == "m" and client.calls[0]["temperature"] == 0.7
