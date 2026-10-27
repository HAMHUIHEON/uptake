from forecaster.types import Reading
from forecaster.score import is_miss, miss_rate, per_receiver_rates, top_triggers


def R(rid, act="request", tone=0, nxt="comply", trig=""):
    return Reading(receiver_id=rid, sample=0, speech_act=act, tone=tone, next_action=nxt, trigger=trig, gloss="")


def test_is_miss_rules(intent):
    assert not is_miss(R("a"), intent)
    assert is_miss(R("a", act="command"), intent)
    assert is_miss(R("a", nxt="refuse"), intent)
    assert not is_miss(R("a", tone=-1), intent)
    assert is_miss(R("a", tone=-2), intent)


def test_rates_and_triggers(intent):
    rs = [R("a"), R("a", act="command", trig="by Friday"), R("b", tone=2, trig="No pressure"),
          R("b", act="command", trig="by Friday"), R("b", nxt="escalate", trig="No pressure")]
    assert miss_rate(rs, intent) == 4 / 5 and miss_rate([], intent) == 0.0
    assert per_receiver_rates(rs, intent) == {"a": 0.5, "b": 1.0}
    assert top_triggers(rs, intent, n=3) == [("No pressure", 2), ("by Friday", 2)]
