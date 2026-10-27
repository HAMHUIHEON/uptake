from forecaster.forecast import run_forecast

MSG = "Can you get the report to me by Friday? No pressure."


def test_run_forecast_end_to_end(intent, base_receiver, fake_client_cls):
    hit = '{"speech_act":"request","tone":0,"next_action":"comply","trigger":"by Friday","gloss":"a friendly ask"}'
    miss = '{"speech_act":"command","tone":-1,"next_action":"comply","trigger":"No pressure","gloss":"it is actually an order"}'
    client = fake_client_cls([hit, miss, miss, miss, hit, hit])
    f = run_forecast(client, MSG, intent, base_receiver, n=3, k=2, seed=0, b=500)
    assert len(f.readings) == 6 and f.miss_rate == 0.5
    assert f.ci_low <= 0.5 <= f.ci_high
    assert f.per_receiver == {"r-0": 0.5, "r-1": 1.0, "r-2": 0.0}
    assert f.top_triggers[0] == ("No pressure", 3)
