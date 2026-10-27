import json
import pytest
from types import SimpleNamespace
from forecaster.usage import Meter, BudgetExceeded, PRICES
from forecaster.reader import MuseClient
from forecaster.batch import model_for, run_batch, genre_report, cohen_kappa, spearman


def test_meter_and_budget():
    m = Meter(budget_usd=0.01)
    m.add("muse-spark-1.3-contributor", 1_000_000, 1_000_000)
    assert m.cost() == pytest.approx(0.30)
    assert m.by_model()["muse-spark-1.3-contributor"]["calls"] == 1
    with pytest.raises(BudgetExceeded):
        m.check()
    assert PRICES["muse-spark-1.3"] == (1.25, 4.25)


class Busy(Exception):
    status_code = 429


def _resp(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50))


def test_client_retry_and_metering(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "fake")
    calls, sleeps = [], []

    def create(**kw):
        calls.append(kw)
        if len(calls) < 3:
            raise Busy()
        return _resp("ok")

    meter = Meter()
    c = MuseClient(meter=meter, default_effort="low", create=create, sleep=sleeps.append)
    assert c.complete([{"role": "user", "content": "x"}], model="muse-spark-1.3") == "ok"
    assert sleeps == [1, 2] and calls[-1]["extra_body"] == {"reasoning_effort": "low"}
    assert meter.by_model()["muse-spark-1.3"]["prompt_tokens"] == 100


def test_model_for():
    assert model_for({"public": True}) == "muse-spark-1.3-contributor"
    assert model_for({"public": False}) == "muse-spark-1.3"


class Fake:
    def complete(self, messages, model, temperature, reasoning_effort=None):
        text = json.dumps(messages)
        if "relationship_shift" in text:
            return '{"next_action":"proceed","relationship_shift":0,"face_threat":0,"understood":"ok"}'
        return '{"moves":["smalltalk"],"addressed_to":"all","quote":""}'


def test_run_batch_resume_and_report(tmp_path):
    casino = [{"dialogue_id": 1, "chat_logs": [{"id": "mturk_agent_1", "text": "hello"},
                                               {"id": "mturk_agent_2", "text": "hi there"}],
               "annotations": [["hello", "small-talk"], ["hi there", "small-talk"]]}]
    cp = tmp_path / "casino.json"
    cp.write_text(json.dumps(casino))
    tp = tmp_path / "t.jsonl"
    tp.write_text('{"speaker":"A","text":"yo","chunk":0}\n{"speaker":"B","text":"sup","chunk":0}\n')
    items = [{"id": "c1", "genre": "negotiation", "kind": "casino", "dialogue_id": 1, "public": True},
             {"id": "y1", "genre": "youtube", "kind": "turns", "path": str(tp), "public": False}]
    out = tmp_path / "out"
    s1 = run_batch(items, Fake(), out, casino_path=cp, workers=2, n=1, k=1, inner_workers=1)
    assert sorted(x["id"] for x in s1) == ["c1", "y1"] and (out / "c1.json").exists()
    assert json.loads((out / "y1.json").read_text())["model"] == "muse-spark-1.3"
    s2 = run_batch(items, None, out, casino_path=cp, workers=2, n=1, k=1, inner_workers=1)  # resume: no client needed
    assert sorted(x["id"] for x in s2) == ["c1", "y1"]
    rep = genre_report(s2, b=100)
    assert rep["negotiation"]["n_conv"] == 1 and "micro_f1" in rep["negotiation"]
    assert rep["youtube"]["other_rate"][0] == 0.0


def test_kappa_spearman():
    assert cohen_kappa(["x", "x", "y", "y"], ["x", "y", "y", "y"]) == pytest.approx(0.5)
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0
    assert spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
