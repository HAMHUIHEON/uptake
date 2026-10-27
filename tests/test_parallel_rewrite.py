import json
import threading
import time

from forecaster.panel import build_panel
from forecaster.reader import read_panel
from forecaster.runlog import save_readings, load_readings
from forecaster.rewrite import run_rewrite
from forecaster.types import Reading

HIT = '{"speech_act":"request","tone":0,"next_action":"comply","trigger":"","gloss":"fine"}'
MISS = '{"speech_act":"command","tone":-2,"next_action":"comply","trigger":"","gloss":"pressure"}'


class RoutingClient:
    """Thread-safe fake: answer depends on message text; tracks concurrency."""

    def __init__(self, bad_text, delay=0.02):
        self.bad_text, self.delay = bad_text, delay
        self.lock = threading.Lock()
        self.active = self.max_active = self.calls = 0

    def complete(self, messages, model, temperature):
        with self.lock:
            self.active += 1
            self.calls += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay)
        with self.lock:
            self.active -= 1
        text = json.dumps(messages, ensure_ascii=False)
        return MISS if self.bad_text in text else HIT


def test_parallel_read_panel_keeps_order_and_limits_concurrency(base_receiver):
    panel = build_panel(base_receiver, n=4)
    client = RoutingClient(bad_text="NEVER-MATCHES")
    out = read_panel(client, "hello", panel, k=3, model="m", max_workers=3)
    assert client.calls == 12
    assert 1 < client.max_active <= 3
    assert [(r.receiver_id, r.sample) for r in out] == [(f"r-{i}", s) for i in range(4) for s in range(3)]


def test_runlog_roundtrip(tmp_path):
    rs = [Reading("r-0", 0, "request", 0, "comply", "by Friday", "g1"),
          Reading("r-1", 2, "command", -2, "refuse", "", "g2")]
    p = tmp_path / "run.jsonl"
    save_readings(p, rs, meta={"message": "m", "model": "x"})
    lines = p.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["meta"] == {"message": "m", "model": "x"}
    assert len(lines) == 3
    meta, back = load_readings(p)
    assert meta == {"message": "m", "model": "x"} and back == rs


def test_rewrite_paired_comparison(intent, base_receiver):
    original = "Can you get the report to me by Friday? No pressure."
    rewrite = "Could you aim for Friday? If that's tight, tell me and we'll adjust."
    client = RoutingClient(bad_text="No pressure")
    res = run_rewrite(client, original, rewrite, intent, base_receiver, n=3, k=2, seed=0, b=500, max_workers=2)
    assert res["before"].miss_rate == 1.0 and res["after"].miss_rate == 0.0
    assert [r.id for r in res["panel"]] == ["r-0", "r-1", "r-2"]
    point, low, high = res["diff"]
    assert point == -1.0 and low <= point <= high and high <= 0.0
