import pytest
from forecaster.types import Intent, Receiver


@pytest.fixture
def intent():
    return Intent(speech_act="request", next_action="comply", tone=0)


@pytest.fixture
def base_receiver():
    return Receiver(id="r", relationship="team lead to new hire", power="sender_higher",
                    channel="slack", state="new hire, worried about making a bad first impression")


class FakeClient:
    """Returns queued raw strings in order; records every call."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def complete(self, messages, model, temperature):
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        return self.outputs.pop(0) if self.outputs else "{}"


@pytest.fixture
def fake_client_cls():
    return FakeClient
