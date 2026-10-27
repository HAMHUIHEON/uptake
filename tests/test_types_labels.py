import pytest
from forecaster.labels import SPEECH_ACTS, NEXT_ACTIONS, TONE_MIN, TONE_MAX
from forecaster.types import Intent


def test_label_sets():
    assert "probe" in SPEECH_ACTS and "disclosure" in SPEECH_ACTS and len(SPEECH_ACTS) == 14
    assert "ask_clarification" in NEXT_ACTIONS and len(NEXT_ACTIONS) == 10
    assert (TONE_MIN, TONE_MAX) == (-2, 2)


@pytest.mark.parametrize("kw", [
    {"speech_act": "nonsense", "next_action": "comply", "tone": 0},
    {"speech_act": "request", "next_action": "nonsense", "tone": 0},
    {"speech_act": "request", "next_action": "comply", "tone": 3},
])
def test_intent_validation(kw):
    with pytest.raises(ValueError):
        Intent(**kw)
