"""U2: README matches the product frame.

The product sentence leads (turn-by-turn wanted-vs-happened reading on
four stakes), with the Avianca case, measured reliability,
limits, and Meta models. Prediction and advance claims stay out.
"""
import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"

PRODUCT_START = "Reads important conversations turn by turn"


def _text():
    return README.read_text(encoding="utf-8")


def test_readme_front_is_product_sentence():
    lines = _text().splitlines()
    assert lines[0].strip() == "# Uptake"
    first_para = lines[2].strip()
    assert first_para.startswith(PRODUCT_START), first_para
    front = _text()[:2000].lower()
    for stake in ("outcome", "relationship", "face", "understanding"):
        assert stake in front, stake
    assert "wanted" in front
    assert "happened" in front
    assert not re.search(r"\bgame\b", front)
    for banned in ("broadcast", "chess", "go/chess", "!!", "?!"):
        assert banned not in front, banned


def test_readme_has_no_forecast_or_advance_claims():
    import re
    text = " ".join(_text().lower().split())
    # After-the-reply readings are the method; claims of knowing the
    # future or reading before the reply are out.
    text = text.replace("after the reply is seen", "")
    for banned in ("forecast", "predict", "in advance", "before the reply",
                   "before_reply", "next-turn", "could have been known"):
        assert banned not in text, banned
    assert re.search(r"no claim about what will happen next", text)


def test_readme_game_only_as_denial():
    text = _text().lower()
    assert "game" not in text


def test_readme_names_avianca_case():
    assert "avianca" in _text().lower()


def test_readme_reports_reliability():
    text = _text().lower()
    assert "reliab" in text
    assert "0.863" in text


def test_readme_states_limits_and_meta_models():
    text = _text().lower()
    assert "limit" in text
    assert "muse-spark-1.3-contributor" in text
    assert "reads the utterance, not the person" in text
