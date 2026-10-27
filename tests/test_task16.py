import json

from forecaster.analyze import build_report


def _negotiation(other_rate):
    return {
        "n_conv": 2,
        "other_rate": other_rate,
        "failed_rate": 0.0,
        "micro_f1": {"micro_f1": 0.8, "micro_f1_ci": [0.6, 0.9]},
        "skill": {"skill_mean": 0.1, "skill_ci": [0.05, 0.15], "n": 5},
    }


def test_build_report_skips_non_genre_keys(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({
        "genre_report": {"negotiation": _negotiation([0.1, 0.0, 0.2]),
                         "n_errors": 0},
    }))
    text = build_report(tmp_path)
    assert "Q2/negotiation" in text
    assert "Q2/n_errors" not in text


def test_other_rate_plain_number(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({
        "genre_report": {"negotiation": _negotiation(0.05),
                         "n_errors": 0},
    }))
    text = build_report(tmp_path)
    assert "Q2/negotiation" in text
    assert "Q2/n_errors" not in text
