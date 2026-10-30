"""Contract for SPEC_task45_standing_logistic (synthetic data only)."""
import json

import pytest

from forecaster import commentary as CM
from forecaster import viewer2 as V2


def _eff(idx, speaker, value, text="", n_readings=3):
    return {"index": idx, "speaker": speaker, "text": text or f"t{idx}",
            "value": value, "disagreement": 0.0,
            "n_readings": n_readings, "listeners": ["X"]}


def _item(cid, effects, moves, scale="pairwise", genre="negotiation"):
    return {"id": cid, "genre": genre, "scale": scale,
            "moves": moves, "effects": effects, "hits": []}


def _pw_norm(mu=0.0, sd=0.6):
    return {"mu": mu, "sd": sd, "scale": "pairwise",
            "table": {"propose": 0.0, "counter": 0.0}, "sd_resid": 0.5}


# --- sigmoid mapping ------------------------------------------------------

def test_logistic_mapping_values():
    assert CM.logistic_standing(0) == 50
    assert CM.logistic_standing(1.0) == 73
    assert CM.logistic_standing(-1.0) == 27
    assert CM.logistic_standing(2.0) == 88
    assert CM.logistic_standing(-2.0) == 12
    assert CM.standing_from_z(0) == 50
    assert CM.sigmoid(0) == pytest.approx(0.5)


def test_pairwise_standing_uses_logistic():
    assert CM.pairwise_standing(0.0, 0.0, 0.6) == pytest.approx(50.0)
    assert CM.pairwise_standing(0.6, 0.0, 0.6) == pytest.approx(73.0)
    assert CM.pairwise_standing(-0.6, 0.0, 0.6) == pytest.approx(27.0)
    assert CM.pairwise_standing(1.2, 0.0, 0.6) == pytest.approx(88.0)
    assert CM.pairwise_standing(-1.2, 0.0, 0.6) == pytest.approx(12.0)
    assert CM.pairwise_standing(0.6, 0.0, 0.0) == pytest.approx(50.0)
    assert CM.STANDING_SCALE_PAIRWISE == "pairwise→0-100 via logistic(z)"


# --- no pinning for ordinary turns -----------------------------------------

def test_commentary_standings_not_pinned_for_small_z():
    # z = ±2.5 would clip to 0/100 under 50+25z; logistic gives ~92/8.
    item = _item("pw-log",
                 [_eff(0, "A", 1.5, text="a"),
                  _eff(1, "B", -1.5, text="b"),
                  _eff(2, "A", 0.2, text="c")],
                 [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
                  {"turn_index": 1, "speaker": "B", "moves": ["counter"]},
                  {"turn_index": 2, "speaker": "A", "moves": ["propose"]}])
    norm = _pw_norm()
    pairs = CM.standing_pairs(item, norm)
    assert len(pairs) == 3
    for before, after in pairs:
        assert 0.0 <= before <= 100.0
        assert 0.0 <= after <= 100.0
        assert after not in (0.0, 100.0)
    assert pairs[0][1] == pytest.approx(92.0)
    assert pairs[1][1] == pytest.approx(8.0)
    com = CM.comment(item, norm, None, k=3)
    assert com["standing_scale"] == "pairwise→0-100 via logistic(z)"
    for t in com["turns"]:
        assert 0.0 < t["standing_after"] < 100.0
        assert "잔고" in t["line_ko"]
    # line numbers match the JSON standings via the shared rounder
    by_index = {t["index"]: t for t in com["turns"]}
    for t in by_index.values():
        rb = CM.standing_round(t["standing_before"])
        ra = CM.standing_round(t["standing_after"])
        assert f"잔고 {rb} → {ra}" in t["line_ko"]


# --- viewer uses the same numbers ------------------------------------------

def test_viewer2_matches_commentary_numbers():
    a = _eff(0, "A", 0.6, text="a good move")
    b = _eff(1, "B", -0.6, text="b replies")
    item = _item("casino-7", [a, b],
                 [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
                  {"turn_index": 1, "speaker": "B", "moves": ["counter"]}])
    norm = _pw_norm()
    com = CM.comment(item, norm, None, k=3)
    view = V2.build_view2(item, com)
    # logistic side labels, not the old 75/25
    assert view["standings"] == [
        {"A": pytest.approx(73.0), "B": pytest.approx(50.0)},
        {"A": pytest.approx(73.0), "B": pytest.approx(27.0)},
    ]
    assert view["turns"][0]["standing"] == pytest.approx(73.0)
    assert view["turns"][1]["standing"] == pytest.approx(27.0)
    # commentary JSON standings agree with the viewer trail
    pairs = CM.standing_pairs(item, norm)
    assert view["turns"][0]["standing"] == pytest.approx(pairs[0][1])
    assert view["turns"][1]["standing"] == pytest.approx(pairs[1][1])
    # settlement sentence uses the same rounded numbers
    assert view["settlement"] == "A 잔고 50 → 73 · B 잔고 50 → 27"
    assert view["square"]["xs"] == pytest.approx([73.0, 73.0])
    assert view["square"]["ys"] == pytest.approx([50.0, 27.0])
    # badge text unchanged
    html = V2.render_html(view)
    assert "읽기 척도: 쌍대 비교(BT 평점)" in html


def test_settlement_and_hover_share_helper():
    assert V2.standing_round is CM.standing_round
    assert V2._r100(62.5) == CM.standing_round(62.5) == 63
    assert V2._r100(62.4) == 62


# --- per-move mu/sd ----------------------------------------------------------

def test_pairwise_per_move_stats():
    item = _item("pw-move",
                 [_eff(0, "A", 0.5, text="a"),
                  _eff(1, "B", 0.5, text="b")],
                 [{"turn_index": 0, "speaker": "A", "moves": ["propose"]},
                  {"turn_index": 1, "speaker": "B", "moves": ["counter"]}])
    norm = {"mu": 0.0, "sd": 1.0, "scale": "pairwise",
            "table": {"propose": 0.0, "counter": 0.0}, "sd_resid": 0.5,
            "move_stats": {"propose": {"mean": 0.5, "sd": 0.5, "n": 4}}}
    pairs = CM.standing_pairs(item, norm)
    # propose turn: z = (0.5-0.5)/0.5 = 0 → 50; counter falls back to global.
    assert pairs[0][1] == pytest.approx(50.0)
    assert pairs[1][1] == pytest.approx(62.0)
    mu, sd = CM.pairwise_mu_sd_for_moves(["propose"], norm, 0.0, 1.0)
    assert mu == pytest.approx(0.5)
    assert sd == pytest.approx(0.5)
    mu2, sd2 = CM.pairwise_mu_sd_for_moves(["counter"], norm, 0.0, 1.0)
    assert (mu2, sd2) == pytest.approx((0.0, 1.0))
    # viewer agrees on the per-move turn
    com = CM.comment(item, norm, None, k=3)
    view = V2.build_view2(item, com)
    assert view["turns"][0]["standing"] == pytest.approx(50.0)
    assert view["turns"][1]["standing"] == pytest.approx(62.0)


def test_load_fit_table_keeps_move_stats(tmp_path):
    fit = tmp_path / "fit"
    fit.mkdir()
    (fit / "table.json").write_text(json.dumps({
        "scale": "pairwise", "mu": 0.0, "sd": 1.0, "sd_resid": 0.5,
        "table": {"propose": 0.1},
        "move_stats": {"propose": {"mean": 0.1, "sd": 0.4, "n": 3}},
    }), encoding="utf-8")
    norm = CM._load_fit_table(fit)
    assert norm["move_stats"]["propose"]["sd"] == pytest.approx(0.4)


# --- 0–100 records unchanged -------------------------------------------------

def test_zero_to_100_records_unchanged():
    e = _eff(0, "A", 2.0, text="a")
    e["relationship"] = [1.0, 1.0, 1.0]
    item = _item("casino-7", [e],
                 [{"turn_index": 0, "speaker": "A", "moves": ["propose"]}],
                 scale="fine100")
    norm = {"mu": 0.0, "sd": 1.0, "scale": "fine100",
            "table": {"propose": 0.0}, "sd_resid": 1.0}
    assert CM.standing_pairs(item, norm) == [(50.0, 75.0)]
    com = CM.comment(item, norm, None, k=3)
    assert "standing_scale" not in com
    view = V2.build_view2(item, com)
    assert view["standings"] == [{"A": pytest.approx(75.0)}]
    assert view["turns"][0]["standing"] == pytest.approx(75.0)
    assert "rating" not in view["turns"][0]
    html = V2.render_html(view)
    assert "panel 0–100" in html
