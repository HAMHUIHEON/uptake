"""Contract for task41 viewer v2 regressions (synthetic data only)."""
import re

import pytest

from forecaster import viewer2 as V2


def _ease_out_back(t):
    t = max(0.0, min(1.0, float(t)))
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2


def _mark_scale(age):
    if not (age >= 0) or age >= 350:
        return 1.0
    return _ease_out_back(age / 350.0)


def _fn_body(name):
    mm = re.search(r"function %s\(.*?\)\{(.*?)\n\}" % name, V2._JS, re.S)
    assert mm, name + " missing from page JS"
    return mm.group(1)


# --- mark glyphs ------------------------------------------------------

def test_easing_math_mirror():
    assert _ease_out_back(0.0) == pytest.approx(0.0)
    assert _ease_out_back(1.0) == pytest.approx(1.0)
    assert "1.70158" in V2._JS  # same constant in the page
    # pop-in starts at 0 and settles at identity by 350ms, stays there
    assert _mark_scale(0.0) == pytest.approx(0.0)
    assert _mark_scale(350) == 1.0
    assert _mark_scale(1000) == 1.0
    assert _mark_scale(-5) == 1.0
    assert _mark_scale(float("nan")) == 1.0


def test_mark_glyph_element_halo_and_steady_state():
    html = V2.render_html({})
    assert "mark-g" in html  # glyph element ships in the page JS
    assert 'svgEl("text")' in V2._JS
    m = re.search(r"\.mark-g\{([^}]*)\}", V2._CSS)
    assert m, ".mark-g rule missing"
    body = m.group(1)
    assert "font-size:14px" in body
    assert "font-weight:700" in body
    assert "paint-order:stroke" in body
    assert "stroke:var(--bg)" in body
    assert "stroke-width:3px" in body
    # steady state is identity: pure helper, no transform once age>=350
    assert "function markScale(age)" in V2._JS
    assert "age>=350" in V2._JS
    assert "markScale(ma)" in V2._JS
    assert "ma<350" in V2._JS
    # pop-in drives itself to completion instead of freezing at scale 0
    assert "requestAnimationFrame" in V2._JS
    # REDUCED motion shows the glyph immediately (no transform branch)
    add_mark = _fn_body("addMark")
    assert "!REDUCED" in add_mark
    assert "markBorn" in add_mark


def test_mark_glyphs_painted_above_bands():
    tide = _fn_body("renderTide")
    assert tide.index("addMark(") > tide.index("addXLabels(svg)")
    assert tide.index("addMark(") > tide.index("tide-line")
    bands = _fn_body("renderBands")
    assert bands.index("addMark(") > bands.index("addXLabels(svg)")
    assert bands.index("addMark(") > bands.index('"band"')


# --- square text ------------------------------------------------------

def test_square_explicit_size_and_wrap():
    m = re.search(r"SW=(\d+),SH=(\d+)", V2._JS)
    assert m, "square viewBox size missing from page JS"
    sw, sh = m.group(1), m.group(2)
    html = V2.render_html({})
    tag = re.search(r'<svg id="square"[^>]*>', html)
    assert tag, "square svg missing"
    assert ('width="%s"' % sw) in tag.group(0)
    assert ('height="%s"' % sh) in tag.group(0)
    assert 'setAttribute("width"' in V2._JS
    assert 'setAttribute("height"' in V2._JS
    assert "sq-wrap" in html
    assert ".sq-wrap" in V2._CSS
    assert "#tide,#square" not in V2._CSS  # no shared 100% scaling rule
    assert re.search(r"\.sq-axis\{[^}]*font-size:13px", V2._CSS)
    assert '"sq-axis"' in V2._JS
