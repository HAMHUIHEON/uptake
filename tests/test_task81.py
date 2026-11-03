"""Task81: end read shown on the -1 to +1 scale used in section 8."""
from forecaster import report as R


def test_end_read_cell_signed_scale():
    assert R._end_read_cell({"mean": 50, "scores": [50, 50, 50]}) == (
        "+0.00 (reads +0.00, +0.00, +0.00)")
    assert R._end_read_cell({"mean": 25, "scores": [25, 25, 25]}) == (
        "-0.50 (reads -0.50, -0.50, -0.50)")
