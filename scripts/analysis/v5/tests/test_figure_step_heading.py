"""Step + heading: the frame, the fold, the bootstrap, the shared window."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import figure_step_heading as F
from scripts.analysis.v5.modules.cell_margin import destination


def _answer(bearing_off_deg, step):
    """An answer `step` gaps from X_in=(0,0), `bearing_off_deg` off the TG direction (TG due east, 500 km)."""
    lat, lon = destination(0.0, 0.0, 90.0 - bearing_off_deg, 500.0 * step)
    return float(lat), float(lon)


TG = destination(0.0, 0.0, 90.0, 500.0)


@pytest.mark.parametrize("off, step", [(0.0, 1.0), (30.0, 0.5), (-30.0, 0.5), (120.0, 1.2), (0.0, 0.0)])
def test_step_and_unsigned_heading(off, step):
    p = _answer(off, step)
    g = F.step_heading(np.array([0.0]), np.array([0.0]), np.array([TG[0]]), np.array([TG[1]]),
                       np.array([p[0]]), np.array([p[1]])).iloc[0]
    assert g.step == pytest.approx(step, abs=1e-6)
    assert g.heading == pytest.approx(abs(off) if step > 0 else 0.0, abs=1e-4)


def test_answer_at_tg_closes_the_gap_and_at_xin_closes_none():
    at_tg = F.step_heading(*(np.array([v]) for v in (0.0, 0.0, TG[0], TG[1], TG[0], TG[1]))).iloc[0]
    at_x = F.step_heading(*(np.array([v]) for v in (0.0, 0.0, TG[0], TG[1], 0.0, 0.0))).iloc[0]
    assert at_tg.gap_closed == pytest.approx(1.0) and at_x.gap_closed == pytest.approx(0.0)


def test_mirror_images_fold_onto_one_point():
    a, b = _answer(40.0, 0.8), _answer(-40.0, 0.8)
    g = F.step_heading(np.zeros(2), np.zeros(2), np.full(2, TG[0]), np.full(2, TG[1]),
                       np.array([a[0], b[0]]), np.array([a[1], b[1]]))
    x, y = F.frame_xy(g.step, g.heading)
    assert np.allclose(x[0], x[1]) and np.allclose(y[0], y[1]) and y[0] > 0


def test_site_bootstrap_is_deterministic_and_brackets_the_median():
    v = np.r_[np.full(20, 1.0), np.full(20, 2.0), np.full(20, 3.0)]
    s = np.repeat(["a", "b", "c"], 20)
    m, lo, hi = F.site_bootstrap_median(v, s, n=300)
    assert m == 2.0 and lo <= m <= hi
    assert F.site_bootstrap_median(v, s, n=300) == [m, lo, hi]


def test_view_holds_the_half_circle_and_rounds_to_the_grid():
    df = pd.DataFrame({"step": [0.1, 0.2], "heading": [10.0, 20.0]})
    e = F.view_extent(df)
    assert e == {"x_min": -1.0, "x_max": 1.0, "y_max": 1.0}
    big = pd.DataFrame({"step": np.linspace(0, 1.6, 200), "heading": np.zeros(200)})
    assert F.view_extent(big)["x_max"] == pytest.approx(1.75)


def test_shared_extent_is_the_union_of_each_methods_window():
    df = pd.DataFrame({"term": ["A"] * 100 + ["B"] * 100,
                       "step": np.r_[np.full(100, 0.5), np.linspace(0, 2.0, 100)], "heading": 0.0})
    e = F.shared_extent(df, coverage=0.95)
    assert e["x_max"] == F.view_extent(df[df.term == "B"], coverage=0.95)["x_max"] >= 1.75
