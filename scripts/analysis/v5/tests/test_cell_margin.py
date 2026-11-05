"""Ray-cast cell margins: spherical primitives, the bisection, the start-outside flag, convergence."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import cell_margin as M
from scripts.analysis.v5.modules.geodesy import haversine_km
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM

# Two seeds on the equator, 1,000 km apart (about 8.993 degrees).
SEP_DEG = np.degrees(1000.0 / EARTH_RADIUS_KM)
SEEDS = pd.DataFrame({"seed_id": [0, 1], "seed_lat": [0.0, 0.0], "seed_lon": [0.0, SEP_DEG]})


def test_destination_and_bearing_round_trip():
    lat, lon = M.destination(40.0, -100.0, 63.0, 750.0)
    assert haversine_km(40.0, -100.0, lat, lon) == pytest.approx(750.0, rel=1e-6)
    assert M.initial_bearing(40.0, -100.0, lat, lon) == pytest.approx(63.0, abs=1e-6)


def test_margin_toward_the_other_seed_is_half_the_separation():
    m = M.margins(0.0, 0.0, 90.0, 0, SEEDS, 3000.0)
    assert m[0] == pytest.approx(500.0, abs=3000.0 * M.TOL)


def test_margin_away_from_the_other_seed_is_open():
    assert M.margins(0.0, 0.0, 270.0, 0, SEEDS, 3000.0)[0] == 3000.0


def test_oblique_ray_meets_the_bisector_farther_than_half():
    # At 45 degrees the perpendicular bisector is reached at ~ 500 / cos 45 (planar intuition).
    m = M.margins(0.0, 0.0, 45.0, 0, SEEDS, 3000.0)[0]
    assert 690.0 < m < 730.0


def test_start_outside_is_zero_in_every_direction():
    m = M.margins(0.0, SEP_DEG * 0.9, np.arange(0, 360, 15.0), 0, SEEDS, 3000.0)
    assert (m == 0.0).all()


def test_margin_is_stable_to_the_tolerance():
    a = M.margins(0.0, 0.0, 90.0, 0, SEEDS, 3000.0, tol=1e-4)[0]
    b = M.margins(0.0, 0.0, 90.0, 0, SEEDS, 3000.0, tol=1e-6)[0]
    assert abs(a - b) <= 3000.0 * 1e-4


def test_correct_iff_error_below_margin():
    # A point 400 km east of seed 0's origin is in cell 0, 600 km east is in cell 1.
    for e, own in ((400.0, 0), (600.0, 1)):
        lat, lon = M.destination(0.0, 0.0, 90.0, e)
        assert M.nearest_seed(lat, lon, SEEDS)[0] == own
    assert M.margins(0.0, 0.0, 90.0, 0, SEEDS, 3000.0)[0] == pytest.approx(500.0, abs=0.5)


def test_mbar_of_a_half_plane_cell():
    # With only these two seeds, cell 0 is a hemisphere-like half: every ray with a westward
    # component is open, eastward rays exit at 500 / cos(theta - 90).
    b = np.arange(0, 360, 15.0)
    m = M.margins(np.zeros_like(b), np.zeros_like(b), b, np.zeros(len(b), int), SEEDS, 3000.0)
    assert (m[(b > 180) & (b < 360)] == 3000.0).all()
    assert m.min() == pytest.approx(500.0, abs=0.5)
