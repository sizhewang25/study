"""The fitting LTDs learn RTT against `FitSample.distance_km` when it is set.

A source may bake a non-great-circle distance into each sample (e.g. the
routing distance through the target's nearest interconnect). Each test fits
one LTD twice: once on the helper samples, whose probes sit at the intended
great-circle distance, and once on copies whose probe is moved onto the VP
(great-circle 0) but which carry that same distance in `distance_km`. If the
LTD reads `distance_km`, the two fits predict identically.
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from scripts.framework.v2.ltd.base import FitSample, sample_distance_km
from scripts.framework.v2.ltd.bounded_spline import BoundedSplineLTD
from scripts.framework.v2.ltd.low_envelope import LowEnvelopeLTD
from scripts.framework.v2.ltd.normal_dist import NormalDistLTD
from scripts.framework.v2.ltd.tests.helpers import (
    ANCHOR_COORDS,
    make_bounded_spline_fit_samples,
    make_low_envelope_fit_samples,
    make_normal_dist_fit_samples,
)
from scripts.framework.v2.types import Latency, VpId

_VP = VpId("anchor-a")


def _collapsed(samples: list[FitSample]) -> list[FitSample]:
    """Probe on the VP, the great-circle distance carried in `distance_km`."""
    return [
        replace(s, probe_coord=s.vp_coord, distance_km=sample_distance_km(s))
        for s in samples
    ]


class TestSampleDistance(unittest.TestCase):
    def test_falls_back_to_great_circle(self):
        s = make_low_envelope_fit_samples(distances_km=[500.0])[0]
        self.assertIsNone(s.distance_km)
        self.assertAlmostEqual(sample_distance_km(s), 500.0, places=5)

    def test_prefers_baked_distance(self):
        s = replace(make_low_envelope_fit_samples(distances_km=[500.0])[0], distance_km=1234.5)
        self.assertEqual(sample_distance_km(s), 1234.5)


class TestFitReadsBakedDistance(unittest.TestCase):
    def _assert_same_prediction(self, make_ltd, samples, rtt: float):
        a, b = make_ltd(), make_ltd()
        self.assertTrue(a.fit(samples).success)
        self.assertTrue(b.fit(_collapsed(samples)).success)
        pa = a.predict(_VP, ANCHOR_COORDS[_VP], Latency(rtt))
        pb = b.predict(_VP, ANCHOR_COORDS[_VP], Latency(rtt))
        self.assertTrue(pa.success and pb.success)
        self.assertAlmostEqual(pa.tg_distance.lower_km, pb.tg_distance.lower_km, places=6)
        self.assertAlmostEqual(pa.tg_distance.upper_km, pb.tg_distance.upper_km, places=6)

    def test_low_envelope(self):
        self._assert_same_prediction(LowEnvelopeLTD, make_low_envelope_fit_samples(), 25.0)

    def test_bounded_spline(self):
        self._assert_same_prediction(
            lambda: BoundedSplineLTD(
                target_coverage=0.8, cutoff_min_points=1, spline_n_knots=4, bin_size_ms=1000,
            ),
            make_bounded_spline_fit_samples(),
            20.0,
        )

    def test_normal_dist(self):
        self._assert_same_prediction(
            lambda: NormalDistLTD(deg_mu=3, deg_sigma=0, cutoff_min_points=1),
            make_normal_dist_fit_samples(n_per_rtt=4, spread_km=100.0),
            20.0,
        )


if __name__ == "__main__":
    unittest.main()
