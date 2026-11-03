"""spotter_cbg end to end on targets with one or two VPs.

The MTL unit tests pin what GaussianDensityMTL returns; this pins what the
benchmark records. The property that matters downstream is `status`: v5's
`classify` counts FALLBACK as "no answer", so a single-VP target must come out
as SUCCESS through the pipeline, not as the identical coordinate via fallback.
"""

from __future__ import annotations

import unittest

from scripts.framework.v2 import CBGModel, GeoStatus
from scripts.framework.v2.ctr.density_point import DensityArgmaxCTR
from scripts.framework.v2.ltd.normal_dist import NormalDistLTD
from scripts.framework.v2.ltd.tests.helpers import make_fitted_spotter_model
from scripts.framework.v2.mtl.gaussian_density import GaussianDensityMTL
from scripts.framework.v2.types import Coord, Latency, VpId

SEATTLE = Coord(47.61, -122.33)
CHICAGO = Coord(41.88, -87.63)


def spotter_cbg() -> CBGModel:
    """`spotter_cbg` with a hand-fitted LTD: mu = 50 km/ms, sigma = 50 km."""
    ltd = NormalDistLTD()
    ltd._model = make_fitted_spotter_model()
    return CBGModel(ltd, GaussianDensityMTL(grid="healpix"), DensityArgmaxCTR())


class TestSpotterSingleVp(unittest.TestCase):
    def test_one_vp_is_a_success_at_the_vp(self) -> None:
        result = spotter_cbg().geolocate([(VpId("vp"), SEATTLE, Latency(20.0))])
        self.assertEqual(result.status, GeoStatus.SUCCESS)
        self.assertEqual(result.coord, SEATTLE)
        self.assertIsNone(result.error)

    def test_one_vp_does_not_need_the_fallback(self) -> None:
        """Same answer with fallback off: the MTL answers, not CBGModel."""
        ltd = NormalDistLTD()
        ltd._model = make_fitted_spotter_model()
        model = CBGModel(
            ltd, GaussianDensityMTL(grid="healpix"), DensityArgmaxCTR(),
            enable_fallback=False,
        )
        result = model.geolocate([(VpId("vp"), SEATTLE, Latency(20.0))])
        self.assertEqual(result.status, GeoStatus.SUCCESS)
        self.assertEqual(result.coord, SEATTLE)

    def test_two_vps_are_a_success_through_the_surface(self) -> None:
        obs = [
            (VpId("a"), SEATTLE, Latency(20.0)),
            (VpId("b"), CHICAGO, Latency(25.0)),
        ]
        result = spotter_cbg().geolocate(obs)
        self.assertEqual(result.status, GeoStatus.SUCCESS)
        self.assertEqual(result.mtl_result.density.grid, "healpix")
        self.assertNotIn(result.coord, (SEATTLE, CHICAGO))


if __name__ == "__main__":
    unittest.main()
