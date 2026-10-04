"""The bipartite maps and the great-circle primitive they draw with. Ported from v3.

Rendering tests are thin, as in `test_map_answer_space.py`: cartopy is slow and
pixel assertions are brittle. `great_circle_segments` gets real assertions,
because it is geometry and the way it goes wrong is silent: a straight lon/lat
chord renders as a plausible line that is simply in the wrong place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import map_bipartite as MB
from scripts.analysis.v5.modules.bipartite import build_bipartite
from scripts.analysis.v5.modules.geodesy import elementwise_km
from scripts.analysis.v5.modules.mapping import great_circle_segments

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
DFW = (32.8968, -97.0380)
SEA = (47.4502, -122.3088)


def _fixture(scored=None):
    tg = [CHI, SJC, NYC, DFW]
    vp = [SEA, DFW, NYC]
    space = A.build_answer_space(
        pd.DataFrame(
            {
                "tg_id": [f"tg-{i}" for i in range(len(tg))],
                "tg_lat": [c[0] for c in tg],
                "tg_lon": [c[1] for c in tg],
            }
        ),
        run_id="syn",
    )
    if scored is not None:
        space.sites[A.SCORED_COL] = scored
    vps = pd.DataFrame(
        {
            "vp_id": [f"vp-{i}" for i in range(len(vp))],
            "vp_lat": [c[0] for c in vp],
            "vp_lon": [c[1] for c in vp],
        }
    )
    t = space.tgs.set_index("tg_id")
    rows = [
        {
            "vp_id": f"vp-{vi}", "vp_lat": vp[vi][0], "vp_lon": vp[vi][1],
            "tg_id": f"tg-{ti}",
            "tg_lat": t.loc[f"tg-{ti}", "tg_lat"], "tg_lon": t.loc[f"tg-{ti}", "tg_lon"],
        }
        for vi in range(len(vp))
        for ti in range(len(tg))
    ]
    return space, build_bipartite(space, vps, pd.DataFrame(rows))


# ---- great-circle interpolation ---------------------------------------------


def test_the_path_bows_where_an_independent_geodesic_says_it_does():
    """CHI->SEA bows ~1.3 deg (~146 km) north of its lon/lat chord."""
    from pyproj import Geod

    path = great_circle_segments(
        np.array([CHI[0]]), np.array([CHI[1]]), np.array([SEA[0]]), np.array([SEA[1]])
    )[0]
    mid_lon, mid_lat = path[len(path) // 2]
    ref_lon, ref_lat = Geod(ellps="WGS84").npts(CHI[1], CHI[0], SEA[1], SEA[0], 1)[0]
    assert mid_lat == pytest.approx(ref_lat, abs=0.01)
    assert mid_lon == pytest.approx(ref_lon, abs=0.01)
    assert mid_lat > (CHI[0] + SEA[0]) / 2


def test_every_interpolated_point_lies_on_the_path():
    a, b = np.array([CHI[0]]), np.array([CHI[1]])
    c, d = np.array([SJC[0]]), np.array([SJC[1]])
    path = great_circle_segments(a, b, c, d, n_points=33)[0]
    hops = elementwise_km(path[:-1, 1], path[:-1, 0], path[1:, 1], path[1:, 0])
    assert hops.sum() == pytest.approx(float(elementwise_km(a, b, c, d)[0]), rel=1e-6)


def test_the_endpoints_are_exact():
    path = great_circle_segments(
        np.array([CHI[0]]), np.array([CHI[1]]), np.array([NYC[0]]), np.array([NYC[1]])
    )[0]
    assert path[0] == pytest.approx((CHI[1], CHI[0]), abs=1e-9)
    assert path[-1] == pytest.approx((NYC[1], NYC[0]), abs=1e-9)


def test_a_path_across_the_antimeridian_stays_contiguous():
    path = great_circle_segments(
        np.array([20.0]), np.array([179.0]), np.array([20.0]), np.array([-179.0])
    )[0]
    assert np.abs(np.diff(path[:, 0])).max() < 10.0
    assert path[:, 0].max() > 180.0


def test_coincident_endpoints_do_not_divide_by_zero():
    path = great_circle_segments(
        np.array([CHI[0]]), np.array([CHI[1]]), np.array([CHI[0]]), np.array([CHI[1]])
    )[0]
    assert np.all(np.isfinite(path))
    assert path[:, 0] == pytest.approx(CHI[1])


def test_mismatched_endpoint_arrays_are_refused():
    with pytest.raises(ValueError, match="disagree"):
        great_circle_segments(
            np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([3.0]), np.array([3.0])
        )


# ---- the figures ------------------------------------------------------------


def test_both_maps_render(tmp_path):
    space, graph = _fixture()
    for p in (
        MB.plot_bipartite_nodes(space, graph, tmp_path / "nodes.png"),
        MB.plot_bipartite_flows(space, graph, tmp_path / "flows.png"),
    ):
        assert p.exists() and p.stat().st_size > 0


def test_the_optional_layers_can_be_dropped(tmp_path):
    space, graph = _fixture()
    out = MB.plot_bipartite_nodes(space, graph, tmp_path / "bare.png", cells=False, vp_grids=False)
    assert out.exists() and out.stat().st_size > 0


def test_the_flow_maps_cells_can_be_dropped(tmp_path):
    space, graph = _fixture()
    on = MB.plot_bipartite_flows(space, graph, tmp_path / "on.png", cells=True)
    off = MB.plot_bipartite_flows(space, graph, tmp_path / "off.png", cells=False)
    assert on.read_bytes() != off.read_bytes()


def test_the_flow_map_can_be_capped_deterministically(tmp_path):
    space, graph = _fixture()
    a = MB.plot_bipartite_flows(space, graph, tmp_path / "a.png", max_segments=4)
    b = MB.plot_bipartite_flows(space, graph, tmp_path / "b.png", max_segments=4)
    assert a.read_bytes() == b.read_bytes()


def test_unscored_mesh_sites_render(tmp_path):
    """A weighted run's mesh-only sites are drawn hollow, not dropped."""
    space, graph = _fixture(scored=[0, 1, 1, 1])
    full_space, _ = _fixture()
    hollow = MB.plot_bipartite_nodes(space, graph, tmp_path / "hollow.png")
    full = MB.plot_bipartite_nodes(full_space, graph, tmp_path / "full.png")
    assert hollow.read_bytes() != full.read_bytes()


def test_the_frame_covers_vps_outside_the_site_hull():
    """Anchorage sits north and west of every TG; a site-only frame crops it."""
    space, graph = _fixture()
    graph.vp_nodes.loc[0, ["vp_lat", "vp_lon"]] = [61.17, -149.99]
    lon_min, lon_max, lat_min, lat_max = MB.extent_for(space, graph)
    assert lat_max > 61.17 and lon_min < -149.99
