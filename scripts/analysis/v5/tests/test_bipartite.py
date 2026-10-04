"""The bipartite-graph step: VP nodes, TG nodes, measured edges. Ported from v3.

Two things are worth pinning beyond the arithmetic.

**The observed/latent split.** Every observed distance comes off a measured
edge, and the failure mode is silent: adding an unmeasured VP to the roster
must move `edge_density` and must *not* move any TG's `nearest_measured_vp_km`,
even when it is parked on top of a TG. A latent leak passes every other test.

**The `angular_features` duplicate.** `partvp/extract_features.py` has a private
copy of the same maths, so the two are compared directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import bipartite as bp
from scripts.analysis.v5.modules.paths import MissingArtifactError

CHI = (41.9742, -87.9073)
SJC = (37.4675, -121.9215)
NYC = (40.7085, -74.0095)
DFW = (32.8968, -97.0380)


def _space(coords, provenance=None):
    return A.build_answer_space(
        pd.DataFrame(
            {
                "tg_id": [f"tg-{i}" for i in range(len(coords))],
                "tg_lat": [c[0] for c in coords],
                "tg_lon": [c[1] for c in coords],
            }
        ),
        run_id="syn",
        provenance=provenance,
    )


def _vps(coords, asn=7018):
    return pd.DataFrame(
        {
            "vp_id": [f"vp-{i}" for i in range(len(coords))],
            "vp_lat": [c[0] for c in coords],
            "vp_lon": [c[1] for c in coords],
            "vp_asn": [asn] * len(coords),
        }
    )


def _edges(pairs, vps, space):
    """`pairs` are `(vp_index, tg_index)` tuples."""
    v = vps.set_index("vp_id")
    t = space.tgs.set_index("tg_id")
    return pd.DataFrame(
        [
            {
                "vp_id": f"vp-{vi}",
                "vp_lat": v.loc[f"vp-{vi}", "vp_lat"],
                "vp_lon": v.loc[f"vp-{vi}", "vp_lon"],
                "tg_id": f"tg-{ti}",
                "tg_lat": t.loc[f"tg-{ti}", "tg_lat"],
                "tg_lon": t.loc[f"tg-{ti}", "tg_lon"],
            }
            for vi, ti in pairs
        ]
    )


def _graph(vp_coords, tg_coords, pairs):
    space = _space(tg_coords)
    vps = _vps(vp_coords)
    return space, vps, bp.build_bipartite(space, vps, _edges(pairs, vps, space))


# ---- angular geometry -------------------------------------------------------


def test_a_single_vp_leaves_the_whole_turn_empty():
    """360/0, and not NaN: one VP does constrain, just from one direction."""
    assert bp.angular_features([12.0]) == (360.0, 0.0)


def test_max_gap_includes_the_wraparound():
    gap, _ = bp.angular_features([10.0, 40.0, 75.0])
    assert gap == pytest.approx(295.0)


def test_bracketing_vps_beat_clustered_ones_at_equal_degree():
    clustered = bp.angular_features([10.0, 40.0, 75.0])
    bracketed = bp.angular_features([20.0, 140.0, 260.0])
    assert bracketed[0] < clustered[0]
    assert bracketed[1] > clustered[1]


def test_circular_variance_is_zero_when_every_bearing_coincides():
    _, cv = bp.angular_features([33.0, 33.0, 33.0])
    assert cv == pytest.approx(0.0)


def test_no_bearings_is_nan_not_a_full_turn():
    gap, cv = bp.angular_features([])
    assert np.isnan(gap) and np.isnan(cv)


def test_angular_features_matches_the_partvp_copy():
    from scripts.analysis.partvp.extract_features import _angular_features, _bearings_deg

    vlats = np.array([c[0] for c in (CHI, SJC, NYC, DFW)])
    vlons = np.array([c[1] for c in (CHI, SJC, NYC, DFW)])
    mine = bp.bearings_deg(*DFW, vlats, vlons)
    theirs = _bearings_deg(DFW[0], DFW[1], vlats, vlons)
    assert mine == pytest.approx(theirs)
    assert bp.angular_features(mine) == pytest.approx(_angular_features(theirs))


# ---- edge and node arithmetic -----------------------------------------------


def test_density_is_over_the_roster_not_over_the_edges():
    """|VP roster| x |answer-space TGs|: an unmeasured VP is measurement unspent."""
    _, _, graph = _graph([CHI, SJC, NYC], [DFW, NYC], [(0, 0), (0, 1), (1, 0)])
    assert graph.meta["edges"]["n_edges"] == 3
    assert graph.meta["edges"]["edge_density"] == pytest.approx(3 / 6)
    assert graph.meta["nodes"]["vps"]["n_with_no_edge"] == 1


def test_an_unmeasured_vp_never_moves_an_observed_distance():
    """The extra VP sits *on* tg-0; `nearest_measured_vp_km` must not budge."""
    _, _, without = _graph([CHI, SJC], [DFW, NYC], [(0, 0), (0, 1)])
    _, _, with_extra = _graph([CHI, SJC, DFW], [DFW, NYC], [(0, 0), (0, 1)])
    a = without.tg_nodes.set_index("tg_id")["nearest_measured_vp_km"]
    b = with_extra.tg_nodes.set_index("tg_id")["nearest_measured_vp_km"]
    assert a.to_dict() == b.to_dict()
    assert b["tg-0"] > 100.0
    assert with_extra.meta["edges"]["edge_density"] < without.meta["edges"]["edge_density"]


def test_distances_are_reported_as_a_nested_pair():
    _, _, graph = _graph([CHI, SJC], [DFW, NYC], [(0, 0), (0, 1)])
    e = graph.meta["edges"]
    for key in ("length_km", "nearest_vp_km"):
        assert set(e[key]) >= {"observed", "latent", "note"}, key
        assert e[key]["observed"]["n"] > 0 and e[key]["latent"]["n"] > 0
    # Degree gets no latent half, and each side says why in its own block.
    for side, key in (("vps", "degree_to_tg"), ("tgs", "degree_to_vp")):
        block = graph.meta["nodes"][side][key]
        assert "latent" not in block
        assert "carries nothing" in block["note"]


def test_the_latent_pair_count_is_the_full_cross_product():
    _, _, graph = _graph([CHI, SJC, NYC], [DFW, NYC], [(0, 0)])
    e = graph.meta["edges"]
    assert e["length_km"]["n_latent_pairs"] == 6
    assert e["length_km"]["latent"]["n"] == 6
    assert e["length_km"]["observed"]["n"] == 1


def test_efficiency_is_one_when_the_nearest_vp_was_measured():
    _, _, graph = _graph([CHI, SJC], [DFW], [(0, 0), (1, 0)])
    assert graph.tg_nodes["measured_nearest_vp_ratio"].tolist() == [1.0]
    assert graph.meta["edges"]["measured_nearest_vp_ratio_per_tg"]["max"] == 1.0
    assert bool(graph.tg_nodes["nearest_vp_is_measured"].iloc[0])


def test_efficiency_exceeds_one_when_the_campaign_missed_the_nearest_vp():
    space = _space([(41.05, -87.05)])
    vps = _vps([(41.0, -87.0), SJC])
    graph = bp.build_bipartite(space, vps, _edges([(1, 0)], vps, space))
    row = graph.tg_nodes.iloc[0]
    assert row["nearest_vp_km"] < 10.0 and row["nearest_measured_vp_km"] > 2_000.0
    assert row["measured_nearest_vp_ratio"] == pytest.approx(
        row["nearest_measured_vp_km"] / row["nearest_vp_km"], rel=1e-3
    )
    assert not bool(row["nearest_vp_is_measured"])


def test_efficiency_is_never_below_one():
    """It broke once: edge lengths were rounded before the division."""
    vp = [CHI, SJC, NYC, DFW, (33.9, -118.4), (47.45, -122.31)]
    tg = [(32.95, -97.10), (40.75, -74.05), (39.10, -94.60), (25.79, -80.29)]
    pairs = [(v, t) for v in range(len(vp)) for t in range(len(tg)) if (v + t) % 3]
    _, _, graph = _graph(vp, tg, pairs)
    eff = graph.tg_nodes["measured_nearest_vp_ratio"].to_numpy(dtype=float)
    assert np.isfinite(eff).all(), eff
    assert (eff >= 1.0).all(), eff
    assert (eff > 1.0).any(), eff


def test_a_coincident_but_unmeasured_vp_is_counted_not_coerced():
    space = _space([CHI])
    vps = _vps([CHI, SJC])
    graph = bp.build_bipartite(space, vps, _edges([(1, 0)], vps, space))
    assert np.isnan(graph.tg_nodes["measured_nearest_vp_ratio"].iloc[0])
    assert graph.meta["edges"]["measured_nearest_vp_ratio_per_tg"]["n"] == 0
    assert graph.meta["nodes"]["tgs"]["count"] == 1


def test_a_coincident_and_measured_vp_is_exactly_one_not_zero_over_zero():
    space = _space([CHI])
    vps = _vps([CHI])
    graph = bp.build_bipartite(space, vps, _edges([(0, 0)], vps, space))
    assert graph.tg_nodes["measured_nearest_vp_ratio"].tolist() == [1.0]


def test_efficiency_snaps_the_two_reductions_residue():
    ratio, undefined = bp.measurement_efficiency(
        np.array([100.0 + 1.24e-7, 100.0, 110.0]), np.array([100.0, 100.0, 100.0])
    )
    assert undefined == 0
    assert ratio[0] == 1.0 and ratio[1] == 1.0
    assert ratio[2] == pytest.approx(1.1)


def test_the_latent_nearest_vp_ignores_the_edge_set():
    _, _, graph = _graph([CHI, SJC, DFW], [DFW, NYC], [(0, 0), (0, 1)])
    row = graph.tg_nodes.set_index("tg_id").loc["tg-0"]
    assert row["nearest_vp_km"] == pytest.approx(0.0)
    assert row["nearest_measured_vp_km"] > 100.0


def test_nearest_across_is_exact_under_chunking():
    rng = np.random.default_rng(7)
    a_lat, a_lon = rng.uniform(25.0, 49.0, 500), rng.uniform(-124.0, -67.0, 500)
    b_lat, b_lon = rng.uniform(25.0, 49.0, 37), rng.uniform(-124.0, -67.0, 37)
    whole = bp.nearest_across_km(a_lat, a_lon, b_lat, b_lon, chunk=10_000)
    chunked = bp.nearest_across_km(a_lat, a_lon, b_lat, b_lon, chunk=7)
    assert chunked[0] == pytest.approx(whole[0], rel=0, abs=0)
    assert chunked[1].tolist() == whole[1].tolist()


def test_nearest_across_an_empty_side_is_nan_not_a_crash():
    km, idx = bp.nearest_across_km(np.array([40.0]), np.array([-80.0]), np.array([]), np.array([]))
    assert np.isnan(km).all() and idx.tolist() == [-1]


def test_describe_p90_agrees_with_the_shared_describe_at_three_digits():
    v = np.concatenate([np.arange(97, dtype=float), [1234.5678, 0.00049, 5.5]])
    mine, theirs = bp.describe_p90(v), A._describe(v)
    for k in ("n", "min", "max", "mean"):
        assert mine[k] == theirs[k], k
    for k, want in theirs["percentiles"].items():
        assert mine["percentiles"][k] == want, k


def test_describe_p90_reads_monotonically():
    d = bp.describe_p90(np.arange(101, dtype=float))
    assert list(d["percentiles"]) == ["p5", "p25", "p50", "p75", "p90", "p95"]
    assert d["percentiles"]["p90"] == pytest.approx(90.0)


def test_edge_length_is_great_circle():
    """Against haversine, a differently derived formula."""
    from scripts.analysis.v5.modules.geodesy import haversine_km

    _, _, graph = _graph([CHI], [SJC], [(0, 0)])
    expected = float(haversine_km(CHI[0], CHI[1], SJC[0], SJC[1]))
    assert graph.edge_segments["length_km"].iloc[0] == pytest.approx(expected, abs=1e-3)


def test_occupied_grids_never_exceed_the_node_count():
    """Co-located VPs collapse -- the reason the count is the honest denominator."""
    _, _, graph = _graph([CHI, CHI, SJC], [DFW], [(0, 0), (1, 0), (2, 0)])
    disp = graph.meta["nodes"]["vps"]["dispersion"]
    assert disp["effective_count"] == 2
    assert disp["occupancy_ratio"] == pytest.approx(2 / 3, abs=1e-4)
    assert set(disp) == {"effective_count", "occupancy_ratio", "note"}
    assert graph.vp_nodes["vp_grid_id"].nunique() == 2


def test_tg_dispersion_is_the_answer_spaces_grid_count():
    """TGs are quantized by the answer space, so the two counts cannot drift."""
    space, _, graph = _graph([CHI], [DFW, NYC, SJC, SJC], [(0, 0), (0, 1), (0, 2), (0, 3)])
    tg = graph.meta["nodes"]["tgs"]
    assert tg["dispersion"]["effective_count"] == space.meta["n_grids"] == tg["n_grids"] == 3
    assert tg["n_sites"] == 3 and tg["n_seeds"] == space.n_seeds
    assert graph.tg_nodes["tg_grid_id"].tolist() == space.tgs["tg_grid_id"].tolist()


def test_diameter_is_reported_with_its_p95():
    _, _, graph = _graph([CHI, SJC, NYC], [DFW], [(0, 0), (1, 0), (2, 0)])
    v = graph.meta["nodes"]["vps"]
    assert v["geographic_diameter_km"] == pytest.approx(v["pairwise_km"]["max"])
    assert v["pairwise_p95_km"] <= v["geographic_diameter_km"]


# ---- components -------------------------------------------------------------


def test_a_split_edge_set_reports_two_components():
    _, _, graph = _graph([CHI, SJC], [DFW, NYC], [(0, 0), (1, 1)])
    cc = graph.meta["edges"]["connected_components"]
    assert cc["n_components"] == 2
    assert cc["largest_component_node_share"] == pytest.approx(0.5)


def test_an_unmeasured_vp_is_its_own_component():
    _, _, graph = _graph([CHI, SJC], [DFW], [(0, 0)])
    cc = graph.meta["edges"]["connected_components"]
    assert cc["n_components"] == 2 and cc["n_isolated_nodes"] == 1


def test_more_than_127_nodes_do_not_overflow_the_incidence_matrix():
    """pandas hands back int8 codes for <= 127 categories."""
    lats = 30.0 + np.arange(60) * 0.4
    space = _space([(float(a), -95.0) for a in lats])
    vps = _vps([(float(a), -90.0) for a in lats])
    graph = bp.build_bipartite(space, vps, _edges([(i, i) for i in range(60)], vps, space))
    assert graph.meta["edges"]["connected_components"]["n_components"] == 60


# ---- flow segments ----------------------------------------------------------


def test_coincident_edges_collapse_into_one_segment():
    """Replicas at one site are one line."""
    space = _space([CHI, CHI, CHI])
    vps = _vps([SJC])
    graph = bp.build_bipartite(space, vps, _edges([(0, 0), (0, 1), (0, 2)], vps, space))
    assert graph.meta["edges"]["n_edges"] == 3
    assert graph.meta["edges"]["n_distinct_geometry_edge"] == 1
    assert graph.edge_segments["n_edges"].tolist() == [3]


def test_distinct_tgs_do_not_collapse():
    _, _, graph = _graph([SJC], [CHI, NYC], [(0, 0), (0, 1)])
    assert graph.meta["edges"]["n_distinct_geometry_edge"] == 2


# ---- joins and guards -------------------------------------------------------


def test_an_edge_naming_an_unknown_tg_is_dropped_and_counted():
    space = _space([DFW])
    vps = _vps([CHI])
    edges = _edges([(0, 0)], vps, space)
    stray = edges.iloc[[0]].assign(tg_id="tg-not-in-the-answer-space")
    graph = bp.build_bipartite(space, vps, pd.concat([edges, stray], ignore_index=True))
    assert graph.meta["inputs"]["n_edges_dropped_tg_not_in_answer_space"] == 1
    assert graph.meta["edges"]["n_edges"] == 1


def test_an_edge_naming_an_unknown_vp_is_dropped_and_counted():
    space = _space([DFW])
    vps = _vps([CHI])
    edges = _edges([(0, 0)], vps, space)
    stray = edges.iloc[[0]].assign(vp_id="vp-not-in-the-roster")
    graph = bp.build_bipartite(space, vps, pd.concat([edges, stray], ignore_index=True))
    assert graph.meta["inputs"]["n_edges_dropped_vp_not_in_roster"] == 1
    assert graph.meta["edges"]["n_edges"] == 1


def test_a_graph_sharing_nothing_with_the_run_is_refused():
    space = _space([DFW])
    vps = _vps([CHI])
    edges = _edges([(0, 0)], vps, space).assign(vp_id="vp-elsewhere")
    with pytest.raises(ValueError, match="do not describe the same graph"):
        bp.build_bipartite(space, vps, edges)


def test_a_weighted_space_keeps_mesh_tgs_at_degree_zero():
    """The weighted arm's space is its mesh; the TGs the filter dropped stay."""
    space = _space([DFW, NYC], provenance={"targets_source": "datasets/mesh.csv"})
    vps = _vps([CHI])
    graph = bp.build_bipartite(space, vps, _edges([(0, 0)], vps, space))
    assert graph.n_tgs == 2
    assert graph.meta["nodes"]["tgs"]["n_with_no_edge"] == 1
    assert graph.meta["inputs"]["tgs_source"] == "datasets/mesh.csv"
    # No edge, so no angular geometry -- NaN, not a full turn.
    row = graph.tg_nodes.set_index("tg_id").loc["tg-1"]
    assert row["degree_to_vp"] == 0 and np.isnan(row["max_angular_gap_deg"])


def test_one_id_with_two_coordinates_is_refused(tmp_path):
    csv = tmp_path / "canonical.csv"
    csv.write_text(
        "vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms\n"
        "vp-0,41.0,-87.0,tg-0,32.0,-97.0,10.0\n"
        "vp-0,42.0,-88.0,tg-1,40.0,-74.0,20.0\n"
    )
    with pytest.raises(ValueError, match="more than one coordinate"):
        bp.load_edges(csv)


def test_load_edges_collapses_repeated_observations_and_speaks_tg(tmp_path):
    csv = tmp_path / "canonical.csv"
    csv.write_text(
        "vp_id,vp_lat,vp_lon,target_id,target_lat,target_lon,rtt_ms\n"
        "vp-0,41.0,-87.0,tg-0,32.0,-97.0,10.0\n"
        "vp-0,41.0,-87.0,tg-0,32.0,-97.0,11.0\n"
        "vp-0,41.0,-87.0,tg-0,32.0,-97.0,-1.0\n"
    )
    edges, n_obs = bp.load_edges(csv)
    assert n_obs == 2  # the non-positive RTT is not an observation
    assert len(edges) == 1
    assert "rtt_ms" not in edges.columns
    assert {"tg_id", "tg_lat", "tg_lon"} <= set(edges.columns)
    assert not any(c.startswith("target_") for c in edges.columns)


# ---- round trip -------------------------------------------------------------


def test_write_then_load_round_trips(tmp_path):
    _, _, graph = _graph([CHI, SJC], [DFW, NYC], [(0, 0), (0, 1), (1, 1)])
    back = bp.load_bipartite(graph.write(tmp_path / "bg"))
    pd.testing.assert_frame_equal(back.edge_segments, graph.edge_segments)
    pd.testing.assert_frame_equal(back.edge_length_cdf, graph.edge_length_cdf)
    assert back.meta == graph.meta
    assert back.vp_nodes["vp_grid_id"].dtype == np.int64
    assert back.vp_nodes["vp_grid_id"].tolist() == graph.vp_nodes["vp_grid_id"].tolist()
    assert back.tg_nodes["tg_seed_id"].tolist() == graph.tg_nodes["tg_seed_id"].tolist()


def test_load_without_the_artifacts_says_which_command_writes_them(tmp_path):
    with pytest.raises(MissingArtifactError, match="build-bipartite-graph"):
        bp.load_bipartite(tmp_path)


def test_the_cdfs_are_monotone_on_a_fixed_quantile_grid():
    _, _, graph = _graph([CHI, SJC, NYC], [DFW, NYC], [(0, 0), (1, 0), (2, 1)])
    for frame, col in (
        (graph.edge_length_cdf, "observed_edge_km"),
        (graph.edge_length_cdf, "latent_pair_km"),
        (graph.pairwise_distance_cdf, "vp_pairwise_km"),
        (graph.pairwise_distance_cdf, "tg_pairwise_km"),
    ):
        v = frame[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        assert np.all(np.diff(v) >= -1e-9), col
    assert graph.edge_length_cdf["quantile"].iloc[[0, -1]].tolist() == [0.0, 1.0]


def test_cdf_column_uses_the_quantiles_the_cdf_csvs_are_written_at():
    out = bp.cdf_column(np.arange(101, dtype=float))
    assert out.size == bp.CDF_QUANTILES.size
    assert out[0] == 0.0 and out[-1] == 100.0
    assert np.isnan(bp.cdf_column([])).all()
