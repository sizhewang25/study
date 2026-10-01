"""Fig. B and the S-P subsection's numbers.

The load-bearing classes are `TestAgreesWithTheClusters` and
`TestTheRealPaperNumbers`. The first refuses a run where this module would
describe a different S-P VP than the one `plot-pni-gap` clustered (a
different tie-break is otherwise silent). The second pins the numbers the
paper quotes, so a change in the data or the lists fails loudly instead of
leaving stale numbers in the .tex.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from scripts.analysis.v5.modules import figure_pni_gap as F
from scripts.analysis.v5.modules import figure_sp_interconnect as B
from scripts.analysis.v5.modules import pni_gap as P
from scripts.analysis.v5.modules.geodesy import haversine_km
from scripts.analysis.v5.modules.paths import MissingArtifactError


@pytest.fixture
def clustered(pni_inputs):
    run, edge_csv, pni_csv, tg_group, root = pni_inputs
    [png] = F.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
    return run, edge_csv, pni_csv, tg_group, root, png.parent


def _load(clustered):
    run, edge_csv, pni_csv, _, root, _ = clustered
    return B.load_runs([run], {run.run_id: pni_csv}, analysis_root=root,
                       source_csvs={run.run_id: edge_csv})


def _brute(edge_csv, pni_csv):
    """Per-(TG, VP) distances to the TG and to the nearest interconnect, from scratch."""
    e = pd.read_csv(edge_csv)
    pn = pd.read_csv(pni_csv)
    e["d"] = haversine_km(e.vp_lat, e.vp_lon, e.target_lat, e.target_lon)
    e["d_x"] = haversine_km(e.vp_lat.values[:, None], e.vp_lon.values[:, None],
                            pn.pni_lat.values[None], pn.pni_lon.values[None]).min(1)
    return e, pn


class TestWritesBesideTheClusters:
    def test_png_csv_and_report(self, clustered):
        run, edge_csv, pni_csv, _, root, out = clustered
        [png] = B.build_for_runs([run], {run.run_id: pni_csv}, analysis_root=root,
                                 source_csvs={run.run_id: edge_csv})
        assert png.parent == out
        for name in (B.PNG_NAME, B.CSV_NAME, B.REPORT_NAME):
            assert (out / name).stat().st_size > 0

    def test_no_clusters_is_a_missing_artifact(self, pni_inputs):
        run, edge_csv, pni_csv, _, root = pni_inputs
        with pytest.raises(MissingArtifactError, match="plot-pni-gap"):
            B.load_runs([run], {run.run_id: pni_csv}, analysis_root=root,
                        source_csvs={run.run_id: edge_csv})


class TestAgreesWithTheClusters:
    def test_every_tg_carries_its_cluster(self, clustered):
        tgs, _, meta, _ = _load(clustered)
        assert tgs[P.CLUSTER_COL].notna().all() and len(tgs) == meta["n_tgs"]

    def test_a_different_sp_vp_is_refused(self, clustered):
        """Same TGs, but the clusters say the S-P VP is elsewhere."""
        out = clustered[-1]
        csv = out / P.CLUSTERS_CSV
        df = pd.read_csv(csv)
        df.loc[0, P.D_SP] += 5.0
        df.to_csv(csv, index=False)
        with pytest.raises(ValueError, match="different VP than the one clustered"):
            _load(clustered)

    def test_an_edited_interconnect_list_is_refused(self, clustered):
        pni_csv = clustered[2]
        with open(pni_csv, "a") as f:
            f.write("pni-z,30.0,-90.0\n")
        with pytest.raises(ValueError, match="has changed since the clusters"):
            _load(clustered)


class TestTheFramesAgainstBruteForce:
    def test_paths_through_an_interconnect_are_never_shorter_than_direct(self, clustered):
        tgs, _, _, _ = _load(clustered)
        assert (tgs.d_via_tg_km >= tgs.d_sp_km - 1e-6).all()
        assert (tgs.d_via_sp_km >= tgs.d_sp_km - 1e-6).all()

    def test_at_interconnect_and_the_random_baseline(self, clustered):
        run, edge_csv, pni_csv, *_ = clustered
        tgs, _, _, _ = _load(clustered)
        e, _ = _brute(edge_csv, pni_csv)
        for r in B.RADII_KM:
            base = e.assign(at=e.d_x <= r).groupby("target_id").at.mean()
            got = tgs.set_index("tg_id")[f"random_at_{r:g}km"].sort_index()
            np.testing.assert_allclose(got.to_numpy(), base.sort_index().to_numpy())
            sp = e.loc[e.groupby("target_id").rtt_ms.idxmin()].set_index("target_id")
            assert (tgs.set_index("tg_id")[f"sp_at_{r:g}km"].sort_index().to_numpy()
                    == (sp.d_x <= r).sort_index().to_numpy()).all()

    def test_rho_and_ties_per_tg(self, clustered):
        run, edge_csv, pni_csv, *_ = clustered
        tgs, _, _, _ = _load(clustered)
        e, _ = _brute(edge_csv, pni_csv)
        tg = tgs.tg_id.iloc[0]
        x = e[e.target_id == tg]
        row = tgs.set_index("tg_id").loc[tg]
        assert row.rho == pytest.approx(spearmanr(x.d, x.rtt_ms).statistic)
        assert row.n_tied == int((x.rtt_ms <= x.rtt_ms.min() + B.TIE_MS).sum())

    def test_violation_counts_match_the_points(self, clustered):
        tgs, coverage, meta, _ = _load(clustered)
        pts = B.points(tgs)
        rep = B.report(tgs, pts, coverage, meta)
        for name, col in B.PATHS.items():
            below = pts.rtt_sp_ms < pts[col] / B.FLOOR_KM_PER_MS
            assert rep["path_fit"][name]["n_points_below_floor"] == int(below.sum())
            assert rep["path_fit"][name]["n_tgs_below_floor"] == int(pts.n_tgs[below].sum())

    def test_every_interconnect_is_covered_in_the_fixture(self, clustered):
        """The fixture puts a VP on each of its two interconnects."""
        _, coverage, _, _ = _load(clustered)
        [d] = coverage.values()
        assert len(d) == 2 and d.max() == pytest.approx(0.0, abs=1e-6)


class TestPooled:
    def test_pooled_report_covers_both_runs(self, pni_two_runs):
        two = pni_two_runs
        F.build_for_runs(two["runs"], two["pni_csvs"], layouts=(P.POOLED,),
                         analysis_root=two["root"], source_csvs=two["edge_csvs"])
        [png] = B.build_for_runs(two["runs"], two["pni_csvs"], layouts=(P.POOLED,),
                                 analysis_root=two["root"], source_csvs=two["edge_csvs"])
        rep = json.loads((png.parent / B.REPORT_NAME).read_text())
        assert sorted(rep["run_ids"]) == ["run-a", "run-b"]
        assert rep["distribution"]["n_tgs"] == 54
        assert set(rep["interconnect_coverage"]["per_run"]) == {"run-a", "run-b"}


class TestPrivacy:
    def test_no_coordinate_vp_or_interconnect_id_in_outputs(self, clustered):
        run, edge_csv, pni_csv, _, root, out = clustered
        B.build_for_runs([run], {run.run_id: pni_csv}, analysis_root=root,
                         source_csvs={run.run_id: edge_csv})
        forbidden = ("lat", "lon", "city", "vp_id", "pni_id", "site_key")
        for col in pd.read_csv(out / B.CSV_NAME).columns:
            assert not any(f in col.lower() for f in forbidden), col
        text = (out / B.REPORT_NAME).read_text()
        assert "pni-a" not in text and "pni-b" not in text and "vp-" not in text
        # No number in the report is any coordinate the fixture used. A regex for
        # "two decimals, comma, two decimals" fires on distance ranges instead.
        coords = set()
        e = pd.read_csv(edge_csv)
        for col in ("vp_lat", "vp_lon", "target_lat", "target_lon"):
            coords |= {round(float(v), 4) for v in e[col]}
        coords |= {round(float(v), 4) for c in ("pni_lat", "pni_lon") for v in pd.read_csv(pni_csv)[c]}
        # Round values (40.0, -100.0) collide with counts and medians; the fixture's
        # seven non-integer coordinates are the meaningful check.
        coords = {c for c in coords if c != int(c)}
        assert len(coords) >= 5

        def numbers(node):
            if isinstance(node, dict):
                for v in node.values():
                    yield from numbers(v)
            elif isinstance(node, list):
                for v in node:
                    yield from numbers(v)
            elif isinstance(node, float) and np.isfinite(node):
                yield round(node, 4)

        leaked = coords & set(numbers(json.loads(text)))
        assert not leaked, leaked


class TestTheRealPaperNumbers:
    """The numbers `sections/eval-on-error-distance.tex` quotes, pinned.

    Skips where the benchmark tree or the interconnect lists are absent. If a
    number here moves, the paper is stale: update both together.
    """

    RUNS = ("pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh")

    @pytest.fixture(scope="class")
    def rep(self, tmp_path_factory):
        from scripts.analysis.v5.modules.labels import declared_pni_csv
        from scripts.analysis.v5.modules.paths import resolve_run

        try:
            runs = [resolve_run(r) for r in self.RUNS]
        except MissingArtifactError as exc:
            pytest.skip(f"benchmark runs not available: {exc}")
        pnis = {r: declared_pni_csv(r) for r in self.RUNS}
        if any(p is None or not p.exists() for p in pnis.values()):
            pytest.skip("a pro-as0* config declares no existing interconnect list")
        root = tmp_path_factory.mktemp("analysis")
        F.build_for_runs(runs, pnis, layouts=(P.POOLED,), analysis_root=root)
        [png] = B.build_for_runs(runs, pnis, layouts=(P.POOLED,), analysis_root=root)
        return json.loads((png.parent / B.REPORT_NAME).read_text())

    def test_distribution(self, rep):
        d = rep["distribution"]
        assert (d["n_tgs"], d["n_sites"]) == (1269, 65)
        assert round(d["zero_gap_pct"], 1) == 19.1
        assert round(d["gap_gt_1000km_pct"], 1) == 6.8

    def test_sp_locates_the_interconnect(self, rep):
        at = rep["at_interconnect"]["by_radius"]["50km"]
        assert round(at["sp_vp_at_interconnect_pct"], 1) == 92.3
        assert round(at["random_vp_at_interconnect_pct"], 1) == 31.5
        for r in rep["interconnect_coverage"]["per_run"].values():
            assert r["with_vp_within_quoted_radius"] == r["n_interconnects"]

    def test_path_fit_and_violations(self, rep):
        fit = rep["path_fit"]
        assert fit["via_sp_vp_nearest_interconnect"]["n_points_below_floor"] == 0
        assert fit["via_tg_nearest_interconnect"]["n_tgs_below_floor"] == 57
        assert round(fit["via_sp_vp_nearest_interconnect"]["rho_long_range"], 2) == 0.95

    def test_latency_condition(self, rep):
        c = rep["latency_condition"]
        assert round(c["holds_pct"], 1) == 95.6
        assert c["by_cluster"]["3"]["n_tied_median"] == 15
