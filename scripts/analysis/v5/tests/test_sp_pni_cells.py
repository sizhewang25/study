"""S-P's cell label against its interconnect's cell.

The load-bearing classes are `TestTheDesignedGroups`, `TestAgreesWithTheClusters`
and `TestTheRealPaperNumbers`. The first checks the rule on the shared PNI
fixture, whose three groups are built to land in each corner of it: `near` is
right and the rule says so, `far` is wrong and the rule says so, `trombone` sits
in its interconnect's cell but S-P answers across the map (the C3 analogue), so
the rule misses exactly there. The second refuses a `classify` S-P VP the
clusters could not have picked, while keeping an exact RTT tie. The third pins
the numbers the paper quotes.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import figure_pni_gap as F
from scripts.analysis.v5.modules import pni_gap as P
from scripts.analysis.v5.modules import sp_pni_cells as M
from scripts.analysis.v5.modules.geodesy import haversine_km, pairwise_km
from scripts.analysis.v5.modules.labels import load_group
from scripts.analysis.v5.modules.paths import MissingArtifactError
from scripts.analysis.v5.modules.status import SHORTEST_PING
from scripts.analysis.v5.tests.conftest import PNI_VPS

NSIDE = M.SOURCE_NSIDE


def write_classify(run, edge_csv, root):
    """An answer space over the edge CSV's TGs and a scored S-P parquet, under `root`.

    The S-P VP is the first `idxmin` in `(tg_id, vp_id)` order, the tie-break
    `plot-pni-gap` uses, so the fixture agrees with the clusters by construction.
    """
    e = pd.read_csv(edge_csv).sort_values(["target_id", "vp_id"])
    tgs = (e.groupby("target_id", as_index=False)[["target_lat", "target_lon"]].first()
           .rename(columns={"target_id": "tg_id", "target_lat": "tg_lat", "target_lon": "tg_lon"}))
    space = A.build_answer_space(tgs, nside=NSIDE, run_id=run.run_id)
    space.write(run.answer_space_dir(NSIDE, root=root))
    sp = e.loc[e.groupby("target_id").rtt_ms.idxmin()]
    frame = pd.DataFrame({"tg_id": sp.target_id, "tg_lat": sp.target_lat, "tg_lon": sp.target_lon,
                          "pred_lat": sp.vp_lat, "pred_lon": sp.vp_lon, "status": "BASELINE"})
    parquet = run.classify_dir(NSIDE, root=root) / C.TGS_PARQUET.format(method=SHORTEST_PING)
    C.score_method(frame.reset_index(drop=True), space).to_parquet(parquet, index=False)
    return space, parquet


@pytest.fixture
def ready(pni_inputs):
    run, edge_csv, pni_csv, tg_group, root = pni_inputs
    F.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
    space, parquet = write_classify(run, edge_csv, root)
    return {"run": run, "edge_csv": edge_csv, "pni_csv": pni_csv, "tg_group": tg_group,
            "root": root, "space": space, "parquet": parquet,
            "out": P.output_dir(run.run_id, pni_csv, analysis_root=root)}


def _load(r):
    return M.load_runs([r["run"]], {r["run"].run_id: r["pni_csv"]}, analysis_root=r["root"],
                       source_csvs={r["run"].run_id: r["edge_csv"]})


class TestWritesBesideTheClusters:
    def test_csv_and_report(self, ready):
        [path] = M.build_for_runs([ready["run"]], {ready["run"].run_id: ready["pni_csv"]},
                                  analysis_root=ready["root"],
                                  source_csvs={ready["run"].run_id: ready["edge_csv"]})
        assert path.parent == ready["out"] and path.name == M.REPORT_NAME
        assert (ready["out"] / M.CSV_NAME).stat().st_size > 0
        assert json.loads(path.read_text())["shares"]["all"]["n_tgs"] == 27

    def test_no_classify_is_a_missing_artifact(self, ready):
        ready["parquet"].unlink()
        with pytest.raises(MissingArtifactError, match="classify"):
            _load(ready)

    def test_no_clusters_is_a_missing_artifact(self, pni_inputs):
        run, edge_csv, pni_csv, _, root = pni_inputs
        write_classify(run, edge_csv, root)
        with pytest.raises(MissingArtifactError, match="plot-pni-gap"):
            M.load_runs([run], {run.run_id: pni_csv}, analysis_root=root,
                        source_csvs={run.run_id: edge_csv})


class TestTheDesignedGroups:
    def test_near_is_right_and_predicted(self, ready):
        tgs = _load(ready)[0].set_index("tg_id")
        near = [t for t, g in ready["tg_group"].items() if g == "near"]
        assert tgs.loc[near, "correct"].all() and tgs.loc[near, "rule_correct"].all()
        assert tgs.loc[near, "pred_in_x_cell"].all()

    def test_far_is_wrong_and_predicted_wrong(self, ready):
        tgs = _load(ready)[0].set_index("tg_id")
        far = [t for t, g in ready["tg_group"].items() if g == "far"]
        assert not tgs.loc[far, "correct"].any() and not tgs.loc[far, "rule_correct"].any()

    def test_the_rule_misses_exactly_the_trombone(self, ready):
        tgs, cell_meta, meta, _ = _load(ready)
        trombone = {t for t, g in ready["tg_group"].items() if g == "trombone"}
        missed = set(tgs.tg_id[tgs.rule_correct != tgs.correct])
        assert missed == trombone
        rep = M.report(tgs, cell_meta, meta)
        assert sum(p["n_tgs"] for p in rep["rule_misses"]["points"]) == len(trombone)
        assert all(p["rule_correct"] and p["n_correct"] == 0 for p in rep["rule_misses"]["points"])


class TestTheFramesAgainstBruteForce:
    def test_cells_and_flags(self, ready):
        tgs = _load(ready)[0].set_index("tg_id").sort_index()
        seeds = ready["space"].seeds
        pn = pd.read_csv(ready["pni_csv"])
        sp = pd.read_parquet(ready["parquet"]).set_index("tg_id").sort_index()

        def seed_of(lat, lon):
            d = pairwise_km(np.asarray(lat), np.asarray(lon), seeds.seed_lat, seeds.seed_lon)
            return seeds.seed_id.to_numpy()[d.argmin(1)]

        pni_seed = seed_of(pn.pni_lat, pn.pni_lon)
        x = haversine_km(sp.tg_lat.values[:, None], sp.tg_lon.values[:, None],
                         pn.pni_lat.values[None], pn.pni_lon.values[None]).argmin(1)
        pred_seed = seed_of(sp.pred_lat, sp.pred_lon)
        np.testing.assert_array_equal(tgs.pred_in_x_cell.to_numpy(), pred_seed == pni_seed[x])
        np.testing.assert_array_equal(tgs.tg_cell_holds_x.to_numpy(), sp.tg_seed_id.to_numpy() == pni_seed[x])
        np.testing.assert_array_equal(tgs.pred_in_interconnect_cell.to_numpy(), np.isin(pred_seed, pni_seed))
        np.testing.assert_array_equal(tgs.correct.to_numpy(), (sp.cell_label == "correct").to_numpy())

    def test_random_baseline(self, ready):
        tgs = _load(ready)[0].set_index("tg_id")
        seeds = ready["space"].seeds
        e = pd.read_csv(ready["edge_csv"])
        d = pairwise_km(e.vp_lat.values, e.vp_lon.values, seeds.seed_lat, seeds.seed_lon)
        e["vp_seed"] = seeds.seed_id.to_numpy()[d.argmin(1)]
        e = e.merge(tgs[["x_seed_id"]], left_on="target_id", right_index=True)
        want = e.assign(hit=e.vp_seed == e.x_seed_id).groupby("target_id").hit.mean()
        np.testing.assert_allclose(tgs.random_in_x_cell.sort_index().to_numpy(), want.sort_index().to_numpy())

    def test_contingency_partitions_every_tg(self, ready):
        tgs, cell_meta, meta, _ = _load(ready)
        rep = M.report(tgs, cell_meta, meta)
        assert sum(c["n_tgs"] for c in rep["contingency"]["all"]) == len(tgs)
        assert sum(c["n_correct"] for c in rep["contingency"]["all"]) == int(tgs.correct.sum())


class TestAgreesWithTheClusters:
    def _move_sp(self, ready, tg_id, where):
        df = pd.read_parquet(ready["parquet"])
        df.loc[df.tg_id == tg_id, ["pred_lat", "pred_lon"]] = where
        df.to_parquet(ready["parquet"], index=False)

    def test_a_vp_that_is_not_lowest_rtt_is_refused(self, ready):
        """`classify` answers at a VP that measured the TG but was not its fastest."""
        tg = next(t for t, g in ready["tg_group"].items() if g == "near")
        self._move_sp(ready, tg, PNI_VPS["vp-south"])
        with pytest.raises(ValueError, match="not a lowest-RTT VP"):
            _load(ready)

    def test_a_coordinate_no_vp_measured_from_is_refused(self, ready):
        tg = next(t for t, g in ready["tg_group"].items() if g == "near")
        self._move_sp(ready, tg, (33.0, -97.0))
        with pytest.raises(ValueError, match="not a lowest-RTT VP"):
            _load(ready)

    def test_an_edited_interconnect_list_is_refused(self, ready):
        with open(ready["pni_csv"], "a") as f:
            f.write("pni-z,30.0,-90.0\n")
        with pytest.raises(ValueError, match="has changed since the clusters"):
            _load(ready)

    def test_a_tie_is_kept_and_flagged(self):
        tgs = pd.DataFrame({"run_id": "r", "tg_id": ["a", "b"], "d_sp_km_classify": [100.0, 50.0],
                            "sp_rtt_ms": [5.0, 3.0], "min_rtt_ms": [5.0, 3.0]})
        clusters = pd.DataFrame({"run_id": "r", "tg_id": ["a", "b"], P.CLUSTER_COL: 1, P.POINT_COL: [0, 1],
                                 P.D_SP: [200.0, 50.0], P.D_PNI: 1.0, P.GAP: 0.0})
        out = M.check_against_clusters(tgs, clusters).set_index("tg_id")
        assert out.sp_tie_broken_differently.tolist() == [True, False]

    def test_a_slower_vp_is_refused(self):
        tgs = pd.DataFrame({"run_id": "r", "tg_id": ["a"], "d_sp_km_classify": [100.0],
                            "sp_rtt_ms": [5.1], "min_rtt_ms": [5.0]})
        clusters = pd.DataFrame({"run_id": "r", "tg_id": ["a"], P.CLUSTER_COL: 1, P.POINT_COL: [0],
                                 P.D_SP: [200.0], P.D_PNI: 1.0, P.GAP: 0.0})
        with pytest.raises(ValueError, match="not a lowest-RTT VP"):
            M.check_against_clusters(tgs, clusters)


class TestPooled:
    def test_pooled_report_covers_both_runs(self, pni_two_runs):
        two = pni_two_runs
        F.build_for_runs(two["runs"], two["pni_csvs"], layouts=(P.POOLED,),
                         analysis_root=two["root"], source_csvs=two["edge_csvs"])
        for run in two["runs"]:
            write_classify(run, two["edge_csvs"][run.run_id], two["root"])
        [path] = M.build_for_runs(two["runs"], two["pni_csvs"], layouts=(P.POOLED,),
                                  analysis_root=two["root"], source_csvs=two["edge_csvs"])
        rep = json.loads(path.read_text())
        assert sorted(rep["run_ids"]) == ["run-a", "run-b"]
        assert rep["shares"]["all"]["n_tgs"] == 54
        assert set(rep["cells"]["per_run"]) == {"run-a", "run-b"}
        # run-b peers only at pni-b, so its cells hold fewer interconnects.
        per = rep["cells"]["per_run"]
        assert per["run-b"]["n_cells_holding_interconnect"] < per["run-a"]["n_cells_holding_interconnect"]


class TestPrivacy:
    def test_no_coordinate_vp_or_interconnect_id_in_outputs(self, ready):
        M.build_for_runs([ready["run"]], {ready["run"].run_id: ready["pni_csv"]},
                         analysis_root=ready["root"], source_csvs={ready["run"].run_id: ready["edge_csv"]})
        forbidden = ("lat", "lon", "city", "vp_id", "pni_id", "site_key")
        for col in pd.read_csv(ready["out"] / M.CSV_NAME).columns:
            assert not any(f in col.lower() for f in forbidden), col
        text = (ready["out"] / M.REPORT_NAME).read_text()
        assert "pni-a" not in text and "pni-b" not in text and "vp-" not in text


#: The group whose pooled folders the real-data numbers are read from.
GROUP = "pro-paper"


class TestTheRealPaperNumbers:
    """The numbers the S-P subsection quotes, pinned against the default analysis tree.

    Reads the existing pooled clusters and `classify` output without writing.
    Skips where the runs, lists or artifacts are absent. If a number here
    moves, the paper is stale: update both together.
    """

    RUNS = ("pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh")

    @pytest.fixture(scope="class")
    def rep(self):
        from scripts.analysis.v5.modules.labels import declared_pni_csv
        from scripts.analysis.v5.modules.paths import resolve_run

        try:
            runs = [resolve_run(r) for r in self.RUNS]
        except MissingArtifactError as exc:
            pytest.skip(f"benchmark runs not available: {exc}")
        pnis = {r: declared_pni_csv(r) for r in self.RUNS}
        if any(p is None or not p.exists() for p in pnis.values()):
            pytest.skip("a pro-as0* config declares no existing interconnect list")
        try:
            with cross.using_group(load_group(GROUP)):  # the pooled folders are group-named
                tgs, cell_meta, meta, _ = M.load_runs(runs, pnis, layout=P.POOLED)
        except MissingArtifactError as exc:
            pytest.skip(f"run `classify` and `plot-pni-gap --layout pooled` first: {exc}")
        return M.report(tgs, cell_meta, meta)

    def test_population(self, rep):
        a = rep["shares"]["all"]
        assert (a["n_tgs"], a["n_sites"]) == (1269, 65)
        assert rep["consistency"]["n_tgs_sp_tie_broken_differently"] == 5

    def test_sp_answers_the_interconnect_cell(self, rep):
        a = rep["shares"]["all"]
        assert round(a["pred_in_x_cell_pct"], 1) == 85.0
        assert round(a["random_in_x_cell_pct"], 1) == 6.7
        assert round(a["pred_in_interconnect_cell_pct"], 1) == 99.9
        assert round(a["random_in_interconnect_cell_pct"], 1) == 67.3

    def test_clusters_split_on_the_tg_cell(self, rep):
        c = rep["shares"]["by_cluster"]
        assert [round(c[k]["tg_cell_holds_x_pct"], 1) for k in "123"] == [100.0, 0.0, 100.0]
        assert [round(c[k]["correct_pct"], 1) for k in "123"] == [96.9, 0.2, 0.0]
        assert [round(c[k]["pred_in_x_cell_pct"], 1) for k in "123"] == [96.9, 75.9, 0.0]

    def test_the_rule(self, rep):
        assert round(rep["rule"]["all"]["agree_pct"], 1) == 96.4
        [hit] = [c for c in rep["contingency"]["all"] if c["pred_in_x_cell"] and c["tg_cell_holds_x"]]
        assert (hit["n_tgs"], hit["correct_pct"]) == (623, 100.0)
        assert sum(p["n_tgs"] for p in rep["rule_misses"]["points"]) == 46
