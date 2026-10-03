"""Has-X against no-X smallest RTTs, per content network.

The load-bearing classes are `TestTheGroups` and `TestOneValuePerTG`. The first
checks the split is `report-sp-pni-cells`'s own flag on the shared PNI fixture
(`near` and `trombone` hold their interconnect, `far` does not). The second
checks the box is over TGs, not sites, and keeps an empty side as a row.
`TestTheRealPaperNumbers` pins what the paper quotes.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import figure_pni_gap as F
from scripts.analysis.v5.modules import figure_x_cell_rtt as X
from scripts.analysis.v5.modules import pni_gap as P
from scripts.analysis.v5.modules import sites as S
from scripts.analysis.v5.modules import sp_pni_cells as M
from scripts.analysis.v5.modules.paths import MissingArtifactError
from scripts.analysis.v5.tests.test_sp_pni_cells import write_classify


@pytest.fixture
def ready(pni_inputs):
    run, edge_csv, pni_csv, tg_group, root = pni_inputs
    F.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
    write_classify(run, edge_csv, root)
    return {"run": run, "edge_csv": edge_csv, "pni_csv": pni_csv, "tg_group": tg_group,
            "root": root, "out": P.output_dir(run.run_id, pni_csv, analysis_root=root)}


def _load(r):
    return X.load_runs([r["run"]], {r["run"].run_id: r["pni_csv"]}, analysis_root=r["root"],
                       source_csvs={r["run"].run_id: r["edge_csv"]})


def _build(r):
    return X.build_for_run(r["run"], r["pni_csv"], analysis_root=r["root"], source_csv=r["edge_csv"])


def _frame(rows):
    """`(run_id, has_x, site, rtt, n)` rows -> a `load_runs`-shaped frame, `n` TGs per row."""
    out = []
    for run_id, has_x, site, rtt, n in rows:
        out += [{"run_id": run_id, "tg_id": f"{run_id}-{site}-{i}", X.GROUP_COL: has_x,
                 X.ANY_COL: has_x, S.SITE_KEY_COL: f"{run_id}|{site}", X.FLOOR_COL: rtt}
                for i in range(n)]
    return pd.DataFrame(out)


class TestWritesBesideTheClusters:
    def test_png_csv_manifest(self, ready):
        [png] = _build(ready)
        assert png.parent == ready["out"] and png.name == X.PNG_NAME
        for name in (X.PNG_NAME, X.CSV_NAME, X.MANIFEST_NAME):
            assert (ready["out"] / name).stat().st_size > 0
        man = json.loads((ready["out"] / X.MANIFEST_NAME).read_text())
        assert man["split_ms"] == X.SPLIT_MS and man["n_tgs_x_vs_any_interconnect_disagree"] == 0

    def test_no_classify_is_a_missing_artifact(self, ready):
        for p in ready["run"].classify_dir(M.SOURCE_NSIDE, root=ready["root"]).glob("*.parquet"):
            p.unlink()
        with pytest.raises(MissingArtifactError, match="classify"):
            _load(ready)

    def test_no_clusters_is_a_missing_artifact(self, pni_inputs):
        run, edge_csv, pni_csv, _, root = pni_inputs
        write_classify(run, edge_csv, root)
        with pytest.raises(MissingArtifactError, match="plot-pni-gap"):
            X.load_runs([run], {run.run_id: pni_csv}, analysis_root=root,
                        source_csvs={run.run_id: edge_csv})


class TestTheGroups:
    def test_the_flag_is_report_sp_pni_cells_own(self, ready):
        tgs = _load(ready)[0].set_index("tg_id").sort_index()
        flags = M.load_runs([ready["run"]], {ready["run"].run_id: ready["pni_csv"]},
                            analysis_root=ready["root"],
                            source_csvs={ready["run"].run_id: ready["edge_csv"]})[0]
        flags = flags.set_index("tg_id").sort_index()
        np.testing.assert_array_equal(tgs[X.GROUP_COL].to_numpy(), flags.tg_cell_holds_x.to_numpy())

    def test_designed_groups_land_on_their_side(self, ready):
        tgs = _load(ready)[0].set_index("tg_id")
        side = {g: set(tgs.loc[[t for t, gg in ready["tg_group"].items() if gg == g], X.GROUP_COL])
                for g in set(ready["tg_group"].values())}
        assert side == {"near": {True}, "trombone": {True}, "far": {False}}

    def test_every_tg_once_at_its_smallest_rtt(self, ready):
        """The fixture gives every TG's S-P VP 0.5 ms; the floor is that, once per TG."""
        tgs = _load(ready)[0]
        assert len(tgs) == len(ready["tg_group"]) and tgs.tg_id.is_unique
        e = pd.read_csv(ready["edge_csv"]).groupby("target_id").rtt_ms.min()
        np.testing.assert_allclose(tgs.set_index("tg_id")[X.FLOOR_COL].sort_index(), e.sort_index())


class TestOneValuePerTG:
    def test_the_box_is_over_tgs_not_sites(self):
        """20 replicas at 1 ms and one TG at 100 ms: the median is 1 ms, as a TG box says."""
        tgs = _frame([("r", True, "a", 1.0, 20), ("r", True, "b", 100.0, 1), ("r", False, "c", 10.0, 5)])
        row = X.stats_table(tgs, ["r"]).set_index("side").loc["has-X"]
        assert (row.n_tgs, row.n_sites, row.p50_ms) == (21, 2, 1.0)
        assert row.n_le_split == 20 and row.le_split_pct == pytest.approx(100 * 20 / 21)

    def test_an_empty_side_is_a_row(self, tmp_path):
        stats = X.stats_table(_frame([("r", True, "a", 1.0, 3)]), ["r"])
        no_x = stats.set_index("side").loc["no-X"]
        assert no_x.n_tgs == 0 and no_x.n_sites == 0 and np.isnan(no_x.p50_ms)
        X.plot(stats, run_ids=["r"], out_png=tmp_path / X.PNG_NAME, fliers={})   # draws without the empty box

    def test_all_rows_only_when_pooled(self):
        tgs = _frame([("r1", True, "a", 1.0, 2), ("r1", False, "b", 9.0, 2),
                      ("r2", True, "a", 2.0, 2), ("r2", False, "b", 8.0, 2)])
        assert X.ALL_RUNS not in set(X.stats_table(tgs[tgs.run_id == "r1"], ["r1"]).run_id)
        stats = X.stats_table(tgs, ["r1", "r2"])
        assert list(stats.run_id) == ["r1", "r1", "r2", "r2", X.ALL_RUNS, X.ALL_RUNS]
        assert list(stats.side) == ["has-X", "no-X"] * 3
        assert stats[stats.run_id == X.ALL_RUNS].n_tgs.tolist() == [4, 4]

    def test_outliers_are_beyond_p5_p95(self):
        tgs = _frame([("r", True, f"s{i}", float(i + 1), 1) for i in range(40)])
        stats = X.stats_table(tgs, ["r"])
        row = stats.set_index("side").loc["has-X"]
        fl = X.outliers(tgs, stats)[("r", True)]
        assert len(fl) == row.n_outliers == 4
        assert all(v < row.p5_ms or v > row.p95_ms for v in fl)

    def test_has_x_left_of_no_x_with_a_gap_between_runs(self):
        pos = X.positions(["r1", "r2"])
        assert pos[("r1", True)] < pos[("r1", False)] < pos[("r2", True)] < pos[("r2", False)]
        assert pos[("r2", True)] - pos[("r1", False)] > pos[("r1", False)] - pos[("r1", True)]


class TestStaleInputsAreRefused:
    def test_an_edited_interconnect_list(self, ready):
        with open(ready["pni_csv"], "a") as f:
            f.write("pni-z,30.0,-90.0\n")
        with pytest.raises(ValueError, match="has changed since the clusters"):
            _load(ready)

    def test_an_edited_edge_csv(self, ready):
        with open(ready["edge_csv"], "a") as f:
            f.write(f"vp-extra,30.0,-90.0,{next(iter(ready['tg_group']))},30.0,-90.0,7.0\n")
        with pytest.raises(ValueError, match="is not the CSV the clusters were computed from"):
            _load(ready)


class TestPooled:
    def test_one_pair_per_run_and_all_rows(self, pni_two_runs):
        two = pni_two_runs
        F.build_for_runs(two["runs"], two["pni_csvs"], layouts=(P.POOLED,),
                         analysis_root=two["root"], source_csvs=two["edge_csvs"])
        for run in two["runs"]:
            write_classify(run, two["edge_csvs"][run.run_id], two["root"])
        [png] = X.build_for_runs(two["runs"], two["pni_csvs"], layouts=(P.POOLED,),
                                 analysis_root=two["root"], source_csvs=two["edge_csvs"])
        stats = pd.read_csv(png.parent / X.CSV_NAME)
        assert list(stats.run_id) == ["run-a", "run-a", "run-b", "run-b", X.ALL_RUNS, X.ALL_RUNS]
        per_run = stats[stats.run_id != X.ALL_RUNS]
        assert per_run.n_tgs.sum() == stats[stats.run_id == X.ALL_RUNS].n_tgs.sum() == 54


class TestPrivacy:
    def test_no_coordinate_vp_or_interconnect_id_in_outputs(self, ready):
        _build(ready)
        forbidden = ("lat", "lon", "city", "vp_id", "pni_id", "site_key")
        for col in pd.read_csv(ready["out"] / X.CSV_NAME).columns:
            assert not any(f in col.lower() for f in forbidden), col
        text = (ready["out"] / X.MANIFEST_NAME).read_text()
        assert "pni-a" not in text and "pni-b" not in text and "vp-" not in text


class TestTheRealPaperNumbers:
    """The numbers the region-classification section quotes, against the default tree.

    Skips where the runs, lists or artifacts are absent. If a number here
    moves, the paper is stale: update both together.
    """

    RUNS = ("pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh")

    @pytest.fixture(scope="class")
    def stats(self):
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
            tgs, meta, _ = X.load_runs(runs, pnis, layout=P.POOLED)
        except MissingArtifactError as exc:
            pytest.skip(f"run `classify` and `plot-pni-gap --layout pooled` first: {exc}")
        assert (tgs[X.GROUP_COL] != tgs[X.ANY_COL]).sum() == 0
        return X.stats_table(tgs, list(meta["run_ids"])).set_index(["run_id", "side"])

    def test_pooled(self, stats):
        h, n = stats.loc[(X.ALL_RUNS, "has-X")], stats.loc[(X.ALL_RUNS, "no-X")]
        assert (h.n_tgs, h.n_sites, n.n_tgs, n.n_sites) == (668, 34, 601, 31)
        assert (round(h.p50_ms, 2), round(n.p50_ms, 1)) == (1.55, 10.5)
        assert (h.n_le_split, n.n_le_split) == (613, 0)

    def test_per_network_medians_and_sites(self, stats):
        got = {(r, s): (round(stats.loc[(r, s)].p50_ms, 1), int(stats.loc[(r, s)].n_sites))
               for r in self.RUNS for s in ("has-X", "no-X")}
        assert got == {
            ("pro-as01-mesh", "has-X"): (1.5, 15), ("pro-as01-mesh", "no-X"): (13.7, 5),
            ("pro-as02-mesh", "has-X"): (1.4, 9), ("pro-as02-mesh", "no-X"): (13.3, 13),
            ("pro-as03-mesh", "has-X"): (1.6, 10), ("pro-as03-mesh", "no-X"): (8.8, 13),
        }
