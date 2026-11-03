"""VP-count distribution and its has-X / no-X split (`figure_vp_obs`)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import figure_vp_obs as V
from scripts.analysis.v5.modules.paths import MissingArtifactError
from scripts.analysis.v5.modules.status import SHORTEST_PING
from scripts.analysis.v5.tests.conftest import FakeRun


def _run(root, run_id, prefix, n_obs, has_x=None, sites=None, *, with_n_obs=True):
    """A run whose S-P parquet holds `n_obs` per TG, and optionally the
    `report-sp-pni-cells` CSV holding `has_x` (bools) per TG."""
    run = FakeRun(run_id, root)
    tg = [f"{prefix}-{i}" for i in range(len(n_obs))]
    frame = pd.DataFrame({"tg_id": tg, "site_id": sites or list(range(len(tg)))})
    if with_n_obs:
        frame[C.N_OBS] = list(n_obs)
    frame.to_parquet(
        run.classify_dir(V.SOURCE_NSIDE) / C.TGS_PARQUET.format(method=SHORTEST_PING), index=False
    )
    if has_x is not None:
        d = run.analysis_dir("pni-gap") / "some-pni-list"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"run_id": run_id, "tg_id": tg, "tg_cell_holds_x": list(has_x)}).to_csv(
            d / "sp_pni_cells_tgs.csv", index=False
        )
    return run


class TestBins:
    def test_every_count_lands_in_exactly_one_bin(self):
        n = pd.Series([1, 2, 3, 4, 5, 6, 10, 11, 20, 21, 50, 51, 100, 101, 134])
        got = V.bin_of(n, V.HIST_BINS).astype(str).tolist()
        assert got == ["1", "2", "3", "4–5", "4–5", "6–10", "6–10",
                       "11–20", "11–20", "21–50", "21–50",
                       "51–100", "51–100", ">100", ">100"]

    def test_split_bins_isolate_one_and_two(self):
        got = V.bin_of(pd.Series([1, 2, 3, 10, 11]), V.SPLIT_BINS).astype(str).tolist()
        assert got == ["1 VP", "2 VPs", "3–10 VPs", "3–10 VPs", ">10 VPs"]

    def test_a_zero_count_is_refused(self):
        with pytest.raises(ValueError, match="n_obs < 1"):
            V.bin_of(pd.Series([0, 1]), V.HIST_BINS)


class TestTables:
    def test_hist_shares_close_and_single_vp_share_matches(self, tmp_path):
        run = _run(tmp_path, "syn1-000000-000000-mesh", "a", (1, 1, 2, 134))
        t = V.hist_table([V.load_tgs(run)], V.COMPARE)
        V.guard_closes(t, "share", ["panel"])
        assert t.set_index("bin").at["1", "share"] == pytest.approx(0.5)
        assert t.set_index("bin").at[">100", "share"] == pytest.approx(0.25)

    def test_pooled_is_the_concatenation_not_a_mean_of_panels(self, tmp_path):
        a = _run(tmp_path, "syn1-000000-000000-mesh", "a", (1, 5))
        b = _run(tmp_path, "syn2-000000-000000-mesh", "b", (1, 5, 5, 5))
        frames = [V.load_tgs(a), V.load_tgs(b)]
        pooled = V.hist_table(frames, V.POOLED).set_index("bin")
        assert pooled.at["1", "share"] == pytest.approx(2 / 6)  # not (1/2 + 1/4) / 2

    def test_pooled_refuses_shared_tg_ids(self, tmp_path):
        a = _run(tmp_path, "syn1-000000-000000-mesh", "same", (1, 5))
        b = _run(tmp_path, "syn2-000000-000000-mesh", "same", (1, 5))
        with pytest.raises(ValueError):
            V.hist_table([V.load_tgs(a), V.load_tgs(b)], V.POOLED)

    def test_has_x_split_reads_both_directions(self, tmp_path):
        # has-X: 1, 1, 5 VPs; no-X: 1, 5, 5, 5 VPs.
        run = _run(tmp_path, "syn1-000000-000000-mesh", "a",
                   (1, 1, 5, 1, 5, 5, 5), (True, True, True, False, False, False, False))
        t = V.has_x_table([V.load_tgs(run, with_has_x=True)], V.COMPARE)
        V.guard_closes(t, "share_of_side", ["panel", "side"])
        V.guard_closes(t, "share_of_bin", ["panel", "bin"])
        one = t[t["bin"] == "1 VP"].set_index("side")
        assert one.at[V.HAS_X, "share_of_side"] == pytest.approx(2 / 3)
        assert one.at[V.NO_X, "share_of_side"] == pytest.approx(1 / 4)
        # Of the three single-VP TGs, two are has-X.
        assert one.at[V.HAS_X, "share_of_bin"] == pytest.approx(2 / 3)
        assert one.at[V.HAS_X, "side_share"] == pytest.approx(3 / 7)

    def test_panel_stats_count_sites_holding_a_single_vp_tg(self, tmp_path):
        run = _run(tmp_path, "syn1-000000-000000-mesh", "a", (1, 5, 5, 5), sites=[0, 0, 1, 2])
        s = V.panel_stats(V.load_tgs(run))
        assert s["single_vp_share"] == pytest.approx(0.25)
        assert s["single_vp_sites_share"] == pytest.approx(1 / 3, abs=1e-4)  # rounded to 4


class TestRefusals:
    def test_a_parquet_without_n_obs_says_rerun_classify(self, tmp_path):
        run = _run(tmp_path, "syn1-000000-000000-mesh", "a", (1, 5), with_n_obs=False)
        with pytest.raises(MissingArtifactError, match="re-run"):
            V.load_tgs(run)

    def test_has_x_needs_the_sp_pni_cells_csv(self, tmp_path):
        run = _run(tmp_path, "syn1-000000-000000-mesh", "a", (1, 5))
        with pytest.raises(MissingArtifactError, match="report-sp-pni-cells"):
            V.load_tgs(run, with_has_x=True)

    def test_a_scored_tg_missing_from_the_has_x_csv_is_refused(self, tmp_path):
        run = _run(tmp_path, "syn1-000000-000000-mesh", "a", (1, 5), (True, False))
        extra = pd.read_parquet(
            run.classify_dir(V.SOURCE_NSIDE) / C.TGS_PARQUET.format(method=SHORTEST_PING)
        )
        extra = pd.concat([extra, pd.DataFrame({"tg_id": ["a-9"], "site_id": [9], C.N_OBS: [1]})])
        extra.to_parquet(
            run.classify_dir(V.SOURCE_NSIDE) / C.TGS_PARQUET.format(method=SHORTEST_PING),
            index=False,
        )
        with pytest.raises(ValueError, match="not in"):
            V.load_tgs(run, with_has_x=True)


@pytest.mark.parametrize("figure", V.FIGURES)
def test_build_writes_each_layout(tmp_path, figure):
    a = _run(tmp_path, "syn1-000000-000000-mesh", "a", (1, 1, 5, 120), (True, False, True, False))
    b = _run(tmp_path, "syn2-000000-000000-mesh", "b", (1, 2, 3, 60), (False, False, True, True))
    pngs = V.build_for_runs([a, b], figure, analysis_root=tmp_path)
    assert len(pngs) == len(V.LAYOUTS) and all(p.exists() for p in pngs)
    for png in pngs:
        man = json.loads(png.with_name(png.name.replace(".png", ".manifest.json")).read_text())
        assert man["kind"] == figure
        assert set(man["panels"]) and all(
            0 <= p["single_vp_share"] <= 1 for p in man["panels"].values()
        )
        assert png.with_suffix(".csv").exists()


def test_the_single_vp_fill_is_the_vp_outcome_bars_fill():
    from scripts.analysis.v5.modules import figure_outcome_bars as F

    assert V.VP1_INK == F.TIER_INK["vp1"] == V.SPLIT_INK["1 VP"]
    assert set(V.SPLIT_INK) == set(V.bin_labels(V.SPLIT_BINS))
