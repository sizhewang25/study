"""The paper's derived numbers: every reading a module writes beside its figure.

`create_paper_artifacts.sh` exists so that no number in the paper is a hand
calculation. Each test here pins one of those readings on a frame small
enough to check by eye: ratios between percentiles, shares at a threshold,
gaps between methods, the cohorts nested in S-P's correct set, the gain kept
under LOSO, the batch budget, the Pareto leads, the dataset shape and the
one-phase variant comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import dataset_summary as D
from scripts.analysis.v5.modules import figure_correct_upset as U
from scripts.analysis.v5.modules import figure_cost_box as K
from scripts.analysis.v5.modules import figure_error_cdf as E
from scripts.analysis.v5.modules import figure_outcome_bars as B
from scripts.analysis.v5.modules import figure_pareto as F
from scripts.analysis.v5.modules import figure_vp_distance_cdf as V
from scripts.analysis.v5.modules import figure_x_cell_rtt as X
from scripts.analysis.v5.modules import loso_delta as L
from scripts.analysis.v5.modules import sp_pni_cells as M
from scripts.analysis.v5.modules import variant_delta as VD
from scripts.analysis.v5.modules.figure_vp_proximity import GEO, MEASURE_COLUMNS, SPING
from scripts.analysis.v5.modules.status import SHORTEST_PING

OCT, SPO, SOI = "octant_cbg_hull", "spotter_cbg", "million_scale_cbg"


# ---- §4: percentile ratios, threshold shares --------------------------------------


def test_ratio_table_pairs_and_spreads():
    table = pd.DataFrame({"method": [SHORTEST_PING, OCT]} | {
        E.pcol(p): [10.0 * (i + 1), 2.0 * (i + 1)] for i, p in enumerate(E.PERCENTILES)
    })
    table.loc[1, E.pcol(99)] = np.nan  # undefined under `cut`
    r = E.ratio_table(table)
    row = r[(r.percentile == "p50") & (r.method == OCT) & (r.reference == SHORTEST_PING)].iloc[0]
    assert row.ratio == pytest.approx(0.2) and row.ratio_inv == pytest.approx(5.0)
    assert np.isnan(r[(r.percentile == "p99") & (r.method == OCT) & (r.reference == SHORTEST_PING)].ratio.iloc[0])
    spread = r[(r.percentile == "p50/p5") & (r.method == OCT)].iloc[0]
    assert spread.reference == OCT and spread.ratio == pytest.approx(3.0)  # 6 / 2
    n = len(E.PERCENTILES) * 2 + len(E.SPREADS) * 2
    assert len(r) == n


def test_ratio_table_name():
    assert E.ratios_name("error_cdf.pooled.norm.cut.csv") == "error_cdf.pooled.norm.cut.ratios.csv"


def test_shares_at_thresholds():
    pop = pd.DataFrame({MEASURE_COLUMNS[GEO]: [1.0, 2.0, 3.0, 4.0],
                        MEASURE_COLUMNS[SPING]: [1.0, 5.0, 9.0, 20.0]})
    pop[V.GAP] = pop[MEASURE_COLUMNS[SPING]] - pop[MEASURE_COLUMNS[GEO]]
    s = V.shares_table(pop, [2.0, 10.0], "norm_e3").set_index(["series", "threshold"])
    assert s.loc[(V.GEO, 2.0), "share_le_pct"] == 50.0
    assert s.loc[(V.GAP, 10.0), "share_gt_pct"] == 25.0  # gaps 0, 3, 6, 16
    assert (s["unit"] == "norm_e3").all()


# ---- §5: per-network gaps, correct-share by pixel ------------------------------------


def _bars_table() -> pd.DataFrame:
    t = pd.DataFrame({
        "run_id": ["r1", "r1"], "dataset": ["AS-A", "AS-A"], "method": [SPO, OCT],
        "n_tgs": [10, 10], "n_cell_correct": [8, 6],
        "n_ring0_cell_correct": [0, 4], "n_ring1_cell_correct": [1, 1],
        "n_ring2_cell_correct": [1, 0], "n_beyond_cell_correct": [6, 1],
    })
    t["share_n_cell_correct"] = t["n_cell_correct"] / t["n_tgs"]
    return t


def test_of_correct_shares_and_bounded():
    t = B.add_of_correct(_bars_table()).set_index("method")
    assert t.loc[SPO, "of_correct_beyond"] == pytest.approx(6 / 8)
    assert t.loc[OCT, "of_correct_le_ring1"] == pytest.approx(5 / 6)
    assert t.loc[SPO, "accuracy_bounded"] == pytest.approx(0.2)  # rings 0-2 correct
    assert t.loc[OCT, "accuracy_bounded"] == pytest.approx(0.5)


def test_gap_table_is_unrounded_pp():
    g = B.gap_table(_bars_table())
    row = g[(g.method == SPO) & (g.reference == OCT)].iloc[0]
    assert row.d_acc_pp == pytest.approx(20.0) and row.d_acc_bounded_pp == pytest.approx(-30.0)
    assert len(g) == 2  # both ordered pairs


def test_x_cell_exceptions_list_sites_across_the_split():
    above = X.SPLIT_MS + 1.0
    tgs = pd.DataFrame({
        "run_id": "r1",
        "tg_id": [f"t{i}" for i in range(6)],
        "site_key": ["r1|a", "r1|a", "r1|b", "r1|b", "r1|c", "r1|c"],
        X.GROUP_COL: [True, True, True, True, False, False],
        X.FLOOR_COL: [above, 1.0, 1.0, 1.0, above, above],
        "cluster": [3, 3, 1, 1, 2, 2],
    })
    e = X.exceptions_table(tgs)
    assert len(e) == 1  # only site a's has-X replica is above the line; no-X c is above, as expected
    row = e.iloc[0]
    assert (row.side, row.site, row.clusters, row.n_tgs_site, row.n_tgs_across) == ("has-X", 0, "3", 2, 1)
    assert X.exceptions_table(tgs.assign(**{X.FLOOR_COL: 1.0})).iloc[0].side == "no-X"


def test_sp_sides():
    tgs = pd.DataFrame({
        "run_id": ["r1", "r1", "r2", "r2"],
        "site_key": ["r1|a", "r1|b", "r2|c", "r2|d"],
        "tg_cell_holds_x": [True, True, False, False],
        "correct": [True, False, False, False],
        "pred_in_interconnect_cell": [True, True, True, False],
        "cluster": [1, 3, 2, 2],
        "n_near_tie": [2, 15, 3, 3],
    })
    s = M.sides(tgs)
    assert s["has-X"]["correct_pct"] == 50.0 and s["has-X"]["misses_by_run"] == {"r1": 1}
    assert s["has-X"]["n_near_tie_median"]["wrong_by_cluster"] == {"3": 15.0}
    assert s["no-X"]["wrong_in_interconnect_cell_pct"] == 50.0


def test_nesting_against_the_baseline():
    oct_s = "octant_cbg_spl"
    mask = pd.DataFrame({
        SHORTEST_PING: [True, True, False, False, False, False],
        SOI: [True, True, False, False, False, False],
        OCT: [True, False, True, True, False, False],
        oct_s: [False, False, True, False, False, False],
        SPO: [False, False, False, False, True, False],
    }, index=[f"t{i}" for i in range(6)]).astype(bool)
    sites = pd.Series(["a", "a", "b", "b", "c", "c"], index=mask.index)
    has_x = pd.Series([True, True, False, False, False, True], index=mask.index)
    n = U.nesting_table(mask, sites, has_x).set_index(["kind", "method"])
    miss = n.loc[("misses_baseline", OCT)]
    assert miss.n_tgs == 1 and miss.share_of_reference == pytest.approx(0.5)
    assert n.loc[("adds_over_baseline", U.ANY_CBG)].n_tgs == 3
    assert n.loc[("misses_baseline", U.ANY_CBG)].n_tgs == 0
    fam = n.loc[("only", "octant_family")]
    assert fam.n_tgs == 2 and fam.has_x_share == 0.0  # t2 (both variants), t3 (OCT-H alone)
    assert n.loc[("only", SPO)].n_tgs == 1
    aon = n.loc[("all_or_nothing_sites", OCT)]
    assert aon.n_sites == 2  # a (1 of 2) is mixed; b all; c none


# ---- §5.4: gain kept under LOSO --------------------------------------------------------


def test_gain_kept_over_the_baseline():
    t = pd.DataFrame({
        "scope": "pooled", "has_x": "no-X",
        "method": [SHORTEST_PING, OCT, SPO],
        "acc_base": [0.002, 0.502, 0.40], "acc_loso": [0.002, 0.202, 0.40],
    })
    g = L.add_gain(t, "has_x").set_index("method")
    assert g.loc[OCT, "gain_kept"] == pytest.approx(0.4)
    assert g.loc[SPO, "gain_kept"] == pytest.approx(1.0)
    assert np.isnan(g.loc[SHORTEST_PING, "gain_kept"])


def test_bound_pixels_match_pareto():
    assert L.BOUND_PIXELS == F.BOUND_PIXELS


# ---- §6: the batch budget ---------------------------------------------------------------


def test_extrapolation_uses_the_mean():
    df = pd.DataFrame({
        "run_id": ["r1", "r1", "r2", "r2"],
        "ltd_ms": [1.0, 1.0, 1.0, 1.0], "mtl_ms": [8.0, 8.0, 8.0, 98.0], "ctr_ms": [1.0, 1.0, 1.0, 1.0],
        "ltd_heap_peak_bytes": [1.0, 1.0, 1.0, 1.0],
        "mtl_heap_peak_bytes": [2 * 2**20] * 4, "ctr_heap_peak_bytes": [0.0] * 4,
    })
    x = K.extrapolation_table({OCT: df}, memory="memory_heap", n_tgs=3_600_000, cores=10,
                              rss={"r1": {OCT: 100.0}, "r2": {OCT: 300.0}})
    pooled = x[x.scope == "all"].iloc[0]
    assert pooled.runtime_mean_ms == pytest.approx(32.5)  # (10, 10, 10, 100)
    assert pooled.core_hours == pytest.approx(32.5)      # 32.5 ms x 3.6e6 / 3.6e6
    assert pooled.wall_hours == pytest.approx(3.25)
    assert pooled.peak_per_tg_x_cores_mb == pytest.approx(20.0)
    assert pooled.process_rss_x_cores_mb == pytest.approx(3000.0)
    assert set(x.scope) == {"r1", "r2", "all"}


# ---- §7: Pareto readings ------------------------------------------------------------------


def test_pareto_readings():
    t = pd.DataFrame({
        "regime": "seen", "metric": "cell", "method": [SHORTEST_PING, SOI, SPO, OCT],
        "acc": [0.49, 0.50, 0.64, 0.74], "acc_min": [0.42, 0.41, 0.51, 0.64],
        "acc_max": [0.64, 0.66, 0.89, 0.82],
        "runtime_p50_ms": [np.nan, 34.0, 179.0, 5244.0],
        "runtime_mean_ms": [np.nan, 38.0, 218.0, 13836.0],
        "on_frontier": [False, True, True, True],
    })
    r = F.add_readings(t).set_index("method")
    assert r.loc[OCT, "lead_vs_sp_pp"] == pytest.approx(25.0)
    assert r.loc[SPO, "spread_pp"] == pytest.approx(38.0)
    assert r.loc[OCT, "frontier_prev"] == SPO
    assert r.loc[OCT, "runtime_ratio_vs_frontier_prev"] == pytest.approx(5244 / 179)
    assert r.loc[SOI, "frontier_prev"] == SHORTEST_PING
    pairs = pd.DataFrame({"regime": ["seen"], "metric": ["cell"], "a": [OCT], "b": [SPO]})
    p = F.add_pair_readings(pairs, t).iloc[0]
    assert p.d_acc_pooled_pp == pytest.approx(10.0) and p.runtime_p50_ratio == pytest.approx(5244 / 179)


# ---- §3: dataset shape ---------------------------------------------------------------------


def test_dataset_pooled_and_shares():
    rows = [
        {"run_id": "a", "dataset": "A", "n_tgs": 399, "n_sites": 20, "n_seeds": 18,
         "n_sites_merged": 4, "n_tgs_partial_mesh": 98, "n_sites_below_cap": 1,
         "est_removed_tgs": 1, "est_tgs_before_filter": 400, "kfold_n_sites_all_folds": 20,
         "replicas_min": 19, "vps_per_tg_min": 80, "kfold_fold_size_min": 79,
         "replicas_max": 20, "n_vps": 134, "kfold_fold_size_max": 80,
         "kfold_sites_per_fold_max": 20, "max_error_km": 3946.0, "dist_norm_max_km": 4387.0,
         "replicas_cap": 20},
        {"run_id": "b", "dataset": "B", "n_tgs": 412, "n_sites": 22, "n_seeds": 21,
         "n_sites_merged": 2, "n_tgs_partial_mesh": 79, "n_sites_below_cap": 7,
         "est_removed_tgs": 28, "est_tgs_before_filter": 440, "kfold_n_sites_all_folds": 22,
         "replicas_min": 10, "vps_per_tg_min": 26, "kfold_fold_size_min": 82,
         "replicas_max": 20, "n_vps": 134, "kfold_fold_size_max": 83,
         "kfold_sites_per_fold_max": 22, "max_error_km": 1448.0, "dist_norm_max_km": 4387.0,
         "replicas_cap": 20},
    ]
    t = D.add_shares(pd.DataFrame([*rows, D.pooled_row(rows)])).set_index("run_id")
    assert t.loc[D.ALL, "sites_merged_pct"] == pytest.approx(100 * 6 / 42)
    assert t.loc["b", "est_removed_pct"] == pytest.approx(100 * 28 / 440)
    assert t.loc[D.ALL, "max_error_over_d"] == pytest.approx(3946 / 4387)
    assert t.loc[D.ALL, "vps_per_tg_min"] == 26


# ---- appendix: the one-phase variant ---------------------------------------------------------


def _variant_long() -> pd.DataFrame:
    """Two sites x two replicas, seen and unseen; the variant flips one replica of site a unseen."""
    rows = []
    for regime, var_flip in ((VD.SEEN, False), (VD.UNSEEN, True)):
        for i, site in enumerate(["a", "a", "b", "b"]):
            orig = site == "a" or regime == VD.SEEN
            var = orig and not (var_flip and i == 1)
            rows.append({
                "base_run": "r", "run_id": f"r-{regime}", "regime": regime,
                "original": OCT, "variant": OCT + "_geo", "tg_id": f"t{i}", "site_key": f"r|{site}",
                "orig_solved": True, "var_solved": True,
                "orig_correct": orig, "var_correct": var,
                "orig_err_km": 10.0, "var_err_km": 12.0, "disp_km": 0.5,
                "orig_ltd_ms": 1.0, "orig_mtl_ms": 4000.0, "orig_ctr_ms": 470.0,
                "var_ltd_ms": 1.0, "var_mtl_ms": 4000.0, "var_ctr_ms": 0.47,
            })
    return pd.DataFrame(rows)


def test_variant_stats_drop_and_timing():
    long = _variant_long()
    s = VD.summary_table(long, ["original", "variant", "regime"], None)
    unseen = s[(s.scope == VD.POOLED) & (s.regime == VD.UNSEEN)].iloc[0]
    assert unseen.d_acc_pp == pytest.approx(-25.0) and unseen.n_sites_flip == 1
    assert unseen.sites_flip_pct == 50.0
    d = VD.drop_table(long)
    pooled = d[d.scope == VD.POOLED].iloc[0]
    assert pooled.drop_orig_pp == pytest.approx(-50.0) and pooled.drop_var_pp == pytest.approx(-75.0)
    assert pooled.did_acc_pp == pytest.approx(-25.0) and pooled.n_sites_drop_differs == 1
    t = VD.timing_table(long)
    assert t.ctr_p50_ratio.iloc[0] == pytest.approx(1000.0)


def test_parse_variant():
    assert VD.parse_variant("a=b") == ("a", "b")
    for bad in ("a", "a=", "=b", "a=a"):
        with pytest.raises(ValueError):
            VD.parse_variant(bad)
