"""`report-dataset`'s single-VP counts."""

from __future__ import annotations

import pandas as pd

from scripts.analysis.v5.modules import dataset_summary as D


def _tgs(site_by_tg):
    return pd.DataFrame({"tg_id": list(site_by_tg), "site_id": list(site_by_tg.values())})


def test_single_vp_tgs_and_the_sites_holding_them():
    per_tg = pd.Series({"a": 1, "b": 1, "c": 5, "d": 1, "e": 5})
    tgs = _tgs({"a": 0, "b": 0, "c": 0, "d": 1, "e": 2})
    assert D.single_vp_counts(per_tg, tgs) == {
        "n_tgs_measured": 5,
        "n_tgs_single_vp": 3,
        "n_sites_measured": 3,
        "n_sites_single_vp": 2,  # sites 0 and 1; site 2 has none
    }


def test_denominators_are_the_edges_not_the_answer_space():
    """A traffic-weighted run's answer space is its mesh's: TGs it holds but
    the edges do not are no part of either share."""
    per_tg = pd.Series({"a": 1, "c": 5})
    tgs = _tgs({"a": 0, "b": 0, "c": 1, "z": 9})
    counts = D.single_vp_counts(per_tg, tgs)
    assert counts["n_tgs_measured"] == 2
    assert counts["n_sites_measured"] == 2


def test_shares_are_over_the_measured_population():
    row = {
        "n_sites_merged": 0, "n_sites": 4, "n_tgs_partial_mesh": 0, "n_tgs": 10,
        "est_removed_tgs": 0, "est_tgs_before_filter": 10,
        "kfold_n_sites_all_folds": 4, "max_error_km": 1.0, "dist_norm_max_km": 1.0,
        "n_tgs_measured": 8, "n_tgs_single_vp": 2, "n_sites_measured": 4, "n_sites_single_vp": 1,
    }
    t = D.add_shares(pd.DataFrame([row]))
    assert t.at[0, "single_vp_pct"] == 25.0
    assert t.at[0, "single_vp_sites_pct"] == 25.0


def test_pooled_row_sums_the_counts():
    base = {
        "n_tgs": 4, "n_sites": 2, "n_seeds": 2, "n_sites_merged": 0, "n_tgs_partial_mesh": 0,
        "n_sites_below_cap": 0, "est_removed_tgs": 0, "est_tgs_before_filter": 4,
        "kfold_n_sites_all_folds": 2, "replicas_min": 2, "vps_per_tg_min": 1,
        "kfold_fold_size_min": 1, "replicas_max": 2, "n_vps": 3, "kfold_fold_size_max": 1,
        "kfold_sites_per_fold_max": 2, "max_error_km": 1.0, "dist_norm_max_km": 1.0,
        "replicas_cap": 20,
    }
    a = {**base, "n_tgs_measured": 4, "n_tgs_single_vp": 1, "n_sites_measured": 2, "n_sites_single_vp": 1}
    b = {**base, "n_tgs_measured": 4, "n_tgs_single_vp": 3, "n_sites_measured": 2, "n_sites_single_vp": 2}
    pooled = D.pooled_row([a, b])
    assert (pooled["n_tgs_single_vp"], pooled["n_tgs_measured"]) == (4, 8)
    assert (pooled["n_sites_single_vp"], pooled["n_sites_measured"]) == (3, 4)
