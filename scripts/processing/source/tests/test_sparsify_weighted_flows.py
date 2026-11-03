"""Tests for sparsify_weighted_flows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.processing.source.sparsify_weighted_flows import default_output, sparsify


def _df(n_tgs=20, n_vps=6):
    rows = [
        {"vp_id": f"vp{v}", "target_id": f"tg{t}", "rtt_ms": 10.0 + v, "weight": (v * 7 + t) % 11 / 10}
        for t in range(n_tgs) for v in range(n_vps)
    ]
    return pd.DataFrame(rows)


def test_shares_of_one_and_two_flow_targets():
    kept, s = sparsify(_df(), seed=1, single_frac=0.25, double_frac=0.10)
    per_tg = kept.groupby("target_id").size()
    assert (s["n_tgs_1_flow"], s["n_tgs_2_flows"]) == (5, 2)
    assert (per_tg == 1).sum() == 5 and (per_tg == 2).sum() == 2
    assert (per_tg == 6).sum() == 13  # untouched
    assert set(kept["target_id"]) == set(_df()["target_id"])  # no TG disappears


def test_the_kept_flows_are_the_heaviest():
    df = _df()
    kept, _ = sparsify(df, seed=1, single_frac=1.0, double_frac=0.0)
    top = df.loc[df.groupby("target_id")["weight"].idxmax()]
    # Every kept weight is its TG's maximum (ties may pick another max row).
    merged = kept.merge(top[["target_id", "weight"]], on="target_id", suffixes=("", "_max"))
    assert (merged["weight"] == merged["weight_max"]).all()


def test_output_is_a_subset_of_the_input_flows():
    df = _df()
    kept, _ = sparsify(df, seed=3)
    keys = set(zip(df.vp_id, df.target_id))
    assert set(zip(kept.vp_id, kept.target_id)) <= keys


def test_deterministic_in_the_seed():
    a, _ = sparsify(_df(), seed=7)
    b, _ = sparsify(_df(), seed=7)
    c, _ = sparsify(_df(), seed=8)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
    assert not a.reset_index(drop=True).equals(c.reset_index(drop=True))


@pytest.mark.parametrize("single,double", [(-0.1, 0.0), (0.7, 0.4)])
def test_bad_fractions_are_refused(single, double):
    with pytest.raises(ValueError):
        sparsify(_df(), seed=1, single_frac=single, double_frac=double)


def test_a_weightless_input_is_refused():
    with pytest.raises(ValueError, match="weight"):
        sparsify(_df().drop(columns=["weight"]), seed=1)


def test_default_output_moves_the_stage_tag():
    p = Path("d/x.mainland.sanitized.randweight.traffic-weighted.csv")
    assert default_output(p).name == "x.mainland.sanitized.randweight.sparse.traffic-weighted.csv"
