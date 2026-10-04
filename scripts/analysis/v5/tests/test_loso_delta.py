"""`report-loso-delta`: a K-fold run against its leave-one-site-out twin.

The invariants: a pair is only compared on identical TGs and cells, the
parameter-free methods must not move between the two runs, the bootstrap
resamples sites (not TGs) and is deterministic, and the pooled rows are the
pairs' rows stacked.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import loso_delta as L
from scripts.analysis.v5.modules.paths import RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING

NSIDE = L.SOURCE_NSIDE
OCT = "octant_cbg_hull"
SOI = "million_scale_cbg"

#: Two sites one degree of latitude apart (~111 km), two replicas each.
SITES = {"A": (40.0, -100.0), "B": (41.0, -100.0)}
TGS = [("tg-a1", "A"), ("tg-a2", "A"), ("tg-b1", "B"), ("tg-b2", "B")]


def _frame(correct: list[bool], errs: list[float], *, status: str = "SUCCESS",
           pred_shift: float = 0.0, tgs=TGS) -> pd.DataFrame:
    seed = {"A": 0, "B": 1}
    return pd.DataFrame(
        {
            "tg_id": [t for t, _ in tgs],
            "tg_lat": [SITES[s][0] for _, s in tgs],
            "tg_lon": [SITES[s][1] for _, s in tgs],
            "tg_seed_id": [seed[s] for _, s in tgs],
            "status": status,
            "pred_lat": [SITES[s][0] + pred_shift for _, s in tgs],
            "pred_lon": [SITES[s][1] for _, s in tgs],
            "pred_dist_to_tg_km": errs,
            "cell_label": ["correct" if c else "wrong" for c in correct],
        }
    )


def _run(root, run_id: str, frames: dict[str, pd.DataFrame]) -> RunPaths:
    run = RunPaths(run_id=run_id, root=root, source="s", setup="t")
    out = run.classify_dir(NSIDE, root=root)
    for method, df in frames.items():
        df.to_parquet(out / C.TGS_PARQUET.format(method=method), index=False)
    return run


def _fixed() -> dict[str, pd.DataFrame]:
    """The parameter-free arms, identical in both runs of a pair."""
    sp = _frame([True, True, False, False], [5, 6, 300, 310], status="BASELINE")
    soi = _frame([True, False, True, False], [10, 200, 20, 220])
    return {SHORTEST_PING: sp, SOI: soi}


def _pair(root, tag: str = "", *, oct_loso=None):
    """Base: OCT-H right on all four. LOSO: wrong on site A's two replicas."""
    base = _run(root, f"mesh{tag}", {**_fixed(), OCT: _frame([True] * 4, [3, 4, 5, 6])})
    loso_oct = oct_loso if oct_loso is not None else _frame(
        [False, False, True, True], [400, 420, 7, 8]
    )
    loso = _run(root, f"loso{tag}", {**_fixed(), OCT: loso_oct})
    return base, loso


def test_parse_pair():
    assert L.parse_pair("a:b") == ("a", "b")
    for bad in ("a", "a:", ":b", "a:a", "a:b:c"):
        with pytest.raises(ValueError):
            L.parse_pair(bad)


def test_join_and_stats(tmp_path):
    base, loso = _pair(tmp_path)
    long = L.load_pair(base, loso, [SHORTEST_PING, SOI, OCT], root=tmp_path)
    oct_rows = long[long.method == OCT]
    s = L._stats(oct_rows)
    assert s["acc_base"] == 1.0 and s["acc_loso"] == 0.5 and s["d_acc"] == -0.5
    assert s["n_correct_to_wrong"] == 2 and s["n_sites_correct_to_wrong"] == 1
    assert s["n_wrong_to_correct"] == 0
    assert s["n_tgs"] == 4 and s["n_sites"] == 2
    # Parameter-free arms: zero delta by construction.
    sp = L._stats(long[long.method == SHORTEST_PING])
    assert sp["d_acc"] == 0 and sp["d_p50_km"] == 0


def test_unsolved_rows_are_wrong_but_not_in_error_percentiles(tmp_path):
    loso_oct = _frame([True, True, True, True], [1, 1, 1, 1])
    loso_oct.loc[0, "status"] = "FALLBACK"
    loso_oct.loc[0, "cell_label"] = "unanswered"
    loso_oct.loc[0, "pred_dist_to_tg_km"] = 9999.0
    base, loso = _pair(tmp_path, oct_loso=loso_oct)
    s = L._stats(L.load_pair(base, loso, [OCT], root=tmp_path))
    assert s["acc_loso"] == 0.75
    assert s["unanswered_loso"] == 0.25
    assert s["p90_loso_km"] == 1.0  # the FALLBACK distance never enters


def test_parameter_free_must_not_move(tmp_path):
    base = _run(tmp_path, "mesh", {**_fixed(), OCT: _frame([True] * 4, [1] * 4)})
    moved = _fixed()
    moved[SOI] = _frame([True, False, True, False], [10, 200, 20, 220], pred_shift=0.01)
    loso = _run(tmp_path, "loso", {**moved, OCT: _frame([True] * 4, [1] * 4)})
    with pytest.raises(ValueError, match="fits nothing"):
        L.load_pair(base, loso, [SOI, OCT], root=tmp_path)


def test_different_tgs_refused(tmp_path):
    base = _run(tmp_path, "mesh", {OCT: _frame([True] * 4, [1] * 4)})
    loso = _run(tmp_path, "loso", {OCT: _frame([True] * 3, [1] * 3, tgs=TGS[:3])})
    with pytest.raises(ValueError, match="different TGs"):
        L.load_pair(base, loso, [OCT], root=tmp_path)


def test_different_cells_refused(tmp_path):
    other = _frame([True] * 4, [1] * 4)
    other["tg_seed_id"] = [5, 5, 6, 6]
    base = _run(tmp_path, "mesh", {OCT: _frame([True] * 4, [1] * 4)})
    loso = _run(tmp_path, "loso", {OCT: other})
    with pytest.raises(ValueError, match="tg_seed_id"):
        L.load_pair(base, loso, [OCT], root=tmp_path)


def test_bootstrap_is_site_clustered_and_deterministic(tmp_path):
    base, loso = _pair(tmp_path)
    g = L.load_pair(base, loso, [OCT], root=tmp_path)
    a = L.site_bootstrap(g, n_boot=500, seed=1)
    b = L.site_bootstrap(g, n_boot=500, seed=1)
    assert a == b
    # Two sites: a replicate holds A twice (-1.0), B twice (0.0) or one each
    # (-0.5). A TG-level bootstrap would also produce -0.25 and -0.75.
    assert a["d_acc_ci_lo"] == -1.0 and a["d_acc_ci_hi"] == 0.0
    one_site = g[g.site_key == g.site_key.iat[0]]
    assert np.isnan(L.site_bootstrap(one_site, n_boot=100)["d_acc_ci_lo"])


def test_nearest_site_distance_and_bins(tmp_path):
    base, loso = _pair(tmp_path)
    g = L.load_pair(base, loso, [OCT], root=tmp_path)
    assert g["nearest_site_km"].between(110, 112).all()
    assert set(L.distance_bin(g["nearest_site_km"]).astype(str)) == {"50-200 km"}


def test_build_writes_tables_and_pools(tmp_path):
    p1 = _pair(tmp_path, "1")
    p2 = _pair(tmp_path, "2", oct_loso=_frame([True] * 4, [3, 4, 5, 6]))
    written = L.build([p1, p2], analysis_root=tmp_path, n_boot=50)
    assert "by_has_x" not in written  # no pni-gap output in the fixture
    summary = pd.read_csv(written["summary"])
    oct_rows = summary[summary.method == OCT].set_index("scope")
    assert oct_rows.loc["mesh1->loso1", "d_acc"] == -0.5
    assert oct_rows.loc["mesh2->loso2", "d_acc"] == 0.0
    pooled = oct_rows.loc[L.POOLED]
    assert pooled["n_tgs"] == 8 and pooled["n_sites"] == 4
    assert pooled["d_acc"] == -0.25  # micro: (−2 + 0) / 8
    trans = pd.read_csv(written["transitions"])
    t = trans[(trans.scope == L.POOLED) & (trans.method == OCT)]
    assert t.n_tgs.sum() == 8
    manifest = json.loads(written["manifest"].read_text())
    assert manifest["pairs"] == [{"base": "mesh1", "loso": "loso1"}, {"base": "mesh2", "loso": "loso2"}]
    member = pd.read_csv(written["membership"])
    assert len(member) == 2 * 3 * 4  # pairs x methods x TGs


def test_run_in_two_pairs_refused(tmp_path):
    p1 = _pair(tmp_path, "1")
    with pytest.raises(ValueError, match="more than one pair"):
        L.build([p1, (p1[0], _pair(tmp_path, "2")[1])], analysis_root=tmp_path, n_boot=0)


def test_requested_method_missing_refused(tmp_path):
    base, loso = _pair(tmp_path)
    with pytest.raises(ValueError, match="not scored"):
        L.pair_methods(base, loso, ["spotter_cbg"], NSIDE, tmp_path)
    assert L.pair_methods(base, loso, [OCT], NSIDE, tmp_path) == [OCT]


def test_has_x_breakdown_uses_the_flag(tmp_path):
    """The breakdown keys on `tg_cell_holds_x`, never on the per-run cluster id."""
    base, loso = _pair(tmp_path)
    d = base.analysis_dir("pni-gap", root=tmp_path) / "list.approx"
    d.mkdir(parents=True)
    pd.DataFrame({
        "tg_id": [t for t, _ in TGS],
        "cluster": [7, 7, 7, 7],  # one cluster id, two flag values: the flag must win
        "tg_cell_holds_x": [False, False, True, True],
    }).to_csv(d / "sp_pni_cells_tgs.csv", index=False)
    written = L.build([(base, loso)], analysis_root=tmp_path, n_boot=0)
    t = pd.read_csv(written["by_has_x"])
    o = t[t.method == OCT].set_index("has_x")
    assert o.loc[L.NO_X, "d_acc"] == -1.0 and o.loc[L.HAS_X, "d_acc"] == 0.0
