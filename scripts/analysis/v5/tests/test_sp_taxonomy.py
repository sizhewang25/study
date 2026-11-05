"""The S-P cause taxonomy: the rule, its features, the fixture end to end, the paper's numbers."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import figure_pni_gap as F
from scripts.analysis.v5.modules import pni_gap as P
from scripts.analysis.v5.modules import sp_taxonomy as T
from scripts.analysis.v5.modules.labels import load_group
from scripts.analysis.v5.modules.paths import MissingArtifactError
from scripts.analysis.v5.tests.test_sp_pni_cells import write_classify


# ---- the rule ---------------------------------------------------------------------


def _row(**kw):
    base = dict(correct=False, rho=0.9, x_feasible=True, n_feasible=3, tied_cells=1,
                pred_in_x_cell=False, best_fit_is_x=True, pred_seed_id=1, best_fit_seed_id=2)
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.mark.parametrize("kw, code", [
    (dict(correct=True, rho=0.0), "0"),                        # correct wins over everything
    (dict(rho=0.1, x_feasible=False), "A"),                    # A before B
    (dict(x_feasible=False, n_feasible=2, tied_cells=3), "B"),  # B before C
    (dict(x_feasible=False, n_feasible=0, pred_in_x_cell=True), "E"),  # nothing fits: not B
    (dict(tied_cells=2), "C"),
    (dict(tied_cells=2, pred_in_x_cell=True), "E"),            # a tie inside X's cell is not C
    (dict(best_fit_is_x=False, pred_seed_id=2, best_fit_seed_id=2), "D"),
    (dict(best_fit_is_x=False, pred_seed_id=1, best_fit_seed_id=2), "F"),
    (dict(pred_in_x_cell=True), "E"),
    (dict(), "F"),
])
def test_cause_precedence(kw, code):
    assert T.cause(_row(**kw)) == code


def test_a_rho_on_the_threshold_is_not_a():
    assert T.cause(_row(rho=T.LOW_RHO, pred_in_x_cell=True)) == "E"


def test_spearman_columns_match_scipy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    x[:, 2] = np.round(x[:, 2])          # ties
    y = rng.normal(size=40)
    got = T._spearman_cols(x, y)
    want = [spearmanr(x[:, j], y).statistic for j in range(3)]
    assert np.allclose(got, want)
    assert np.isnan(T._spearman_cols(np.ones((5, 1)), np.arange(5.0))[0])


# ---- the fixture, end to end ----------------------------------------------------------


@pytest.fixture
def ready(pni_inputs):
    run, edge_csv, pni_csv, tg_group, root = pni_inputs
    F.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
    write_classify(run, edge_csv, root)
    return {"run": run, "edge_csv": edge_csv, "pni_csv": pni_csv, "tg_group": tg_group, "root": root,
            "out": P.output_dir(run.run_id, pni_csv, analysis_root=root)}


def _build(r):
    return T.build_for_runs([r["run"]], {r["run"].run_id: r["pni_csv"]}, analysis_root=r["root"],
                            source_csvs={r["run"].run_id: r["edge_csv"]})


def test_writes_csv_and_report_beside_the_clusters(ready):
    [path] = _build(ready)
    assert path == ready["out"] / T.REPORT_NAME
    rep = json.loads(path.read_text())
    tgs = pd.read_csv(ready["out"] / T.CSV_NAME)
    assert rep["population"]["n_tgs"] == len(tgs)
    # Every TG gets exactly one cause, and the side tables partition the TGs.
    assert tgs.cause.astype(str).isin(T.CAUSES).all()
    assert sum(c["n_tgs"] for s in rep["sides"].values() for c in s["causes"].values()) == len(tgs)
    # Correct is cause 0 and nothing else.
    assert (tgs.correct == (tgs.cause.astype(str) == "0")).all()


def test_no_coordinate_vp_interconnect_or_site_key_in_outputs(ready):
    _build(ready)
    cols = set(pd.read_csv(ready["out"] / T.CSV_NAME).columns)
    assert not cols & {"tg_lat", "tg_lon", "pred_lat", "pred_lon", "vp_id", "site_key", "pni_id"}
    text = (ready["out"] / T.REPORT_NAME).read_text()
    assert "_lat" not in text and "_lon" not in text


def test_no_classify_is_a_missing_artifact(pni_inputs):
    run, edge_csv, pni_csv, _, root = pni_inputs
    F.build_for_run(run, pni_csv, analysis_root=root, source_csv=edge_csv)
    with pytest.raises(MissingArtifactError):
        T.build_for_runs([run], {run.run_id: pni_csv}, analysis_root=root,
                         source_csvs={run.run_id: edge_csv})


# ---- the paper's numbers ----------------------------------------------------------------


class TestTheRealPaperNumbers:
    """The §4.2 cause table on the pooled meshes. If a number moves, the paper is stale."""

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
            with cross.using_group(load_group("pro-paper")):
                tgs, meta, _ = T.load_runs(runs, pnis, layout=P.POOLED)
        except MissingArtifactError as exc:
            pytest.skip(f"run `classify` and `plot-pni-gap --layout pooled` first: {exc}")
        return T.report(tgs, meta)

    def _n(self, rep, side):
        return {k: v["n_tgs"] for k, v in rep["sides"][side]["causes"].items() if v["n_tgs"]}

    def test_has_x(self, rep):
        # The prototype's has-X table exactly: 622 correct, 25 RTT-ignores-distance, 20 near-tie.
        assert self._n(rep, "has-X") == {"0": 622, "A": 25, "C": 20}
        c = rep["sides"]["has-X"]["causes"]
        assert (round(c["A"]["side_pct"], 1), round(c["C"]["side_pct"], 1)) == (3.7, 3.0)
        assert c["A"]["runs"] == c["C"]["runs"] == ["pro-as01-mesh"]

    def test_no_x(self, rep):
        assert self._n(rep, "no-X") == {"A": 31, "B": 46, "C": 19, "D": 46, "E": 456, "F": 2}

    def test_the_rho_band_is_empty_around_the_threshold(self, rep):
        b = rep["rho_band"]
        assert b["separated"]
        assert round(b["a_rho_max"], 2) == 0.20 and round(b["correct_has_x_rho_min"], 2) == 0.81

    def test_misses_of_x_cell_by_cause(self, rep):
        m = rep["misses_x_cell"]
        assert round(m["pct"], 1) == 14.9
        assert (round(m["x_in_ne_x_pct"], 1), round(m["rtt_not_discriminative_pct"], 1)) == (7.3, 7.5)
        assert round(m["proven_share_of_x_in_ne_x_pct"]) == 50
        # Every TG in X's cell is correct or the expected no-X miss.
        assert m["in_x_cell_causes"] == ["0", "E"]

    def test_the_nearest_vp_never_wins_a_no_x_tg(self, rep):
        s = rep["rtt_signal"]
        assert s["no-X"]["nearest_vp_is_lowest_pct"] == 0.0
        assert round(s["has-X"]["nearest_vp_is_lowest_pct"], 1) == 36.1
        # Pooled, it is the zero-gap share (S-P answers at the nearest VP location).
        assert round(s["all"]["nearest_vp_is_lowest_pct"], 1) == 19.0
        assert round(s["no-X"]["nearest_over_lowest_rtt_median"], 2) == 1.76
        assert round(s["has-X"]["nearest_over_lowest_rtt_median"], 2) == 1.37

    def test_routing_by_side(self, rep):
        # §5: RTT tracks VP distance on has-X; on no-X even the nearest VP's RTT follows the path through X.
        r = rep["routing"]
        assert (round(r["has-X"]["per_tg_rho_p50"], 2), round(r["has-X"]["per_tg_rho_p10"], 2)) == (0.95, 0.89)
        assert (round(r["no-X"]["per_tg_rho_p50"], 2), round(r["no-X"]["per_tg_rho_p10"], 2)) == (0.86, 0.42)
        assert round(r["no-X"]["nearest_vp_rho_rtt_vs_path_via_x"], 2) == 0.77
        assert round(r["no-X"]["nearest_vp_rho_rtt_vs_own_distance"], 2) == 0.43
