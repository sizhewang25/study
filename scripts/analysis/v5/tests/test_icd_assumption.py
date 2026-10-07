"""The interconnect-distance side report: X* (nearest the TG) against S-P's X.

`TestTwoInterconnects` is the load-bearing class: one TG whose S-P VP sits at
the far interconnect (S-P disagrees with X*, and only the routing distance
breaks the floor) beside one TG where the two agree.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.analysis.v5.modules import icd_assumption as I
from scripts.analysis.v5.modules.geodesy import haversine_km

#: X1 55 km north of TG `t-far`; X2 852 km east of it, beside TG `t-near`.
PNIS = pd.DataFrame({"pni_id": ["x1", "x2"], "pni_lat": [40.5, 40.0], "pni_lon": [-100.0, -90.0]})
TG = {"t-far": (40.0, -100.0), "t-near": (40.0, -89.0)}
VP = {"v-x2": (40.0, -90.0), "v-north": (45.0, -100.0)}


def _hav(a, b) -> float:
    return float(haversine_km(a[0], a[1], b[0], b[1]))


def _edges() -> pd.DataFrame:
    """`figure_sp_interconnect.load_edges`' shape. `t-far`'s S-P VP is `v-x2`,
    at an RTT whose floor lies between its air and its routed distance."""
    x1 = (40.5, -100.0)
    d_air = _hav(VP["v-x2"], TG["t-far"])
    d_route = _hav(VP["v-x2"], x1) + _hav(x1, TG["t-far"])
    rtt_far = (d_air + d_route) / 2 / I.FLOOR_KM_PER_MS
    rows = [
        ("t-far", "v-x2", rtt_far),
        ("t-far", "v-north", 20.0),
        ("t-near", "v-x2", 2.0),
        ("t-near", "v-north", 30.0),
    ]
    df = pd.DataFrame([
        {"tg_id": t, "vp_id": v, "rtt_ms": r,
         "vp_lat": VP[v][0], "vp_lon": VP[v][1], "tg_lat": TG[t][0], "tg_lon": TG[t][1]}
        for t, v, r in rows
    ])
    df["d_km"] = haversine_km(df.vp_lat.values, df.vp_lon.values, df.tg_lat.values, df.tg_lon.values)
    return df


@pytest.fixture
def frames():
    routed = I.route_edges(_edges(), PNIS)
    tgs = I.tg_frame(routed, PNIS, run_id="r")
    routed = I.route_edges_via_sp(routed, tgs, PNIS)
    return routed, tgs, I.report(routed, tgs, {"run_id": "r"})


class TestTwoInterconnects:
    def test_route_goes_through_the_tgs_nearest_interconnect(self, frames):
        routed, _, _ = frames
        row = routed.set_index(["tg_id", "vp_id"]).loc[("t-far", "v-x2")]
        x1 = (40.5, -100.0)
        assert row.d_route_km == pytest.approx(_hav(VP["v-x2"], x1) + _hav(x1, TG["t-far"]))
        assert (routed.d_route_km >= routed.d_km - 1e-9).all()

    def test_sp_disagrees_only_where_its_vp_sits_at_the_other_interconnect(self, frames):
        _, tgs, rep = frames
        same = tgs.set_index("tg_id").same_interconnect
        assert not same["t-far"] and same["t-near"]
        assert rep["agreement"]["all"]["disagree_tgs_pct"] == 50.0
        assert rep["agreement"]["all"]["disagree_sites"] == {"any_pct": 50.0, "all_pct": 50.0}
        sep = tgs.set_index("tg_id").x_separation_km
        assert sep["t-far"] == pytest.approx(_hav((40.5, -100.0), (40.0, -90.0)))
        assert sep["t-near"] == 0.0

    def test_floor_broken_by_the_routing_distance_only(self, frames):
        _, _, rep = frames
        fv = rep["floor_violations"]
        assert fv["edges"] == {"air_pct": 0.0, "route_pct": 25.0, "route_only_pct": 25.0}
        assert fv["edges_where_sp_disagrees"]["route_only_pct"] == 50.0
        assert fv["edges_where_sp_agrees"]["route_only_pct"] == 0.0
        # The S-P VP's own edge: broken through X*, not through its own X.
        assert fv["sp_edge"]["via_tg_nearest_pct"] == 50.0
        assert fv["sp_edge"]["via_sp_nearest_pct"] == 0.0

    def test_routing_through_sp_interconnect_keeps_the_floor(self, frames):
        """Both TGs route through X2 under X_sp; t-far's S-P edge is then its
        air path, so nothing breaks the floor."""
        routed, _, rep = frames
        row = routed.set_index(["tg_id", "vp_id"]).loc[("t-far", "v-x2")]
        assert row.d_route_sp_km == pytest.approx(row.d_km)
        assert rep["floor_violations"]["edges_via_sp_interconnect"]["route_pct"] == 0.0
        assert (routed.d_route_sp_km >= routed.d_km - 1e-9).all()

    def test_no_credible_sp_without_rho(self, frames):
        """Two VPs per TG leave rho undefined, so no TG counts as credible."""
        _, tgs, rep = frames
        assert not tgs.sp_credible.any()
        assert rep["agreement"]["sp_credible"]["share_of_tgs_pct"] == 0.0

    def test_report_is_json_and_counts_free(self, frames):
        _, _, rep = frames
        text = json.dumps(rep)
        assert "n_tgs" not in text and "n_sites" not in text


class TestBuild:
    def test_writes_report_and_csv(self, pni_inputs, monkeypatch):
        run, edge_csv, pni_csv, _, root = pni_inputs
        monkeypatch.setattr(I, "_fit_distance", lambda run, root: ("air_distance", None))
        path = I.build(run, pni_csv, analysis_root=root, source_csv=edge_csv)
        rep = json.loads(path.read_text())
        assert rep["fit_distance"] == "air_distance"
        assert 0.0 <= rep["agreement"]["all"]["disagree_tgs_pct"] <= 100.0
        csv = pd.read_csv(path.parent / I.CSV_NAME)
        assert list(csv.columns) == list(I.CSV_COLUMNS)

    def test_refuses_a_different_baked_list(self, pni_inputs, monkeypatch, tmp_path):
        run, edge_csv, pni_csv, _, root = pni_inputs
        monkeypatch.setattr(I, "_fit_distance",
                            lambda run, root: ("interconnect_distance", str(tmp_path / "other.csv")))
        with pytest.raises(ValueError, match="fits through"):
            I.build(run, pni_csv, analysis_root=root, source_csv=edge_csv)


class TestFitDistance:
    """Read off the inputs manifests: the record of what the LTDs were fit with."""

    def _manifests(self, root, run, records):
        for i, rec in enumerate(records):
            d = root / run.source / run.run_id / run.setup / f"fold_{i}"
            d.mkdir(parents=True)
            (d / "manifest.json").write_text(json.dumps(rec))

    def test_icd_and_legacy(self, pni_inputs, tmp_path):
        run = pni_inputs[0]
        icd = {"distance": "interconnect_distance", "interconnect_csv_path": "x.csv"}
        self._manifests(tmp_path / "icd", run, [icd, icd])
        assert I._fit_distance(run, tmp_path / "icd") == ("interconnect_distance", "x.csv")
        self._manifests(tmp_path / "old", run, [{}])
        assert I._fit_distance(run, tmp_path / "old") == ("air_distance", None)

    def test_mixed_folds_and_missing_inputs_refused(self, pni_inputs, tmp_path):
        from scripts.analysis.v5.modules.paths import MissingArtifactError

        run = pni_inputs[0]
        self._manifests(tmp_path / "mix", run, [{}, {"distance": "interconnect_distance",
                                                     "interconnect_csv_path": "x.csv"}])
        with pytest.raises(ValueError, match="different distances"):
            I._fit_distance(run, tmp_path / "mix")
        with pytest.raises(MissingArtifactError, match="no materialized inputs"):
            I._fit_distance(run, tmp_path / "none")
