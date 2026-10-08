"""The benchmark's materialized inputs, and the config-named edge CSV.

What v5 used to take from `eval-source` (the Shortest-Ping VP, `min_inflation`)
and `materialize-target-space` (`vps.csv`, the CSV path) now comes from the
benchmark's inputs tree and the run's config. These pin that the reduction is
eval-source's, over the observations the run actually scored.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.analysis.v5.modules import bench_inputs as B
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import edges as E
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths

RUN = "bi-run"
SOURCE, SETUP = "generic_csv", "anchors_to_probes"
COMBOS = ("alpha", "beta")

#: Two TGs per fold. Seattle-ish VPs; `vp-c` sits on `tg-1`, so its inflation
#: is undefined.
VPS = {"vp-a": (47.0, -122.0), "vp-b": (45.0, -122.0), "vp-c": (40.0, -100.0)}
TGS = {"tg-0": (47.5, -122.3), "tg-1": (40.0, -100.0), "tg-2": (34.0, -118.0), "tg-3": (33.0, -97.0)}
FOLD_OF = {"tg-0": 0, "tg-1": 0, "tg-2": 1, "tg-3": 1}


def _obs(rows):
    """`(tg, vp, rtt, weight)` -> an `eval_observations.parquet` frame."""
    return pd.DataFrame([
        {"target_id": t, "target_lat": TGS[t][0], "target_lon": TGS[t][1],
         "vp_id": v, "vp_lat": VPS[v][0], "vp_lon": VPS[v][1],
         "latency_ms": rtt, "weight": w}
        for t, v, rtt, w in rows
    ])


def _tree(tmp_path, obs_by_fold, *, pair_weight_min=None, vps_by_fold=None):
    """Outputs (folds, combos, run.json, targets.parquet) and inputs on disk."""
    outputs, inputs = tmp_path / "outputs", tmp_path / "inputs"
    pwm = pair_weight_min if isinstance(pair_weight_min, dict) else {}
    for fold, rows in obs_by_fold.items():
        fid = f"fold_{fold}"
        in_dir = inputs / SOURCE / RUN / SETUP / fid
        in_dir.mkdir(parents=True)
        _obs(rows).to_parquet(in_dir / B.EVAL_OBSERVATIONS, index=False)
        vps = (vps_by_fold or {}).get(fold, VPS)
        pd.DataFrame([{"vp_id": k, "lat": la, "lon": lo, "asn": 1, "country": "US"}
                      for k, (la, lo) in vps.items()]).to_parquet(in_dir / B.VP_CONFIGS, index=False)
        tgs = sorted({r[0] for r in rows} | {t for t, f in FOLD_OF.items() if f == fold})
        for combo in COMBOS:
            d = outputs / RUN / SOURCE / SETUP / fid / combo
            d.mkdir(parents=True)
            thr = pwm.get(combo, pair_weight_min if not isinstance(pair_weight_min, dict) else None)
            (d / "run.json").write_text(json.dumps({"combo_id": combo, "pair_weight_min": thr}))
            pd.DataFrame({
                "target_id": tgs,
                "target_lat": [TGS[t][0] for t in tgs],
                "target_lon": [TGS[t][1] for t in tgs],
                "pred_lat": [TGS[t][0] for t in tgs],
                "pred_lon": [TGS[t][1] for t in tgs],
                "status": "SUCCESS",
            }).to_parquet(d / "targets.parquet", index=False)
    return RunPaths(run_id=RUN, root=outputs, source=SOURCE, setup=SETUP), inputs


BASE = {
    0: [("tg-0", "vp-a", 2.0, 1.0), ("tg-0", "vp-b", 5.0, 1.0),
        ("tg-1", "vp-c", 1.0, 1.0), ("tg-1", "vp-a", 30.0, 1.0)],
    1: [("tg-2", "vp-b", 20.0, 1.0), ("tg-2", "vp-a", 25.0, 1.0),
        ("tg-3", "vp-c", 12.0, 1.0), ("tg-3", "vp-b", 40.0, 1.0)],
}


class TestShortestPing:
    def test_min_rtt_vp_per_tg_and_fold(self, tmp_path):
        run, inputs = _tree(tmp_path, BASE)
        sp = B.shortest_ping(run, inputs).set_index("tg_id")
        assert sp["sping_vp_id"].to_dict() == {"tg-0": "vp-a", "tg-1": "vp-c", "tg-2": "vp-b", "tg-3": "vp-c"}
        assert sp["fold"].to_dict() == FOLD_OF
        assert sp.loc["tg-2", "min_rtt_ms"] == 20.0
        assert (sp.loc["tg-0", "sping_vp_lat"], sp.loc["tg-0", "sping_vp_lon"]) == VPS["vp-a"]

    def test_repeated_pair_keeps_its_minimum(self, tmp_path):
        obs = {0: BASE[0] + [("tg-0", "vp-b", 1.0, 1.0)], 1: BASE[1]}
        run, inputs = _tree(tmp_path, obs)
        sp = B.shortest_ping(run, inputs).set_index("tg_id")
        assert sp.loc["tg-0", "sping_vp_id"] == "vp-b"
        assert sp.loc["tg-0", "min_rtt_ms"] == 1.0

    def test_a_tie_goes_to_the_first_row(self, tmp_path):
        """Eval-source's rule: stable sort on RTT, then the first row."""
        obs = {0: [("tg-0", "vp-b", 3.0, 1.0), ("tg-0", "vp-a", 3.0, 1.0),
                   ("tg-1", "vp-c", 1.0, 1.0)], 1: BASE[1]}
        run, inputs = _tree(tmp_path, obs)
        assert B.shortest_ping(run, inputs).set_index("tg_id").loc["tg-0", "sping_vp_id"] == "vp-b"

    def test_min_inflation_skips_colocated_pairs(self, tmp_path):
        run, inputs = _tree(tmp_path, BASE)
        sp = B.shortest_ping(run, inputs).set_index("tg_id")
        # tg-1's colocated vp-c has no inflation; its other pair (vp-a) does.
        assert np.isfinite(sp.loc["tg-1", "min_inflation"])
        only_colocated = {0: [("tg-1", "vp-c", 1.0, 1.0), ("tg-0", "vp-a", 2.0, 1.0)], 1: BASE[1]}
        run, inputs = _tree(tmp_path / "b", only_colocated)
        assert np.isnan(B.shortest_ping(run, inputs).set_index("tg_id").loc["tg-1", "min_inflation"])


class TestRunnerWeightFilter:
    """`pair_weight_min` is applied by the runner after the parquet is read."""

    def test_it_drops_light_observations(self, tmp_path):
        obs = {0: [("tg-0", "vp-a", 2.0, 0.1), ("tg-0", "vp-b", 5.0, 0.9),
                   ("tg-1", "vp-c", 1.0, 0.9)], 1: BASE[1]}
        run, inputs = _tree(tmp_path, obs, pair_weight_min=0.5)
        sp = B.shortest_ping(run, inputs).set_index("tg_id")
        assert sp.loc["tg-0", "sping_vp_id"] == "vp-b"

    def test_a_tg_left_empty_has_no_baseline(self, tmp_path):
        obs = {0: [("tg-0", "vp-a", 2.0, 0.1), ("tg-1", "vp-c", 1.0, 0.9)],
               1: [(t, v, r, 0.9) for t, v, r, _ in BASE[1]]}
        run, inputs = _tree(tmp_path, obs, pair_weight_min=0.5)
        assert "tg-0" not in set(B.shortest_ping(run, inputs)["tg_id"])
        roster = C.load_method_frame(run, "alpha")
        frame = C.load_shortest_ping_frame(run, roster, inputs_root=inputs)
        assert frame.attrs["n_evaluated_without_baseline"] == 1
        assert frame.set_index("tg_id").loc["tg-0", ["pred_lat", "pred_lon"]].isna().all()

    def test_combos_that_disagree_are_refused(self, tmp_path):
        run, inputs = _tree(tmp_path, BASE, pair_weight_min={"alpha": None, "beta": 0.5})
        with pytest.raises(ValueError, match="different pair_weight_min"):
            B.shortest_ping(run, inputs)


class TestTheClassifyFrame:
    def test_joined_on_tg_and_fold(self, tmp_path):
        run, inputs = _tree(tmp_path, BASE)
        roster = C.load_method_frame(run, "alpha")
        frame = C.load_shortest_ping_frame(run, roster, inputs_root=inputs).set_index("tg_id")
        assert frame.loc["tg-3", ["pred_lat", "pred_lon"]].tolist() == list(VPS["vp-c"])
        assert (frame["status"] == "BASELINE").all()

    def test_a_tg_in_two_folds_takes_its_own_folds_observations(self, tmp_path):
        """Keyed on (tg, fold): fold 1's tg-0 rows must not leak into fold 0."""
        obs = {0: BASE[0], 1: BASE[1] + [("tg-0", "vp-c", 0.5, 1.0)]}
        run, inputs = _tree(tmp_path, obs)
        sp = B.shortest_ping(run, inputs)
        got = sp.set_index(["tg_id", "fold"])["sping_vp_id"]
        assert got[("tg-0", 0)] == "vp-a" and got[("tg-0", 1)] == "vp-c"

    def test_missing_inputs_name_the_fold(self, tmp_path):
        run, inputs = _tree(tmp_path, BASE)
        (inputs / SOURCE / RUN / SETUP / "fold_1" / B.EVAL_OBSERVATIONS).unlink()
        with pytest.raises(MissingArtifactError, match="1 of 2 folds"):
            B.shortest_ping(run, inputs)


class TestVps:
    def test_union_over_folds_in_v5_names(self, tmp_path):
        run, inputs = _tree(tmp_path, BASE, vps_by_fold={1: {"vp-a": VPS["vp-a"], "vp-z": (1.0, 2.0)}})
        vps = B.load_vps(run, inputs)
        assert sorted(vps["vp_id"]) == ["vp-a", "vp-b", "vp-c", "vp-z"]
        assert {"vp_lat", "vp_lon", "vp_asn", "vp_country"} <= set(vps.columns)

    def test_a_vp_that_moves_between_folds_is_refused(self, tmp_path):
        run, inputs = _tree(tmp_path, BASE, vps_by_fold={1: {**VPS, "vp-a": (0.0, 0.0)}})
        with pytest.raises(ValueError, match="different coordinates"):
            B.load_vps(run, inputs)


class TestSourceCsvFromTheConfig:
    def _config(self, tmp_path, top=None, **source_kwargs):
        d = tmp_path / "configs"
        d.mkdir(exist_ok=True)
        (d / f"{RUN}.yaml").write_text(yaml.safe_dump({
            "run_id": RUN, **(top or {}), "benchmark": {"source_kwargs": source_kwargs},
        }))
        return RunPaths(run_id=RUN, root=tmp_path / "outputs", source=SOURCE, setup=SETUP)

    def test_csv_path(self, tmp_path):
        csv = tmp_path / "mesh.csv"
        csv.touch()
        assert E.resolve_source_csv(self._config(tmp_path, csv_path=str(csv))) == csv

    def test_precomputed_weighted_csv_wins(self, tmp_path):
        mesh, weighted = tmp_path / "mesh.csv", tmp_path / "weighted.csv"
        mesh.touch()
        weighted.touch()
        run = self._config(tmp_path, mesh_csv_path=str(mesh), weighted_csv_path=str(weighted))
        assert E.resolve_source_csv(run) == weighted

    def test_on_the_fly_weighting_is_refused(self, tmp_path):
        mesh = tmp_path / "mesh.csv"
        mesh.touch()
        run = self._config(tmp_path, top={"eval_kept_traffic_fraction": 0.95}, mesh_csv_path=str(mesh))
        with pytest.raises(E.MeshSupersetError):
            E.resolve_source_csv(run)

    def test_an_override_wins(self, tmp_path):
        given = tmp_path / "given.csv"
        given.touch()
        run = self._config(tmp_path, csv_path=str(tmp_path / "absent.csv"))
        assert E.resolve_source_csv(run, given) == given

    def test_a_named_csv_that_is_gone(self, tmp_path):
        run = self._config(tmp_path, csv_path=str(tmp_path / "absent.csv"))
        with pytest.raises(MissingArtifactError, match="does not exist"):
            E.resolve_source_csv(run)
