"""The TG coverage floor: `analysis.common.min_vps_per_tg`, scoring only."""

from __future__ import annotations

import pandas as pd
import pytest
import yaml

from scripts.analysis.v5.modules import coverage as COV
from scripts.analysis.v5.modules import labels as L
from scripts.analysis.v5.modules import pni_gap as P
from scripts.analysis.v5.tests.conftest import PNI_VPS


def _thin(edge_csv, tg: str, keep: int) -> None:
    """Keep only the first `keep` VPs of `tg` in the edge CSV."""
    df = pd.read_csv(edge_csv)
    vps = list(PNI_VPS)[:keep]
    df = df[(df.target_id != tg) | df.vp_id.isin(vps)]
    df.to_csv(edge_csv, index=False)


# ---- the declaration ----------------------------------------------------------------


def _config(tmp_path, value) -> None:
    d = tmp_path / "configs"
    d.mkdir(exist_ok=True)
    body = {"run_id": "run-a", "analysis": {"common": {"min_vps_per_tg": value}}}
    (d / "run-a.yaml").write_text(yaml.safe_dump(body))


def test_an_undeclared_floor_is_none(tmp_path):
    assert L.declared_min_vps_per_tg("run-nothing") is None


def test_a_declared_floor_is_read(tmp_path):
    _config(tmp_path, 130)
    assert L.declared_min_vps_per_tg("run-a") == 130


@pytest.mark.parametrize("bad", [0, -5, 1.5, "130", True])
def test_a_bad_floor_is_refused(tmp_path, bad):
    _config(tmp_path, bad)
    with pytest.raises(ValueError, match="min_vps_per_tg"):
        L.declared_min_vps_per_tg("run-a")


def test_pro_paper_declares_130():
    group = L.load_group("pro-paper")
    assert group["analysis"]["common"]["min_vps_per_tg"] == 130


# ---- the drop ---------------------------------------------------------------------------


def test_no_floor_drops_nothing(pni_inputs):
    run, edge_csv, *_ = pni_inputs
    assert COV.low_coverage(run, None, source_csv=edge_csv).empty


def test_counts_are_distinct_vps(pni_inputs):
    run, edge_csv, *_ = pni_inputs
    n = COV.vp_counts(run, source_csv=edge_csv)
    assert (n == len(PNI_VPS)).all()


def test_the_floor_drops_exactly_the_thin_tg(pni_inputs):
    run, edge_csv, _, tg_group, _ = pni_inputs
    thin = sorted(tg_group)[0]
    _thin(edge_csv, thin, keep=2)
    low = COV.low_coverage(run, len(PNI_VPS), source_csv=edge_csv)
    assert low.to_dict("records") == [{"tg_id": thin, "n_vps": 2}]
    # A floor at the thin TG's own count keeps it: the test is `<`, not `<=`.
    assert COV.low_coverage(run, 2, source_csv=edge_csv).empty


def test_record_names_each_drop():
    low = pd.DataFrame({"tg_id": ["t1"], "n_vps": [26]})
    rec = COV.record(130, low)
    assert rec["floor"] == 130 and rec["n_tgs_dropped"] == 1
    assert rec["dropped"] == [{"tg_id": "t1", "n_vps": 26}]
    assert COV.record(None, low.iloc[:0])["n_tgs_dropped"] == 0


def test_the_pni_population_drops_and_records(pni_inputs):
    run, edge_csv, pni_csv, tg_group, _ = pni_inputs
    thin = sorted(tg_group)[0]
    _thin(edge_csv, thin, keep=2)
    full, _ = P.run_population(run, pni_csv, source_csv=edge_csv)
    pop, record = P.run_population(run, pni_csv, source_csv=edge_csv, min_vps=len(PNI_VPS))
    assert thin in set(full.tg_id) and thin not in set(pop.tg_id)
    assert len(pop) == len(full) - 1
    assert record["n_tgs"] == len(pop)
    assert record[COV.RECORD_KEY]["dropped"] == [{"tg_id": thin, "n_vps": 2}]
