"""The correctness UpSet: which methods put each TG in its own cell.

A FALLBACK row is not an answer even when classify labels it correct; a TG
every method gets wrong is a real `(none)` combination; the set sizes are the
methods' accuracies; small combinations share one lumped column.
"""

from __future__ import annotations

import json

import matplotlib.image as mpimg
import pandas as pd
import pytest

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import figure_champion_upset as U
from scripts.analysis.v5.modules import figure_correct_upset as K
from scripts.analysis.v5.modules.paths import RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING

OCT = "octant_cbg_hull"
VAN = "vanilla_cbg"


def _tgs(labels, *, status=None, offsets=None, first_id: int = 0) -> pd.DataFrame:
    n = len(labels)
    return pd.DataFrame({
        "tg_id": [f"tg-{first_id + i}" for i in range(n)],
        "tg_lat": [40.0 + i for i in range(n)],
        "tg_lon": [-100.0] * n,
        "status": list(status) if status is not None else ["SUCCESS"] * n,
        "cell_label": list(labels),
        C.GRID_OFFSET: list(offsets) if offsets is not None else [0] * n,
    })


def _write_run(root, run_id: str, frames: dict[str, pd.DataFrame]) -> RunPaths:
    run = RunPaths(run_id=run_id, root=root, source="s", setup="t")
    out = run.classify_dir(K.SOURCE_NSIDE, root=root)
    for method, df in frames.items():
        df.to_parquet(out / C.TGS_PARQUET.format(method=method), index=False)
    return run


@pytest.fixture
def run(tmp_path):
    sp = _tgs(["correct", "correct", "wrong", "wrong"])
    sp["status"] = "BASELINE"
    return _write_run(tmp_path, "r1", {
        SHORTEST_PING: sp,
        VAN: _tgs(["correct", "correct", "wrong", "wrong"],
                  status=["SUCCESS", "FALLBACK", "SUCCESS", "SUCCESS"]),
        OCT: _tgs(["correct", "wrong", "correct", "wrong"], offsets=[0, 0, 5, 0]),
    })


def test_fallback_is_not_correct_and_none_is_a_combination(run, tmp_path):
    mask, _ = K.correct_matrix(run, analysis_root=tmp_path)
    assert mask.loc["tg-1"].to_dict() == {SHORTEST_PING: True, VAN: False, OCT: False}
    combos = U.combination(mask)
    assert combos.loc["tg-3"] == U.NONE_LABEL
    assert combos.loc["tg-0"] == "S-P+VAN+OCT-H"


def test_bounded_drops_far_correct_cells(run, tmp_path):
    mask, _ = K.correct_matrix(run, "bounded", analysis_root=tmp_path)
    assert not mask.loc["tg-2", OCT]


def test_set_share_is_accuracy_and_tables_partition(run, tmp_path):
    mask, sites = K.correct_matrix(run, analysis_root=tmp_path)
    sets = K.set_table(mask, sites).set_index("method")
    assert sets.loc[SHORTEST_PING, "share"] == 0.5
    assert sets.loc[VAN, "n_correct"] == 1
    assert sets.loc[OCT, "n_only"] == 1
    inter = U.intersection_table(mask, sites)
    assert inter["n_tgs"].sum() == len(mask)


def test_pooled_artifacts_are_written(tmp_path, run):
    other = _write_run(tmp_path, "r2", {
        m: _tgs(["correct"], first_id=10) for m in (SHORTEST_PING, VAN, OCT)
    })
    written = K.build_for_runs([run, other], layouts=(K.POOLED,), analysis_root=tmp_path)
    paths = written[0]
    assert paths["png"].name == "correct_upset.pooled.cell.png"
    assert mpimg.imread(paths["png"]).size > 0
    manifest = json.loads(paths["manifest"].read_text())
    assert manifest["n_tgs"] == 5 and manifest["share_none"] == 0.2


def test_lumping_needs_two_small_combinations(tmp_path):
    # Five distinct patterns, each 20%: none below 25% is kept alone.
    mask = pd.DataFrame({
        SHORTEST_PING: [True, False, True, False, True],
        OCT: [True, True, False, False, False],
    }, index=[f"t{i}" for i in range(5)])
    out = U.plot_upset(mask, tmp_path / "u.png", title=None, subtitle=None, min_share=0.25)
    assert out.exists()


def test_unknown_metric_is_refused():
    with pytest.raises(ValueError):
        K.validate_metric("nearest")
