"""report-failures: the error bin, the seed each failure belongs to, the listing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from scripts.analysis.v5.modules import answer_space as A
from scripts.analysis.v5.modules import failures as F
from scripts.analysis.v5.modules.paths import ANSWER_SPACE_KIND, grid_slug

SEATTLE = (47.449, -122.309)
MIAMI = (25.7932, -80.29)
NSIDE = F.SOURCE_NSIDE


@dataclass(frozen=True)
class _Run:
    """The slice of `RunPaths` the report touches."""

    run_id: str
    root: Path
    source: str = "src"
    setup: str = "setup"

    @property
    def setup_dir(self) -> Path:
        return self.root / "bench" / self.run_id

    @property
    def fold_ids(self) -> list[str]:
        return sorted(p.name for p in self.setup_dir.glob("fold_*"))

    @property
    def combo_ids(self) -> list[str]:
        return sorted({p.parent.name for p in self.setup_dir.glob("fold_*/*/targets.parquet")})

    def combo_dir(self, combo_id, fold_id) -> Path:
        return self.setup_dir / fold_id / combo_id

    def analysis_dir(self, kind, *, root=None):
        d = (root or self.root) / self.run_id / kind
        d.mkdir(parents=True, exist_ok=True)
        return d

    def answer_space_dir(self, nside, *, root=None):
        d = self.analysis_dir(ANSWER_SPACE_KIND, root=root) / grid_slug(nside)
        d.mkdir(parents=True, exist_ok=True)
        return d


def _vp(ok: bool, error: str | None = None) -> dict:
    return {"vp_id": "vp", "success": ok, "error": error, "upper_km": 1.0, "lower_km": 0.0}


def _row(tg, status, *, n_obs, ltd, mtl_ok, mtl_error=None, ctr_error=None):
    return {
        "target_id": tg, "status": status, "error": mtl_error or ctr_error,
        "n_obs": n_obs, "n_ltd_success": sum(p["success"] for p in ltd),
        "ltd_predictions": ltd, "mtl_success": mtl_ok, "mtl_error": mtl_error,
        "n_mtl_participants": 0, "ctr_error": ctr_error,
    }


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Two sites, one SPO-like and one VAN-like combo, one row per bin."""
    root = tmp_path_factory.mktemp("v5-failures")
    r = _Run("syn-mesh", root)
    tgs = pd.DataFrame({
        "tg_id": ["tg-a", "tg-b", "tg-c", "tg-d", "tg-e"],
        "tg_lat": [SEATTLE[0]] * 3 + [MIAMI[0]] * 2,
        "tg_lon": [SEATTLE[1]] * 3 + [MIAMI[1]] * 2,
    })
    A.build_answer_space(tgs, nside=NSIDE, run_id=r.run_id).write(r.answer_space_dir(NSIDE))

    spo = [
        # Two VPs, both kept by the LTD: too few to begin with.
        _row("tg-a", "FALLBACK", n_obs=2, ltd=[_vp(True)] * 2, mtl_ok=False,
             mtl_error="INSUFFICIENT_CONSTRAINTS"),
        # Four VPs, three dropped by the band check: the method's doing.
        _row("tg-b", "FALLBACK", n_obs=4,
             ltd=[_vp(True), _vp(False, "DEGENERATE_REGION"), _vp(False, "DEGENERATE_REGION"),
                  _vp(False, "NUMERICAL_FAILURE")],
             mtl_ok=False, mtl_error="INSUFFICIENT_CONSTRAINTS"),
        _row("tg-c", "SUCCESS", n_obs=50, ltd=[_vp(True)] * 50, mtl_ok=True),
    ]
    van = [
        _row("tg-d", "FALLBACK", n_obs=90, ltd=[_vp(True)] * 90, mtl_ok=False,
             mtl_error="EMPTY_REGION"),
        _row("tg-e", "ERROR", n_obs=90, ltd=[_vp(True)] * 90, mtl_ok=True,
             ctr_error="INSUFFICIENT_DATA"),
    ]
    for combo, rows in (("spotter_cbg", spo), ("vanilla_cbg", van)):
        d = r.combo_dir(combo, "fold_0")
        d.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(d / "targets.parquet", index=False)
    return r


@pytest.fixture(scope="module")
def built(run):
    return F.build(run, nside=NSIDE, analysis_root=run.root)


def test_only_the_failed_rows_are_listed(built):
    tgs, _ = built
    assert sorted(tgs["tg_id"]) == ["tg-a", "tg-b", "tg-d", "tg-e"]
    assert list(tgs.columns) == F.TGS_COLUMNS


def test_the_bins_separate_few_vps_from_ltd_drops(built):
    tgs, _ = built
    bins = dict(zip(tgs["tg_id"], tgs["error_bin"]))
    assert bins["tg-a"] == "MTL:insufficient_constraints / few VPs"
    assert bins["tg-b"] == "MTL:insufficient_constraints / LTD dropped DEGENERATE_REGION"
    assert bins["tg-d"] == "MTL:empty_region"
    assert bins["tg-e"] == "CTR:insufficient_data"
    drops = tgs.set_index("tg_id").loc["tg-b", "ltd_drops"]
    assert drops == "degenerate_region:2;numerical_failure:1"


def test_each_failure_is_placed_at_its_own_seed(run, built):
    tgs, clusters = built
    space = A.load_answer_space(run.answer_space_dir(NSIDE))
    want = space.tgs.set_index("tg_id")["tg_seed_id"]
    for r in tgs.itertuples():
        assert r.seed_id == want[r.tg_id]
    # Without a PNI list the place name is blank, not invented.
    assert tgs["nearest_pni"].isna().all()
    # A cluster is one (method, bin, seed); its ids are its TGs.
    assert clusters["n_tgs"].sum() == len(tgs)
    for c in clusters.itertuples():
        g = tgs[(tgs.method == c.method) & (tgs.error_bin == c.error_bin) & (tgs.seed_id == c.seed_id)]
        assert c.tg_ids.split(";") == sorted(g["tg_id"])


def test_methods_come_in_term_order(built):
    _, clusters = built
    # VAN before SPO in methods.TERM_ORDER, whatever order the tree lists them.
    assert list(dict.fromkeys(clusters["method_label"])) == ["VAN", "SPO"]


def test_the_listing_names_every_tg(built):
    _, clusters = built
    text = F.render_text(clusters)
    for tg in ("tg-a", "tg-b", "tg-d", "tg-e"):
        assert tg in text
    assert "VPs 2" in text


def test_a_run_with_no_failure_writes_empty_files(tmp_path):
    r = _Run("clean", tmp_path)
    A.build_answer_space(
        pd.DataFrame({"tg_id": ["tg-a"], "tg_lat": [SEATTLE[0]], "tg_lon": [SEATTLE[1]]}),
        nside=NSIDE, run_id="clean",
    ).write(r.answer_space_dir(NSIDE))
    d = r.combo_dir("vanilla_cbg", "fold_0")
    d.mkdir(parents=True)
    pd.DataFrame([_row("tg-a", "SUCCESS", n_obs=9, ltd=[_vp(True)] * 9, mtl_ok=True)]).to_parquet(
        d / "targets.parquet", index=False
    )
    paths, text = F.build_for_run(r, nside=NSIDE, analysis_root=tmp_path)
    assert text == "no failed TG"
    assert all(p.exists() for p in paths)
    body = json.loads(paths[1].read_text())
    assert body["n_failed"] == 0 and body["methods"] == {}


def test_the_json_nests_method_bin_seed(built):
    tgs, clusters = built
    body = F.to_json(clusters, "syn-mesh")
    assert list(body["methods"]) == ["vanilla_cbg", "spotter_cbg"]
    assert body["n_failed"] == len(tgs)
    spo = body["methods"]["spotter_cbg"]
    assert spo["label"] == "SPO" and spo["n_failed"] == 2
    few = spo["bins"]["MTL:insufficient_constraints / few VPs"]
    assert few["n_tgs"] == 1
    assert few["sites"][0]["tg_ids"] == ["tg-a"]
    assert few["sites"][0]["n_obs"] == [2, 2]
    assert few["sites"][0]["nearest_pni"] is None
    json.dumps(body)  # every value is plain JSON


def test_the_map_counts_failures_per_seed(run, built):
    tgs, _ = built
    van = F.failed_by_seed(tgs, "vanilla_cbg")
    assert int(van.sum()) == 2 and len(van) == 1  # both VAN failures at Miami
    assert F.failed_by_seed(tgs, "no_such_combo").empty


def test_the_map_labels_each_failed_seed(run, built):
    import cartopy.crs as ccrs

    tgs, _ = built
    space = A.load_answer_space(run.answer_space_dir(NSIDE))
    polygons = F.CL.cell_polygons(space.seeds, F.MAP_EXTENT)
    counts = F.failed_by_seed(tgs, "spotter_cbg")
    fig = F.plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    drawn = F.draw_panel(ax, space.seeds, polygons, frozenset(), counts, title="t")
    labels = sorted(t.get_text() for t in ax.texts)
    F.plt.close(fig)
    assert drawn == len(counts)
    assert labels == sorted(str(int(n)) for n in counts)


def test_build_for_run_writes_the_map(run):
    paths, _ = F.build_for_run(run, nside=NSIDE, analysis_root=run.root)
    assert [p.name for p in paths] == [F.TGS_CSV, F.SUMMARY_JSON, F.MAP_PNG]
    assert paths[2].stat().st_size > 0


def test_the_map_draws_the_interconnects(run, built, tmp_path):
    import cartopy.crs as ccrs

    tgs, _ = built
    pni_csv = tmp_path / "pni.csv"
    pd.DataFrame({"pni_id": ["pni-sea", "pni-mia"], "pni_lat": [47.6, 25.8],
                  "pni_lon": [-122.3, -80.2]}).to_csv(pni_csv, index=False)
    space = A.load_answer_space(run.answer_space_dir(NSIDE))
    polygons = F.CL.cell_polygons(space.seeds, F.MAP_EXTENT)
    fig = F.plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    F.draw_panel(ax, space.seeds, polygons, frozenset(), F.failed_by_seed(tgs, "vanilla_cbg"),
                 title="t", pnis=F.P.load_pnis(pni_csv))
    pni = [c for c in ax.collections if c.get_zorder() == F._Z_PNI]
    F.plt.close(fig)
    assert len(pni) == 1 and len(pni[0].get_offsets()) == 2
