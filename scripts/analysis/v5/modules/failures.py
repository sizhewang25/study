"""Where each method fails, and which TGs: every non-SUCCESS row, binned.

`classify` counts a method's failures (`n_failed`, the `(nF)` on the outcome
map); this lists them. One row per `(method, TG)` the method did not answer
itself -- FALLBACK or ERROR -- read straight off the benchmark's
`targets.parquet`, with the TG's site and seed from the answer space.

## The error bin

`model.CBGModel.geolocate` runs LTD, then MTL on the VPs whose LTD succeeded,
then CTR, and records the first stage that failed. LTD never fails a TG on
its own: a VP whose LTD fails is dropped, and too few left surface as an MTL
error. So the bin is `<phase>:<code>` for the stage that failed, and an
`insufficient_*` MTL error is split by whether the LTD dropped anyone:

* `MTL:insufficient_constraints / few VPs` -- the LTD kept every VP; the TG
  had too few to begin with. A property of the data.
* `MTL:insufficient_constraints / LTD dropped <code>` -- the LTD dropped VPs,
  named by its most common per-VP error. A property of the method.

No threshold is read from the method: the split only compares `n_ltd_ok`
against `n_obs`, so it holds for any MTL's minimum.

## Where on the map

`tg_seed_id` and the seed's coordinate, the same key the outcome map counts
on, plus the interconnect nearest the seed from `pni_csv` -- the CLI passes
the run's declared `analysis.common.pni_csv` (blank when none is declared). The interconnect is
a place name for reading the list, not a claim about routing.

Command: `report-failures`. Needs the benchmark and `build-answer-space`.
## The map

`failures.png`, one panel per method, on the US mainland frame: the run's
Voronoi cells, the no-data cells grey (`figure_outcome_map.no_data_seeds`, so
only a traffic-weighted run has any), every seed a small dot, and each seed
with failures red with its count of failed TGs, and the interconnects of
`pni_csv` as small orange triangles when one is given. A method with no failure
keeps its panel, so a clean method is visibly clean rather than missing.

Writes `<run>/failures/failures.tgs.csv` (one row per failed TG),
`failures.json` -- method, then bin, then seed, with the TG ids -- the
listing the command prints, nested, and `failures.png`.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.v5.modules import cells as CL  # noqa: E402
from scripts.analysis.v5.modules import grid as G  # noqa: E402
from scripts.analysis.v5.modules import mapping as M  # noqa: E402
from scripts.analysis.v5.modules import pni_gap as P  # noqa: E402
from scripts.analysis.v5.modules.figure_outcome_map import (  # noqa: E402
    LEADER_MIN_DEG,
    NO_DATA_FILL,
    no_data_seeds,
    place_labels,
)
from scripts.analysis.v5.modules.geodesy import pairwise_km  # noqa: E402
from scripts.analysis.v5.modules.map_answer_space import load_rung  # noqa: E402
from scripts.analysis.v5.modules.methods import method_label, method_order  # noqa: E402
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths  # noqa: E402

KIND = "failures"
TGS_CSV = "failures.tgs.csv"
SUMMARY_JSON = "failures.json"
MAP_PNG = "failures.png"

#: The map's frame. The failures are counted per seed and every seed sits at
#: a US site, so unlike the outcome map there is no stray prediction to fit.
MAP_EXTENT: tuple[float, float, float, float] = M.US_MAINLAND_EXTENT

#: A seed with failures. Red is the only hue on the map besides the grey.
FAIL_INK = "#c62828"
FAIL_AREA = 34.0
SEED_AREA = 7.0

#: An interconnect from `pni_csv`. Orange and a triangle, so it reads apart
#: from both the black seed dots and the red failure marks.
PNI_INK = "#f28e2b"
PNI_AREA = 28.0

#: Panel size and columns, as the outcome map's per-dataset column.
_PANEL_W, _PANEL_H, _MAX_COLS = 17.5 / 3, 3.3, 3
_Z_NO_DATA, _Z_CELLS, _Z_SEEDS, _Z_PNI, _Z_FAIL, _Z_LEADER, _Z_LABEL = 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0
_HALO = [pe.withStroke(linewidth=2.4, foreground="white")]

#: Which rung's answer space supplies the site and seed. v5 runs one.
SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: The statuses that are failures: the method did not answer itself.
FAILED_STATUSES = ("FALLBACK", "ERROR")

#: MTL errors that mean "too few constraints", which the bin splits by cause.
_INSUFFICIENT = ("INSUFFICIENT_CONSTRAINTS", "INSUFFICIENT_DATA")

_READ = (
    "target_id", "status", "error", "n_obs", "n_ltd_success", "ltd_predictions",
    "mtl_success", "mtl_error", "n_mtl_participants", "ctr_error",
)

#: What `targets.parquet` gives, then where the answer space places the TG.
ROW_COLUMNS = [
    "run_id", "method", "method_label", "fold", "tg_id", "status", "error_bin",
    "phase", "error", "n_obs", "n_ltd_ok", "ltd_drops", "n_mtl_participants",
]
PLACE_COLUMNS = ["site_id", "seed_id", "seed_lat", "seed_lon", "nearest_pni", "nearest_pni_km"]
TGS_COLUMNS = ROW_COLUMNS + PLACE_COLUMNS

CLUSTERS_COLUMNS = [
    "run_id", "method", "method_label", "error_bin", "n_bin_tgs", "seed_id",
    "seed_lat", "seed_lon", "nearest_pni", "nearest_pni_km", "n_tgs",
    "n_obs_min", "n_obs_max", "tg_ids",
]


def ltd_drops(predictions) -> Counter:
    """Per-VP LTD error codes of one TG, `{code: n_vps}`; successes left out."""
    if predictions is None:
        return Counter()
    return Counter(
        str(p.get("error") or "unknown") for p in predictions if not p.get("success")
    )


def error_bin(row) -> tuple[str, str]:
    """`(phase, bin)` for one failed row of `targets.parquet`; see the docstring."""
    if not bool(row["mtl_success"]):
        code = str(row["mtl_error"] or row["error"])
        if code in _INSUFFICIENT:
            drops = ltd_drops(row["ltd_predictions"])
            why = f"LTD dropped {drops.most_common(1)[0][0]}" if drops else "few VPs"
            return "MTL", f"MTL:{code.lower()} / {why}"
        return "MTL", f"MTL:{code.lower()}"
    code = str(row["ctr_error"] or row["error"])
    return "CTR", f"CTR:{code.lower()}"


def _format_drops(drops: Counter) -> str:
    return ";".join(f"{k.lower()}:{v}" for k, v in sorted(drops.items()))


def failed_rows(run: RunPaths) -> pd.DataFrame:
    """Every FALLBACK / ERROR row of every combo and fold, binned."""
    out = []
    for combo in run.combo_ids:
        for fold in run.fold_ids:
            path = run.combo_dir(combo, fold) / "targets.parquet"
            if not path.exists():
                continue
            t = pd.read_parquet(path, columns=list(_READ))
            t = t[t["status"].isin(FAILED_STATUSES)]
            for _, r in t.iterrows():
                phase, bin_ = error_bin(r)
                out.append({
                    "run_id": run.run_id, "method": combo,
                    "method_label": method_label(combo), "fold": fold,
                    "tg_id": str(r["target_id"]), "status": r["status"],
                    "error_bin": bin_, "phase": phase, "error": r["error"],
                    "n_obs": int(r["n_obs"]), "n_ltd_ok": int(r["n_ltd_success"]),
                    "ltd_drops": _format_drops(ltd_drops(r["ltd_predictions"])),
                    "n_mtl_participants": r["n_mtl_participants"],
                })
    return pd.DataFrame(out, columns=ROW_COLUMNS)


def _places(run: RunPaths, nside: int, analysis_root: Path | None,
            pni_csv: Path | None) -> pd.DataFrame:
    """`tg_id -> site, seed, seed coordinate, nearest interconnect`."""
    space = load_rung(run, nside, analysis_root=analysis_root)
    seeds = space.seeds[["seed_id", "seed_lat", "seed_lon"]].copy()
    seeds["nearest_pni"], seeds["nearest_pni_km"] = None, np.nan
    if pni_csv is not None:
        pnis = P.load_pnis(pni_csv)
        d = pairwise_km(seeds["seed_lat"], seeds["seed_lon"], pnis["pni_lat"], pnis["pni_lon"])
        seeds["nearest_pni"] = pnis["pni_id"].to_numpy()[d.argmin(axis=1)]
        seeds["nearest_pni_km"] = np.round(d.min(axis=1), 1)
    tgs = space.tgs[["tg_id", "site_id", "tg_seed_id"]].rename(columns={"tg_seed_id": "seed_id"})
    return tgs.merge(seeds, on="seed_id", how="left")


def build(run: RunPaths, *, nside: int = SOURCE_NSIDE, analysis_root: Path | None = None,
          pni_csv: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(tgs, clusters)`: one row per failed `(method, TG)`, and per `(method, bin, seed)`.

    `pni_csv` names the interconnects; `None` leaves `nearest_pni` blank.
    """
    rows = failed_rows(run)
    places = _places(run, G.validate_nside(nside), analysis_root, pni_csv)
    tgs = rows.merge(places, on="tg_id", how="left")
    unplaced = tgs["seed_id"].isna()
    if unplaced.any():
        raise MissingArtifactError(
            f"{run.run_id}: {int(unplaced.sum())} failed TG(s) are not in the answer space "
            f"(e.g. {sorted(tgs.loc[unplaced, 'tg_id'])[:3]}); rerun `build-answer-space`."
        )
    tgs["seed_id"] = tgs["seed_id"].astype("int64")
    order = {m: i for i, m in enumerate(method_order(tgs["method"].unique()))}
    tgs["_m"] = tgs["method"].map(order)
    tgs = tgs.sort_values(["_m", "error_bin", "seed_id", "tg_id"]).reset_index(drop=True)[TGS_COLUMNS]

    if tgs.empty:
        return tgs, pd.DataFrame(columns=CLUSTERS_COLUMNS)
    keys = ["run_id", "method", "method_label", "error_bin", "seed_id",
            "seed_lat", "seed_lon", "nearest_pni", "nearest_pni_km"]
    clusters = (
        tgs.groupby(keys, dropna=False, sort=False)
        .agg(n_tgs=("tg_id", "size"), n_obs_min=("n_obs", "min"),
             n_obs_max=("n_obs", "max"), tg_ids=("tg_id", lambda s: ";".join(sorted(s))))
        .reset_index()
    )
    clusters["n_bin_tgs"] = clusters.groupby(["method", "error_bin"])["n_tgs"].transform("sum")
    clusters["_m"] = clusters["method"].map(order)
    clusters = clusters.sort_values(
        ["_m", "n_bin_tgs", "error_bin", "n_tgs", "seed_id"],
        ascending=[True, False, True, False, True],
    )
    return tgs, clusters[CLUSTERS_COLUMNS].reset_index(drop=True)


def render_text(clusters: pd.DataFrame) -> str:
    """The clusters as an indented listing: method, then bin, then seed."""
    if clusters.empty:
        return "no failed TG"
    lines = []
    for (run_id, label), m in clusters.groupby(["run_id", "method_label"], sort=False):
        lines.append(f"{label}  ({run_id}) -- {int(m['n_tgs'].sum())} failed")
        for bin_, b in m.groupby("error_bin", sort=False):
            lines.append(f"  {bin_}: {int(b['n_tgs'].sum())}")
            for r in b.itertuples():
                where = (f"{r.nearest_pni}, {r.nearest_pni_km:.0f} km"
                         if isinstance(r.nearest_pni, str) else f"{r.seed_lat:.2f}, {r.seed_lon:.2f}")
                obs = (f"{r.n_obs_min}" if r.n_obs_min == r.n_obs_max
                       else f"{r.n_obs_min}-{r.n_obs_max}")
                lines.append(f"    seed {r.seed_id} ({where}) · {r.n_tgs} TGs · VPs {obs}: "
                             f"{r.tg_ids.replace(';', ' ')}")
    return "\n".join(lines)


def to_json(clusters: pd.DataFrame, run_id: str) -> dict:
    """The clusters nested as `{combo id: {label, bins: {bin: {sites: [...]}}}}`.

    Keyed by combo id and ordered as `build` sorts: methods in term order, bins
    and seeds by size. JSON objects keep their insertion order, so the file
    reads in the same order as the printed listing.
    """
    methods: dict = {}
    for method, m in clusters.groupby("method", sort=False):
        bins: dict = {}
        for bin_, b in m.groupby("error_bin", sort=False):
            bins[bin_] = {
                "n_tgs": int(b["n_tgs"].sum()),
                "sites": [
                    {
                        "seed_id": int(r.seed_id),
                        "seed_lat": round(float(r.seed_lat), 4),
                        "seed_lon": round(float(r.seed_lon), 4),
                        "nearest_pni": r.nearest_pni if isinstance(r.nearest_pni, str) else None,
                        "nearest_pni_km": (None if pd.isna(r.nearest_pni_km)
                                           else float(r.nearest_pni_km)),
                        "n_tgs": int(r.n_tgs),
                        "n_obs": [int(r.n_obs_min), int(r.n_obs_max)],
                        "tg_ids": r.tg_ids.split(";"),
                    }
                    for r in b.itertuples()
                ],
            }
        methods[method] = {
            "label": m["method_label"].iloc[0], "n_failed": int(m["n_tgs"].sum()), "bins": bins,
        }
    return {
        "run_id": run_id,
        "n_failed": int(clusters["n_tgs"].sum()) if len(clusters) else 0,
        "failed": "status FALLBACK or ERROR: the method did not answer itself",
        "n_obs": "[min, max] VPs the site's failed TGs had after any filtering",
        "methods": methods,
    }


def failed_by_seed(tgs: pd.DataFrame, method: str) -> pd.Series:
    """Failed TGs per seed for one method, `{seed_id: n}`, largest first."""
    sel = tgs[tgs["method"] == method]
    return sel.groupby("seed_id").size().sort_values(ascending=False).astype("int64")


def draw_panel(ax, seeds: pd.DataFrame, polygons: dict, no_data: frozenset[int],
               counts: pd.Series, *, title: str,
               extent: tuple[float, float, float, float] = MAP_EXTENT,
               pnis: pd.DataFrame | None = None) -> int:
    """One method's map; returns how many seeds are labelled."""
    import cartopy.crs as ccrs

    pc = ccrs.PlateCarree()
    ax.set_extent(extent, crs=pc)
    M.draw_basemap(ax)
    empty = [polygons[s] for s in sorted(no_data) if s in polygons]
    if empty:
        ax.add_geometries(empty, crs=pc, facecolor=NO_DATA_FILL, edgecolor="none",
                          zorder=_Z_NO_DATA)
    M.draw_cells(ax, polygons, zorder=_Z_CELLS)
    ax.scatter(seeds["seed_lon"], seeds["seed_lat"], s=SEED_AREA, c=M.INK, linewidths=0,
               transform=pc, zorder=_Z_SEEDS)
    if pnis is not None and len(pnis):
        ax.scatter(pnis["pni_lon"], pnis["pni_lat"], s=PNI_AREA, c=PNI_INK, marker="^",
                   edgecolors="white", linewidths=0.4, transform=pc, zorder=_Z_PNI)
    ax.set_title(title, fontsize=8.5)
    pos = seeds.set_index("seed_id")
    ids = [k for k in counts.index if k in pos.index]
    if not ids:
        return 0
    anchors = np.c_[pos.loc[ids, "seed_lon"].to_numpy(), pos.loc[ids, "seed_lat"].to_numpy()]
    ax.scatter(anchors[:, 0], anchors[:, 1], s=FAIL_AREA, c=FAIL_INK, edgecolors="white",
               linewidths=0.6, transform=pc, zorder=_Z_FAIL)
    for (x0, y0), (lx, ly), sid in zip(anchors, place_labels(anchors, extent), ids):
        if np.hypot(lx - x0, ly - y0) > LEADER_MIN_DEG:
            ax.plot([x0, lx], [y0, ly], lw=0.55, color=FAIL_INK, transform=pc, zorder=_Z_LEADER)
        ax.text(lx, ly, str(int(counts[sid])), fontsize=8, fontweight="bold", color=FAIL_INK,
                ha="center", va="center", transform=pc, zorder=_Z_LABEL, path_effects=_HALO)
    return len(ids)


def render_map(run: RunPaths, tgs: pd.DataFrame, out_png: Path, *, nside: int = SOURCE_NSIDE,
               analysis_root: Path | None = None,
               extent: tuple[float, float, float, float] = MAP_EXTENT,
               pni_csv: Path | None = None) -> Path:
    """`failures.png`: a panel per method the run scored, failed TGs per seed."""
    import cartopy.crs as ccrs

    space = load_rung(run, G.validate_nside(nside), analysis_root=analysis_root)
    polygons = CL.cell_polygons(space.seeds, extent)
    no_data = no_data_seeds(space)
    pnis = P.load_pnis(pni_csv) if pni_csv is not None else None
    methods = method_order(run.combo_ids)
    ncols = max(1, min(_MAX_COLS, len(methods)))
    nrows = max(1, math.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(_PANEL_W * ncols, _PANEL_H * nrows),
                             squeeze=False, subplot_kw={"projection": ccrs.PlateCarree()})
    for ax, method in zip(axes.flat, methods):
        counts = failed_by_seed(tgs, method)
        title = (f"{method_label(method)} · {int(counts.sum())} failed"
                 + (f" at {len(counts)} sites" if len(counts) else ""))
        draw_panel(ax, space.seeds, polygons, no_data, counts, title=title, extent=extent,
                   pnis=pnis)
    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    subject = (f"{run.run_id} · TGs a method did not answer itself (FALLBACK / ERROR) · "
               f"red dot = seed with failures, number = its failed TGs"
               + (" · orange ▲ = interconnect" if pnis is not None else ""))
    if no_data:
        subject += "\ngrey cell = no data (site has no TG after the traffic filter)"
    fig.suptitle(subject, fontsize=10)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return out_png


def build_for_run(run: RunPaths, *, nside: int = SOURCE_NSIDE, analysis_root: Path | None = None,
                  pni_csv: Path | None = None) -> tuple[list[Path], str]:
    """Write the per-TG CSV, the JSON and the map; return their paths and the text listing."""
    tgs, clusters = build(run, nside=nside, analysis_root=analysis_root, pni_csv=pni_csv)
    out_dir = run.analysis_dir(KIND, root=analysis_root)
    paths = [out_dir / TGS_CSV, out_dir / SUMMARY_JSON]
    tgs.to_csv(paths[0], index=False)
    paths[1].write_text(json.dumps(to_json(clusters, run.run_id), indent=2) + "\n")
    paths.append(render_map(run, tgs, out_dir / MAP_PNG, nside=nside,
                            analysis_root=analysis_root, pni_csv=pni_csv))
    return paths, render_text(clusters)
