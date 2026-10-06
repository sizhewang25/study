"""The datasets' shape, as the methodology section states it -- `report-dataset`.

Every other v5 command answers a question about the methods. This one answers
the questions the methodology asks about the data before any method runs: how
many VPs measured each network, and whether they are the same fleet; how many
replicas a site holds; how many sites share a seed; how the K-fold and LOSO
folds are laid out; and the footprint span D against the largest error any
method made. Each is otherwise a sum or a comparison done by hand across the
runs' answer-space metas, classify parquets and edge CSVs.

## Per run, then pooled

One row per `--run-id` and one `all` row over them. Counts are written beside
shares: the paper prints shares only, the artifact keeps both so a share can
be checked.

* **VPs.** Distinct `vp_id`s in the run's edge CSV, the fewest and median VPs
  measuring a TG, and the TGs measured by fewer than all of them
  (`partial_mesh`). The manifest says whether every run's VP set is identical.
* **Sites.** TGs per site (min / median / max); sites merged into a shared
  seed, from the answer space's `n_sites_merged`.
* **Filter removal (estimated).** No pre-filter CSV survives (the final CSVs
  are reconstructions), so how many TGs the SOI sanity filter removed cannot
  be counted. `est_removed_*` assumes every site started at
  `--replicas-per-site` replicas (default 20, the collection's design) and
  counts the shortfall. An estimate under that assumption, and named so.
* **Folds.** K-fold count and sizes from the classify parquets' `fold`, and the
  share of sites whose replicas span every fold. With `--pair BASE:LOSO`, the
  LOSO twin's fold count, whether it equals the site count, and the most sites
  any fold holds out (1 for leave-one-site-out).
* **D.** The configs' `analysis.common.dist_norm_km` max, and the largest
  `pred_dist_to_tg_km` over every scored method (both runs of a pair), so
  "no error exceeds D" is a number.

No coordinate, VP id or site key is written.

Command: `report-dataset`. Needs `build-answer-space` and `classify` on every
run. Writes `_cross/dataset/<n>-runs-<hash>/dataset.{csv,manifest.json}`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import edges as EDGES
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules import sites as SITES
from scripts.analysis.v5.modules.figure_sp_interconnect import load_edges
from scripts.analysis.v5.modules.labels import dataset_label
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths

KIND = "dataset"
CSV_NAME = "dataset.csv"
MANIFEST_NAME = "dataset.manifest.json"
ALL = "all"

SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: Replicas a site was collected with (10 IPv4 + 10 IPv6). The estimate of
#: what the filter removed counts each site's shortfall from this.
REPLICAS_PER_SITE = 20


def _parquets(run: RunPaths, nside: int, root: Path | None) -> list[Path]:
    found = sorted(run.classify_dir(nside, root=root).glob(C.TGS_PARQUET.format(method="*")))
    if not found:
        raise MissingArtifactError(
            f"{run.classify_dir(nside, root=root)} holds no *_tgs.parquet; "
            f"run `classify --run-id {run.run_id}` first"
        )
    return found


def max_error_km(run: RunPaths, nside: int, root: Path | None) -> float:
    """The largest prediction-to-TG distance over every scored method of a run."""
    return float(max(
        pd.read_parquet(p, columns=["pred_dist_to_tg_km"])["pred_dist_to_tg_km"].max()
        for p in _parquets(run, nside, root)
    ))


def fold_layout(run: RunPaths, nside: int, root: Path | None) -> dict:
    """Fold count, sizes, sites per fold and sites spanning every fold, off one parquet."""
    df = pd.read_parquet(_parquets(run, nside, root)[0], columns=["tg_id", "tg_lat", "tg_lon", "fold"])
    df["site"] = SITES.site_key(df, run_id=run.run_id)
    sizes = df.groupby("fold").size()
    folds_per_site = df.groupby("site")["fold"].nunique()
    sites_per_fold = df.groupby("fold")["site"].nunique()
    return {
        "n_folds": int(len(sizes)),
        "fold_size_min": int(sizes.min()),
        "fold_size_max": int(sizes.max()),
        "sites_per_fold_max": int(sites_per_fold.max()),
        "n_sites_all_folds": int((folds_per_site == len(sizes)).sum()),
    }


def run_row(
    run: RunPaths, *, nside: int, root: Path | None, cap: int,
    loso: RunPaths | None = None, dist_norm_km: tuple[float, float] | None = None,
) -> tuple[dict, frozenset]:
    """One run's row, and its VP id set (kept out of the row)."""
    meta = json.loads((run.answer_space_dir(nside, root=root) / "meta.json").read_text())
    sites = pd.read_csv(run.answer_space_dir(nside, root=root) / "sites.csv")
    edges = load_edges(EDGES.resolve_source_csv(run))
    vps = frozenset(edges["vp_id"].astype(str))
    per_tg = edges.groupby("tg_id")["vp_id"].nunique()
    shortfall = (cap - sites["n_tgs"]).clip(lower=0)
    removed = int(shortfall.sum())
    folds = fold_layout(run, nside, root)
    row = {
        "run_id": run.run_id,
        "dataset": dataset_label(run.run_id),
        "n_tgs": int(meta["n_tgs"]),
        "n_sites": int(meta["n_sites"]),
        "replicas_min": int(sites["n_tgs"].min()),
        "replicas_median": float(sites["n_tgs"].median()),
        "replicas_max": int(sites["n_tgs"].max()),
        "n_seeds": int(meta["n_seeds"]),
        "n_sites_merged": int(meta["n_sites_merged"]),
        "n_vps": int(len(vps)),
        "vps_per_tg_min": int(per_tg.min()),
        "vps_per_tg_median": float(per_tg.median()),
        "n_tgs_partial_mesh": int((per_tg < len(vps)).sum()),
        "replicas_cap": cap,
        "n_sites_below_cap": int((shortfall > 0).sum()),
        "est_removed_tgs": removed,
        "est_tgs_before_filter": int(meta["n_tgs"]) + removed,
        **{f"kfold_{k}": v for k, v in folds.items()},
        "dist_norm_max_km": float(dist_norm_km[1]) if dist_norm_km else np.nan,
        "max_error_km": max_error_km(run, nside, root),
    }
    if loso is not None:
        lf = fold_layout(loso, nside, root)
        row.update({
            "loso_run_id": loso.run_id,
            "loso_n_folds": lf["n_folds"],
            "loso_sites_per_fold_max": lf["sites_per_fold_max"],
            "loso_folds_equal_sites": lf["n_folds"] == row["n_sites"],
        })
        row["max_error_km"] = max(row["max_error_km"], max_error_km(loso, nside, root))
    return row, vps


def add_shares(t: pd.DataFrame) -> pd.DataFrame:
    """Shares of every count, in percent: what the paper prints."""
    t = t.copy()
    t["sites_merged_pct"] = 100 * t["n_sites_merged"] / t["n_sites"]
    t["partial_mesh_pct"] = 100 * t["n_tgs_partial_mesh"] / t["n_tgs"]
    t["est_removed_pct"] = 100 * t["est_removed_tgs"] / t["est_tgs_before_filter"]
    t["kfold_sites_all_folds_pct"] = 100 * t["kfold_n_sites_all_folds"] / t["n_sites"]
    t["max_error_over_d"] = t["max_error_km"] / t["dist_norm_max_km"]
    return t


def pooled_row(rows: list[dict]) -> dict:
    """The `all` row: sums of counts, extremes of extremes, medians left out."""
    t = pd.DataFrame(rows)
    out = {"run_id": ALL, "dataset": ALL}
    for col in ("n_tgs", "n_sites", "n_seeds", "n_sites_merged", "n_tgs_partial_mesh",
                "n_sites_below_cap", "est_removed_tgs", "est_tgs_before_filter",
                "kfold_n_sites_all_folds"):
        out[col] = int(t[col].sum())
    for col in ("replicas_min", "vps_per_tg_min", "kfold_fold_size_min"):
        out[col] = int(t[col].min())
    for col in ("replicas_max", "n_vps", "kfold_fold_size_max", "kfold_sites_per_fold_max",
                "max_error_km", "dist_norm_max_km"):
        out[col] = t[col].max()
    out["replicas_cap"] = int(t["replicas_cap"].iloc[0])
    if "loso_folds_equal_sites" in t:
        out["loso_folds_equal_sites"] = bool(t["loso_folds_equal_sites"].all())
        out["loso_sites_per_fold_max"] = int(t["loso_sites_per_fold_max"].max())
    return out


def _manifest(run_ids, loso_ids, vp_sets, cap, nside) -> str:
    sets = list(vp_sets.values())
    return json.dumps({
        "csv": CSV_NAME,
        "kind": KIND,
        "run_ids": run_ids,
        "loso_run_ids": loso_ids,
        "nside": nside,
        "vp_sets_identical": all(s == sets[0] for s in sets),
        "n_vps_union": len(frozenset().union(*sets)) if sets else 0,
        "replicas_cap": cap,
        "estimate_note": (
            "est_removed_* assumes every site was collected with replicas_cap replicas "
            "and the SOI sanity filter removed the shortfall. No pre-filter CSV survives, "
            "so this is an estimate under that assumption, not a count."
        ),
        "partial_mesh_note": (
            "n_tgs_partial_mesh: TGs measured by fewer than n_vps VPs -- the filter drops "
            "a TG's violating pairs as well as whole TGs, so the mesh is not full."
        ),
        "d_note": (
            "dist_norm_max_km is the configs' analysis.common.dist_norm_km max (D, "
            "confidential in the paper); max_error_km covers every scored method of the "
            "run and of its LOSO twin."
        ),
    }, indent=2) + "\n"


def build(
    runs: list[RunPaths], *, pairs: dict[str, RunPaths] | None = None,
    nside: int = SOURCE_NSIDE, analysis_root: Path | None = None,
    cap: int = REPLICAS_PER_SITE,
    dist_norm_km: dict[str, tuple[float, float] | None] | None = None,
) -> dict[str, Path]:
    """The table and manifest for these runs. `pairs` maps a run id to its LOSO twin."""
    if not runs:
        raise ValueError("pass at least one --run-id")
    nside = G.validate_nside(nside)
    pairs = pairs or {}
    unknown = sorted(set(pairs) - {r.run_id for r in runs})
    if unknown:
        raise ValueError(f"--pair names a base run not given as --run-id: {unknown}")
    rows, vp_sets = [], {}
    for run in runs:
        row, vps = run_row(run, nside=nside, root=analysis_root, cap=cap,
                           loso=pairs.get(run.run_id),
                           dist_norm_km=(dist_norm_km or {}).get(run.run_id))
        rows.append(row)
        vp_sets[run.run_id] = vps
    if len(rows) > 1:
        rows.append(pooled_row(rows))
    table = add_shares(pd.DataFrame(rows))
    run_ids = [r.run_id for r in runs]
    loso_ids = [pairs[r].run_id for r in run_ids if r in pairs]
    out_dir = cross.cross_dir(run_ids + loso_ids, analysis_root=analysis_root, kind=KIND)
    written = {"csv": out_dir / CSV_NAME, "manifest": out_dir / MANIFEST_NAME}
    table.to_csv(written["csv"], index=False)
    written["manifest"].write_text(_manifest(run_ids, loso_ids, vp_sets, cap, nside))
    return written
