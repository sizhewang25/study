"""Does S-P land in its interconnect's cell? The cell-accuracy half of the S-P subsection.

`plot-sp-interconnect` shows that the S-P VP sits at the interconnect the
target's traffic crosses. This module carries that onto the **cell axis**
`classify` scores: if S-P answers the interconnect, its `cell_label` is
`correct` exactly when the TG shares a cell with that interconnect. It writes
the numbers that test this, per `plot-pni-gap` cluster, into one report.

## Which cell an interconnect is in

The answer space's cells are the unbounded Voronoi cells of its seeds, and
`classify` puts a prediction in the cell of its nearest seed. An interconnect
gets the same rule: **its cell is its nearest seed's**. Seeds sit only at
target sites, so an interconnect in a metro with no TG falls in whichever
cell is nearest, however far. `cells.per_run` reports each list's
interconnect-to-seed distance so a loose one is visible (AS03's Dallas sits
367 km from its seed).

## Two interconnects per TG, and only one is a test

* **X**, the interconnect nearest the TG. Chosen without looking at S-P, so
  "S-P answers X's cell" is a real test. This is the paper's assumed `X`.
* **K**, the interconnect nearest the S-P VP. Chosen *from* the S-P VP, so
  "S-P answers K's cell" holds almost by construction. It is reported because
  it separates "S-P answers an interconnect" from "S-P answers the TG's
  nearest one", not as evidence. Do not headline it.

Landing in *any* interconnect's cell is reported too, against its baseline:
on the meshes about half the cells hold an interconnect, so a random VP lands
in one two times in three.

## The rule, and what it predicts

`rule_correct = (the TG's own cell holds X)`. If S-P answers X's cell, this is
S-P's `cell_label`. `rule` reports how often it agrees with the label, and
`rule_misses` lists every clusters point where it does not, with its distances
and nothing that names a place.

## The random-VP baseline

Per TG, the share of the VPs that measured it whose nearest seed is the
interconnect cell in question, read off the same edge CSV the clusters were
computed from. The pooled figure is the mean of those per-TG shares, as in
`plot-sp-interconnect`'s `random_vp_at_interconnect_pct`.

## Consistency with the clusters

The S-P coordinate comes from `classify`'s `shortest_ping_tgs.parquet`, since
its `cell_label` is what is being explained. Where its distance to the TG
differs from the clusters CSV's `d_sp_km` by more than `D_SP_TOL_KM`, the two
picked different VPs. That is allowed only for an **exact RTT tie**: the
`classify` VP must measure the TG's minimum RTT in the edge CSV, or the module
refuses. On the meshes 5 of 1,269 TGs tie this way (all at 19-64 ms, three of
them in C3), counted as `n_tgs_sp_tie_broken_differently`. The edge CSV and
PNI list go through `pni_gap.checked_source_csv` like every other consumer of
the clusters.

## Units

Every share is per TG and carries its site count, since ~20 replicas of a
site are one observation repeated. `rule_misses` is per clusters point.
No coordinate, VP id or interconnect id is written (ids name cities).

Command: `report-sp-pni-cells`. Needs `classify` and `plot-pni-gap`. Writes
`sp_pni_cells_tgs.csv` and `sp_pni_cells.report.json` beside the clusters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules import pni_gap as P
from scripts.analysis.v5.modules import sites as S
from scripts.analysis.v5.modules.figure_sp_interconnect import load_edges
from scripts.analysis.v5.modules.geodesy import elementwise_km, haversine_km, pairwise_km
from scripts.analysis.v5.modules.map_answer_space import load_rung
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING, solved_mask

CSV_NAME = "sp_pni_cells_tgs.csv"
REPORT_NAME = "sp_pni_cells.report.json"

#: The rung whose cells are scored. Read off the ladder, as `figure_outcome_map` does.
SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: `d(TG, S-P coordinate)` from `classify` against the clusters' `d_sp_km`.
#: The parquet stores coordinates to 6 decimals (~0.1 m); 10 m is slack for
#: that, and far below the distance between any two VPs. Also the radius in
#: which an edge-CSV VP is taken to be the `classify` S-P coordinate.
D_SP_TOL_KM = 0.01
#: Two RTTs equal to within this are a tie (the CSV carries 0.1-1 us digits).
RTT_TIE_MS = 1e-9

_SP_COLUMNS = ("tg_id", "tg_lat", "tg_lon", "pred_lat", "pred_lon", "status",
               "tg_seed_id", "pred_seed_id", "cell_label")

#: Per-TG flags, in report order. `random_*` are shares, the rest booleans.
FLAGS = ("correct", "pred_in_interconnect_cell", "pred_in_x_cell", "pred_in_k_cell",
         "tg_cell_holds_x", "tg_cell_holds_interconnect", "same_interconnect")
RANDOM = ("random_in_interconnect_cell", "random_in_x_cell", "random_in_own_cell")

DEFINITIONS = {
    "cell": "unbounded Voronoi cell of an answer-space seed; a point is in its nearest seed's cell",
    "interconnect_cell": "the cell of the interconnect's nearest seed",
    "x": "the interconnect nearest the TG, chosen without S-P (the paper's X)",
    "k": "the interconnect nearest the S-P VP; pred_in_k_cell is near-circular, not evidence",
    "correct": "classify's cell_label == correct on a solved row",
    "rule_correct": "the TG's own cell holds X",
    "random_*": "per TG, the share of its measuring VPs in that cell; pooled as the mean over TGs",
}


# -- frames -------------------------------------------------------------------


def nearest_seed(lat, lon, seeds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(seed_id, km)` of each point's nearest seed, by `classify`'s own rule."""
    d = pairwise_km(np.asarray(lat, float), np.asarray(lon, float),
                    seeds.seed_lat.to_numpy(), seeds.seed_lon.to_numpy())
    return seeds.seed_id.to_numpy()[d.argmin(axis=1)], d.min(axis=1)


def interconnect_cells(pnis: pd.DataFrame, seeds: pd.DataFrame) -> pd.DataFrame:
    """The interconnect list with each interconnect's cell and its distance to that seed."""
    seed, km = nearest_seed(pnis.pni_lat, pnis.pni_lon, seeds)
    return pnis.assign(seed_id=seed, d_seed_km=km)


def _nearest_interconnect(lat, lon, pnis: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(row index, km)` of each point's nearest interconnect, haversine like `pni_gap`."""
    d = haversine_km(np.asarray(lat, float)[:, None], np.asarray(lon, float)[:, None],
                     pnis.pni_lat.to_numpy()[None, :], pnis.pni_lon.to_numpy()[None, :])
    return d.argmin(axis=1), d.min(axis=1)


def load_sp_frame(run: RunPaths, nside: int = SOURCE_NSIDE, *,
                  analysis_root: Path | None = None) -> pd.DataFrame:
    path = run.classify_dir(nside, root=analysis_root) / C.TGS_PARQUET.format(method=SHORTEST_PING)
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `classify --run-id {run.run_id}` first")
    return pd.read_parquet(path, columns=list(_SP_COLUMNS))


def tg_frame(sp: pd.DataFrame, seeds: pd.DataFrame, pnis: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    """One row per TG: its cell, S-P's cell, X's and K's cells, and the flags."""
    cells = interconnect_cells(pnis, seeds)
    held = set(cells.seed_id)
    x, d_x = _nearest_interconnect(sp.tg_lat, sp.tg_lon, pnis)
    k, d_k = _nearest_interconnect(sp.pred_lat, sp.pred_lon, pnis)
    seed_of = cells.seed_id.to_numpy()
    out = pd.DataFrame({
        "run_id": run_id,
        "tg_id": sp.tg_id.to_numpy(),
        "tg_lat": sp.tg_lat.to_numpy(), "tg_lon": sp.tg_lon.to_numpy(),
        "pred_lat": sp.pred_lat.to_numpy(), "pred_lon": sp.pred_lon.to_numpy(),
        "tg_seed_id": sp.tg_seed_id.to_numpy(),
        "pred_seed_id": sp.pred_seed_id.to_numpy(),
        "x_seed_id": seed_of[x], "k_seed_id": seed_of[k],
        "d_tg_x_km": d_x, "d_sp_vp_k_km": d_k,
        "d_x_seed_km": cells.d_seed_km.to_numpy()[x],
        "d_sp_km_classify": elementwise_km(sp.tg_lat, sp.tg_lon, sp.pred_lat, sp.pred_lon),
    })
    out["correct"] = (sp.cell_label.eq("correct") & solved_mask(sp)).to_numpy()
    out["pred_in_interconnect_cell"] = out.pred_seed_id.isin(held)
    out["pred_in_x_cell"] = out.pred_seed_id.eq(out.x_seed_id)
    out["pred_in_k_cell"] = out.pred_seed_id.eq(out.k_seed_id)
    out["tg_cell_holds_x"] = out.tg_seed_id.eq(out.x_seed_id)
    out["tg_cell_holds_interconnect"] = out.tg_seed_id.isin(held)
    out["same_interconnect"] = x == k
    out["rule_correct"] = out.tg_cell_holds_x
    out[S.SITE_KEY_COL] = S.site_key(out, run_id=run_id)
    return out


def random_baseline(edges: pd.DataFrame, tgs: pd.DataFrame, seeds: pd.DataFrame,
                    pnis: pd.DataFrame) -> pd.DataFrame:
    """Per TG, the share of its measuring VPs in an interconnect cell, X's cell, its own cell."""
    vps = edges.drop_duplicates("vp_id")
    vp_seed = dict(zip(vps.vp_id, nearest_seed(vps.vp_lat, vps.vp_lon, seeds)[0]))
    held = set(interconnect_cells(pnis, seeds).seed_id)
    e = edges[["tg_id", "vp_id"]].merge(tgs[["tg_id", "x_seed_id", "tg_seed_id"]], on="tg_id",
                                        how="inner", validate="m:1")
    seed = e.vp_id.map(vp_seed)
    flags = pd.DataFrame({
        "tg_id": e.tg_id,
        "random_in_interconnect_cell": seed.isin(held),
        "random_in_x_cell": seed.eq(e.x_seed_id),
        "random_in_own_cell": seed.eq(e.tg_seed_id),
    })
    return flags.groupby("tg_id", as_index=False)[list(RANDOM)].mean()


def sp_rtt(edges: pd.DataFrame, tgs: pd.DataFrame) -> pd.DataFrame:
    """Per TG: the RTT of the VP at `classify`'s S-P coordinate, and the TG's minimum RTT.

    `sp_rtt_ms` is NaN where no measuring VP sits at that coordinate, which
    `check_against_clusters` refuses like any other mismatch.
    """
    e = edges[["tg_id", "vp_lat", "vp_lon", "rtt_ms"]].merge(
        tgs[["tg_id", "pred_lat", "pred_lon"]], on="tg_id", how="inner", validate="m:1")
    at = elementwise_km(e.vp_lat, e.vp_lon, e.pred_lat, e.pred_lon) <= D_SP_TOL_KM
    return pd.DataFrame({
        "sp_rtt_ms": e[at].groupby("tg_id").rtt_ms.min(),
        "min_rtt_ms": e.groupby("tg_id").rtt_ms.min(),
    }).rename_axis("tg_id").reset_index()


def check_against_clusters(tgs: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """Join the clusters; refuse a TG in only one side, or an S-P VP that is not a lowest-RTT VP.

    A `classify` S-P VP other than the clustered one is kept when it ties the
    minimum RTT exactly, and flagged `sp_tie_broken_differently`.
    """
    key = ["run_id", "tg_id"]
    cols = key + [P.CLUSTER_COL, P.POINT_COL, P.D_SP, P.D_PNI, P.GAP]
    out = tgs.merge(clusters[cols], on=key, how="outer", validate="1:1", indicator=True)
    lost = out[out._merge != "both"]
    if len(lost):
        raise ValueError(f"{len(lost)} TGs are in only one of `classify` and the clusters, e.g. "
                         f"{lost.tg_id.iloc[0]!r}; re-run `classify` and `plot-pni-gap` on the same runs.")
    differs = (out.d_sp_km_classify - out[P.D_SP]).abs() > D_SP_TOL_KM
    tied = out.sp_rtt_ms <= out.min_rtt_ms + RTT_TIE_MS  # False where sp_rtt_ms is NaN
    bad = differs & ~tied
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} TGs' S-P coordinate in `classify` is not a lowest-RTT VP the "
            f"clusters could have picked, e.g. {out.tg_id[bad].iloc[0]!r}. Re-run `classify` "
            f"or `plot-pni-gap` on the same edge CSV."
        )
    out["sp_tie_broken_differently"] = differs
    return out.drop(columns=["_merge", "d_sp_km_classify", "sp_rtt_ms", "min_rtt_ms"])


# -- the report ---------------------------------------------------------------


def _pct(mask) -> float:
    mask = np.asarray(mask, dtype=float)
    return 100.0 * float(mask.mean()) if mask.size else float("nan")


def block(g: pd.DataFrame) -> dict:
    """The per-TG shares for one group, with its TG and site counts."""
    out = {"n_tgs": int(len(g)), "n_sites": int(g[S.SITE_KEY_COL].nunique())}
    out.update({f"{c}_pct": _pct(g[c]) for c in FLAGS + RANDOM})
    return out


def contingency(g: pd.DataFrame) -> list[dict]:
    """S-P correct rate by (S-P in X's cell) x (the TG's cell holds X). Rows with no TGs are listed as 0."""
    rows = []
    for pred_in_x in (True, False):
        for holds_x in (True, False):
            cell = g[(g.pred_in_x_cell == pred_in_x) & (g.tg_cell_holds_x == holds_x)]
            rows.append({"pred_in_x_cell": pred_in_x, "tg_cell_holds_x": holds_x,
                         "n_tgs": int(len(cell)), "n_correct": int(cell.correct.sum()),
                         "correct_pct": _pct(cell.correct)})
    return rows


def rule(g: pd.DataFrame) -> dict:
    return {"n_tgs": int(len(g)), "actual_correct_pct": _pct(g.correct),
            "rule_correct_pct": _pct(g.rule_correct),
            "agree_pct": _pct(g.rule_correct == g.correct)}


def rule_misses(tgs: pd.DataFrame) -> list[dict]:
    """Every clusters point where the rule and the label disagree on at least one TG."""
    miss = tgs[tgs.rule_correct != tgs.correct]
    rows = []
    for (run_id, _), p in miss.groupby(["run_id", P.POINT_COL], sort=True):
        rows.append({
            "run_id": run_id, "cluster": int(p[P.CLUSTER_COL].iloc[0]), "n_tgs": int(len(p)),
            "n_correct": int(p.correct.sum()), "rule_correct": bool(p.rule_correct.iloc[0]),
            "d_tg_x_km": round(float(p.d_tg_x_km.iloc[0]), 1),
            "gap_km": round(float(p[P.GAP].median()), 1),
            "pred_in_x_cell_pct": _pct(p.pred_in_x_cell),
            "pred_in_interconnect_cell_pct": _pct(p.pred_in_interconnect_cell),
        })
    return rows


def report(tgs: pd.DataFrame, cell_meta: dict, meta: dict) -> dict:
    by_cluster = {str(int(c)): g for c, g in tgs.groupby(P.CLUSTER_COL)}
    by_run = dict(tuple(tgs.groupby("run_id")))
    return {
        "layout": meta.get("layout"),
        "run_ids": meta["run_ids"],
        "nside": SOURCE_NSIDE,
        "constants": {"d_sp_tol_km": D_SP_TOL_KM, "rtt_tie_ms": RTT_TIE_MS},
        "consistency": {
            "unit": "TG",
            "n_tgs_sp_tie_broken_differently": int(tgs.sp_tie_broken_differently.sum()),
            "note": "classify and plot-pni-gap picked different VPs at an exact minimum-RTT tie; "
                    "classify's is used, since its cell_label is the one explained",
        },
        "definitions": DEFINITIONS,
        "cells": {"unit": "cell; interconnect_to_seed_km per interconnect; vps per distinct VP",
                  "per_run": cell_meta},
        "shares": {
            "unit": "TG, % (n_sites alongside)",
            "all": block(tgs),
            "by_cluster": {c: block(g) for c, g in by_cluster.items()},
            "by_run": {r: block(g) for r, g in by_run.items()},
            "by_run_cluster": {r: {str(int(c)): block(b) for c, b in g.groupby(P.CLUSTER_COL)}
                               for r, g in by_run.items()},
        },
        "contingency": {
            "unit": "TG",
            "all": contingency(tgs),
            "by_cluster": {c: contingency(g) for c, g in by_cluster.items()},
        },
        "rule": {
            "unit": "TG",
            "rule": DEFINITIONS["rule_correct"],
            "all": rule(tgs),
            "by_cluster": {c: rule(g) for c, g in by_cluster.items()},
            "by_run": {r: rule(g) for r, g in by_run.items()},
        },
        "rule_misses": {"unit": "clusters point", "points": rule_misses(tgs)},
    }


# -- the whole step -----------------------------------------------------------


def cell_summary(seeds: pd.DataFrame, pnis: pd.DataFrame, edges: pd.DataFrame) -> dict:
    """How much of the answer space the interconnects hold, for one run."""
    cells = interconnect_cells(pnis, seeds)
    held = set(cells.seed_id)
    vps = edges.drop_duplicates("vp_id")
    vp_seed = nearest_seed(vps.vp_lat, vps.vp_lon, seeds)[0]
    return {
        "n_cells": int(len(seeds)),
        "n_interconnects": int(len(pnis)),
        "n_cells_holding_interconnect": int(len(held)),
        "interconnect_to_seed_km": {"median": float(cells.d_seed_km.median()),
                                    "max": float(cells.d_seed_km.max())},
        "n_vps": int(len(vps)),
        "vps_in_interconnect_cell_pct": _pct(pd.Series(vp_seed).isin(held)),
    }


def load_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layout: str = P.PER_RUN,
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict, dict, Path]:
    """`(tgs, cell_meta, cluster_meta, out_dir)`, every run checked against the clusters."""
    run_ids = [r.run_id for r in runs]
    if layout == P.POOLED:
        out_dir = P.pooled_output_dir(run_ids, analysis_root=analysis_root)
    elif len(runs) == 1:
        out_dir = P.output_dir(runs[0].run_id, pni_csvs[runs[0].run_id], analysis_root=analysis_root)
    else:
        raise ValueError(f"layout {layout!r} takes one run; got {len(runs)}")
    clusters, meta = P.read_clusters(out_dir, run_ids=run_ids)
    records = {r["run_id"]: r for r in meta.get("runs", [])}

    frames, cell_meta = [], {}
    for run in runs:
        csv = P.checked_source_csv(run, pni_csvs[run.run_id], records.get(run.run_id, {}),
                                   (source_csvs or {}).get(run.run_id))
        pnis = P.load_pnis(pni_csvs[run.run_id])
        seeds = load_rung(run, SOURCE_NSIDE, analysis_root=analysis_root).seeds
        edges = load_edges(csv)
        tgs = tg_frame(load_sp_frame(run, analysis_root=analysis_root), seeds, pnis, run_id=run.run_id)
        tgs = tgs.merge(random_baseline(edges, tgs, seeds, pnis), on="tg_id", how="left", validate="1:1")
        frames.append(tgs.merge(sp_rtt(edges, tgs), on="tg_id", how="left", validate="1:1"))
        cell_meta[run.run_id] = cell_summary(seeds, pnis, edges)
    tgs = check_against_clusters(pd.concat(frames, ignore_index=True), clusters)
    if tgs[list(RANDOM)].isna().any().any():
        raise ValueError("a TG in `classify` has no edge in the clusters' edge CSV")
    return tgs, cell_meta, meta, out_dir


#: Per-TG columns written to the CSV. No coordinate, no VP or interconnect id.
CSV_COLUMNS = ("run_id", "tg_id", P.POINT_COL, P.CLUSTER_COL, "tg_seed_id", "pred_seed_id",
               "x_seed_id", "k_seed_id", "d_tg_x_km", "d_sp_vp_k_km", "d_x_seed_km", P.GAP,
               *FLAGS, "rule_correct", *RANDOM, "sp_tie_broken_differently")


def _write(tgs, cell_meta, meta, out_dir) -> Path:
    tgs.sort_values(["run_id", "tg_id"])[list(CSV_COLUMNS)].to_csv(out_dir / CSV_NAME, index=False)
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(report(tgs, cell_meta, meta), indent=2) + "\n")
    return path


def build_for_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layouts: tuple[str, ...] = (P.PER_RUN,),
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> list[Path]:
    """The per-TG CSV and the report, beside each layout's clusters. Returns the reports."""
    bad = [lay for lay in layouts if lay not in P.LAYOUTS]
    if bad:
        raise ValueError(f"unknown layout {bad}; expected {list(P.LAYOUTS)}")
    written = []
    if P.PER_RUN in layouts:
        for run in runs:
            one = {run.run_id: source_csvs[run.run_id]} if source_csvs and run.run_id in source_csvs else None
            written.append(_write(*load_runs([run], pni_csvs, analysis_root=analysis_root, source_csvs=one)))
    if P.POOLED in layouts:
        written.append(_write(*load_runs(runs, pni_csvs, layout=P.POOLED, analysis_root=analysis_root,
                                         source_csvs=source_csvs)))
    return written
