"""Why each S-P prediction lands where it does: a cause per TG, per side of the has-X split.

`report-sp-pni-cells` shows that S-P answers the cell of its interconnect, so
S-P is correct on a has-X TG and wrong on a no-X TG. This module gives every
TG a **cause**, first match wins, so the deviations from that expectation
are counted instead of read off by hand:

| Code | Cause | Test (on a wrong TG) |
|---|---|---|
| 0 | correct | `classify` `cell_label == correct` on a solved row |
| A | RTT ignores distance | per-TG Spearman rho(RTT, d(VP, TG)) over its VPs < `LOW_RHO` |
| B | X_in != X, proven | some VP measures below the propagation floor through X, and some other interconnect fits every VP |
| C | near-tie | the VPs within `TIE_MS` of the lowest RTT span more than one cell, and S-P is not in X's cell |
| D | X_in != X, by fit | S-P is not in X's cell; the interconnect whose path best fits the RTTs is not X, and S-P is in its cell |
| E | expected no-X | S-P is in X's cell, which is not the TG's |
| F | residual | anything else |

X is the interconnect nearest the TG, chosen without S-P (as in
`sp_pni_cells`). "Fits every VP" means no VP's RTT is below
`d(VP, I) + d(I, TG)` at `FLOOR_KM_PER_MS`. A TG where *no* listed
interconnect fits is flagged `no_interconnect_fits` (a list gap or a VP
location, not an ingress), whatever its cause.

## A is one test, and the report checks it is not a knife-edge

A used to need a large excess delay as well. On the meshes rho alone picks
the same TGs: every correct has-X TG has rho >= 0.81 and every A TG <= 0.08.
`rho_band` reports the largest rho of an A TG and the smallest of a correct
has-X TG, and `rho_band.separated` says whether the threshold sits in an
empty band. Excess delays in ms are not reported (absolute RTTs are
confidential).

## The RTT signal on each side

`rtt_signal` gives, per side, the median ratio of the RTT measured by the VP
nearest the TG to the TG's lowest RTT, the nearest VP being any VP at the
nearest VP location (VPs share sites): 1 where S-P answers there, above 1
where a VP farther away (at the interconnect) beats it. Pooled under `all`
(its share at 1 equals the zero-gap share of `plot-sp-interconnect`). Ratios only.

## Units and privacy

Shares are per TG with site counts alongside (~20 replicas per site). The CSV
and report carry no coordinate, VP id, interconnect id or site key.

Command: `report-sp-taxonomy`. Needs `classify` and `plot-pni-gap` (via
`sp_pni_cells.load_runs`). Writes `sp_taxonomy_tgs.csv` and
`sp_taxonomy.report.json` beside the clusters.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import pni_gap as P
from scripts.analysis.v5.modules import sites as S
from scripts.analysis.v5.modules import sp_pni_cells as SPC
from scripts.analysis.v5.modules.figure_sp_interconnect import FLOOR_KM_PER_MS, LOW_RHO, TIE_MS, load_edges
from scripts.analysis.v5.modules.geodesy import haversine_km
from scripts.analysis.v5.modules.map_answer_space import load_rung
from scripts.analysis.v5.modules.paths import RunPaths

CSV_NAME = "sp_taxonomy_tgs.csv"
REPORT_NAME = "sp_taxonomy.report.json"
#: The CDF of per-TG rho, has-X and no-X (pooled layouts only): §5's Fig. 9b, drawn in
#: the hues and height of `figure_x_cell_rtt`'s pooled min-RTT boxes it sits beside.
RHO_PNG_NAME = "sp_rho_by_side.png"
RHO_CDF_WIDTH_IN = 1.45  # with HEIGHT_IN, prints at the height of the min-RTT boxes

CAUSES = {
    "0": "correct",
    "A": "RTT ignores distance",
    "B": "X_in != X, proven",
    "C": "near-tie",
    "D": "X_in != X, by fit",
    "E": "expected no-X",
    "F": "residual",
}
SIDES = ((True, "has-X"), (False, "no-X"))


# -- per-TG features --------------------------------------------------------------


def _ranks(a: np.ndarray) -> np.ndarray:
    """Average ranks down axis 0 (ties share their mean rank), as Spearman needs."""
    a = np.asarray(a, float)
    if a.ndim == 1:
        return pd.Series(a).rank().to_numpy()
    return pd.DataFrame(a).rank().to_numpy()


def _spearman_cols(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman rho of each column of `x` [n, k] against `y` [n]; NaN where a column is constant."""
    rx, ry = _ranks(x), _ranks(y)
    rx = rx - rx.mean(0)
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum(0) * (ry ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, (rx * ry[:, None]).sum(0) / den, np.nan)


def edge_features(edges: pd.DataFrame, tgs: pd.DataFrame, seeds: pd.DataFrame,
                  pnis: pd.DataFrame) -> pd.DataFrame:
    """Per TG, over every VP that measured it: rho, the tie's cells, path feasibility, the best-fit interconnect."""
    pnis = pnis.reset_index(drop=True)
    vps = edges.drop_duplicates("vp_id")
    vp_seed = dict(zip(vps.vp_id, SPC.nearest_seed(vps.vp_lat, vps.vp_lon, seeds)[0]))
    pni_seed = SPC.interconnect_cells(pnis, seeds).seed_id.to_numpy()
    plat, plon = pnis.pni_lat.to_numpy(), pnis.pni_lon.to_numpy()
    keep = set(tgs.tg_id)

    rows = []
    for tg_id, x in edges[edges.tg_id.isin(keep)].groupby("tg_id", sort=True):
        rtt = x.rtt_ms.to_numpy()
        lo = rtt.min()
        vi = haversine_km(x.vp_lat.to_numpy()[:, None], x.vp_lon.to_numpy()[:, None], plat[None, :], plon[None, :])
        ti = haversine_km(x.tg_lat.to_numpy()[:1, None], x.tg_lon.to_numpy()[:1, None],
                          plat[None, :], plon[None, :])[0]
        path = vi + ti[None, :]
        feasible = (rtt[:, None] >= path / FLOOR_KM_PER_MS).all(0)
        rho_path = _spearman_cols(path, rtt)
        xi = int(ti.argmin())
        best = int(np.nanargmax(rho_path)) if np.isfinite(rho_path).any() else xi
        tied = rtt <= lo + TIE_MS
        # The nearest VP *location*: several VPs can share a site, and any of
        # them counts (the same tolerance `sp_pni_cells` uses for a coordinate).
        d = x.d_km.to_numpy()
        at_nearest = d <= d.min() + SPC.D_SP_TOL_KM
        rows.append({
            "tg_id": tg_id,
            "n_vps": int(len(x)),
            "rho": float(_spearman_cols(x.d_km.to_numpy()[:, None], rtt)[0]) if len(x) > 2 else np.nan,
            "tied_cells": int(pd.Series(x.vp_id[tied].map(vp_seed)).nunique()),
            "x_feasible": bool(feasible[xi]),
            "n_feasible": int(feasible.sum()),
            "best_fit_is_x": best == xi,
            "best_fit_seed_id": pni_seed[best],
            "nearest_over_lowest_rtt": float(rtt[at_nearest].min() / lo) if lo > 0 else np.nan,
            # For the routing block of the report only (km / ms are never written).
            "rtt_nearest_ms": float(rtt[at_nearest].min()),
            "d_nearest_km": float(d.min()),
        })
    return pd.DataFrame(rows)


# -- the cause rule ---------------------------------------------------------------


def cause(r) -> str:
    """The first cause that applies to one TG row (see the module table)."""
    if r.correct:
        return "0"
    if r.rho < LOW_RHO:
        return "A"
    if not r.x_feasible and r.n_feasible > 0:
        return "B"
    if r.tied_cells > 1 and not r.pred_in_x_cell:
        return "C"
    if not r.pred_in_x_cell and not r.best_fit_is_x and r.pred_seed_id == r.best_fit_seed_id:
        return "D"
    if r.pred_in_x_cell:
        return "E"
    return "F"


def classify_causes(tgs: pd.DataFrame) -> pd.DataFrame:
    out = tgs.copy()
    out["side"] = np.where(out.tg_cell_holds_x, "has-X", "no-X")
    out["cause"] = [cause(r) for r in out.itertuples(index=False)]
    out["no_interconnect_fits"] = out.n_feasible == 0
    return out


# -- the report -------------------------------------------------------------------


def _pct(n: int, d: int) -> float:
    return 100.0 * n / d if d else float("nan")


def _table(g: pd.DataFrame, n_all: int) -> dict:
    n_side = len(g)
    out = {}
    for code, label in CAUSES.items():
        b = g[g.cause == code]
        out[code] = {"cause": label, "n_tgs": int(len(b)), "n_sites": int(b[S.SITE_KEY_COL].nunique()),
                     "side_pct": _pct(len(b), n_side), "all_pct": _pct(len(b), n_all),
                     "runs": sorted(b.run_id.unique())}
    return out


def report(tgs: pd.DataFrame, meta: dict) -> dict:
    n = len(tgs)
    sides = {}
    for flag, name in SIDES:
        g = tgs[tgs.tg_cell_holds_x == flag]
        sides[name] = {"n_tgs": int(len(g)), "n_sites": int(g[S.SITE_KEY_COL].nunique()),
                       "tgs_pct": _pct(len(g), n), "causes": _table(g, n)}
    by_run = {}
    for rid, g in tgs.groupby("run_id"):
        by_run[rid] = {name: _table(g[g.tg_cell_holds_x == flag], len(g)) for flag, name in SIDES}

    ok = tgs[tgs.correct & tgs.tg_cell_holds_x]
    a = tgs[tgs.cause == "A"]
    band = {
        "low_rho": LOW_RHO,
        "a_rho_max": float(a.rho.max()) if len(a) else None,
        "correct_has_x_rho_min": float(ok.rho.min()) if len(ok) else None,
    }
    band["separated"] = bool(band["a_rho_max"] is None or band["correct_has_x_rho_min"] is None
                             or band["a_rho_max"] < LOW_RHO <= band["correct_has_x_rho_min"])

    signal = {}
    for flag, name in (*SIDES, (None, "all")):
        g = tgs if flag is None else tgs[tgs.tg_cell_holds_x == flag]
        signal[name] = {
            "nearest_over_lowest_rtt_median": float(g.nearest_over_lowest_rtt.median()),
            "nearest_vp_is_lowest_pct": _pct(int((g.nearest_over_lowest_rtt <= 1.0).sum()), len(g)),
        }

    # Direct vs indirect routing, per side: how well each TG's RTTs track VP distance
    # (percentiles of the per-TG rho), and whether the nearest VP's RTT follows the path
    # through X or its own distance to the TG (Spearman across TGs; `d_tg_x_km` is
    # `sp_pni_cells`' distance from the TG to its nearest interconnect).
    from scipy.stats import spearmanr

    routing = {}
    for flag, name in SIDES:
        g = tgs[tgs.tg_cell_holds_x == flag]
        routing[name] = {
            "per_tg_rho_p10": float(g.rho.quantile(0.1)), "per_tg_rho_p50": float(g.rho.median()),
            "nearest_vp_rho_rtt_vs_path_via_x": float(spearmanr(g.rtt_nearest_ms, g.d_nearest_km + g.d_tg_x_km).statistic),
            "nearest_vp_rho_rtt_vs_own_distance": float(spearmanr(g.rtt_nearest_ms, g.d_nearest_km).statistic),
        }

    # Where S-P misses the cell of X (the TG's nearest interconnect), by cause:
    # the deviations from "S-P answers X's cell" that §4.2 accounts for.
    miss = tgs[~tgs.pred_in_x_cell.astype(bool)]
    by = miss.cause.astype(str).value_counts()
    x_in_ne_x = int(by.get("B", 0) + by.get("D", 0))
    misses_x_cell = {
        "unit": "TG, % of all TGs",
        "pct": _pct(len(miss), n),
        "by_cause_pct": {c: _pct(int(by.get(c, 0)), n) for c in CAUSES if by.get(c, 0)},
        "x_in_ne_x_pct": _pct(x_in_ne_x, n),                         # B + D
        "rtt_not_discriminative_pct": _pct(int(by.get("A", 0) + by.get("C", 0)), n),  # A + C
        "proven_share_of_x_in_ne_x_pct": _pct(int(by.get("B", 0)), x_in_ne_x),
        "in_x_cell_causes": sorted(tgs[tgs.pred_in_x_cell.astype(bool)].cause.astype(str).unique()),
    }

    return {
        "layout": meta.get("layout"),
        "run_ids": meta["run_ids"],
        "constants": {"low_rho": LOW_RHO, "tie_ms": TIE_MS, "floor_km_per_ms": FLOOR_KM_PER_MS},
        "misses_x_cell": misses_x_cell,
        "causes": CAUSES,
        "population": {"unit": "TG", "n_tgs": int(n), "n_sites": int(tgs[S.SITE_KEY_COL].nunique())},
        "sides": sides,
        "by_run": by_run,
        "rho_band": band,
        "rtt_signal": {"unit": "TG; RTT of the VP nearest the TG over the TG's lowest RTT", **signal},
        "routing": {"unit": "TG; per-TG Spearman rho(RTT, VP distance) percentiles, and Spearman across "
                            "TGs of the nearest VP's RTT against d(VP,X)+d(X,TG) and against d(VP,TG)",
                    **routing},
        "no_interconnect_fits": {
            "unit": "TG", "n_tgs": int(tgs.no_interconnect_fits.sum()),
            "pct": _pct(int(tgs.no_interconnect_fits.sum()), n),
            "by_cause": {c: int(b.no_interconnect_fits.sum()) for c, b in tgs.groupby("cause")},
        },
    }


# -- the whole step ---------------------------------------------------------------


def load_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layout: str = P.PER_RUN,
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict, Path]:
    """`(tgs with causes, cluster_meta, out_dir)`, every run checked against the clusters."""
    tgs, _, meta, out_dir = SPC.load_runs(runs, pni_csvs, layout=layout, analysis_root=analysis_root,
                                          source_csvs=source_csvs)
    records = {r["run_id"]: r for r in meta.get("runs", [])}
    feats = []
    for run in runs:
        csv = P.checked_source_csv(run, pni_csvs[run.run_id], records.get(run.run_id, {}),
                                   (source_csvs or {}).get(run.run_id))
        seeds = load_rung(run, SPC.SOURCE_NSIDE, analysis_root=analysis_root).seeds
        f = edge_features(load_edges(csv), tgs[tgs.run_id == run.run_id], seeds,
                          P.load_pnis(pni_csvs[run.run_id]))
        feats.append(f.assign(run_id=run.run_id))
    tgs = tgs.merge(pd.concat(feats, ignore_index=True), on=["run_id", "tg_id"], how="left", validate="1:1")
    if tgs.n_vps.isna().any():
        raise ValueError("a TG in `classify` has no edge in the clusters' edge CSV")
    return classify_causes(tgs), meta, out_dir


#: Per-TG columns written to the CSV. No coordinate, VP, interconnect id or site key.
CSV_COLUMNS = ("run_id", "tg_id", P.POINT_COL, P.CLUSTER_COL, "side", "correct", "cause",
               "rho", "tied_cells", "x_feasible", "n_feasible", "best_fit_is_x",
               "pred_in_x_cell", "nearest_over_lowest_rtt", "no_interconnect_fits", "n_vps")


def plot_rho_by_side(tgs: pd.DataFrame, out_png: Path) -> Path:
    """CDF over TGs of the per-TG Spearman rho(RTT, VP distance), one curve per side.

    §5's Fig. 9b, beside `figure_x_cell_rtt`'s pooled min-RTT boxes: same height and
    hues. The legend sits top left; the sides' shares are in Table 1. Every percentile
    the text quotes (median, p10) reads straight off the curves.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from scripts.analysis.v5.modules import figure_x_cell_rtt as XR

    fig, ax = plt.subplots(figsize=(RHO_CDF_WIDTH_IN, XR.HEIGHT_IN))
    for flag, tick, _ in XR.GROUPS:
        v = np.sort(tgs.loc[tgs.tg_cell_holds_x == flag, "rho"].dropna().to_numpy(float))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where="post", color=XR.HUES[flag], lw=1.1, label=tick)
    ax.set_xlim(min(-0.05, float(np.nanmin(tgs.rho)) - 0.05), 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel(r"per-TG Spearman's $\rho$", fontsize=6.5)
    ax.set_ylabel("CDF over TGs", fontsize=6.5)
    ax.legend(loc="upper left", fontsize=6, frameon=False, handlelength=1.2, borderaxespad=0.2)
    ax.tick_params(axis="both", labelsize=6, length=2, pad=1.5)
    ax.grid(alpha=0.25, lw=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_png


def _write(tgs: pd.DataFrame, meta: dict, out_dir: Path) -> Path:
    tgs.sort_values(["run_id", "tg_id"])[list(CSV_COLUMNS)].to_csv(out_dir / CSV_NAME, index=False)
    if len(meta["run_ids"]) > 1:
        plot_rho_by_side(tgs, out_dir / RHO_PNG_NAME)
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(report(tgs, meta), indent=2) + "\n")
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
