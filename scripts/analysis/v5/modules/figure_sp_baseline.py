"""The baseline, explained without the nearest-interconnect proxy: four panels.

The paper's baseline section asks what sets Shortest Ping's error and its
region accuracy. The earlier figures answered through Δ_VP against the
distance to the TG's *nearest* interconnect, which assumes every TG is served
through that interconnect -- something the meshes never observe. These three
panels need only the TG's own cell: **has-X** means the cell holds an
interconnect location (`tg_cell_holds_interconnect`), a fact of the answer
space and the interconnect list alone, never which interconnect serves the TG.

* `sp_vp_scatter` / `sp_x_scatter` -- S-P's error (`d_sp`, the distance to
  the VP S-P returns) against two candidate explanations, one panel each, on
  the same axes: the distance to the TG's geographically nearest VP (`d_geo`,
  the conjecture that a close VP makes S-P accurate) and the distance to the
  TG's nearest interconnect (`d_x`). The `d_geo` points scatter far above the
  diagonal; the `d_x` points sit on it. `d_x` enters only as a predictor tested
  against the error, not as an assumption about which interconnect serves the TG.
* `sp_accuracy_by_x` -- S-P's region classification, has-X against no-X,
  pooled over the runs: correct or wrong cell, in the outcome bars' inks. The
  per-TG CSV and the manifest still split the wrong answers by where they
  land: the cell next to the TG's (Voronoi neighbours, `adjacent_seeds`) or
  farther.
* `sp_rtt_error_has_x` -- the has-X TGs only: each TG's minimum RTT against
  S-P's error, correct against wrong. It explains the has-X misses the bars
  show. A wrong TG whose minimum RTT is above every correct has-X TG's is
  outlined as *inflated RTT* (the RTT no longer tracks distance); the other
  wrong TGs as *near-tie* (a VP outside the cell wins by a fraction of a ms).

## Distances

`d_geo` and `d_sp` are read from the clusters file `plot-pni-gap` writes
(`pni_gap_clusters.csv`), through `sp_pni_cells.load_runs`, which refuses a
`d_sp` that disagrees with `classify`'s S-P coordinate. Normalized by the
runs' `analysis.common.dist_norm_km` (`dist_norm`), in units of 10^-3. The
clusters themselves are not used. Minimum RTTs come from the same edge CSV
(`sp_pni_cells.sp_rtt`), normalized by `analysis.common.rtt_norm_ms`.

## Replicas

~20 replicas share a site's geometry, so the distance scatters draw one marker
per distinct `(run, site, d_geo, d_sp)` with its area proportional to the TGs
it carries. Replicas differ slightly in RTT, so the RTT scatter draws one
marker per `(run, site, correct)` at its TGs' medians. The bars count TGs, as
the paper's accuracy does.

## Adjacency

Two seeds are neighbours when they share an edge of the spherical Delaunay
triangulation: a facet of the 3-D convex hull of the seeds' unit vectors whose
plane leaves the origin on its inner side. The other facets are the flat lid
that closes a set of points in one hemisphere; they would join peripheral
seeds across the far side of the globe, which the unbounded cells never do.
Every seed's nearest seed must come out adjacent (a Delaunay property); the
function refuses otherwise.

Command: `plot-sp-baseline`. Needs `classify` and `plot-pni-gap`. Writes
`sp_baseline.*` beside the clusters.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.v5.modules import dist_norm as DN  # noqa: E402
from scripts.analysis.v5.modules import figure_pni_cluster_rtt as R  # noqa: E402
from scripts.analysis.v5.modules import pni_gap as P  # noqa: E402
from scripts.analysis.v5.modules import sites as S  # noqa: E402
from scripts.analysis.v5.modules import sp_pni_cells as M  # noqa: E402
from scripts.analysis.v5.modules.figure_outcome_bars import CELL_HATCH, CELL_LEGEND, MODE_INK  # noqa: E402
from scripts.analysis.v5.modules.geodesy import unit_vectors  # noqa: E402
from scripts.analysis.v5.modules.map_answer_space import load_rung  # noqa: E402
from scripts.analysis.v5.modules.mapping import INK, INK_2  # noqa: E402
from scripts.analysis.v5.modules.methods import method_colors  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402
from scripts.analysis.v5.modules.status import SHORTEST_PING  # noqa: E402

SCATTER_VP = "sp_vp_scatter"
SCATTER_X = "sp_x_scatter"
ACCURACY = "sp_accuracy_by_x"
RTT_SCATTER = "sp_rtt_error_has_x"
FIGURES = (SCATTER_VP, SCATTER_X, ACCURACY, RTT_SCATTER)
CSV_NAME = "sp_baseline.tgs.csv"
MANIFEST_NAME = "sp_baseline.manifest.json"

HAS_X = "tg_cell_holds_interconnect"
D_GEO = "d_geo_norm_e3"
D_X = "d_x_norm_e3"
D_SP = "d_sp_norm_e3"
D_RTT = "min_rtt_norm"
WHERE = "where"
#: `WHERE` values, in stack order bottom to top.
CORRECT, ADJACENT, FARTHER = "correct", "wrong_adjacent", "wrong_farther"
WHERE_ORDER = (CORRECT, ADJACENT, FARTHER)
#: The bars' two outcomes, bottom to top: the outcome bars' correct and wrong cell.
OUTCOMES = ("correct", "wrong")
#: Why a has-X TG is wrong, in the RTT scatter's outlines.
INFLATED, NEAR_TIE = "inflated RTT", "near-tie"

#: `(flag, tick label, legend label)`, has-X first, as in `figure_x_cell_rtt`.
GROUPS = ((True, "has-$X$", "has-X"), (False, "no-$X$", "no-X"))

#: Every panel of the four-panel two-column row, so they print at one size.
NARROW_SIZE = (1.75, 1.4)
LABEL_PT, TICK_PT, LEGEND_PT, NOTE_PT = 6.5, 6.0, 5.5, 5.0
#: Scatter marker area per TG, in pt^2 (a site of 20 replicas -> 24 pt^2).
AREA_PER_TG = 1.2
#: The two scatter series: (column, legend term, marker, hue). Both take S-P's
#: colour (hue None), so the panels differ only in their x axis.
SCATTER_SERIES = {SCATTER_VP: (D_GEO, "VP", "o", None), SCATTER_X: (D_X, "$X$", "o", None)}
#: Scatter axes, in 10^-3 of the footprint span: both distances fit, and
#: the diagonal runs corner to corner.
SCATTER_LIM = (0.1, 1000.0)
#: The RTT scatter's normalized min-RTT axis, log.
RTT_LIM = (0.005, 1.0)
#: An outline's margin around its markers, as a factor on each log axis.
OUTLINE_PAD = 1.7
#: The conjecture test's thresholds, reported in the manifest.
CLOSE_VP_QUANTILE = 0.5
FAR_FACTOR = 10.0
#: Bar segments below this share are left unlabelled.
LABEL_MIN_PCT = 12.0
BAR_WIDTH = 0.55


# -- adjacency ------------------------------------------------------------------


def adjacent_seeds(seeds: pd.DataFrame) -> set[tuple[int, int]]:
    """Unordered seed-id pairs that are Voronoi neighbours (see the module docstring)."""
    from scipy.spatial import ConvexHull

    ids = seeds["seed_id"].to_numpy()
    if len(ids) < 4:
        return {(int(a), int(b)) for i, a in enumerate(ids) for b in ids[i + 1:]}
    u = unit_vectors(seeds["seed_lat"].to_numpy(float), seeds["seed_lon"].to_numpy(float))
    hull = ConvexHull(u)
    pairs: set[tuple[int, int]] = set()
    for simplex, eq in zip(hull.simplices, hull.equations):
        if eq[-1] >= 0:  # the origin is outside this facet's half-space: the lid
            continue
        for i in range(3):
            a, b = int(ids[simplex[i]]), int(ids[simplex[(i + 1) % 3]])
            pairs.add((min(a, b), max(a, b)))
    # Delaunay property: every seed's nearest seed is a neighbour.
    d = u @ u.T
    np.fill_diagonal(d, -np.inf)
    for i, j in enumerate(np.argmax(d, axis=1)):
        a, b = int(ids[i]), int(ids[j])
        if (min(a, b), max(a, b)) not in pairs:
            raise ValueError(f"seed {a}'s nearest seed {b} is not adjacent: the triangulation is wrong")
    return pairs


# -- frames ---------------------------------------------------------------------


def load_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    bounds: DN.Bounds,
    rtt_bounds: R.Bounds,
    layout: str = P.POOLED,
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict, Path]:
    """`(tgs, meta, out_dir)`: one row per TG with has-X, both distances, and where S-P landed."""
    tgs, _, meta, out_dir = M.load_runs(runs, pni_csvs, layout=layout, analysis_root=analysis_root,
                                        source_csvs=source_csvs)
    # `tgs` already carries `d_sp_km`, checked against `classify`; only `d_geo` comes from here.
    clusters, _ = P.read_clusters(out_dir, run_ids=[r.run_id for r in runs])
    key = ["run_id", "tg_id"]
    out = tgs.merge(clusters[key + [P.D_GEO]], on=key, how="left", validate="1:1")
    if out[[P.D_GEO, P.D_SP]].isna().any().any():
        raise ValueError("a TG has an S-P flag but no distances in the clusters file; re-run `plot-pni-gap`")
    if (out[P.D_SP] < out[P.D_GEO] - 1e-9).any():
        raise ValueError("d_sp < d_geo for some TG: the smallest-RTT VP cannot be closer than the closest VP")
    out[D_GEO] = DN.distance(out[P.D_GEO], bounds, what=P.D_GEO)
    out[D_SP] = DN.distance(out[P.D_SP], bounds, what=P.D_SP)
    out[D_X] = DN.distance(out["d_tg_x_km"], bounds, what="d_tg_x_km")
    out[D_RTT] = R.to_norm(out["min_rtt_ms"], rtt_bounds)

    adj = {r.run_id: adjacent_seeds(load_rung(r, M.SOURCE_NSIDE, analysis_root=analysis_root).seeds)
           for r in runs}
    pair = [(min(int(a), int(b)), max(int(a), int(b))) for a, b in zip(out.tg_seed_id, out.pred_seed_id)]
    is_adj = np.array([p in adj[r] for r, p in zip(out.run_id, pair)])
    out[WHERE] = np.where(out["correct"], CORRECT, np.where(is_adj, ADJACENT, FARTHER))
    out[HAS_X] = out[HAS_X].astype(bool)
    return out, meta, out_dir


def scatter_points(tgs: pd.DataFrame, x_col: str) -> pd.DataFrame:
    """One row per distinct `(run, site, x, d_sp)`, with the TGs it carries."""
    t = tgs.assign(_x=tgs[x_col].round(6), _s=tgs[D_SP].round(6))
    return (t.groupby(["run_id", S.SITE_KEY_COL, "_x", "_s"], as_index=False)
            .size().rename(columns={"_x": x_col, "_s": D_SP, "size": "n_tgs"}))


def has_x_misses(tgs: pd.DataFrame) -> pd.Series:
    """Per wrong has-X TG: `INFLATED` above every correct has-X TG's min RTT, else `NEAR_TIE`."""
    has = tgs[tgs[HAS_X]]
    ceiling = has.loc[has["correct"], D_RTT].max()
    wrong = has[~has["correct"]]
    return pd.Series(np.where(wrong[D_RTT] > ceiling, INFLATED, NEAR_TIE), index=wrong.index)


def rtt_points(tgs: pd.DataFrame) -> pd.DataFrame:
    """has-X only: one row per `(run, site, correct)` at its TGs' median min RTT and error."""
    has = tgs[tgs[HAS_X]]
    return (has.groupby(["run_id", S.SITE_KEY_COL, "correct"], as_index=False)
            .agg(**{D_RTT: (D_RTT, "median"), D_SP: (D_SP, "median"), "n_tgs": ("tg_id", "size")}))


def accuracy_table(tgs: pd.DataFrame, run_ids: list[str]) -> pd.DataFrame:
    """Per `(run, side)` plus pooled `all` rows: TG counts and shares of each `WHERE`."""
    rows = []
    for scope, block in [(r, tgs[tgs.run_id == r]) for r in run_ids] + [("all", tgs)]:
        for flag, _, name in GROUPS:
            g = block[block[HAS_X] == flag]
            n = len(g)
            row = {"run_id": scope, "side": name, "n_tgs": n,
                   "tgs_pct": 100.0 * n / len(block) if len(block) else float("nan")}
            for w in WHERE_ORDER:
                row[f"{w}_pct"] = 100.0 * float((g[WHERE] == w).sum()) / n if n else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


# -- figures --------------------------------------------------------------------


def _axes_style(ax) -> None:
    ax.tick_params(labelsize=TICK_PT, colors=INK, length=2, width=0.6, pad=1.5)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, out_png: Path) -> Path:
    fig.tight_layout(pad=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_png


def plot_scatter(tgs: pd.DataFrame, figure: str, out_png: Path) -> Path:
    """One series on the shared log-log axes, with its Spearman rho in the empty lower right."""
    col, term, marker, hue = SCATTER_SERIES[figure]
    fig, ax = plt.subplots(figsize=NARROW_SIZE)
    lo, hi = SCATTER_LIM
    ax.plot([lo, hi], [lo, hi], color=INK_2, lw=0.6, ls=":", zorder=1)
    pts = scatter_points(tgs, col)
    rho = tgs[col].corr(tgs[D_SP], method="spearman")
    ax.scatter(pts[col].clip(lower=lo), pts[D_SP], s=pts["n_tgs"] * AREA_PER_TG, marker=marker,
               color=hue or method_colors([SHORTEST_PING])[SHORTEST_PING], alpha=0.55, linewidths=0,
               zorder=2, clip_on=False)
    # Below the diagonal on the right: d_sp >= d_geo leaves it empty of VP points.
    ax.text(0.97, 0.05, rf"$\rho={rho:.2f}$", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=LEGEND_PT + 0.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    fmt = ticker.FuncFormatter(lambda v, _: DN.power_label(v))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel(rf"Dist. to nearest {term} ($\times10^{{-3}}$)", fontsize=LABEL_PT, color=INK)
    ax.set_ylabel("S-P error distance\n" + r"($\times10^{-3}$)", fontsize=LABEL_PT, color=INK)
    _axes_style(ax)
    return _save(fig, out_png)


def plot_accuracy(table: pd.DataFrame, out_png: Path) -> Path:
    """Pooled has-X and no-X bars: correct below, wrong (adjacent and farther together) on top."""
    fig, ax = plt.subplots(figsize=NARROW_SIZE)
    labels = []
    for x, (_, tick, name) in enumerate(GROUPS):
        r = table[(table.run_id == "all") & (table.side == name)].iloc[0]
        labels.append(f"{tick}\n{r.tgs_pct:.0f}%")
        shares = {"correct": float(r[f"{CORRECT}_pct"]),
                  "wrong": float(r[f"{ADJACENT}_pct"]) + float(r[f"{FARTHER}_pct"])}
        bottom = 0.0
        for o in OUTCOMES:
            h = shares[o] if r.n_tgs else 0.0
            ax.bar(x, h, bottom=bottom, width=BAR_WIDTH, color=MODE_INK[o], hatch=CELL_HATCH[o],
                   edgecolor="white" if CELL_HATCH[o] else MODE_INK[o], linewidth=0.0,
                   label=CELL_LEGEND[o] if x == 0 else None)
            if h >= LABEL_MIN_PCT:
                ax.text(x, bottom + h / 2, f"{h:.0f}", ha="center", va="center",
                        fontsize=NOTE_PT, color="white")
            bottom += h
    ax.set_xticks(range(len(GROUPS)), labels, fontsize=TICK_PT)
    ax.set_xlim(-0.6, len(GROUPS) - 0.4)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.set_ylabel("Share of TGs (%)", fontsize=LABEL_PT, color=INK)
    _axes_style(ax)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=LEGEND_PT,
              frameon=False, handlelength=1.0, handleheight=0.8, columnspacing=0.8, borderaxespad=0.0)
    return _save(fig, out_png)


def plot_rtt_scatter(tgs: pd.DataFrame, out_png: Path) -> Path:
    """has-X TGs: min RTT against S-P's error, correct and wrong, the misses outlined by cause."""
    fig, ax = plt.subplots(figsize=NARROW_SIZE)
    pts = rtt_points(tgs)
    for ok, o in ((True, "correct"), (False, "wrong")):
        g = pts[pts["correct"] == ok]
        ax.scatter(g[D_RTT], g[D_SP], s=g["n_tgs"] * AREA_PER_TG, color=MODE_INK[o], alpha=0.6,
                   linewidths=0, zorder=2 if ok else 3, clip_on=False)
    kinds = has_x_misses(tgs)
    for kind in (INFLATED, NEAR_TIE):
        w = tgs.loc[kinds.index[kinds == kind]]
        if w.empty:
            continue
        # The inflated TGs sit in the top-right corner, where the spines are
        # hidden, so their outline may run past the axes.
        x0, x1 = w[D_RTT].min() / OUTLINE_PAD, w[D_RTT].max() * OUTLINE_PAD
        y0, y1 = w[D_SP].min() / OUTLINE_PAD, w[D_SP].max() * OUTLINE_PAD
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls="--", lw=0.6,
                                   edgecolor=INK_2, zorder=4, clip_on=False))
        # Inflated sits at the right edge, so its label goes below the box; near-tie's to its right.
        if kind == INFLATED:
            ax.text(x1, y0 / 1.3, kind, ha="right", va="top", fontsize=NOTE_PT, color=INK)
        else:
            ax.text(x1 * 1.1, (y0 * y1) ** 0.5, kind, ha="left", va="center", fontsize=NOTE_PT, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*RTT_LIM)
    ax.set_ylim(*SCATTER_LIM)
    fmt = ticker.FuncFormatter(lambda v, _: DN.power_label(v))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("Normalized min RTT", fontsize=LABEL_PT, color=INK)
    ax.set_ylabel("S-P error distance\n" + r"($\times10^{-3}$)", fontsize=LABEL_PT, color=INK)
    # One fixed marker per outcome: the drawn markers' areas vary with their TGs.
    handles = [plt.Line2D([], [], ls="", marker="o", ms=3.5, color=MODE_INK[o], alpha=0.6, mew=0,
                          label=f"{o} cell") for o in OUTCOMES]
    # Lower right: no has-X TG has both a high min RTT and a small error.
    ax.legend(handles=handles, loc="lower right", fontsize=LEGEND_PT, frameon=False, handletextpad=0.2,
              borderaxespad=0.3)
    _axes_style(ax)
    return _save(fig, out_png)


# -- the whole step -------------------------------------------------------------


def _misses(tgs: pd.DataFrame) -> dict:
    has = tgs[tgs[HAS_X]]
    kinds = has_x_misses(tgs)
    out = {"rule": f"{INFLATED}: min RTT above every correct has-X TG's; else {NEAR_TIE}",
           "correct_min_rtt_norm_max": float(has.loc[has["correct"], D_RTT].max()),
           "correct_d_sp_max": float(has.loc[has["correct"], D_SP].max())}
    for kind in (INFLATED, NEAR_TIE):
        w = tgs.loc[kinds.index[kinds == kind]]
        out[kind] = {"n_tgs": int(len(w)), "n_sites": int(w[S.SITE_KEY_COL].nunique()),
                     "runs": sorted(w.run_id.unique()),
                     "min_rtt_norm": [float(w[D_RTT].min()), float(w[D_RTT].max())] if len(w) else None,
                     "d_sp": [float(w[D_SP].min()), float(w[D_SP].max())] if len(w) else None}
    return out


def _manifest(tgs: pd.DataFrame, table: pd.DataFrame, meta: dict, bounds: DN.Bounds,
              rtt_bounds: R.Bounds) -> dict:
    close = tgs[D_GEO] <= tgs[D_GEO].quantile(CLOSE_VP_QUANTILE)
    by_x = {}
    for flag, _, name in GROUPS:
        v = tgs.loc[tgs[HAS_X] == flag, D_SP]
        by_x[name] = {"n_tgs": int(v.size), **{f"p{q}": float(v.quantile(q / 100)) for q in (5, 25, 50, 75, 95)}}
    has, nox = tgs[tgs[HAS_X]], tgs[~tgs[HAS_X]]
    return {
        "figures": {f: f"{f}.png" for f in FIGURES},
        "csv": CSV_NAME,
        "run_ids": meta["run_ids"],
        "layout": meta.get("layout"),
        "has_x": f"{HAS_X}: the TG's own cell holds an interconnect location (no nearest-interconnect proxy)",
        "n_tgs_disagree_with_tg_cell_holds_x": int((tgs[HAS_X] != tgs["tg_cell_holds_x"].astype(bool)).sum()),
        "dist_norm_km": {"min": bounds[0], "max": bounds[1], "unit": "1e-3 of the span"},
        "rtt_norm_ms": {"min": rtt_bounds[0], "max": rtt_bounds[1]},
        "scatter": {
            "spearman_d_geo_d_sp": float(tgs[D_GEO].corr(tgs[D_SP], method="spearman")),
            "spearman_d_x_d_sp": float(tgs[D_X].corr(tgs[D_SP], method="spearman")),
            "median_d_sp_over_d_geo": float((tgs[D_SP] / tgs[D_GEO]).median()),
            "median_d_sp_over_d_x": float((tgs[D_SP] / tgs[D_X]).median()),
            "median_d_sp_over_d_x_no_x": float((tgs[D_SP] / tgs[D_X])[~tgs[HAS_X]].median()),
            "within_2x_of_diagonal_pct": {
                "d_geo": 100.0 * float(((tgs[D_SP] / tgs[D_GEO]).between(0.5, 2.0)).mean()),
                "d_x": 100.0 * float(((tgs[D_SP] / tgs[D_X]).between(0.5, 2.0)).mean()),
            },
            "d_x_note": "d_x = distance to the TG's nearest interconnect, a predictor tested here, not an assumption",
            "zero_gap_pct": 100.0 * float((np.isclose(tgs[D_SP], tgs[D_GEO], atol=1e-9)).mean()),
            "close_vp": {
                "rule": f"d_geo <= its p{int(CLOSE_VP_QUANTILE * 100)}",
                "threshold_norm_e3": float(tgs[D_GEO].quantile(CLOSE_VP_QUANTILE)),
                "n_tgs": int(close.sum()),
                f"d_sp_over_{FAR_FACTOR:g}x_d_geo_pct": 100.0 * float((tgs[D_SP] > FAR_FACTOR * tgs[D_GEO])[close].mean()),
                "d_sp_over_overall_median_pct": 100.0 * float((tgs[D_SP] > tgs[D_SP].median())[close].mean()),
            },
            "d_geo_min_norm_e3": float(tgs[D_GEO].min()),
            "n_drawn_at_axis_floor": int((tgs[D_GEO] < SCATTER_LIM[0]).sum()),
            "markers": f"one per distinct (run, site, x, d_sp) per series; area = {AREA_PER_TG} pt^2 per TG",
        },
        "interconnect_cell": {
            "sp_in_interconnect_cell_pct": 100.0 * float(tgs["pred_in_interconnect_cell"].mean()),
            "random_vp_in_interconnect_cell_pct": 100.0 * float(tgs["random_in_interconnect_cell"].mean()),
        },
        "error_by_x": by_x,
        "error_by_x_note": "no-X TGs with S-P error below the has-X p75: "
                           f"{100.0 * float((nox[D_SP] < has[D_SP].quantile(0.75)).mean()):.1f}%",
        "accuracy": table.to_dict(orient="records"),
        "has_x_misses": _misses(tgs),
        "adjacency": "spherical Delaunay edges of the seeds (hull facets with the origin inside); "
                     "each seed's nearest seed checked adjacent",
    }


def build_for_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    bounds: DN.Bounds,
    rtt_bounds: R.Bounds,
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> list[Path]:
    """The four PNGs, the per-TG twin and the manifest, pooled, beside the clusters."""
    tgs, meta, out_dir = load_runs(runs, pni_csvs, bounds=bounds, rtt_bounds=rtt_bounds,
                                   analysis_root=analysis_root, source_csvs=source_csvs)
    run_ids = list(meta["run_ids"])
    table = accuracy_table(tgs, run_ids)
    cols = ["run_id", "tg_id", HAS_X, D_GEO, D_X, D_SP, D_RTT, "correct", WHERE]
    tgs.sort_values(["run_id", "tg_id"])[cols].to_csv(out_dir / CSV_NAME, index=False)
    (out_dir / MANIFEST_NAME).write_text(json.dumps(_manifest(tgs, table, meta, bounds, rtt_bounds), indent=2) + "\n")
    return [
        plot_scatter(tgs, SCATTER_VP, out_dir / f"{SCATTER_VP}.png"),
        plot_scatter(tgs, SCATTER_X, out_dir / f"{SCATTER_X}.png"),
        plot_accuracy(table, out_dir / f"{ACCURACY}.png"),
        plot_rtt_scatter(tgs, out_dir / f"{RTT_SCATTER}.png"),
    ]
