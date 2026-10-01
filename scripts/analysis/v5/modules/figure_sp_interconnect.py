"""Does S-P locate the interconnect? Fig. B of the S-P subsection, and its numbers.

The S-P subsection argues that, given a VP at every interconnect of the target
network and latency that still reflects propagation delay, the lowest-RTT VP
is the VP at the interconnect the target's traffic crosses. Its ceiling and
floor then follow from where the target sits relative to that interconnect.
This module draws the figure behind that claim and writes **every number the
subsection quotes** into one report, so the paper has a single reproducible
source for them.

## The figure

The S-P VP's RTT, one marker per `(site, gap)` point coloured by the
`plot-pni-gap` cluster, against two distances:

* **direct**: `d(S-P VP, TG)`;
* **through the interconnect nearest the S-P VP**:
  `d(S-P VP, X) + d(X, TG)`, X being the interconnect the S-P VP sits beside.

Both panels carry the propagation floor `RTT = d / FLOOR_KM_PER_MS` (a round
trip at 2/3 c) and twice it. A point below the floor on the right would mean
traffic cannot cross at X; on the meshes this happens nowhere, while through
each TG's *nearest* interconnect it happens for four sites (in the report).

## By construction, and why that is the point

X is chosen from the S-P VP's own location, so the right panel's fit partly
follows from the choice. That is the claim, not a flaw: if S-P is the VP at
the crossing, routing through it must fit. What makes it non-trivial is in
the report: the share of TGs whose S-P VP sits at an interconnect against the
share of *all* VPs measuring that TG that do (`at_interconnect`), and the
independent test of `plot-pni-gap`, which uses each TG's nearest interconnect
chosen without the S-P result.

## The condition: RTT must vary with VP distance

Per TG, `rho` is the Spearman correlation between VP distance and VP RTT over
every VP that measured it. Where a delay shared by all VPs dominates, `rho`
collapses and many VPs tie within `TIE_MS` of the lowest RTT, so S-P's choice
carries no location signal (C3 on the meshes: `rho ~ 0`, 15 ties). The report
gives `rho` and the ties per cluster. "Inflated" is the wrong word for this:
the lowest RTT of those TGs is ordinary path stretch over its own path.

## Units

* distribution and the floor paragraph: per TG;
* the coverage of the VP set: per site;
* the path fit and the floor violations: per `point_id` from the clusters
  CSV, since ~20 replicas of a site are one observation repeated.

Each block of the report names its unit.

## Consistency with the clusters

`d_geo` and `d_sp` are recomputed here from the edge CSV, with the S-P VP's
identity, which `vp_distances` does not return. They must equal the clusters
CSV's to 1e-6 km for every TG, or the module refuses: a different tie-break
would silently describe a different S-P VP than the one clustered.

Command: `plot-sp-interconnect`. Writes beside the clusters it reads.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import ConstantInputWarning, spearmanr  # noqa: E402

from scripts.analysis.v5.modules import pni_gap as P  # noqa: E402
from scripts.analysis.v5.modules import sites as S  # noqa: E402
from scripts.analysis.v5.modules.answer_space import BENCHMARK_TG_COLUMNS  # noqa: E402
from scripts.analysis.v5.modules.figure_pni_gap import cluster_label, cluster_style  # noqa: E402
from scripts.analysis.v5.modules.geodesy import haversine_km  # noqa: E402
from scripts.analysis.v5.modules.mapping import INK_2, MUTED  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402

PNG_NAME = "sp_interconnect_rtt.png"
CSV_NAME = "sp_interconnect_tgs.csv"
REPORT_NAME = "sp_interconnect.report.json"

#: RTT >= d / FLOOR_KM_PER_MS: a round trip at 2/3 c covers 100 km of one-way
#: distance per ms.
FLOOR_KM_PER_MS = 100.0
#: "At an interconnect" radii. 50 km is quoted; 25 and 100 are the sensitivity.
RADII_KM = (25.0, 50.0, 100.0)
QUOTED_RADIUS_KM = 50.0
#: Beyond this distance propagation, not per-VP fixed delay, dominates RTT.
LONG_RANGE_KM = 100.0
#: Below this per-TG rho, RTT is taken as no longer tracking VP distance.
LOW_RHO = 0.3
#: VPs whose RTT is within this of the lowest count as tied with it.
TIE_MS = 1.0
#: "Largest errors" in the floor paragraph.
LARGE_ERROR_KM = 3000.0
#: Coverage of the VP set, in the distribution paragraph.
COVERAGE_KM = 15.0

#: Distances drawn on the two panels, in order.
PANELS = (
    ("d_sp_km", r"$d$(S-P VP, TG) (km)", "direct"),
    ("d_via_sp_km", r"$d$(S-P VP, $X$) + $d$($X$, TG) (km)",
     r"through the interconnect $X$ nearest the S-P VP"),
)
#: The three paths whose fit the report compares.
PATHS = {
    "direct": "d_sp_km",
    "via_tg_nearest_interconnect": "d_via_tg_km",
    "via_sp_vp_nearest_interconnect": "d_via_sp_km",
}

FIGSIZE = (7.2, 3.1)

#: x axis: linear to X_LINTHRESH_KM with a tick every X_LINEAR_STEP_KM, log
#: beyond. Most points sit under 1,000 km; the log tail keeps the 3,300-3,950 km
#: C3 points on the panel without squeezing everything else into its left edge.
X_LINTHRESH_KM = 1000.0
X_LINEAR_STEP_KM = 200.0


# -- frames -------------------------------------------------------------------


def load_edges(source_csv: Path) -> pd.DataFrame:
    """One row per `(tg_id, vp_id)` at its minimum RTT, with both coordinates and `d_km`."""
    from scripts.libs.canonical.schema import load_canonical_csv

    df = load_canonical_csv(source_csv).rename(columns=BENCHMARK_TG_COLUMNS)
    df = df.groupby(["tg_id", "vp_id"], as_index=False).agg(
        rtt_ms=("rtt_ms", "min"), vp_lat=("vp_lat", "first"), vp_lon=("vp_lon", "first"),
        tg_lat=("tg_lat", "first"), tg_lon=("tg_lon", "first"),
    )
    df["d_km"] = haversine_km(df.vp_lat.values, df.vp_lon.values, df.tg_lat.values, df.tg_lon.values)
    return df


def _to_interconnects(lat, lon, pnis: pd.DataFrame) -> np.ndarray:
    """`[n, n_interconnects]` great-circle km."""
    return haversine_km(np.asarray(lat, float)[:, None], np.asarray(lon, float)[:, None],
                        pnis.pni_lat.to_numpy()[None, :], pnis.pni_lon.to_numpy()[None, :])


def tg_frame(edges: pd.DataFrame, pnis: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    """One row per TG: both VPs, the paths through interconnects, and the latency condition.

    Tie-breaks follow `figure_vp_proximity.vp_distances` (first `idxmin` in
    `(tg_id, vp_id)` order), and `check_against_clusters` holds them to it.
    """
    by_tg = edges.groupby("tg_id", sort=False)
    geo = edges.loc[by_tg["d_km"].idxmin()].set_index("tg_id")
    sp = edges.loc[by_tg["rtt_ms"].idxmin()].set_index("tg_id")
    out = pd.DataFrame({
        "tg_lat": geo.tg_lat, "tg_lon": geo.tg_lon,
        "d_geo_km": geo.d_km, "rtt_geo_ms": geo.rtt_ms,
        "sp_vp_id": sp.vp_id, "d_sp_km": sp.d_km, "rtt_sp_ms": sp.rtt_ms,
    })
    out[P.GAP] = (out.d_sp_km - out.d_geo_km).clip(lower=0.0)

    tg_x = _to_interconnects(out.tg_lat, out.tg_lon, pnis)
    vp_x = _to_interconnects(sp.vp_lat.reindex(out.index), sp.vp_lon.reindex(out.index), pnis)
    rows = np.arange(len(out))
    k_tg, k_vp = tg_x.argmin(1), vp_x.argmin(1)
    out[P.D_PNI] = tg_x[rows, k_tg]
    out["d_sp_vp_interconnect_km"] = vp_x[rows, k_vp]
    out["same_interconnect"] = k_tg == k_vp
    # VP -> X -> TG: through the TG's nearest X, and through the S-P VP's nearest X.
    out["d_via_tg_km"] = vp_x[rows, k_tg] + tg_x[rows, k_tg]
    out["d_via_sp_km"] = vp_x[rows, k_vp] + tg_x[rows, k_vp]

    # The latency condition, per TG over every VP that measured it.
    stats = by_tg.apply(_latency_condition, include_groups=False)
    out = out.join(stats)

    # Random-VP baseline: share of the TG's measuring VPs that sit at an interconnect.
    vp_at = _to_interconnects(edges.vp_lat, edges.vp_lon, pnis).min(1)
    for r in RADII_KM:
        out[f"sp_at_{r:g}km"] = out.d_sp_vp_interconnect_km <= r
        out[f"random_at_{r:g}km"] = pd.Series(vp_at <= r, index=edges.index).groupby(edges.tg_id).mean()
    out.insert(0, "run_id", run_id)
    out = out.rename_axis("tg_id").reset_index()
    out[S.SITE_KEY_COL] = S.site_key(out, run_id=run_id)
    return out


def _latency_condition(x: pd.DataFrame) -> pd.Series:
    rtt = x.rtt_ms.to_numpy()
    lo = rtt.min()
    rho = _rho(x.d_km, rtt) if len(x) > 2 else np.nan
    return pd.Series({
        "rho": float(rho),
        "n_tied": int((rtt <= lo + TIE_MS).sum()),
        "rtt_range_ms": float(np.percentile(rtt, 90) - lo),
    })


def interconnect_coverage(edges: pd.DataFrame, pnis: pd.DataFrame) -> np.ndarray:
    """Distance from each interconnect to its nearest measuring VP, km."""
    vps = edges.drop_duplicates("vp_id")
    return _to_interconnects(vps.vp_lat, vps.vp_lon, pnis).min(axis=0)


def check_against_clusters(tgs: pd.DataFrame, clusters: pd.DataFrame) -> pd.DataFrame:
    """Join the clusters and refuse any TG whose `d_geo`/`d_sp` disagree with them."""
    key = ["run_id", "tg_id"]
    cols = key + [P.CLUSTER_COL, P.POINT_COL, P.D_GEO, P.D_SP]
    out = tgs.merge(clusters[cols], on=key, how="outer", suffixes=("", "_clustered"),
                    validate="1:1", indicator=True)
    lost = out[out._merge != "both"]
    if len(lost):
        raise ValueError(f"{len(lost)} TGs are in only one of the edge CSV and the clusters; "
                         f"re-run `plot-pni-gap`.")
    for col in (P.D_GEO, P.D_SP):
        bad = (out[col] - out[f"{col}_clustered"]).abs() > 1e-6
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} TGs' {col} differ from the clusters CSV, e.g. "
                f"{out.tg_id[bad].iloc[0]!r}: this module picked a different VP than "
                f"the one clustered. Re-run `plot-pni-gap`."
            )
    return out.drop(columns=["_merge", f"{P.D_GEO}_clustered", f"{P.D_SP}_clustered"])


def points(tgs: pd.DataFrame) -> pd.DataFrame:
    """One row per clusters point: its TG count and the values the figure draws."""
    return (
        tgs.groupby(["run_id", P.POINT_COL], as_index=False)
        .agg(n_tgs=("tg_id", "size"), cluster=(P.CLUSTER_COL, "first"),
             rtt_sp_ms=("rtt_sp_ms", "median"), d_sp_km=("d_sp_km", "first"),
             d_via_tg_km=("d_via_tg_km", "first"), d_via_sp_km=("d_via_sp_km", "first"),
             d_pni_km=(P.D_PNI, "first"))
    )


# -- the report ---------------------------------------------------------------


def _q(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if len(s) else float("nan")


def _pct(mask) -> float:
    mask = np.asarray(mask, dtype=bool)
    return 100.0 * float(mask.mean()) if mask.size else float("nan")


def _rho(x, y) -> float:
    """Spearman rho; NaN, silently, where an input is constant (rho is undefined there)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return float(spearmanr(x, y).statistic)


def _path_fit(pts: pd.DataFrame, col: str) -> dict:
    ratio = pts.rtt_sp_ms / (pts[col] / FLOOR_KM_PER_MS)
    long = pts[col] > LONG_RANGE_KM
    below = pts.rtt_sp_ms < pts[col] / FLOOR_KM_PER_MS
    lr = np.log10(ratio[long])
    return {
        "rho_all": _rho(pts[col], pts.rtt_sp_ms),
        "rho_long_range": _rho(pts[col][long], pts.rtt_sp_ms[long]),
        "rtt_over_floor_median_long_range": float(ratio[long].median()),
        "log10_ratio_iqr_long_range": float(lr.quantile(0.75) - lr.quantile(0.25)),
        "n_points_below_floor": int(below.sum()),
        "n_tgs_below_floor": int(pts.n_tgs[below].sum()),
    }


def report(tgs: pd.DataFrame, pts: pd.DataFrame, coverage: dict[str, np.ndarray], meta: dict) -> dict:
    """Every number the S-P subsection quotes, by paragraph. No coordinate, no interconnect name."""
    site = tgs.groupby(S.SITE_KEY_COL).d_geo_km.first()
    long = tgs.d_sp_km > LONG_RANGE_KM
    far_gap = tgs[P.GAP] > LONG_RANGE_KM
    near_gap = ~far_gap

    at = {}
    for r in RADII_KM:
        sp, rnd = tgs[f"sp_at_{r:g}km"], tgs[f"random_at_{r:g}km"]
        at[f"{r:g}km"] = {
            "sp_vp_at_interconnect_pct": _pct(sp),
            "random_vp_at_interconnect_pct": 100.0 * float(rnd.mean()),
            "ratio": float(sp.mean() / rnd.mean()) if rnd.mean() else float("nan"),
        }
    r50 = f"sp_at_{QUOTED_RADIUS_KM:g}km"
    distinct = tgs.drop_duplicates(["run_id", "sp_vp_id"])

    by_cluster = {}
    for c, b in tgs.groupby(P.CLUSTER_COL):
        by_cluster[str(int(c))] = {
            "n_tgs": int(len(b)),
            "rho_median": float(b.rho.median()), "rho_p10": _q(b.rho, 0.10),
            "low_rho_pct": _pct(b.rho < LOW_RHO),
            "n_tied_median": float(b.n_tied.median()),
            "rtt_range_ms_median": float(b.rtt_range_ms.median()),
        }

    large = pts[pts.d_sp_km > LARGE_ERROR_KM].merge(
        tgs.groupby(["run_id", P.POINT_COL], as_index=False).d_geo_km.first(), on=["run_id", P.POINT_COL])
    worst = tgs[tgs.d_geo_km == tgs.d_geo_km.max()]
    violations = pts[pts.rtt_sp_ms < pts.d_via_tg_km / FLOOR_KM_PER_MS]

    return {
        "layout": meta.get("layout"),
        "run_ids": meta["run_ids"],
        "constants": {
            "floor_km_per_ms": FLOOR_KM_PER_MS, "radii_km": list(RADII_KM),
            "quoted_radius_km": QUOTED_RADIUS_KM, "long_range_km": LONG_RANGE_KM,
            "low_rho": LOW_RHO, "tie_ms": TIE_MS, "coverage_km": COVERAGE_KM,
        },
        "distribution": {
            "unit": "TG (d_geo: site)",
            "n_tgs": int(len(tgs)), "n_sites": int(len(site)),
            "zero_gap_pct": _pct(tgs[P.GAP] == 0),
            "gap_gt_100km_pct": _pct(tgs[P.GAP] > 100), "gap_gt_1000km_pct": _pct(tgs[P.GAP] > 1000),
            "gap_max_km": float(tgs[P.GAP].max()),
            "sites_with_vp_within_coverage_km_pct": _pct(site <= COVERAGE_KM),
            "site_d_geo_median_km": float(site.median()), "site_d_geo_p90_km": _q(site, 0.9),
            "site_d_geo_max_km": float(site.max()),
        },
        "interconnect_coverage": {
            "unit": "interconnect",
            "per_run": {
                rid: {"n_interconnects": int(len(d)), "nearest_vp_max_km": float(d.max()),
                      "with_vp_within_quoted_radius": int((d <= QUOTED_RADIUS_KM).sum())}
                for rid, d in coverage.items()
            },
        },
        "at_interconnect": {
            "unit": "TG; random = mean over TGs of the share of its measuring VPs at an interconnect",
            "by_radius": at,
            "per_run_at_quoted_radius": {
                rid: {"sp_vp_at_interconnect_pct": _pct(b[r50]),
                      "random_vp_at_interconnect_pct": 100.0 * float(b[f"random_at_{QUOTED_RADIUS_KM:g}km"].mean()),
                      "distinct_sp_vps": int(len(distinct[distinct.run_id == rid])),
                      "distinct_sp_vps_at_interconnect": int(distinct[(distinct.run_id == rid)][r50].sum())}
                for rid, b in tgs.groupby("run_id")
            },
            "same_as_tg_nearest_interconnect": {
                "n_tgs": int(tgs.same_interconnect.sum()), "pct": _pct(tgs.same_interconnect),
            },
        },
        "path_fit": {"unit": "clusters point", **{name: _path_fit(pts, col) for name, col in PATHS.items()}},
        "latency_condition": {
            "unit": "TG",
            "low_rho_pct": _pct(tgs.rho < LOW_RHO),
            "holds_pct": _pct(tgs.rho >= LOW_RHO),
            "by_cluster": by_cluster,
        },
        "ceiling": {
            "unit": "TG",
            "gap_below_100km": {"n_tgs": int(near_gap.sum()),
                                "rtt_sp_median_ms": float(tgs.rtt_sp_ms[near_gap].median()),
                                "rtt_sp_p90_ms": _q(tgs.rtt_sp_ms[near_gap], 0.9)},
        },
        "floor": {
            "unit": "TG, except largest_errors (clusters point)",
            "largest_errors": {
                "n_points": int(len(large)), "n_tgs": int(large.n_tgs.sum()),
                "d_sp_km_range": [float(large.d_sp_km.min()), float(large.d_sp_km.max())] if len(large) else None,
                "d_geo_km_range": [float(large.d_geo_km.min()), float(large.d_geo_km.max())] if len(large) else None,
            },
            "worst_covered": {"d_geo_km": float(worst.d_geo_km.iloc[0]),
                              "d_sp_km_range": [float(worst.d_sp_km.min()), float(worst.d_sp_km.max())]},
            "gap_above_100km": {
                "n_tgs": int(far_gap.sum()),
                "nearest_vp_d_geo_median_km": float(tgs.d_geo_km[far_gap].median()),
                "nearest_vp_rtt_median_ms": float(tgs.rtt_geo_ms[far_gap].median()),
                "rtt_sp_median_ms": float(tgs.rtt_sp_ms[far_gap].median()),
                "nearest_over_lowest_rtt_median": float((tgs.rtt_geo_ms / tgs.rtt_sp_ms)[far_gap].median()),
            },
            "lowest_rtt_over_floor_long_range": {
                "n_tgs": int(long.sum()),
                "median": float((tgs.rtt_sp_ms / (tgs.d_sp_km / FLOOR_KM_PER_MS))[long].median()),
                "iqr": [_q((tgs.rtt_sp_ms / (tgs.d_sp_km / FLOOR_KM_PER_MS))[long], q) for q in (0.25, 0.75)],
            },
        },
        "nearest_interconnect_violations": {
            "unit": "clusters point; RTT below the floor of the path through the TG's nearest interconnect",
            "n_points": int(len(violations)), "n_tgs": int(violations.n_tgs.sum()),
            "points": [
                {"run_id": r.run_id, "n_tgs": int(r.n_tgs), "d_tg_nearest_interconnect_km": round(float(r.d_pni_km), 1),
                 "d_sp_km": round(float(r.d_sp_km), 1), "rtt_sp_ms": round(float(r.rtt_sp_ms), 2),
                 "floor_via_tg_nearest_ms": round(float(r.d_via_tg_km / FLOOR_KM_PER_MS), 2),
                 "floor_via_sp_vp_nearest_ms": round(float(r.d_via_sp_km / FLOOR_KM_PER_MS), 2)}
                for r in violations.itertuples()
            ],
        },
    }


# -- the figure ---------------------------------------------------------------


def plot(pts: pd.DataFrame, cluster_meta: dict, *, out_png: Path) -> Path:
    x_max = float(np.ceil(max(pts.d_sp_km.max(), pts.d_via_sp_km.max()) / 1000.0) * 1000.0)
    y_max = float(np.ceil(pts.rtt_sp_ms.max() / 10.0) * 10.0)
    totals = (cluster_meta["n_tgs"], cluster_meta["n_sites"])
    sizes = {c["cluster"]: c for c in cluster_meta["clusters"]}
    km = np.linspace(0.0, x_max, 400)  # dense: the floor bends on the log part

    fig, axes = plt.subplots(1, len(PANELS), figsize=FIGSIZE, sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, (col, xlabel, title) in zip(axes, PANELS):
        for c, block in pts.groupby("cluster"):
            hue, marker = cluster_style(c)
            row = sizes[int(c)]
            ax.scatter(block[col], block.rtt_sp_ms, s=8 + 2.0 * block.n_tgs, c=hue, marker=marker,
                       alpha=0.75, edgecolor="white", linewidth=0.5, clip_on=False, zorder=3,
                       label=(f"{cluster_label(c)}  {P.count_label(row['n_tgs'], totals[0], 'TGs')}, "
                              f"{P.count_label(row['n_sites'], totals[1], 'sites')}"))
        ax.plot(km, km / FLOOR_KM_PER_MS, color=INK_2, lw=0.8, ls=":", zorder=2)
        ax.plot(km, 2 * km / FLOOR_KM_PER_MS, color=MUTED, lw=0.6, ls=":", zorder=2)
        ax.text(x_max * 0.97, x_max / FLOOR_KM_PER_MS * 0.97, "floor", fontsize=6, color=INK_2,
                ha="right", va="top")
        two_x = min(x_max, y_max * FLOOR_KM_PER_MS / 2)
        ax.text(two_x * 0.97, 2 * two_x / FLOOR_KM_PER_MS * 0.97, "2× floor", fontsize=6, color=MUTED,
                ha="right", va="top")
        ax.set_xscale("symlog", linthresh=X_LINTHRESH_KM, linscale=1.0)
        ax.set_xlim(0, x_max)
        majors = list(np.arange(0.0, X_LINTHRESH_KM + 1, X_LINEAR_STEP_KM))
        majors += [v for v in (2000.0, 4000.0, 8000.0) if v <= x_max]
        ax.xaxis.set_major_locator(ticker.FixedLocator(majors))
        ax.xaxis.set_minor_locator(ticker.FixedLocator([v for v in (3000.0, 5000.0, 6000.0, 7000.0) if v <= x_max]))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.axvline(X_LINTHRESH_KM, color=MUTED, lw=0.5, ls=":", zorder=1)
        ax.set_xlabel(xlabel, fontsize=7.5)
        ax.set_title(title, fontsize=7.5)
        ax.tick_params(labelsize=7, length=3)
        ax.grid(alpha=0.25, lw=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    axes[0].set_ylabel("RTT of the S-P VP (ms)", fontsize=8)
    axes[0].set_ylim(0, y_max)
    axes[0].yaxis.set_major_locator(ticker.MultipleLocator(10))
    leg = axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.3, frameon=False,
                          borderaxespad=0.0, handletextpad=0.3, labelspacing=0.5)
    for text in leg.get_texts():
        text.set_color(INK_2)
    for handle in leg.legend_handles:
        handle.set_sizes([24.0])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


# -- the whole step -----------------------------------------------------------


def load_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layout: str = P.PER_RUN,
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict, Path]:
    """`(tgs, coverage, cluster_meta, out_dir)`, every run checked against the clusters."""
    run_ids = [r.run_id for r in runs]
    if layout == P.POOLED:
        out_dir = P.pooled_output_dir(run_ids, analysis_root=analysis_root)
    elif len(runs) == 1:
        out_dir = P.output_dir(runs[0].run_id, pni_csvs[runs[0].run_id], analysis_root=analysis_root)
    else:
        raise ValueError(f"layout {layout!r} takes one run; got {len(runs)}")
    clusters, meta = P.read_clusters(out_dir, run_ids=run_ids)
    records = {r["run_id"]: r for r in meta.get("runs", [])}

    frames, coverage = [], {}
    for run in runs:
        csv = P.checked_source_csv(run, pni_csvs[run.run_id], records.get(run.run_id, {}),
                                   (source_csvs or {}).get(run.run_id))
        pnis = P.load_pnis(pni_csvs[run.run_id])
        edges = load_edges(csv)
        frames.append(tg_frame(edges, pnis, run_id=run.run_id))
        coverage[run.run_id] = interconnect_coverage(edges, pnis)
    tgs = check_against_clusters(pd.concat(frames, ignore_index=True), clusters)
    return tgs, coverage, meta, out_dir


#: Per-TG columns written to the CSV. No coordinate, no VP or interconnect id.
CSV_COLUMNS = ("run_id", "tg_id", P.POINT_COL, P.CLUSTER_COL, "d_geo_km", "rtt_geo_ms", "d_sp_km",
               "rtt_sp_ms", P.GAP, P.D_PNI, "d_sp_vp_interconnect_km", "same_interconnect",
               "d_via_tg_km", "d_via_sp_km", "rho", "n_tied", "rtt_range_ms")


def _write(tgs, coverage, meta, out_dir) -> Path:
    pts = points(tgs)
    tgs.sort_values(["run_id", "tg_id"])[list(CSV_COLUMNS)].to_csv(out_dir / CSV_NAME, index=False)
    (out_dir / REPORT_NAME).write_text(json.dumps(report(tgs, pts, coverage, meta), indent=2))
    return plot(pts, meta, out_png=out_dir / PNG_NAME)


def build_for_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layouts: tuple[str, ...] = (P.PER_RUN,),
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> list[Path]:
    """Fig. B, the per-TG CSV and the report, beside each layout's clusters."""
    bad = [lay for lay in layouts if lay not in P.LAYOUTS]
    if bad:
        raise ValueError(f"unknown layout {bad}; expected {list(P.LAYOUTS)}")
    pngs = []
    if P.PER_RUN in layouts:
        for run in runs:
            one = {run.run_id: source_csvs[run.run_id]} if source_csvs and run.run_id in source_csvs else None
            pngs.append(_write(*load_runs([run], pni_csvs, analysis_root=analysis_root, source_csvs=one)))
    if P.POOLED in layouts:
        pngs.append(_write(*load_runs(runs, pni_csvs, layout=P.POOLED, analysis_root=analysis_root,
                                      source_csvs=source_csvs)))
    return pngs
