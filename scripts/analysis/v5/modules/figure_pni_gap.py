"""`d_pni` against the S-P gap, one marker per scatter point, coloured by cluster.

The data and the clustering are `pni_gap`'s. This module draws them, and it is
where `plot-pni-gap` writes the three artifacts together: the clusters the RTT
figure reads, the points, and this PNG. One command writes all three, so the
picture cannot show a partition the CSV does not hold.

## Axes

Both axes are symlog with `pni_gap`'s parameters: linear 0-100 km with ticks
every 25, log from 100 to 4,000 km, and equal aspect, since both axes are
distances on the same transform. The paper panel drops the equal aspect: it
shares one printed height with its two row neighbours, so square axes could
only be as wide as they are tall and the legend would sit on the clusters. A dotted line marks the 100 km switch on
each axis, because the change of scale is invisible otherwise and a reader
would compare a 0-100 km step with a 100-1,000 km one.

A gap of exactly zero sits on the x axis; symlog draws zero, so there is no
fold-onto-the-floor trick here as in `figure_vp_distance_cdf`. Markers are
drawn unclipped so a point on an axis is a whole marker, not half of one.

## Normalized by declared bounds

When the runs' configs declare `analysis.common.dist_norm_km`, the panel is
drawn in units of 10^-3 (`dist_norm`): `d_pni` as a distance, the gap as a
difference of distances. **Only the drawing changes.** The clustering, the
CSVs and the RTT figure that reads them stay in km, on their own 100 km symlog
switch. The panel switches at `NORM_LINTHRESH` (10) instead, so it is a
monotone re-scaling of the clustered geometry, not the same transform: the
cluster membership is the km one, and a boundary read off this panel is
approximate. Ticks are 0 and the decades, labelled 10^k; the axes end at the
declared max. The legend sits inside the panel and gives each cluster's TG
share only (no counts), as "**C1** (51% TGs)"; there are no direct labels. A point beyond the axes is reported without its km
value. The PNG takes a `.norm.` infix.

## Encoding

Marker area grows with `n_tgs`, because a point holds up to ~20 replicas
and a split site's 5-replica point is a real minority. Cluster is carried by
**hue and shape together** plus a direct label, because three of the six
palette hues are under 3:1 against the surface (validated:
`validate_palette.js --mode light`, CVD worst adjacent dE 9.1). The hues are
the reference categorical palette in fixed order, so cluster 1 is always the
same blue whatever k is.

## Pooled

`--layout pooled` draws every run's points on one panel, clustered once. The
marker encodes cluster only, not dataset: the question is whether the regimes
recur across operators, and `clusters_by_run` in the manifest says which runs
each cluster is made of.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.v5.modules import dist_norm as DN  # noqa: E402
from scripts.analysis.v5.modules import pni_gap as P  # noqa: E402
from scripts.analysis.v5.modules.mapping import INK_2, MUTED  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402

PNG_NAME = "pni_gap_scatter.png"
PNG_NAME_NORM = "pni_gap_scatter.norm.png"

#: Symlog switch on a normalized axis (units of 10^-3): linear below, log above.
NORM_LINTHRESH = 10.0

#: Panel styles. `PANEL_PAPER` is the normalized panel, printed at its own
#: size as one of three in a row; marker areas scale by `marker`.
PANEL_KM = {"figsize": (5.3, 3.8), "label": 8.0, "tick": 7.5, "legend": 6.5, "marker": 1.0}
PANEL_PAPER = {"figsize": (2.4, 1.8), "label": 6.0, "tick": 5.5, "legend": 5.0, "marker": 0.35}

#: Axis-name suffix on a normalized axis.
NORM_AXIS_UNIT = r"Normalized Dist. $\times 10^{-3}$"

#: Reference categorical palette, fixed order (dataviz `palette.md`, light).
CLUSTER_HUES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300")
CLUSTER_MARKERS = ("o", "s", "^", "D", "v", "P")

MAJOR_TICKS_KM = (0, 25, 50, 75, 100, 1000, 4000)
MINOR_TICKS_KM = (200, 300, 400, 500, 600, 700, 800, 900, 2000, 3000)


def cluster_style(cluster: int) -> tuple[str, str]:
    """`(hue, marker)` for cluster `1..k`. Refuses k beyond the palette."""
    i = int(cluster) - 1
    if not 0 <= i < len(CLUSTER_HUES):
        raise ValueError(
            f"cluster {cluster} has no colour: the palette holds {len(CLUSTER_HUES)} "
            f"(generating more hues breaks the CVD validation). Use a smaller --k."
        )
    return CLUSTER_HUES[i], CLUSTER_MARKERS[i]


def cluster_label(cluster: int) -> str:
    return f"C{int(cluster)}"


def marker_area(n_tgs) -> np.ndarray:
    return 10.0 + 2.5 * np.asarray(n_tgs, dtype=float)


def style_symlog_axis(axis, major=MAJOR_TICKS_KM, minor=MINOR_TICKS_KM, label=None) -> None:
    axis.set_major_locator(ticker.FixedLocator(major))
    axis.set_minor_locator(ticker.FixedLocator(minor))
    label = label or (lambda v: f"{v:g}")
    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: label(v)))


def norm_ticks(linthresh: float, axis_max: float) -> tuple[list[float], list[float]]:
    """`(major, minor)` for a normalized symlog axis: 0 and every decade from
    `linthresh` up, with the 2-9 multiples of each decade above `linthresh`
    as minors, all within `axis_max`."""
    major = [0.0]
    minor = []
    decade = 10.0 ** np.floor(np.log10(linthresh))
    while decade <= axis_max:
        if linthresh <= decade:
            major.append(decade)
        minor += [m * decade for m in range(2, 10) if linthresh < m * decade <= axis_max]
        decade *= 10
    return [float(v) for v in major], [float(v) for v in minor]


def plot(
    pts: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    meta: dict,
    out_png: Path,
    bounds: tuple[float, float] | None = None,
) -> Path:
    """The scatter, in km or, with `bounds`, in units of 10^-3 (drawing only)."""
    if bounds is None:
        x_of, y_of = (lambda v: np.asarray(v, dtype=float)), (lambda v: np.asarray(v, dtype=float))
        linthresh, axis_max = P.LINTHRESH_KM, P.AXIS_MAX_KM
        major, minor = MAJOR_TICKS_KM, MINOR_TICKS_KM
        tick_label = None
        unit = "km"
    else:
        x_of = lambda v: DN.distance(v, bounds, what=P.D_PNI)  # noqa: E731
        y_of = lambda v: DN.difference(v, bounds)  # noqa: E731
        linthresh, axis_max = NORM_LINTHRESH, DN.X_MAX
        major, minor = norm_ticks(linthresh, axis_max)
        tick_label = lambda v: DN.power_label(v, linthresh)  # noqa: E731
        unit = NORM_AXIS_UNIT
    # km: square panel + legend column on the right. Normalized: the legend
    # sits inside, and the panel is sized to print at its own size as one of
    # three panels in a row (~1/3 of the text width).
    st = PANEL_KM if bounds is None else PANEL_PAPER
    fig, ax = plt.subplots(figsize=st["figsize"])
    for c, block in pts.groupby(P.CLUSTER_COL):
        hue, marker = cluster_style(c)
        row = summary.set_index(P.CLUSTER_COL).loc[c]
        ax.scatter(
            x_of(block[P.D_PNI]), y_of(block[P.GAP]), s=marker_area(block.n_tgs) * st["marker"],
            c=hue, marker=marker,
            alpha=0.75, edgecolor="white", linewidth=0.6 * st["marker"] ** 0.5, clip_on=False, zorder=3,
            label=(f"{cluster_label(c)}  {P.count_label(row.n_tgs, meta['n_tgs'], 'TGs')}, "
                   f"{P.count_label(row.n_sites, meta['n_sites'], 'sites')}"
                   if bounds is None
                   # No counts on a paper figure: the share of TGs only.
                   # The cluster name bold, through mathtext (one label, two weights).
                   else rf"$\mathbf{{{cluster_label(c)}}}$ "
                        f"({P.share_pct(row.n_tgs, meta['n_tgs']):.0f}% TGs)"),
        )
        if bounds is None:
            # Direct label at the cluster's top-right point, nudged off the
            # marker. The normalized panel relies on its legend instead.
            top = block.loc[block[P.GAP].idxmax()]
            ax.annotate(
                cluster_label(c), (float(x_of(top[P.D_PNI])), float(y_of(top[P.GAP]))),
                xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK_2, zorder=4,
            )

    ax.set_xscale("symlog", linthresh=linthresh, linscale=P.LINSCALE)
    ax.set_yscale("symlog", linthresh=linthresh, linscale=P.LINSCALE)
    ax.set_xlim(0, axis_max)
    ax.set_ylim(0, axis_max)
    if bounds is None:
        ax.set_aspect("equal")
    for axis in (ax.xaxis, ax.yaxis):
        style_symlog_axis(axis, major, minor, tick_label)
    ax.axvline(linthresh, color=MUTED, lw=0.5, ls=":", zorder=1)
    ax.axhline(linthresh, color=MUTED, lw=0.5, ls=":", zorder=1)

    if bounds is None:
        beyond = meta["axes"]["n_points_beyond_max"]
        note = f"{beyond} points beyond {P.AXIS_MAX_KM:g} km not drawn"
    else:
        # Counted in the drawn unit, and printed without a km value.
        beyond = int(((x_of(pts[P.D_PNI]) > axis_max) | (y_of(pts[P.GAP]) > axis_max)).sum())
        note = f"{beyond} points beyond the axes not drawn"
    if beyond:
        ax.text(0.98, 0.02, note,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, color=INK_2)

    # "Interconnect", not "PNI": the lists hold private interconnects *and*
    # settlement-free peering locations. Code keeps the `pni_*` names.
    # X is the paper's symbol for the interconnect.
    # The unit on a second line on the paper panel, which is too narrow for one.
    sep = " " if bounds is None else "\n"
    ax.set_xlabel(r"$d(\mathrm{TG},\ \mathrm{nearest}\ X)$" + sep + "(" + unit + ")",
                  fontsize=st["label"])
    ax.set_ylabel(r"$\Delta_\mathrm{VP} = d_\mathrm{sp}-d_\mathrm{geo}$" + sep + "(" + unit + ")",
                  fontsize=st["label"])
    ax.tick_params(labelsize=st["tick"], length=3 * st["tick"] / 7.5)
    ax.tick_params(which="minor", length=1.8 * st["tick"] / 7.5)
    ax.grid(alpha=0.25, lw=0.4)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if bounds is None:
        # Outside the axes: every corner of the panel holds data on the pooled meshes.
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.5, frameon=False,
                        borderaxespad=0.0,
                        handletextpad=0.3, labelspacing=0.5)
    else:
        # Inside, in the log-log quadrant's lower half, which holds no cluster:
        # above the x axis (a point sits on it) and right of the switch.
        leg = ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.07), fontsize=st["legend"],
                        frameon=False, borderaxespad=0.0, handletextpad=0.2, labelspacing=0.4)
    for text in leg.get_texts():
        text.set_color(INK_2)
    for handle in leg.legend_handles:
        handle.set_sizes([28.0 * st["marker"]])  # the key names a cluster, not a TG count

    fig.tight_layout(pad=0.4)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out_png


def _checked_k(k: int | None) -> None:
    if k is not None:
        cluster_style(k)  # refuse an unpaintable --k before fitting anything


def _draw(
    tgs: pd.DataFrame,
    pts: pd.DataFrame,
    meta: dict,
    out_dir: Path,
    bounds: tuple[float, float] | None = None,
) -> Path:
    for c in pts[P.CLUSTER_COL].unique():
        cluster_style(c)  # and a chosen k, before anything is written
    png = PNG_NAME if bounds is None else PNG_NAME_NORM
    meta = {**meta, "scatter_png": png, "dist_norm_km": DN.manifest_entry(bounds)}
    P.write(tgs, pts, meta, out_dir)
    return plot(pts, P.cluster_summary(tgs, pts), meta=meta, out_png=out_dir / png, bounds=bounds)


def build_for_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layouts: tuple[str, ...] = (P.PER_RUN,),
    k: int | None = None,
    method: str = P.DEFAULT_METHOD,
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
    dist_norm_km: dict[str, tuple[float, float] | None] | None = None,
) -> list[Path]:
    """Every requested layout. `per-run` writes one figure per run into that
    run's tree; `pooled` writes one figure over all of them into `_cross/`.
    Each run is measured against its own PNI list either way. `dist_norm_km`
    maps run id to its declared bounds; the pooled runs must agree."""
    _checked_k(k)
    bad = [lay for lay in layouts if lay not in P.LAYOUTS]
    if bad:
        raise ValueError(f"unknown layout {bad}; expected {list(P.LAYOUTS)}")
    pngs = []
    if P.PER_RUN in layouts:
        for run in runs:
            one = {run.run_id: (source_csvs or {}).get(run.run_id)} if source_csvs else None
            tgs, pts, meta = P.compute_runs([run], pni_csvs, layout=P.PER_RUN, k=k, method=method,
                                             source_csvs=one)
            out_dir = P.output_dir(run.run_id, pni_csvs[run.run_id], analysis_root=analysis_root)
            pngs.append(_draw(tgs, pts, meta, out_dir,
                              DN.common_bounds([run.run_id], dist_norm_km)))
    if P.POOLED in layouts:
        bounds = DN.common_bounds([r.run_id for r in runs], dist_norm_km)
        tgs, pts, meta = P.compute_runs(runs, pni_csvs, layout=P.POOLED, k=k, method=method,
                                         source_csvs=source_csvs)
        out_dir = P.pooled_output_dir([r.run_id for r in runs], analysis_root=analysis_root)
        pngs.append(_draw(tgs, pts, meta, out_dir, bounds))
    return pngs


def build_for_run(
    run: RunPaths,
    pni_csv: Path,
    *,
    k: int | None = None,
    method: str = P.DEFAULT_METHOD,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
) -> list[Path]:
    """Clusters CSV, points CSV, manifest and the scatter for one run, per-run layout."""
    return build_for_runs(
        [run], {run.run_id: pni_csv}, k=k, method=method, analysis_root=analysis_root,
        source_csvs={run.run_id: source_csv} if source_csv is not None else None,
    )
