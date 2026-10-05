"""Each TG's smallest RTT, per `pni_gap` cluster, as boxes: one latency regime per cluster?

`plot-pni-gap` groups TGs by where they sit on the `d_pni` x gap scatter. This
figure reads that grouping **off disk** (`pni_gap.read_clusters`) and asks what
the RTTs look like inside each group. It does not recompute the clusters, so
the boxes cannot describe a different partition from the scatter.

## One quantity: each TG's smallest RTT

The box per cluster is over its TGs' **smallest RTT across the fleet**: the
RTT of the S-P VP, read off `edges.load_min_rtt` (each pair at its minimum,
the RTT every LTD was built from). It is the delay no VP avoids, and on as01
it is the quantity that separates the mechanism groups (0.6-4.0 / 13.0-14.5 /
59.1-63.8 ms). The distribution over every (TG, VP) pair was drawn beside it
once; it mostly restated the fleet's geography and was dropped. The y axis is
linear from 0.

## Normalized by declared bounds

Absolute RTTs are confidential. When the run's config declares
`analysis.common.rtt_norm_ms: {min, max}`, every RTT r is drawn as
(r - min) / (max - min), as `plot-error-cdf` does with `dist_norm_km`: min 0
and max the largest (VP, TG) pair RTT pooled over the datasets the paper
reports, so every value is in [0, 1] and ratios between RTTs survive. A pair
RTT outside [min, max] is refused. The CSV keeps the `_ms` columns and adds the
same stats as `_norm`; the manifest records the bounds. Pooled runs must
declare the same bounds. Without them, the figure is in ms as before.

## Whiskers are p5 and p95

As in `figure_cost_box`: boxes are drawn from precomputed percentiles
(`Axes.bxp`), whiskers p5/p95, hinges p25/p75. A Tukey whisker moves with the
IQR, so its end is not a percentile anyone can quote.

Every TG whose smallest RTT falls outside its cluster's p5/p95 is drawn as an
open circle (`outliers`), so the min and max are on the figure as well as in
the CSV. By the whisker's definition that is about a tenth of each cluster's
TGs, and replicas at one site share an RTT, so circles overlap; `n_outliers`
in the CSV counts them.

## What is checked before drawing

The clusters were computed from one canonical CSV; the RTTs here are read
from whatever CSV the run resolves to now. So:

* the manifest's `run_id` must be this run;
* the PNI list's sha256 must be unchanged (the directory is keyed on its file
  stem, so an edited list would otherwise read the old list's clusters);
* the edge CSV's sha256 must equal the one the clusters were computed from;
* the TG sets must be identical, and each TG's smallest RTT here must equal
  the `sp_rtt_ms` the clusters CSV recorded.

Any mismatch raises and says to re-run `plot-pni-gap`.

## Replicas

~20 TGs per site share their VP geometry, so a cluster of 20 TGs at one site
is one site's RTTs repeated, not 20 samples. Each tick label carries the
cluster's site count next to its TG count for that reason.

Pooled (`--layout pooled`), the clusters come from `_cross/pni-gap/...` and
every check above runs per run, each against its own PNI list and edge CSV.

Command: `plot-pni-cluster-rtt`. Writes beside the clusters it reads.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.v5.modules import edges  # noqa: E402
from scripts.analysis.v5.modules import pni_gap as P  # noqa: E402
from scripts.analysis.v5.modules.cost import cost_stats  # noqa: E402
from scripts.analysis.v5.modules.figure_cost_box import _box as bxp_stats  # noqa: E402
from scripts.analysis.v5.modules.figure_pni_gap import cluster_label, cluster_style  # noqa: E402
from scripts.analysis.v5.modules.mapping import INK, INK_2  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402

PNG_NAME = "pni_cluster_rtt.png"
CSV_NAME = "pni_cluster_rtt.csv"
MANIFEST_NAME = "pni_cluster_rtt.manifest.json"

#: The y-axis label: what each box is over. Short, because the figure is small;
#: the caption spells it out ("each TG's smallest RTT over the fleet").
QUANTITY = "min RTT (ms)"

#: The y-axis label when the RTTs are normalized by the declared `rtt_norm_ms`.
QUANTITY_NORM = "normalized min RTT"

#: A small figure. Width is fixed rather than grown with k; at k <= 6 a
#: 0.5 in slot per box still fits a two-line tick label.
FIGSIZE = (3.0, 2.0)

#: y tick spacing, in ms.
Y_STEP_MS = 10.0

#: y tick spacing, normalized.
Y_STEP_NORM = 0.1

#: How far above the x axis, in points, a median label's centre must sit.
MEDIAN_LABEL_CLEARANCE_PT = 4.0

#: Both sides are the same float off the same CSV; this absorbs rounding only.
RTT_MATCH_TOL_MS = 1e-6

#: Outlier circle diameter, in points. Open, so overlapping replicas stay legible.
OUTLIER_MS_PT = 2.0


def _checked_run(run: RunPaths, pni_csv: Path, record: dict, source_csv: Path | None) -> pd.DataFrame:
    """One run's min-RTT edges, after checking its inputs are the clustered ones."""
    csv = P.checked_source_csv(run, pni_csv, record, source_csv)
    rtt = edges.load_min_rtt(run, source_csv=csv)
    rtt.insert(0, "run_id", run.run_id)
    return rtt


def load_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layout: str = P.PER_RUN,
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    """`(pairs, tgs, cluster_meta, out_dir)`: RTT edges labelled by cluster, checked.

    `pairs` is one row per `(run_id, tg_id, vp_id)` with `cluster`; `tgs` is
    the clusters CSV as written. Every run is checked on its own: PNI list,
    edge CSV, TG set, and each TG's smallest RTT.
    """
    run_ids = [r.run_id for r in runs]
    if layout == P.POOLED:
        out_dir = P.pooled_output_dir(run_ids, analysis_root=analysis_root)
    elif len(runs) == 1:
        out_dir = P.output_dir(runs[0].run_id, pni_csvs[runs[0].run_id], analysis_root=analysis_root)
    else:
        raise ValueError(f"layout {layout!r} takes one run; got {len(runs)}")
    tgs, meta = P.read_clusters(out_dir, run_ids=run_ids)
    records = {r["run_id"]: r for r in meta.get("runs", [])}

    rtt = pd.concat(
        [_checked_run(run, pni_csvs[run.run_id], records.get(run.run_id, {}),
                      (source_csvs or {}).get(run.run_id)) for run in runs],
        ignore_index=True,
    )
    key = ["run_id", "tg_id"]
    have = set(map(tuple, rtt[key].drop_duplicates().to_numpy()))
    want = set(map(tuple, tgs[key].to_numpy()))
    if have != want:
        raise ValueError(
            f"the edge CSVs hold {len(have - want)} TGs the clusters do not, and "
            f"lack {len(want - have)} they do. Re-run `plot-pni-gap`."
        )
    floor = rtt.groupby(key).rtt_ms.min()
    recorded = tgs.set_index(key)[P.SP_RTT].reindex(floor.index)
    off = (floor - recorded).abs() > RTT_MATCH_TOL_MS
    if off.any():
        raise ValueError(
            f"{int(off.sum())} TGs' smallest RTT differs from the clusters CSV's "
            f"{P.SP_RTT}, e.g. {floor.index[off][0]!r}. Re-run `plot-pni-gap`."
        )
    pairs = rtt.merge(tgs[key + [P.CLUSTER_COL]], on=key, how="left", validate="m:1")
    return pairs, tgs, meta, out_dir


def load(
    run: RunPaths,
    pni_csv: Path,
    *,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, Path]:
    """`load_runs` for one run, per-run layout."""
    return load_runs(
        [run], {run.run_id: pni_csv}, analysis_root=analysis_root,
        source_csvs={run.run_id: source_csv} if source_csv is not None else None,
    )


def stats_table(pairs: pd.DataFrame, tgs: pd.DataFrame, cluster_meta: dict) -> pd.DataFrame:
    """One row per cluster: percentiles and extrema of its TGs' smallest RTT, and counts.

    `tgs_pct` and `sites_pct` are over the clustered population's totals, as
    in `pni_gap.cluster_summary`; `sites_pct` can sum past 100 where a site's
    replicas split across clusters.
    """
    n_sites = {c["cluster"]: c["n_sites"] for c in cluster_meta["clusters"]}
    total_tgs, total_sites = cluster_meta["n_tgs"], cluster_meta["n_sites"]
    rows = []
    for c, rtt in _floors(pairs).items():
        stats = cost_stats(rtt)
        rows.append(
            {
                P.CLUSTER_COL: c,
                "n_tgs": (n := int(stats.pop("n"))),
                "tgs_pct": P.share_pct(n, total_tgs),
                "n_sites": int(n_sites[c]),
                "sites_pct": P.share_pct(n_sites[c], total_sites),
                **{f"{k}_ms": v for k, v in stats.items()},
                "n_outliers": len(_outside(rtt, stats["p5"], stats["p95"])),
            }
        )
    return pd.DataFrame(rows)


#: `(min, max)` in ms, as `labels.declared_rtt_norm_ms` returns it.
Bounds = tuple[float, float]


def to_norm(values, bounds: Bounds):
    """(r - min) / (max - min)."""
    lo, hi = bounds
    return (values - lo) / (hi - lo)


def check_bounds(pairs: pd.DataFrame, bounds: Bounds) -> None:
    """Refuse any pair RTT outside the bounds: the axis is declared to hold them all."""
    lo, hi = bounds
    rtt = pairs.rtt_ms.to_numpy(dtype=float)
    outside = (rtt < lo) | (rtt > hi)
    if outside.any():
        raise ValueError(
            f"{int(outside.sum())} pair RTTs fall outside the declared "
            f"analysis.common.rtt_norm_ms [{lo:g}, {hi:g}] ms "
            f"(range {rtt.min():g}-{rtt.max():g} ms)"
        )


def normalized(stats: pd.DataFrame, bounds: Bounds) -> pd.DataFrame:
    """`stats` plus every `<stat>_ms` column normalized by `bounds`, as `<stat>_norm`."""
    out = stats.copy()
    for col in [c for c in stats.columns if c.endswith("_ms")]:
        out[col.removesuffix("_ms") + "_norm"] = to_norm(stats[col], bounds)
    return out


def common_norm(run_ids: list[str], rtt_norm_ms: dict[str, Bounds | None] | None) -> Bounds | None:
    """The one `(min, max)` every run in `run_ids` declares, None if none does;
    mixed raises.

    A pooled figure normalizes every run's RTTs by one pair of bounds, so runs
    declaring different values, or some declaring none, cannot share a figure.
    """
    values = {r: (rtt_norm_ms or {}).get(r) for r in run_ids}
    distinct = set(values.values())
    if len(distinct) > 1:
        raise ValueError(
            f"pooled runs must declare the same analysis.common.rtt_norm_ms; got {values}"
        )
    return distinct.pop() if distinct else None


def _floors(pairs: pd.DataFrame) -> dict[int, np.ndarray]:
    """Each cluster's TGs' smallest RTTs, one value per `(run_id, tg_id)`."""
    floors = pairs.groupby(["run_id", "tg_id", P.CLUSTER_COL], as_index=False).rtt_ms.min()
    return {int(c): block.rtt_ms.to_numpy() for c, block in floors.groupby(P.CLUSTER_COL)}


def _outside(rtt: np.ndarray, lo: float, hi: float) -> list[float]:
    """The values beyond the whiskers, sorted."""
    return sorted(float(v) for v in rtt if v < lo or v > hi)


def outliers(pairs: pd.DataFrame, stats: pd.DataFrame) -> dict[int, list[float]]:
    """Per cluster, the TG floors outside that cluster's p5/p95 whiskers."""
    floors = _floors(pairs)
    return {int(r[P.CLUSTER_COL]): _outside(floors[int(r[P.CLUSTER_COL])], r["p5_ms"], r["p95_ms"])
            for _, r in stats.iterrows()}


def plot(
    stats: pd.DataFrame,
    *,
    total_tgs: int,
    total_sites: int,
    out_png: Path,
    fliers: dict[int, list[float]] | None = None,
    rtt_norm_ms: Bounds | None = None,
) -> Path:
    """Boxes per cluster; `fliers` (from `outliers`) are drawn as open circles.

    With `rtt_norm_ms`, every value is drawn normalized by those bounds
    (`stats` and `fliers` stay in ms).
    """
    stats = stats.sort_values(P.CLUSTER_COL).reset_index(drop=True)
    clusters = stats[P.CLUSTER_COL].tolist()

    def scale(v: float) -> float:
        return v if rtt_norm_ms is None else float(to_norm(v, rtt_norm_ms))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    boxes = [{**bxp_stats({k: scale(r[f"{k}_ms"]) for k in ("p5", "p25", "p50", "p75", "p95")}),
              "fliers": [scale(v) for v in (fliers or {}).get(c, [])]}
             for c, (_, r) in zip(clusters, stats.iterrows())]
    art = ax.bxp(boxes, positions=range(1, len(clusters) + 1), widths=0.5, patch_artist=True,
                 showfliers=True, medianprops={"color": INK, "lw": 0.9},
                 whiskerprops={"color": INK_2, "lw": 0.6}, capprops={"color": INK_2, "lw": 0.6})
    for patch, flier, c in zip(art["boxes"], art["fliers"], clusters):
        hue, _ = cluster_style(c)
        patch.set(facecolor=hue, alpha=0.6, edgecolor=hue, linewidth=0.6)
        flier.set(marker="o", markersize=OUTLIER_MS_PT, markerfacecolor="none",
                  markeredgecolor=hue, markeredgewidth=0.5, linestyle="none")
    # TGs then sites, the scatter legend's order. Sites are the count that says
    # how much evidence a box rests on: replicas are one observation repeated.
    ax.set_xticks(
        range(1, len(clusters) + 1),
        [f"{cluster_label(r[P.CLUSTER_COL])}\n{P.count_label(r.n_tgs, total_tgs, 'TGs')}"
         f"\n{P.count_label(r.n_sites, total_sites, 'sites')}" for _, r in stats.iterrows()],
        fontsize=5.5,
    )
    ax.set_ylabel(QUANTITY if rtt_norm_ms is None else QUANTITY_NORM, fontsize=6)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MultipleLocator(Y_STEP_MS if rtt_norm_ms is None else Y_STEP_NORM))
    ax.tick_params(axis="both", labelsize=5.5, length=2, pad=1.5)
    ax.grid(axis="y", alpha=0.25, lw=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    # Median labels, after layout so the axes' height is known. A label centred
    # on a ~1 ms median sits on the axis line, so its centre is kept at least
    # `MEDIAN_LABEL_CLEARANCE_PT` above it, whatever the figure height.
    height_pt = ax.bbox.height * 72.0 / fig.dpi
    floor = ax.get_ylim()[1] * MEDIAN_LABEL_CLEARANCE_PT / height_pt
    for i, (_, r) in enumerate(stats.iterrows(), start=1):
        median = scale(r["p50_ms"])
        ax.text(i + 0.3, max(median, floor), f"{median:.1f}" if rtt_norm_ms is None else f"{median:.2f}",
                fontsize=5.5, va="center", color=INK_2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_png


def _manifest(cluster_meta: dict, stats: pd.DataFrame, rtt_norm_ms: Bounds | None) -> str:
    return json.dumps(
        {
            "figure": PNG_NAME,
            "csv": CSV_NAME,
            "layout": cluster_meta.get("layout"),
            "run_ids": cluster_meta["run_ids"],
            "clusters_from": P.MANIFEST_NAME,
            "k": cluster_meta["clustering"]["k"],
            "inputs": [
                {key: r[key] for key in ("run_id", "source_csv_sha256", "pni_csv_sha256")}
                for r in cluster_meta["runs"]
            ],
            "quantity": "each TG's smallest RTT over the fleet, i.e. its S-P VP's RTT",
            "rtt_norm_ms": (
                None if rtt_norm_ms is None else {
                    "min": rtt_norm_ms[0],
                    "max": rtt_norm_ms[1],
                    "source": "analysis.common.rtt_norm_ms in each run's config",
                    "note": (
                        "every RTT r drawn as (r - min) / (max - min), the _norm columns; "
                        "no pair RTT lies outside [min, max] (refused otherwise). "
                        "max is the pooled largest pair RTT, confidential in the paper."
                    ),
                }
            ),
            "boxes": (
                "whiskers p5/p95, hinges p25/p75, line = median; every TG outside "
                "p5/p95 drawn as an open circle (n_outliers in the CSV)."
            ),
            "replicas_note": (
                "~20 TGs share a site's VP geometry; n_sites, not n_tgs, is the "
                "number of independent observations per cluster."
            ),
            "n_rows": int(len(stats)),
        },
        indent=2,
    )


def _write(pairs: pd.DataFrame, tgs: pd.DataFrame, meta: dict, out_dir: Path,
           rtt_norm_ms: Bounds | None = None) -> Path:
    if rtt_norm_ms is not None:
        check_bounds(pairs, rtt_norm_ms)
    stats = stats_table(pairs, tgs, meta)
    table = stats if rtt_norm_ms is None else normalized(stats, rtt_norm_ms)
    table.to_csv(out_dir / CSV_NAME, index=False)
    (out_dir / MANIFEST_NAME).write_text(_manifest(meta, stats, rtt_norm_ms))
    return plot(stats, total_tgs=meta["n_tgs"], total_sites=meta["n_sites"],
                out_png=out_dir / PNG_NAME, fliers=outliers(pairs, stats), rtt_norm_ms=rtt_norm_ms)


def build_for_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layouts: tuple[str, ...] = (P.PER_RUN,),
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
    rtt_norm_ms: dict[str, Bounds | None] | None = None,
) -> list[Path]:
    """Boxes for every requested layout, each beside the clusters it reads.

    `rtt_norm_ms` maps run id to its declared `(min, max)`
    (`labels.declared_rtt_norm_ms`); a run absent or None is drawn in ms.
    """
    bad = [lay for lay in layouts if lay not in P.LAYOUTS]
    if bad:
        raise ValueError(f"unknown layout {bad}; expected {list(P.LAYOUTS)}")
    pngs = []
    if P.PER_RUN in layouts:
        for run in runs:
            one = {run.run_id: source_csvs[run.run_id]} if source_csvs and run.run_id in source_csvs else None
            pairs, tgs, meta, out_dir = load_runs([run], pni_csvs, analysis_root=analysis_root, source_csvs=one)
            pngs.append(_write(pairs, tgs, meta, out_dir, common_norm([run.run_id], rtt_norm_ms)))
    if P.POOLED in layouts:
        norm = common_norm([r.run_id for r in runs], rtt_norm_ms)
        pairs, tgs, meta, out_dir = load_runs(runs, pni_csvs, layout=P.POOLED,
                                              analysis_root=analysis_root, source_csvs=source_csvs)
        pngs.append(_write(pairs, tgs, meta, out_dir, norm))
    return pngs


def build_for_run(
    run: RunPaths,
    pni_csv: Path,
    *,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
    rtt_norm_ms: Bounds | None = None,
) -> list[Path]:
    """Stats CSV, manifest and the PNG for one run, per-run layout."""
    return build_for_runs(
        [run], {run.run_id: pni_csv}, analysis_root=analysis_root,
        source_csvs={run.run_id: source_csv} if source_csv is not None else None,
        rtt_norm_ms={run.run_id: rtt_norm_ms},
    )
