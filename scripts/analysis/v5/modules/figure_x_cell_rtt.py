"""Each TG's smallest RTT, has-X against no-X, side by side per content network.

`report-sp-pni-cells` splits the TGs by whether their own cell holds X, the
interconnect nearest the TG. That split is geometric: it needs the answer
space and the interconnect list, neither of which a method sees. This figure
asks whether it is also visible in the one thing every method does see, the
RTTs. On the meshes it is: has-X TGs sit at a median 1.5 ms, no-X TGs at
10.5 ms, although both have a VP within ~15 km.

## Who is has-X

`tg_cell_holds_x` from `sp_pni_cells.load_runs`, the flag the paper's has-X /
no-X table is built on, computed here rather than read from its CSV so the
same staleness checks run. The looser "the TG's cell holds *any*
interconnect" (`tg_cell_holds_interconnect`) is not drawn; the manifest
counts the TGs where the two disagree (0 on the meshes).

## One quantity: each TG's smallest RTT

The RTT of the TG's S-P VP over the fleet, from `figure_pni_cluster_rtt.load_runs`,
which refuses it unless it equals the clusters CSV's `sp_rtt_ms`. One value per
TG, not a site median: the box is over TGs, and the tick labels carry the
site count beside the TG count because ~20 replicas at a site are one
observation repeated.

## Boxes, axis and the split line

As in `plot-pni-cluster-rtt`: whiskers p5/p95, hinges p25/p75, every TG
beyond the whiskers an open circle. With ~20 replicas per site a whisker
end can be a single site, so read the circles with the site counts. The
y axis is log: has-X boxes sit near 1 ms and one inflated site sits near
60 ms, and a linear axis flattens the first under the second.

`SPLIT_MS` is drawn as a dotted line and counted per group in the CSV
(`n_le_split`). It is a reading aid chosen by eye on the meshes, not fitted.

Groups are the runs in the clusters manifest's order, has-X left of no-X.
Pooled, the CSV also carries one `run_id = all` row per side; the figure
draws only the per-run pairs. A group's title is its dataset label minus the
words every group shares (`PRO-MESH AS-A` -> `AS-A` when all three are
`PRO-MESH`).

## Normalized RTT, shares not counts

When the runs declare `analysis.common.rtt_norm_ms: {min, max}` (as
`plot-pni-cluster-rtt`), every RTT on the figure -- boxes, circles, median
labels and the split line -- is drawn as (r - min) / (max - min), so no
absolute RTT is printed. The CSV keeps the `_ms` columns and adds `_norm`
twins. Pooled runs must declare one pair of bounds.

Tick labels carry the side's share of its network's sites, not counts: the
paper prints no absolute TG or site count. Counts stay in the CSV.

No coordinate, VP id or interconnect id is written.

## The exceptions, by site

The text names the sites on the wrong side of the split -- has-X TGs above it,
no-X TGs at or below it -- with their RTT range. `x_cell_rtt.exceptions.csv`
lists them: one row per `(run, site, side)` with any TG across the line, the
site as a per-run index in sorted-key order (never its coordinate), its
`plot-pni-gap` cluster(s), how many of its TGs cross, and their min/max RTT
(`_norm` twins when normalized). Empty, with its header, when nothing crosses.

Command: `plot-x-cell-rtt`. Needs `classify` and `plot-pni-gap`. Writes
`x_cell_rtt.{png,csv,manifest.json}` beside the clusters.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.v5.modules import figure_pni_cluster_rtt as R  # noqa: E402
from scripts.analysis.v5.modules import pni_gap as P  # noqa: E402
from scripts.analysis.v5.modules import sites as S  # noqa: E402
from scripts.analysis.v5.modules import sp_pni_cells as M  # noqa: E402
from scripts.analysis.v5.modules.cost import cost_stats  # noqa: E402
from scripts.analysis.v5.modules.figure_cost_box import _box as bxp_stats  # noqa: E402
from scripts.analysis.v5.modules.figure_pni_gap import CLUSTER_HUES  # noqa: E402
from scripts.analysis.v5.modules.labels import dataset_label  # noqa: E402
from scripts.analysis.v5.modules.mapping import INK, INK_2  # noqa: E402
from scripts.analysis.v5.modules.paths import RunPaths  # noqa: E402

PNG_NAME = "x_cell_rtt.png"
CSV_NAME = "x_cell_rtt.csv"
MANIFEST_NAME = "x_cell_rtt.manifest.json"

GROUP_COL = "tg_cell_holds_x"
ANY_COL = "tg_cell_holds_interconnect"
FLOOR_COL = "min_rtt_ms"
#: Pooled rows of the CSV carry this in place of a run id.
ALL_RUNS = "all"

#: `(flag, tick label, CSV label)`, in draw order: has-X left of no-X.
GROUPS = ((True, "has-$X$", "has-X"), (False, "no-$X$", "no-X"))
#: The scatter's first two hues. Fixed by side, not by cluster number:
#: clusters are renumbered by size, so cluster 1 need not be has-X elsewhere.
HUES = {True: CLUSTER_HUES[0], False: CLUSTER_HUES[1]}

QUANTITY = "min RTT (ms), per TG"
QUANTITY_NORM = "normalized min RTT"

#: The dotted line, in ms. A reading aid; see the module docstring.
SPLIT_MS = 3.0

#: Log-axis ticks, in ms; only those inside the data's range are drawn.
Y_TICKS_MS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
#: The normalized axis labels decades only, as 10^k: every median carries its
#: own number, so the axis only has to say the order of magnitude.
Y_TICKS_NORM = (0.0001, 0.001, 0.01, 0.1, 1.0)
#: Headroom either side of the data on the log axis, as a factor.
Y_PAD = 1.5

#: Width per run group and the fixed height, in inches.
GROUP_WIDTH_IN = 1.65
HEIGHT_IN = 1.35
#: Box centres within a group, and the gap between groups, in x units.
SLOT = 1.0
GROUP_GAP = 0.6
BOX_WIDTH = 0.5

OUTLIER_MS_PT = 2.0

STAT_KEYS = ("p5", "p25", "p50", "p75", "p95")


# -- frames -------------------------------------------------------------------


def load_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layout: str = P.PER_RUN,
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> tuple[pd.DataFrame, dict, Path]:
    """`(tgs, cluster_meta, out_dir)`: one row per TG with its has-X flag, site and smallest RTT.

    Both halves are loaded through their own checked loaders, so every
    staleness check of `report-sp-pni-cells` and `plot-pni-cluster-rtt` runs.
    """
    kw = {"layout": layout, "analysis_root": analysis_root, "source_csvs": source_csvs}
    tgs, _, meta, out_dir = M.load_runs(runs, pni_csvs, **kw)
    pairs, _, _, rtt_dir = R.load_runs(runs, pni_csvs, **kw)
    if rtt_dir != out_dir:
        raise ValueError(f"the flags and the RTTs resolved to different directories: {out_dir} vs {rtt_dir}")
    key = ["run_id", "tg_id"]
    floor = pairs.groupby(key, as_index=False).rtt_ms.min().rename(columns={"rtt_ms": FLOOR_COL})
    keep = key + [GROUP_COL, ANY_COL, S.SITE_KEY_COL] + (["cluster"] if "cluster" in tgs else [])
    out = tgs[keep].merge(
        floor, on=key, how="outer", validate="1:1", indicator=True)
    lost = out[out._merge != "both"]
    if len(lost):
        raise ValueError(f"{len(lost)} TGs have a has-X flag or an RTT but not both, e.g. "
                         f"{lost.tg_id.iloc[0]!r}; re-run `classify` and `plot-pni-gap` on the same runs.")
    return out.drop(columns="_merge"), meta, out_dir


def _block(g: pd.DataFrame, n_run_tgs: int, n_run_sites: int) -> dict:
    rtt = g[FLOOR_COL].to_numpy(float)
    stats = cost_stats(rtt)
    n = int(stats.pop("n"))
    n_sites = int(g[S.SITE_KEY_COL].nunique())
    return {
        "n_tgs": n,
        "tgs_pct": P.share_pct(n, n_run_tgs),
        "n_sites": n_sites,
        "sites_pct": P.share_pct(n_sites, n_run_sites),
        **{f"{k}_ms": v for k, v in stats.items()},
        "n_outliers": len(R._outside(rtt, stats["p5"], stats["p95"])) if n else 0,
        "n_le_split": int((rtt <= SPLIT_MS).sum()),
        "le_split_pct": P.share_pct(int((rtt <= SPLIT_MS).sum()), n),
    }


def stats_table(tgs: pd.DataFrame, run_ids: list[str]) -> pd.DataFrame:
    """One row per `(run, side)` in draw order, plus `all` rows when there is more than one run.

    A side with no TGs in a run is kept as a row with `n_tgs = 0`, so the CSV
    says it is empty rather than leaving it out.
    """
    scopes = [(r, tgs[tgs.run_id == r]) for r in run_ids]
    if len(run_ids) > 1:
        scopes.append((ALL_RUNS, tgs))
    rows = []
    for scope, block in scopes:
        for flag, _, name in GROUPS:
            rows.append({
                "run_id": scope,
                "dataset": dataset_label(scope) if scope != ALL_RUNS else ALL_RUNS,
                "side": name,
                GROUP_COL: flag,
                **_block(block[block[GROUP_COL] == flag], len(block),
                         block[["run_id", S.SITE_KEY_COL]].drop_duplicates().shape[0]),
            })
    return pd.DataFrame(rows)


EXCEPTIONS_NAME = "x_cell_rtt.exceptions.csv"


def exceptions_table(tgs: pd.DataFrame, rtt_norm_ms: R.Bounds | None = None) -> pd.DataFrame:
    """The sites whose TGs sit on the wrong side of `SPLIT_MS`.

    Wrong side: a has-X TG above the split, or a no-X TG at or below it. One
    row per `(run, site, side)` with at least one such TG. `site` is the
    site's index among its run's sites in sorted `site_key` order -- stable,
    and free of the coordinate the key embeds. `clusters` joins the TGs'
    `plot-pni-gap` clusters when the frame carries them.
    """
    t = tgs.copy()
    t["site"] = t.groupby("run_id")[S.SITE_KEY_COL].rank(method="dense").astype(int) - 1
    across = np.where(t[GROUP_COL].astype(bool), t[FLOOR_COL] > SPLIT_MS, t[FLOOR_COL] <= SPLIT_MS)
    t = t[across]
    cols = ["run_id", "dataset", "side", "site", "clusters", "n_tgs_site", "n_tgs_across",
            "min_rtt_ms", "max_rtt_ms"]
    if rtt_norm_ms is not None:
        cols += ["min_rtt_norm", "max_rtt_norm"]
    if t.empty:
        return pd.DataFrame(columns=cols)
    sizes = tgs.groupby(["run_id", S.SITE_KEY_COL]).size()
    rows = []
    for (run_id, key, flag), g in t.groupby(["run_id", S.SITE_KEY_COL, GROUP_COL], sort=True):
        row = {
            "run_id": run_id,
            "dataset": dataset_label(run_id),
            "side": dict((f, name) for f, _, name in GROUPS)[bool(flag)],
            "site": int(g["site"].iloc[0]),
            "clusters": (
                "+".join(str(c) for c in sorted(g["cluster"].dropna().unique()))
                if "cluster" in g else ""
            ),
            "n_tgs_site": int(sizes.loc[(run_id, key)]),
            "n_tgs_across": int(len(g)),
            "min_rtt_ms": float(g[FLOOR_COL].min()),
            "max_rtt_ms": float(g[FLOOR_COL].max()),
        }
        if rtt_norm_ms is not None:
            row["min_rtt_norm"] = float(R.to_norm(row["min_rtt_ms"], rtt_norm_ms))
            row["max_rtt_norm"] = float(R.to_norm(row["max_rtt_ms"], rtt_norm_ms))
        rows.append(row)
    return pd.DataFrame(rows, columns=cols).sort_values(["run_id", "side", "site"]).reset_index(drop=True)


def outliers(tgs: pd.DataFrame, stats: pd.DataFrame) -> dict[tuple[str, bool], list[float]]:
    """Per `(run, side)`, the TG floors outside that box's p5/p95 whiskers."""
    out = {}
    for _, r in stats[stats.run_id != ALL_RUNS].iterrows():
        g = tgs[(tgs.run_id == r.run_id) & (tgs[GROUP_COL] == r[GROUP_COL])]
        out[(r.run_id, bool(r[GROUP_COL]))] = (
            R._outside(g[FLOOR_COL].to_numpy(float), r.p5_ms, r.p95_ms) if r.n_tgs else [])
    return out


# -- figure -------------------------------------------------------------------


def positions(run_ids: list[str]) -> dict[tuple[str, bool], float]:
    """Box centre per `(run, side)`: two slots per run, `GROUP_GAP` between runs."""
    step = len(GROUPS) * SLOT + GROUP_GAP
    return {(r, flag): 1.0 + i * step + j * SLOT
            for i, r in enumerate(run_ids) for j, (flag, _, _) in enumerate(GROUPS)}


def _y_limits(values: np.ndarray, split: float, ticks: tuple[float, ...]) -> tuple[float, float]:
    v = values[np.isfinite(values) & (values > 0)]
    if not v.size:
        return ticks[2], ticks[-3]
    return float(min(v.min(), split)) / Y_PAD, float(max(v.max(), split)) * Y_PAD


def panel_names(run_ids: list[str]) -> dict[str, str]:
    """Each run's dataset label minus the leading words every run's label shares.

    One run keeps its whole label: there is nothing to compare it against.
    """
    words = {r: dataset_label(r).split() for r in run_ids}
    shared = 0
    if len(run_ids) > 1:
        for column in zip(*words.values()):
            if len(set(column)) > 1:
                break
            shared += 1
    return {r: " ".join(w[shared:]) or " ".join(w) for r, w in words.items()}


def plot(
    stats: pd.DataFrame,
    *,
    run_ids: list[str],
    out_png: Path,
    fliers: dict[tuple[str, bool], list[float]] | None = None,
    rtt_norm_ms: R.Bounds | None = None,
) -> Path:
    """Two boxes per run, has-X then no-X, on a log axis; `fliers` drawn as open circles.

    With `rtt_norm_ms`, every value is drawn normalized by those bounds.
    """
    def y(v):
        return v if rtt_norm_ms is None else R.to_norm(np.asarray(v, dtype=float), rtt_norm_ms)

    stats = stats[stats.run_id != ALL_RUNS]
    pos = positions(run_ids)
    names = panel_names(run_ids)
    fig, ax = plt.subplots(figsize=(GROUP_WIDTH_IN * len(run_ids) + 0.5, HEIGHT_IN))
    ticks, labels = [], []
    extremes = [y(stats.min_ms.to_numpy(float)), y(stats.max_ms.to_numpy(float))]
    for _, r in stats.iterrows():
        flag = bool(r[GROUP_COL])
        x = pos[(r.run_id, flag)]
        tick = next(t for f, t, _ in GROUPS if f == flag)
        ticks.append(x)
        labels.append(f"{tick}\n{r.sites_pct:.0f}% sites")
        if not r.n_tgs:
            continue
        box = {**bxp_stats({k: float(y(r[f"{k}_ms"])) for k in STAT_KEYS}),
               "fliers": list(y((fliers or {}).get((r.run_id, flag), [])))}
        art = ax.bxp([box], positions=[x], widths=BOX_WIDTH, patch_artist=True, showfliers=True,
                     medianprops={"color": INK, "lw": 0.9},
                     whiskerprops={"color": INK_2, "lw": 0.6}, capprops={"color": INK_2, "lw": 0.6})
        art["boxes"][0].set(facecolor=HUES[flag], alpha=0.45, edgecolor=HUES[flag], linewidth=0.6)
        art["fliers"][0].set(marker="o", markersize=OUTLIER_MS_PT, markerfacecolor="none",
                             markeredgecolor=HUES[flag], markeredgewidth=0.5, linestyle="none")
        median = float(y(r.p50_ms))
        ax.text(x + BOX_WIDTH / 2 + 0.05, median,
                f"{median:.1f}" if rtt_norm_ms is None else f"{median:.3f}", fontsize=5.5,
                va="center", color=INK_2)
    # The network name above its pair, in axes coordinates so it clears every box.
    for rid in run_ids:
        mid = np.mean([pos[(rid, f)] for f, _, _ in GROUPS])
        ax.text(mid, 1.0, names[rid], transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6, color=INK)
    split = float(y(SPLIT_MS))
    ax.axhline(split, color=INK_2, lw=0.4, ls=":", zorder=0)
    ax.set_yscale("log")
    y_ticks = Y_TICKS_MS if rtt_norm_ms is None else Y_TICKS_NORM
    lo, hi = _y_limits(np.concatenate(extremes), split, y_ticks)
    ax.set_ylim(lo, hi)
    shown = [t for t in y_ticks if lo <= t <= hi]
    if shown:
        ax.set_yticks(shown, [f"{t:g}" if rtt_norm_ms is None
                              else rf"$10^{{{int(round(np.log10(t)))}}}$" for t in shown])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks(ticks, labels, fontsize=5.5)
    ax.set_xlim(min(pos.values()) - SLOT * 0.6, max(pos.values()) + SLOT * 0.8)
    ax.set_ylabel(QUANTITY if rtt_norm_ms is None else QUANTITY_NORM, fontsize=6)
    ax.tick_params(axis="both", labelsize=5.5, length=2, pad=1.5)
    ax.grid(axis="y", alpha=0.25, lw=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_png


# -- the whole step -----------------------------------------------------------


def _manifest(meta: dict, tgs: pd.DataFrame, stats: pd.DataFrame,
              rtt_norm_ms: R.Bounds | None = None) -> str:
    return json.dumps(
        {
            "figure": PNG_NAME,
            "csv": CSV_NAME,
            "exceptions_csv": EXCEPTIONS_NAME,
            "layout": meta.get("layout"),
            "run_ids": meta["run_ids"],
            "clusters_from": P.MANIFEST_NAME,
            "inputs": [
                {key: r[key] for key in ("run_id", "source_csv_sha256", "pni_csv_sha256")}
                for r in meta["runs"]
            ],
            "nside": M.SOURCE_NSIDE,
            "quantity": "each TG's smallest RTT over the fleet, i.e. its S-P VP's RTT",
            "grouping": f"{GROUP_COL}: {M.DEFINITIONS['rule_correct']} (X = {M.DEFINITIONS['x']})",
            "n_tgs_x_vs_any_interconnect_disagree": int((tgs[GROUP_COL] != tgs[ANY_COL]).sum()),
            "split_ms": SPLIT_MS,
            "rtt_norm_ms": (
                None if rtt_norm_ms is None else {
                    "min": rtt_norm_ms[0],
                    "max": rtt_norm_ms[1],
                    "source": "analysis.common.rtt_norm_ms in each run's config",
                    "rule": "every drawn RTT r is (r - min) / (max - min); the CSV adds *_norm",
                    "split_norm": float(R.to_norm(SPLIT_MS, rtt_norm_ms)),
                }
            ),
            "tick_labels": "side's share of its network's sites (sites_pct); counts in the CSV",
            "split_note": "a reading aid drawn as a dotted line and counted in n_le_split; not fitted",
            "boxes": (
                "whiskers p5/p95, hinges p25/p75, line = median, log y axis; every TG "
                "outside p5/p95 drawn as an open circle (n_outliers in the CSV)."
            ),
            "replicas_note": (
                "~20 TGs share a site's VP geometry; n_sites, not n_tgs, is the number of "
                "independent observations per box, and a whisker end can be one site."
            ),
            "n_rows": int(len(stats)),
        },
        indent=2,
    )


def _check_bounds(tgs: pd.DataFrame, bounds: R.Bounds) -> None:
    """Refuse a TG floor outside the declared bounds: the axis says it holds them all."""
    R.check_bounds(tgs.rename(columns={FLOOR_COL: "rtt_ms"}), bounds)


def _write(tgs: pd.DataFrame, meta: dict, out_dir: Path,
           rtt_norm_ms: R.Bounds | None = None) -> Path:
    run_ids = list(meta["run_ids"])
    stats = stats_table(tgs, run_ids)
    if rtt_norm_ms is not None:
        _check_bounds(tgs, rtt_norm_ms)
    table = stats if rtt_norm_ms is None else R.normalized(stats, rtt_norm_ms)
    table.to_csv(out_dir / CSV_NAME, index=False)
    exceptions_table(tgs, rtt_norm_ms).to_csv(out_dir / EXCEPTIONS_NAME, index=False)
    (out_dir / MANIFEST_NAME).write_text(_manifest(meta, tgs, stats, rtt_norm_ms) + "\n")
    return plot(stats, run_ids=run_ids, out_png=out_dir / PNG_NAME, fliers=outliers(tgs, stats),
                rtt_norm_ms=rtt_norm_ms)


def build_for_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layouts: tuple[str, ...] = (P.PER_RUN,),
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
    rtt_norm_ms: dict[str, R.Bounds | None] | None = None,
) -> list[Path]:
    """The boxes for every requested layout, each beside the clusters it reads.

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
            pngs.append(_write(*load_runs([run], pni_csvs, analysis_root=analysis_root, source_csvs=one),
                               R.common_norm([run.run_id], rtt_norm_ms)))
    if P.POOLED in layouts:
        norm = R.common_norm([r.run_id for r in runs], rtt_norm_ms)
        pngs.append(_write(*load_runs(runs, pni_csvs, layout=P.POOLED, analysis_root=analysis_root,
                                      source_csvs=source_csvs), norm))
    return pngs


def build_for_run(
    run: RunPaths,
    pni_csv: Path,
    *,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
    rtt_norm_ms: R.Bounds | None = None,
) -> list[Path]:
    """CSV, manifest and PNG for one run, per-run layout."""
    return build_for_runs(
        [run], {run.run_id: pni_csv}, analysis_root=analysis_root,
        source_csvs={run.run_id: source_csv} if source_csv is not None else None,
        rtt_norm_ms={run.run_id: rtt_norm_ms},
    )
