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
draws only the per-run pairs.

No coordinate, VP id or interconnect id is written.

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

#: The dotted line, in ms. A reading aid; see the module docstring.
SPLIT_MS = 3.0

#: Log-axis ticks, in ms; only those inside the data's range are drawn.
Y_TICKS_MS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0)
#: Headroom either side of the data on the log axis, as a factor.
Y_PAD = 1.5

#: Width per run group and the fixed height, in inches.
GROUP_WIDTH_IN = 1.2
HEIGHT_IN = 2.1
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
    out = tgs[key + [GROUP_COL, ANY_COL, S.SITE_KEY_COL]].merge(
        floor, on=key, how="outer", validate="1:1", indicator=True)
    lost = out[out._merge != "both"]
    if len(lost):
        raise ValueError(f"{len(lost)} TGs have a has-X flag or an RTT but not both, e.g. "
                         f"{lost.tg_id.iloc[0]!r}; re-run `classify` and `plot-pni-gap` on the same runs.")
    return out.drop(columns="_merge"), meta, out_dir


def _block(g: pd.DataFrame, n_run_tgs: int) -> dict:
    rtt = g[FLOOR_COL].to_numpy(float)
    stats = cost_stats(rtt)
    n = int(stats.pop("n"))
    return {
        "n_tgs": n,
        "tgs_pct": P.share_pct(n, n_run_tgs),
        "n_sites": int(g[S.SITE_KEY_COL].nunique()),
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
                **_block(block[block[GROUP_COL] == flag], len(block)),
            })
    return pd.DataFrame(rows)


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


def _y_limits(values: np.ndarray) -> tuple[float, float]:
    v = values[np.isfinite(values) & (values > 0)]
    if not v.size:
        return Y_TICKS_MS[2], Y_TICKS_MS[-3]
    return float(min(v.min(), SPLIT_MS)) / Y_PAD, float(max(v.max(), SPLIT_MS)) * Y_PAD


def plot(
    stats: pd.DataFrame,
    *,
    run_ids: list[str],
    out_png: Path,
    fliers: dict[tuple[str, bool], list[float]] | None = None,
) -> Path:
    """Two boxes per run, has-X then no-X, on a log axis; `fliers` drawn as open circles."""
    stats = stats[stats.run_id != ALL_RUNS]
    pos = positions(run_ids)
    fig, ax = plt.subplots(figsize=(GROUP_WIDTH_IN * len(run_ids) + 0.5, HEIGHT_IN))
    ticks, labels, extremes = [], [], [stats.min_ms.to_numpy(float), stats.max_ms.to_numpy(float)]
    for _, r in stats.iterrows():
        flag = bool(r[GROUP_COL])
        x = pos[(r.run_id, flag)]
        tick = next(t for f, t, _ in GROUPS if f == flag)
        ticks.append(x)
        labels.append(f"{tick}\n{int(r.n_tgs)} TGs\n{int(r.n_sites)} sites")
        if not r.n_tgs:
            continue
        box = {**bxp_stats({k: r[f"{k}_ms"] for k in STAT_KEYS}),
               "fliers": (fliers or {}).get((r.run_id, flag), [])}
        art = ax.bxp([box], positions=[x], widths=BOX_WIDTH, patch_artist=True, showfliers=True,
                     medianprops={"color": INK, "lw": 0.9},
                     whiskerprops={"color": INK_2, "lw": 0.6}, capprops={"color": INK_2, "lw": 0.6})
        art["boxes"][0].set(facecolor=HUES[flag], alpha=0.45, edgecolor=HUES[flag], linewidth=0.6)
        art["fliers"][0].set(marker="o", markersize=OUTLIER_MS_PT, markerfacecolor="none",
                             markeredgecolor=HUES[flag], markeredgewidth=0.5, linestyle="none")
        ax.text(x + BOX_WIDTH / 2 + 0.05, r.p50_ms, f"{r.p50_ms:.1f}", fontsize=5.5,
                va="center", color=INK_2)
    # The network name above its pair, in axes coordinates so it clears every box.
    for rid in run_ids:
        mid = np.mean([pos[(rid, f)] for f, _, _ in GROUPS])
        ax.text(mid, 1.0, dataset_label(rid), transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=6, color=INK)
    ax.axhline(SPLIT_MS, color=INK_2, lw=0.4, ls=":", zorder=0)
    ax.set_yscale("log")
    lo, hi = _y_limits(np.concatenate(extremes))
    ax.set_ylim(lo, hi)
    shown = [t for t in Y_TICKS_MS if lo <= t <= hi]
    if shown:
        ax.set_yticks(shown, [f"{t:g}" for t in shown])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xticks(ticks, labels, fontsize=5.5)
    ax.set_xlim(min(pos.values()) - SLOT * 0.6, max(pos.values()) + SLOT * 0.8)
    ax.set_ylabel(QUANTITY, fontsize=6)
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


def _manifest(meta: dict, tgs: pd.DataFrame, stats: pd.DataFrame) -> str:
    return json.dumps(
        {
            "figure": PNG_NAME,
            "csv": CSV_NAME,
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


def _write(tgs: pd.DataFrame, meta: dict, out_dir: Path) -> Path:
    run_ids = list(meta["run_ids"])
    stats = stats_table(tgs, run_ids)
    stats.to_csv(out_dir / CSV_NAME, index=False)
    (out_dir / MANIFEST_NAME).write_text(_manifest(meta, tgs, stats) + "\n")
    return plot(stats, run_ids=run_ids, out_png=out_dir / PNG_NAME, fliers=outliers(tgs, stats))


def build_for_runs(
    runs: list[RunPaths],
    pni_csvs: dict[str, Path],
    *,
    layouts: tuple[str, ...] = (P.PER_RUN,),
    analysis_root: Path | None = None,
    source_csvs: dict[str, Path] | None = None,
) -> list[Path]:
    """The boxes for every requested layout, each beside the clusters it reads."""
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


def build_for_run(
    run: RunPaths,
    pni_csv: Path,
    *,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
) -> list[Path]:
    """CSV, manifest and PNG for one run, per-run layout."""
    return build_for_runs(
        [run], {run.run_id: pni_csv}, analysis_root=analysis_root,
        source_csvs={run.run_id: source_csv} if source_csv is not None else None,
    )
