"""How many VPs measured each TG: the distribution, and its has-X / no-X split.

Two figures over the same per-TG count, `n_obs` -- the observations a TG was
**scored** on, after every eval-side filter (`min_obs`, the traffic mask), as
`classify` carries it into `shortest_ping_tgs.parquet`. That is the count the
single-VP accuracy split (`classify.summarize`) and the `vp` outcome bars
(`figure_outcome_bars`) are cut on, so all three agree on which TGs are
single-VP by construction. On a mesh nearly every TG sits near the full fleet;
on a traffic-filtered (TF) run the filter leaves some with one or two.

* `plot-vp-obs-hist` -- **the distribution**: share of TGs per VP-count bin
  (`HIST_BINS`), one bar per bin. The single-VP bar is orange and every other
  bar blue, the same orange as the `vp` outcome bars' 1-VP tier.
* `plot-vp-obs-has-x` -- **by has-X / no-X**: two 100%-stacked bars per panel,
  the TGs whose own cell holds their nearest interconnect X and those whose
  cell does not, each broken down by `SPLIT_BINS`. It answers whether the
  single-VP TGs concentrate on one side. The CSV twin also carries the other
  direction -- of the TGs in a bin, the share on each side -- which is what
  "where do the single-VP TGs sit" reads off directly.

## The has-X flag

Read from `report-sp-pni-cells`' per-TG CSV (`tg_cell_holds_x`) through
`loso_delta.attach_has_x`, the one reader of that file, so the split is the
paper's has-X / no-X and not a second derivation of it. A run without exactly
one such CSV, or with a scored TG the CSV does not hold, is refused: a TG with
no flag would silently fall out of both bars.

## Unit and pooling

The unit is a **TG**, matching every accuracy denominator. About 20 IP
replicas share a site and most of its VP set, so the curve is close to a
site-level one repeated per replica; the CSV twin and manifest carry the share
of *sites* holding a single-VP TG beside the TG share rather than drawing a
second curve.

Layouts as `figure_outcome_bars`: `compare` is one panel per dataset, `pooled`
one panel over every run's TGs (micro-average, `cross.guard_disjoint_tgs`).
Shares only on the figure; counts stay in the CSV twin and manifest.

Commands: `plot-vp-obs-hist`, `plot-vp-obs-has-x`. Need `classify` on every
run (re-run it if its parquets predate `n_obs`); the has-X figure also needs
`plot-pni-gap` and `report-sp-pni-cells` per run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.figure_outcome_bars import COMPARE, LAYOUTS, POOLED, TIER_INK
from scripts.analysis.v5.modules.loso_delta import HAS_X, NO_X, attach_has_x
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING

#: The one rung read. `n_obs` does not depend on the grid, and the has-X flag
#: is `report-sp-pni-cells`', which is built at this rung.
SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: `(lo, hi, label)`, inclusive; `hi=None` is open. Single counts where the
#: interesting mass is, then widening bins out to the fleet.
HIST_BINS: tuple[tuple[int, int | None, str], ...] = (
    (1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4–5"), (6, 10, "6–10"),
    (11, 20, "11–20"), (21, 50, "21–50"), (51, 100, "51–100"), (101, None, ">100"),
)

#: The has-X figure's stack, bottom-up. Coarser than `HIST_BINS`: a stacked bar
#: holds four segments legibly, and the question is the single-VP share.
SPLIT_BINS: tuple[tuple[int, int | None, str], ...] = (
    (1, 1, "1 VP"), (2, 2, "2 VPs"), (3, 10, "3–10 VPs"), (11, None, ">10 VPs"),
)

#: Single-VP orange, shared with the `vp` outcome bars; the rest blue.
VP1_INK = TIER_INK["vp1"]
MULTI_INK = TIER_INK["vp3plus"]

#: `SPLIT_BINS` fills: the 1-VP orange, then a three-step blue ramp light to
#: dark for the ordered multi-VP bins. Validated (dataviz validate_palette):
#: adjacent CVD dE >= 15.2 (worst deutan 19.0 on the dark pair), normal-vision
#: floor 19.5; as an ordinal ramp the blues are monotone in lightness, single
#: hue, and the light end clears 2:1 on the surface. The light blue fails the
#: categorical chroma floor, as a ramp's light step does; every segment is
#: labelled in place.
SPLIT_INK: dict[str, str] = {
    "1 VP": VP1_INK, "2 VPs": TIER_INK["vp2"], "3–10 VPs": TIER_INK["vp3plus"], ">10 VPs": "#104281",
}

SIDES: tuple[str, ...] = (HAS_X, NO_X)

HIST = "hist"
HAS_X_FIG = "has-x"
FIGURES: tuple[str, ...] = (HIST, HAS_X_FIG)

#: `{figure: {layout: stem}}`; each stem takes `.png`, `.csv`, `.manifest.json`.
STEMS = {
    HIST: {COMPARE: "vp_obs_hist", POOLED: "vp_obs_hist.pooled"},
    HAS_X_FIG: {COMPARE: "vp_obs_has_x", POOLED: "vp_obs_has_x.pooled"},
}

#: Below this share a bar or segment carries no label; the value is in the CSV.
_LABEL_FLOOR = 0.01
_SEGMENT_LABEL_FLOOR = 0.05

_SURFACE = "#ffffff"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"

#: Per-dataset panels at slide-ish size; pooled at one paper column, 300 dpi.
_PANEL_W, _FIG_H = 3.2, 2.6
POOLED_PANEL_W, POOLED_FIG_H = 3.4, 2.0
FONTS = {"label": 7.5, "title": 10.0, "tick": 8.5, "ylabel": 9.5, "legend": 8.5}
COLUMN_FONTS = {"label": 6.0, "title": 8.0, "tick": 6.5, "ylabel": 7.5, "legend": 6.5}
_PANEL_TAGS = "abcdefghijklmnopqrstuvwxyz"


def bin_labels(bins) -> list[str]:
    return [lab for _, _, lab in bins]


def bin_of(n_obs: pd.Series, bins) -> pd.Series:
    """Each count's bin label, as an ordered categorical. A count of 0 or below
    is refused: a scored TG was measured by at least one VP."""
    v = n_obs.to_numpy(dtype=int)
    if (v < 1).any():
        raise ValueError(f"{int((v < 1).sum())} TG(s) carry n_obs < 1; a scored TG has a VP")
    out = np.empty(len(v), dtype=object)
    for lo, hi, lab in bins:
        out[(v >= lo) & ((v <= hi) if hi is not None else True)] = lab
    return pd.Series(pd.Categorical(out, categories=bin_labels(bins), ordered=True), index=n_obs.index)


def load_tgs(run: RunPaths, *, analysis_root: Path | None = None,
             with_has_x: bool = False) -> pd.DataFrame:
    """One row per scored TG of `run`: `run_id, dataset, tg_id, site_id, n_obs`
    (and `has_x` when asked), off its Shortest-Ping parquet."""
    path = run.classify_dir(SOURCE_NSIDE, root=analysis_root) / C.TGS_PARQUET.format(
        method=SHORTEST_PING
    )
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `classify --run-id {run.run_id}` first")
    if C.N_OBS not in pq.read_schema(path).names:
        raise MissingArtifactError(
            f"{path} has no {C.N_OBS!r} column; it predates it -- re-run "
            f"`classify --run-id {run.run_id}`"
        )
    df = pd.read_parquet(path, columns=["tg_id", "site_id", C.N_OBS])
    df.insert(0, "run_id", run.run_id)
    df.insert(1, "dataset", cross.short_dataset(run.run_id))
    if not with_has_x:
        return df
    df, source = attach_has_x(df, run, analysis_root)
    if source is None:
        raise MissingArtifactError(
            f"{run.run_id}: no single `sp_pni_cells_tgs.csv` under its pni-gap/; run "
            f"`plot-pni-gap --run-id {run.run_id}` then `report-sp-pni-cells --run-id {run.run_id}`"
        )
    if df["has_x"].isna().any():
        raise ValueError(
            f"{run.run_id}: {int(df['has_x'].isna().sum())} scored TG(s) are not in {source}; "
            f"rebuild report-sp-pni-cells from the same classify output"
        )
    df.attrs["has_x_source"] = source
    return df


def panels(frames: list[pd.DataFrame], layout: str) -> list[tuple[str, pd.DataFrame]]:
    """`(panel, TGs)`: one per dataset for `compare`, one over all for `pooled`."""
    if layout == COMPARE:
        return [(f["dataset"].iloc[0], f) for f in frames]
    if layout == POOLED:
        cross.guard_disjoint_tgs({f["run_id"].iloc[0]: set(f["tg_id"]) for f in frames})
        return [(cross.dataset_slug([f["run_id"].iloc[0] for f in frames]), pd.concat(frames))]
    raise ValueError(f"unknown layout {layout!r}; pick from {list(LAYOUTS)}")


def panel_stats(tgs: pd.DataFrame) -> dict:
    """The numbers a reader quotes off a panel: single-VP shares of TGs and sites."""
    one = tgs[C.N_OBS] == 1
    return {
        "n_tgs": int(len(tgs)),
        "n_sites": int(tgs["site_id"].nunique()),
        "single_vp_share": round(float(one.mean()), 4),
        "single_vp_sites_share": round(
            float(tgs.loc[one, "site_id"].nunique() / tgs["site_id"].nunique()), 4
        ),
        "n_obs_min": int(tgs[C.N_OBS].min()),
        "n_obs_median": float(tgs[C.N_OBS].median()),
    }


def _site_key(tgs: pd.DataFrame) -> pd.Series:
    """`site_id` is per run, so a pooled panel keys sites on `(run_id, site_id)`."""
    return tgs["run_id"].astype(str) + ":" + tgs["site_id"].astype(str)


def hist_table(frames: list[pd.DataFrame], layout: str) -> pd.DataFrame:
    """One row per panel x bin: TGs in the bin and their share of the panel."""
    rows = []
    for panel, tgs in panels(frames, layout):
        counts = bin_of(tgs[C.N_OBS], HIST_BINS).value_counts(sort=False)
        for lab in bin_labels(HIST_BINS):
            rows.append({"panel": panel, "bin": lab, "n_tgs": int(counts[lab]),
                         "share": float(counts[lab] / len(tgs))})
    return pd.DataFrame(rows)


def has_x_table(frames: list[pd.DataFrame], layout: str) -> pd.DataFrame:
    """One row per panel x side x bin.

    `share_of_side`: of the side's TGs, the share in the bin -- what the bar
    draws. `share_of_bin`: of the bin's TGs, the share on this side -- where
    the single-VP TGs sit. `side_share`: the side's share of the panel.
    """
    rows = []
    for panel, tgs in panels(frames, layout):
        b = bin_of(tgs[C.N_OBS], SPLIT_BINS)
        for side in SIDES:
            on = (tgs["has_x"] == side).to_numpy()
            for lab in bin_labels(SPLIT_BINS):
                in_bin = (b == lab).to_numpy()
                n = int((on & in_bin).sum())
                rows.append({
                    "panel": panel, "side": side, "bin": lab, "n_tgs": n,
                    "share_of_side": n / on.sum() if on.any() else np.nan,
                    "share_of_bin": n / in_bin.sum() if in_bin.any() else np.nan,
                    "side_share": float(on.mean()),
                })
    return pd.DataFrame(rows)


def guard_closes(table: pd.DataFrame, share_col: str, keys: list[str]) -> None:
    """Every drawn bar's shares sum to one."""
    sums = table.groupby(keys)[share_col].sum(min_count=1).dropna()
    bad = sums[~np.isclose(sums, 1.0)]
    if len(bad):
        raise ValueError(f"{share_col} does not close at 1 for {bad.to_dict()}")


# ---- drawing -----------------------------------------------------------------


def _style_axis(ax, *, first: bool, pt: dict, top: float = 1.0) -> None:
    ax.set_facecolor(_SURFACE)
    ax.set_ylim(0, top)
    ticks = np.arange(0, top + 1e-9, 0.2 if top > 0.5 else 0.1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{int(v * 100)}%" for v in ticks])
    ax.grid(True, axis="y", color=_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(first)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_MUTED, labelsize=pt["tick"])
    if not first:
        ax.tick_params(axis="y", length=0)


def _title(ax, panel: str, i: int, n_panels: int, pt: dict) -> None:
    """Dataset name, tagged `(a)` when there is more than one; a pooled panel
    (its slug joins several datasets) is not headed, as in the outcome bars."""
    if "+" in panel:
        return
    tag = f"({_PANEL_TAGS[i]}) " if n_panels > 1 else ""
    ax.set_title(f"{tag}{panel.upper()}", fontsize=pt["title"], color=_INK, pad=4)


def _figure(n_panels: int, layout: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if layout == POOLED:
        size, pt, dpi = (POOLED_PANEL_W, POOLED_FIG_H), COLUMN_FONTS, 300
    else:
        size, pt, dpi = (_PANEL_W * n_panels, _FIG_H), FONTS, 150
    fig, axes = plt.subplots(1, n_panels, figsize=size, sharey=True, squeeze=False)
    fig.patch.set_facecolor(_SURFACE)
    return plt, fig, axes[0], pt, dpi


def _save(plt, fig, png: Path, dpi: int) -> Path:
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=dpi, facecolor=_SURFACE, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return png


def render_hist(table: pd.DataFrame, png: Path, layout: str) -> Path:
    """Share of TGs per VP-count bin; the single-VP bar orange."""
    names = list(dict.fromkeys(table["panel"]))
    plt, fig, axes, pt, dpi = _figure(len(names), layout)
    labels = bin_labels(HIST_BINS)
    xs = np.arange(len(labels))
    # A distribution, not a composition: the axis stops one tick above the
    # tallest bar of any panel, shared so the panels still compare.
    top = min(1.0, np.ceil((table["share"].max() + 0.04) * 10) / 10)
    for i, (ax, panel) in enumerate(zip(axes, names)):
        sub = table[table["panel"] == panel].set_index("bin").reindex(labels)
        shares = sub["share"].to_numpy(dtype=float)
        faces = [VP1_INK if lab == labels[0] else MULTI_INK for lab in labels]
        ax.bar(xs, shares, width=0.72, color=faces, edgecolor=_SURFACE, linewidth=1.0, zorder=3)
        for x, v in zip(xs, shares):
            if v >= _LABEL_FLOOR:
                ax.text(x, v + 0.015, f"{v * 100:.0f}%", ha="center", va="bottom",
                        fontsize=pt["label"], color=_INK_2, zorder=4)
        _style_axis(ax, first=i == 0, pt=pt, top=top)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=pt["tick"], color=_INK_2, rotation=45, ha="right",
                           rotation_mode="anchor")
        ax.set_xlabel("VPs measuring the TG", fontsize=pt["ylabel"], color=_INK_2)
        _title(ax, panel, i, len(names), pt)
    axes[0].set_ylabel("Share of TGs", fontsize=pt["ylabel"], color=_INK_2)
    fig.tight_layout()
    return _save(plt, fig, png, dpi)


def render_has_x(table: pd.DataFrame, png: Path, layout: str) -> Path:
    """Per panel, has-X and no-X as 100% stacks of `SPLIT_BINS`, 1 VP at the base."""
    from matplotlib.patches import Patch

    names = list(dict.fromkeys(table["panel"]))
    plt, fig, axes, pt, dpi = _figure(len(names), layout)
    labels = bin_labels(SPLIT_BINS)
    for i, (ax, panel) in enumerate(zip(axes, names)):
        sub = table[table["panel"] == panel]
        ticks = []
        for x, side in enumerate(SIDES):
            rows = sub[sub["side"] == side].set_index("bin").reindex(labels)
            bottom = 0.0
            for lab in labels:
                v = float(rows.at[lab, "share_of_side"]) if pd.notna(rows.at[lab, "share_of_side"]) else 0.0
                if v > 0:
                    face = SPLIT_INK[lab]
                    ax.bar(x, v, bottom=bottom, width=0.55, color=face,
                           edgecolor=_SURFACE, linewidth=1.0, zorder=3)
                    if v >= _SEGMENT_LABEL_FLOOR:
                        ax.text(x, bottom + v / 2, f"{v * 100:.0f}%", ha="center", va="center",
                                fontsize=pt["label"], color=_contrast_ink(face), zorder=4)
                bottom += v
            ticks.append(f"{side}\n{rows['side_share'].iloc[0] * 100:.0f}% of TGs")
        _style_axis(ax, first=i == 0, pt=pt)
        ax.set_xlim(-0.6, len(SIDES) - 0.4)
        ax.set_xticks(range(len(SIDES)))
        ax.set_xticklabels(ticks, fontsize=pt["tick"], color=_INK_2)
        _title(ax, panel, i, len(names), pt)
    axes[0].set_ylabel("Share of the side's TGs", fontsize=pt["ylabel"], color=_INK_2)
    handles = [Patch(facecolor=SPLIT_INK[lab], label=lab) for lab in labels]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               fontsize=pt["legend"], labelcolor=_INK_2, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return _save(plt, fig, png, dpi)


def _contrast_ink(face: str) -> str:
    """White or ink by the fill's luminance, as `figure_outcome_bars`."""
    r, g, b = (int(face[i : i + 2], 16) / 255 for i in (1, 3, 5))
    return "#ffffff" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.55 else _INK


# ---- writing -----------------------------------------------------------------


def _manifest(figure: str, layout: str, png: Path, csv: Path, frames, stats) -> str:
    body = {
        "figure": png.name,
        "csv": csv.name,
        "kind": figure,
        "layout": layout,
        "runs": [f["run_id"].iloc[0] for f in frames],
        "count": (
            f"{C.N_OBS}: observations each TG was scored on, after every eval-side filter, "
            f"from classify's {SHORTEST_PING} parquet at nside {SOURCE_NSIDE}"
        ),
        "bins": bin_labels(HIST_BINS if figure == HIST else SPLIT_BINS),
        "panels": stats,
        "unit_note": (
            "one row per TG, as every accuracy denominator; ~20 replicas share a site and its "
            "VP set, so single_vp_sites_share is given beside single_vp_share"
        ),
        "confidentiality": "the figure prints shares only; counts are in this file and the CSV",
    }
    if figure == HAS_X_FIG:
        body["has_x"] = {
            "definition": "tg_cell_holds_x: the TG's own cell holds its nearest interconnect X",
            "sources": {f["run_id"].iloc[0]: f.attrs.get("has_x_source") for f in frames},
        }
    return json.dumps(body, indent=2) + "\n"


def build_for_runs(
    runs: list[RunPaths],
    figure: str,
    *,
    layouts: tuple[str, ...] = LAYOUTS,
    analysis_root: Path | None = None,
) -> list[Path]:
    """One PNG, CSV twin and manifest per layout. Returns the PNGs."""
    if figure not in FIGURES:
        raise ValueError(f"unknown figure {figure!r}; pick from {list(FIGURES)}")
    bad = [x for x in layouts if x not in LAYOUTS]
    if bad:
        raise ValueError(f"unknown layout {bad}; pick from {list(LAYOUTS)}")
    frames = [load_tgs(r, analysis_root=analysis_root, with_has_x=figure == HAS_X_FIG) for r in runs]
    out_dir = cross.cross_dir([r.run_id for r in runs], analysis_root=analysis_root)
    written = []
    for layout in dict.fromkeys(layouts):
        stem = STEMS[figure][layout]
        png, csv, man = (out_dir / f"{stem}{ext}" for ext in (".png", ".csv", ".manifest.json"))
        stats = {}
        for panel, tgs in panels(frames, layout):
            stats[panel] = panel_stats(tgs.assign(site_id=_site_key(tgs)))
        if figure == HIST:
            table = hist_table(frames, layout)
            guard_closes(table, "share", ["panel"])
            render_hist(table, png, layout)
        else:
            table = has_x_table(frames, layout)
            guard_closes(table, "share_of_side", ["panel", "side"])
            guard_closes(table, "share_of_bin", ["panel", "bin"])
            render_has_x(table, png, layout)
        table.to_csv(csv, index=False)
        man.write_text(_manifest(figure, layout, png, csv, frames, stats))
        written.append(png)
    return written
