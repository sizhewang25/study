"""Where each CBG method's no-X answers land relative to the ingress point: step and heading.

§4.2 shows S-P answers at a TG's ingress point X_in, so S-P's answer is the
observable X_in here. On a no-X TG, X_in lies outside the TG's cell, so a
method that answers at X_in is wrong. This figure asks how far, and which way,
each CBG method moves off X_in.

## The frame

Per TG: X_in (S-P's answer) at the origin, the TG at (1, 0), unit d(X_in, TG)
(the gap), the x axis along X_in -> TG. A method's answer p is drawn at

    step    = d(X_in, p) / d(X_in, TG)
    heading = the unsigned angle at X_in between X_in -> p and X_in -> TG, in [0°, 180°]

so answers on either side of the X_in–TG line are mirrored onto the upper
half (the side is arbitrary). `gap_closed = 1 - d(p, TG) / d(X_in, TG)` is the
share of the gap a method closes. No distance is written in km.

## One image per method

Paper-ready, no title: gap-unit axes, the upper unit half circle, X_in and the
TG (in a legend above the axes), every no-X answer, and the 95% site-bootstrap CI of the median step (radial segment, at
the median heading) and of the median heading (arc, at the median step).
Every panel shares one fixed window, x in [-1.5, 1.5] and y in [0, 1.5] gaps,
with the y axis through X_in; answers more than 1.5 gaps off X_in are drawn at
radius 1.5 on their own heading, hollow (`capped_pct`). Every number is in `step_heading.json`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Arc  # noqa: E402

from scripts.analysis.v5.modules.figure_io import save_figure
from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules.cell_margin import initial_bearing  # noqa: E402
from scripts.analysis.v5.modules.geodesy import haversine_km  # noqa: E402
from scripts.analysis.v5.modules.methods import method_colors, method_label  # noqa: E402
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths  # noqa: E402
from scripts.analysis.v5.modules.status import SHORTEST_PING  # noqa: E402

KIND = "step-heading"
JSON_NAME = "step_heading.json"
PNG_NAME = "step_heading.{term}.png"

COMBOS = ("million_scale_cbg", "vanilla_cbg", "octant_cbg_hull", "octant_cbg_spl", "spotter_cbg")
VIEW_COVERAGE = 0.95  # of each method's answers
VIEW_GRID = 0.25
NEAR_TG_GAP = 0.1
N_BOOT = 2000
#: Printed at about its own size (a fifth of the text width), so the fonts below are print sizes.
PANEL_IN = 1.4
#: Fixed window and cap, in gap units: x in [-CAP, CAP], y in [0, CAP]; steps beyond CAP drawn at CAP.
CAP = 1.5
DOT_S = 3.0
MARK_S = 14
#: Ticks, in gaps: the unit half circle's ends and X_in.
X_TICKS = (-1, 0, 1)
Y_TICKS = (1,)
FONT = 6.5
TICK_FONT = 6
#: Axes margins (left, bottom, right, top), inches: fixed so every panel is identical.
MARGIN_IN = (0.17, 0.29, 0.05, 0.17)
#: Both axes are in units of d(X_in, TG); the caption names the unit.
AXIS_LABEL_X = r"Dist. along $X_{\mathrm{in}}\!\to$TG"
AXIS_LABEL_Y = r"Dist. across"


# ---- geometry ----------------------------------------------------------------------------


def step_heading(x_lat, x_lon, tg_lat, tg_lon, p_lat, p_lon) -> pd.DataFrame:
    """`step`, `heading` (deg, unsigned), `gap_closed` per row. Vectorised; rows with a zero gap -> NaN."""
    gap = haversine_km(x_lat, x_lon, tg_lat, tg_lon)
    d_xp = haversine_km(x_lat, x_lon, p_lat, p_lon)
    d_pt = haversine_km(p_lat, p_lon, tg_lat, tg_lon)
    head = np.abs((initial_bearing(x_lat, x_lon, p_lat, p_lon)
                   - initial_bearing(x_lat, x_lon, tg_lat, tg_lon) + 180.0) % 360.0 - 180.0)
    ok = gap > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        return pd.DataFrame({
            "step": np.where(ok, d_xp / gap, np.nan),
            "heading": np.where(ok & (d_xp > 0), head, 0.0),
            "gap_closed": np.where(ok, 1.0 - d_pt / gap, np.nan),
        })


def frame_xy(step, heading) -> tuple[np.ndarray, np.ndarray]:
    th = np.radians(np.asarray(heading, float))
    return np.asarray(step, float) * np.cos(th), np.asarray(step, float) * np.sin(th)


# ---- inputs -------------------------------------------------------------------------------


def _classify(run: RunPaths, combo: str, analysis_root: Path | None) -> pd.DataFrame:
    p = run.analysis_dir("classify", root=analysis_root) / "healpix-128" / f"{combo}_tgs.parquet"
    if not p.exists():
        raise MissingArtifactError(f"{p} missing; run `classify --run-id {run.run_id}` first")
    return pd.read_parquet(p)


def load(runs: list[RunPaths], *, analysis_root: Path | None = None) -> pd.DataFrame:
    """One row per (run, method, no-X TG with an answer): step, heading, gap_closed, outcome, site."""
    from scripts.analysis.v5.modules.loso_delta import attach_has_x

    frames = []
    for run in runs:
        sp = _classify(run, SHORTEST_PING, analysis_root)[["tg_id", "pred_lat", "pred_lon"]] \
            .rename(columns={"pred_lat": "x_lat", "pred_lon": "x_lon"})
        for combo in COMBOS:
            c = _classify(run, combo, analysis_root).merge(sp, on="tg_id", validate="1:1")
            c, src = attach_has_x(c, run, analysis_root)
            if src is None:
                raise MissingArtifactError(f"{run.run_id}: no sp_pni_cells_tgs.csv; run `report-sp-pni-cells`")
            c = c[(c.has_x == "no-X") & c.pred_lat.notna() & c.x_lat.notna()].reset_index(drop=True)
            g = step_heading(c.x_lat.to_numpy(), c.x_lon.to_numpy(), c.tg_lat.to_numpy(), c.tg_lon.to_numpy(),
                             c.pred_lat.to_numpy(), c.pred_lon.to_numpy())
            frames.append(pd.concat([c[["tg_id", "site_id", "cell_label", "status"]], g], axis=1)
                          .assign(run_id=run.run_id, combo=combo, term=method_label(combo)))
    df = pd.concat(frames, ignore_index=True)
    df["site_key"] = df.run_id + ":" + df.site_id.astype(str)
    return df.dropna(subset=["step"])


# ---- statistics ---------------------------------------------------------------------------


def site_bootstrap_median(values: np.ndarray, sites: np.ndarray, *, n: int = N_BOOT, seed: int = 0) -> list[float]:
    """`[median, lo, hi]`: the 95% percentile CI of the median under resampling of sites."""
    rng = np.random.default_rng(seed)
    keys, inv = np.unique(sites, return_inverse=True)
    groups = [values[inv == k] for k in range(len(keys))]
    boots = np.empty(n)
    for b in range(n):
        boots[b] = np.median(np.concatenate([groups[i] for i in rng.integers(0, len(keys), len(keys))]))
    return [float(np.median(values)), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]


def view_extent(df: pd.DataFrame, *, coverage: float = VIEW_COVERAGE, grid: float = VIEW_GRID) -> dict:
    """The shared window: holds the unit half circle and `coverage` of all answers, on a `grid`."""
    x, y = frame_xy(df.step, df.heading)
    tail = (1 - coverage) / 2
    up = lambda v: math.ceil(v / grid) * grid  # noqa: E731
    return {"x_min": -max(1.0, up(-np.quantile(x, tail))), "x_max": max(1.0, up(np.quantile(x, 1 - tail))),
            "y_max": max(1.0, up(np.quantile(y, coverage)))}


def shared_extent(df: pd.DataFrame, *, coverage: float = VIEW_COVERAGE) -> dict:
    """The union of every method's own `view_extent`."""
    ext = [view_extent(g, coverage=coverage) for _, g in df.groupby("term")]
    return {"x_min": min(e["x_min"] for e in ext), "x_max": max(e["x_max"] for e in ext),
            "y_max": max(e["y_max"] for e in ext)}


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for term, g in df.groupby("term"):
        beyond = g.step > CAP
        sites = g.site_key.to_numpy()
        correct = g.cell_label.eq("correct")
        out[term] = {
            "step_median_ci": site_bootstrap_median(g.step.to_numpy(), sites),
            "heading_median_ci_deg": site_bootstrap_median(g.heading.to_numpy(), sites),
            "gap_closed_median_ci": site_bootstrap_median(g.gap_closed.to_numpy(), sites),
            "step_quartiles": [float(q) for q in g.step.quantile([.25, .5, .75])],
            "heading_quartiles_deg": [float(q) for q in g.heading.quantile([.25, .5, .75])],
            "near_tg_pct": round(float((g.gap_closed >= 1 - NEAR_TG_GAP).mean()) * 100, 2),
            "correct_pct": round(float(correct.mean()) * 100, 2),
            "capped_pct": round(float(beyond.mean()) * 100, 2),
            "gap_closed_median_correct_vs_wrong": [float(g.gap_closed[correct].median()) if correct.any() else None,
                                                   float(g.gap_closed[~correct].median())],
        }
    return out


# ---- figure -------------------------------------------------------------------------------


def render(g: pd.DataFrame, stats: dict, color: str, out_png: Path) -> Path:
    """One method's panel on the fixed window x in [-CAP, CAP], y in [0, CAP], y axis through x = 0.

    Answers with step > CAP are drawn at radius CAP on their own heading, hollow, so
    every dot keeps its direction. The axes sit at fixed margins (inches), so every
    panel is the same window at the same scale.
    """
    step = g.step.to_numpy(); capped = step > CAP
    x, y = frame_xy(np.minimum(step, CAP), g.heading)
    left, bottom, right, top = MARGIN_IN
    ax_w = PANEL_IN - left - right
    ax_h = ax_w * CAP / (2 * CAP)
    fh = ax_h + bottom + top
    fig = plt.figure(figsize=(PANEL_IN, fh))
    ax = fig.add_axes((left / PANEL_IN, bottom / fh, ax_w / PANEL_IN, ax_h / fh))
    ax.add_patch(Arc((0, 0), 2, 2, theta1=0, theta2=180, color="0.55", lw=0.5, ls=":"))
    ax.scatter(x[~capped], y[~capped], s=DOT_S, color=color, alpha=0.35, lw=0, zorder=3, clip_on=False)
    if capped.any():
        ax.scatter(x[capped], y[capped], s=DOT_S * 1.4, facecolor="none", edgecolor=color, lw=0.4,
                   alpha=0.8, zorder=3, clip_on=False)
    r, lo, hi = stats["step_median_ci"]
    hd, hlo, hhi = stats["heading_median_ci_deg"]
    th = math.radians(hd)
    ax.plot([lo * math.cos(th), hi * math.cos(th)], [lo * math.sin(th), hi * math.sin(th)],
            color="k", lw=1.2, solid_capstyle="butt", zorder=6)
    ax.add_patch(Arc((0, 0), 2 * r, 2 * r, theta1=hlo, theta2=hhi, color="k", lw=1.2, zorder=6))
    ax.scatter([0], [0], marker="^", s=MARK_S, color="k", zorder=8, clip_on=False, label=r"$X_{\mathrm{in}}$")
    ax.scatter([1], [0], marker="*", s=MARK_S * 1.8, color="k", zorder=8, clip_on=False, label="TG")
    # Above the axes, in the fixed top margin: never over an answer, same in every panel.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=FONT, frameon=False,
              handletextpad=0.3, columnspacing=1.2, borderaxespad=0.1)
    ax.set_aspect("equal")
    ax.set_xlim(-CAP, CAP); ax.set_ylim(0, CAP)
    ax.spines["left"].set_position(("data", 0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)
    ax.set_xticks(X_TICKS); ax.set_yticks(Y_TICKS)
    ax.tick_params(labelsize=TICK_FONT, length=2, pad=1.2, width=0.5)
    ax.set_xlabel(AXIS_LABEL_X, fontsize=FONT, labelpad=1.5)
    ax.set_ylabel(AXIS_LABEL_Y, fontsize=FONT, labelpad=1.5)
    # The y spine runs through X_in; its label sits at the frame's left edge, clear of the answers.
    ax.yaxis.set_label_coords(-0.035, 0.5)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out_png, dpi=300)
    plt.close(fig)
    return out_png


def build(runs: list[RunPaths], *, analysis_root: Path | None = None) -> dict[str, Path]:
    df = load(runs, analysis_root=analysis_root)
    stats = summarize(df)
    run_ids = sorted(r.run_id for r in runs)
    out_dir = cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)
    written = {}
    for combo in COMBOS:
        term = method_label(combo)
        written[term] = render(df[df.term == term], stats[term], method_colors([combo])[combo],
                               out_dir / PNG_NAME.format(term=term))
    meta = {
        "run_ids": run_ids, "population": "no-X TGs with an answer, from each run's own classify",
        "frame": "X_in = S-P's answer at 0, TG at (1, 0), unit = d(X_in, TG); heading unsigned (folded)",
        "ci": f"95% percentile, {N_BOOT} site-bootstrap resamples (run, site_id)",
        "view": {"x": [-CAP, CAP], "y": [0, CAP], "y_axis_at_x": 0,
                 "cap": f"steps > {CAP} drawn at radius {CAP} on their own heading, hollow (capped_pct)"},
        "near_tg": f"gap_closed >= {1 - NEAR_TG_GAP}",
        "by_method": stats,
    }
    (out_dir / JSON_NAME).write_text(json.dumps(meta, indent=1) + "\n")
    written["json"] = out_dir / JSON_NAME
    return written
