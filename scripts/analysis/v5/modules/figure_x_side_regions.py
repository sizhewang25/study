"""Error distance and the multilateration region, has-X vs no-X (§4.3).

Three paper panels on seen sites, pooled over the runs:

(a) `x_side_error.png`: normalized error distance d(pred, TG) / D of every
    method, has-X box then no-X box, over the TGs each method answers
    (`status.solved_mask`: VAN's FALLBACK points are not answers).
(b) `x_side_area.png`: region area / D^2 of the methods that multilaterate a
    region (SOI, VAN, OCT-H, OCT-S), from `report-mtl-replay`, over the TGs
    whose region is non-empty.
(c) `x_side_inside.png`: the share of those regions that hold the TG.

S-P has no region, and SPO forms a density, not a region, so both are absent
from (b) and (c). Boxes: hinges p25/p75, whiskers p5/p95, no fliers; colours as
Fig. 9 (`figure_x_cell_rtt.HUES`). Each answer lies inside its own region
(`pred_to_region_over_D` = 0), so d(TG, region) <= the error distance in (a):
a small error with a small region that misses the TG means the region sits
next to the TG.

Every number drawn is in `x_side_regions.json`; no km, no counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules import mtl_replay  # noqa: E402
from scripts.analysis.v5.modules.figure_x_cell_rtt import HUES  # noqa: E402
from scripts.analysis.v5.modules.mapping import INK, INK_2  # noqa: E402
from scripts.analysis.v5.modules.methods import method_label  # noqa: E402
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths  # noqa: E402
from scripts.analysis.v5.modules.status import SHORTEST_PING, solved_mask  # noqa: E402

KIND = "x-side-regions"
JSON_NAME = "x_side_regions.json"
PNG = {"error": "x_side_error.png", "area": "x_side_area.png", "inside": "x_side_inside.png"}

ERROR_COMBOS = (SHORTEST_PING, "million_scale_cbg", "vanilla_cbg", "octant_cbg_hull", "octant_cbg_spl",
                "spotter_cbg")
REGION_COMBOS = ("million_scale_cbg", "vanilla_cbg", "octant_cbg_hull", "octant_cbg_spl")
SIDES = ("has-X", "no-X")
SIDE_TICK = {"has-X": "has-$X$", "no-X": "no-$X$"}
QUANTILES = {"p5": 0.05, "p25": 0.25, "p50": 0.5, "p75": 0.75, "p95": 0.95}

#: Error axis in units of 1e-3 (as Fig. 4); values below the floor are drawn at it.
ERROR_SCALE = 1e3
ERROR_FLOOR = 1e-2
AREA_FLOOR = 1e-10
#: Panel sizes, inches: (a) spans a column, (b) and (c) share the row below.
SIZE_IN = {"error": (3.3, 1.25), "area": (1.62, 1.15), "inside": (1.62, 1.15)}
#: (b) and (c) form their own figure, each with the legend, and smaller method ticks so OCT-H/OCT-S do not touch.
REGION_XFONT = 5.2
BOX_W = 0.34
#: Headroom either side of the data on a log axis, as a factor.
Y_PAD = 1.6
OFFSET = 0.19


# ---- inputs ---------------------------------------------------------------------------


def _classify(run: RunPaths, combo: str, analysis_root: Path | None) -> pd.DataFrame:
    p = run.analysis_dir("classify", root=analysis_root) / "healpix-128" / f"{combo}_tgs.parquet"
    if not p.exists():
        raise MissingArtifactError(f"{p} missing; run `classify --run-id {run.run_id}` first")
    return pd.read_parquet(p)


def load_errors(runs: list[RunPaths], D_km: dict[str, float], *,
                analysis_root: Path | None = None) -> pd.DataFrame:
    """One row per (run, method, TG): side, answered, error / D (NaN when unanswered)."""
    from scripts.analysis.v5.modules.loso_delta import attach_has_x

    frames = []
    for run in runs:
        for combo in ERROR_COMBOS:
            c = _classify(run, combo, analysis_root)
            c, src = attach_has_x(c, run, analysis_root)
            if src is None:
                raise MissingArtifactError(f"{run.run_id}: no sp_pni_cells_tgs.csv; run `report-sp-pni-cells`")
            answered = solved_mask(c) & c.pred_dist_to_tg_km.notna()
            frames.append(pd.DataFrame({
                "run_id": run.run_id, "tg_id": c.tg_id, "term": method_label(combo), "has_x": c.has_x,
                "answered": answered.to_numpy(),
                "error_over_D": np.where(answered, c.pred_dist_to_tg_km / D_km[run.run_id], np.nan),
            }))
    return pd.concat(frames, ignore_index=True)


def load_regions(runs: list[RunPaths], *, analysis_root: Path | None = None) -> pd.DataFrame:
    """The seen-site region rows of `report-mtl-replay` for `REGION_COMBOS`, on `runs`."""
    frames = []
    for run in runs:
        p = run.analysis_dir(mtl_replay.KIND, root=analysis_root) / mtl_replay.TGS_NAME
        if not p.exists():
            raise MissingArtifactError(f"{p} missing; run `report-mtl-replay --run-id {run.run_id}`")
        frames.append(pd.read_parquet(p).assign(run_id=run.run_id))
    r = pd.concat(frames, ignore_index=True)
    r = r[r.variant.isin(REGION_COMBOS)].copy()
    sides = load_sides(runs, analysis_root=analysis_root)
    r = r.merge(sides, on=["run_id", "tg_id"], how="left", validate="m:1")
    if r.has_x.isna().any():
        raise ValueError("mtl-replay TGs without a has-X / no-X side")
    r["term"] = r.variant.map(method_label)
    return r


def load_sides(runs: list[RunPaths], *, analysis_root: Path | None = None) -> pd.DataFrame:
    from scripts.analysis.v5.modules.loso_delta import attach_has_x

    frames = []
    for run in runs:
        c, _ = attach_has_x(_classify(run, SHORTEST_PING, analysis_root)[["tg_id"]], run, analysis_root)
        frames.append(c[["tg_id", "has_x"]].assign(run_id=run.run_id))
    return pd.concat(frames, ignore_index=True)


# ---- statistics -----------------------------------------------------------------------


def quantiles(v: pd.Series) -> dict[str, float]:
    v = v.dropna()
    return {k: float(v.quantile(q)) if len(v) else float("nan") for k, q in QUANTILES.items()}


def summarize(err: pd.DataFrame, reg: pd.DataFrame) -> dict:
    out = {"error_over_D": {}, "area_over_D2": {}, "tg_inside_pct": {}, "answered_pct": {},
           "region_nonempty_pct": {}}
    for (term, side), g in err.groupby(["term", "has_x"]):
        out["error_over_D"].setdefault(term, {})[side] = quantiles(g.error_over_D)
        out["answered_pct"].setdefault(term, {})[side] = round(float(g.answered.mean()) * 100, 1)
    for (term, side), g in reg.groupby(["term", "has_x"]):
        ok = g[g.replay_success.astype(bool)]
        out["area_over_D2"].setdefault(term, {})[side] = quantiles(ok.area_over_D2)
        out["tg_inside_pct"].setdefault(term, {})[side] = round(float(ok.contains_tg.astype(bool).mean()) * 100, 1)
        out["region_nonempty_pct"].setdefault(term, {})[side] = round(float(g.replay_success.mean()) * 100, 1)
    return out


# ---- rendering ------------------------------------------------------------------------


def _style(ax, terms: list[str], ylabel: str, *, xfont: float = 6) -> None:
    ax.set_xticks(range(len(terms)), terms, fontsize=xfont)
    ax.set_xlim(-0.6, len(terms) - 0.4)
    ax.set_ylabel(ylabel, fontsize=6.5)
    ax.tick_params(axis="both", labelsize=6, length=2, pad=1.5)
    ax.grid(axis="y", alpha=0.25, lw=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _decade_ticks(ax, lo: float, hi: float, *, step: int = 1) -> None:
    """Decade ticks inside [lo, hi]; the limits are the data's, padded by `Y_PAD`."""
    lo, hi = lo / Y_PAD, hi * Y_PAD
    ks = range(int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi))) + 1)
    ticks = [10.0**k for k in ks if k % step == 0]
    ax.set_yticks(ticks, [rf"$10^{{{int(np.log10(t))}}}$" for t in ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(lo, hi)


def _legend(fig) -> None:
    handles = [plt.Rectangle((0, 0), 1, 1, fc=HUES[s == "has-X"], alpha=0.45, ec=HUES[s == "has-X"], lw=0.6)
               for s in SIDES]
    fig.legend(handles, [SIDE_TICK[s] for s in SIDES], loc="lower center", ncol=2, fontsize=6,
               frameon=False, bbox_to_anchor=(0.5, 0.98), handlelength=1.0, columnspacing=1.0)


def render_boxes(stats: dict, terms: list[str], *, scale: float, floor: float, ylabel: str,
                 size: tuple[float, float], out_png: Path, decade_step: int = 1,
                 legend: bool = True, xfont: float = 6) -> Path:
    """Two boxes per method (has-X left), log axis, whiskers p5/p95; values below `floor` drawn at it."""
    fig, ax = plt.subplots(figsize=size)
    lo, hi = np.inf, 0.0
    for i, term in enumerate(terms):
        for j, side in enumerate(SIDES):
            q = stats.get(term, {}).get(side)
            if not q or not np.isfinite(q["p50"]):
                continue
            v = {k: max(q[k] * scale, floor) for k in QUANTILES}
            lo, hi = min(lo, v["p5"]), max(hi, v["p95"])
            art = ax.bxp([{"whislo": v["p5"], "q1": v["p25"], "med": v["p50"], "q3": v["p75"],
                           "whishi": v["p95"], "fliers": []}],
                         positions=[i + (2 * j - 1) * OFFSET], widths=BOX_W, patch_artist=True,
                         showfliers=False, medianprops={"color": INK, "lw": 0.9},
                         whiskerprops={"color": INK_2, "lw": 0.6}, capprops={"color": INK_2, "lw": 0.6})
            art["boxes"][0].set(facecolor=HUES[side == "has-X"], alpha=0.45,
                                edgecolor=HUES[side == "has-X"], linewidth=0.6)
    ax.set_yscale("log")
    _decade_ticks(ax, lo, hi, step=decade_step)
    _style(ax, terms, ylabel, xfont=xfont)
    if legend:
        _legend(fig)
    return _save(fig, out_png)


def render_inside(stats: dict, terms: list[str], *, size: tuple[float, float], out_png: Path,
                  legend: bool = True, xfont: float = 6) -> Path:
    fig, ax = plt.subplots(figsize=size)
    for i, term in enumerate(terms):
        for j, side in enumerate(SIDES):
            v = stats.get(term, {}).get(side)
            if v is None:
                continue
            ax.bar(i + (2 * j - 1) * OFFSET, v, width=BOX_W, color=HUES[side == "has-X"], alpha=0.45,
                   edgecolor=HUES[side == "has-X"], linewidth=0.6)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    _style(ax, terms, "TG inside\nintersection (%)", xfont=xfont)
    if legend:
        _legend(fig)
    return _save(fig, out_png)


def _save(fig, out_png: Path) -> Path:
    fig.tight_layout(pad=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_png


# ---- the whole step -------------------------------------------------------------------


def build(runs: list[RunPaths], D_km: dict[str, float], *, analysis_root: Path | None = None) -> dict[str, Path]:
    err = load_errors(runs, D_km, analysis_root=analysis_root)
    reg = load_regions(runs, analysis_root=analysis_root)
    stats = summarize(err, reg)
    run_ids = sorted(r.run_id for r in runs)
    out_dir = cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)
    e_terms = [method_label(c) for c in ERROR_COMBOS]
    r_terms = [method_label(c) for c in REGION_COMBOS]
    written = {
        "error": render_boxes(stats["error_over_D"], e_terms, scale=ERROR_SCALE, floor=ERROR_FLOOR,
                              ylabel=r"Norm. error ($10^{-3}$)", size=SIZE_IN["error"],
                              out_png=out_dir / PNG["error"]),
        "area": render_boxes(stats["area_over_D2"], r_terms, scale=1.0, floor=AREA_FLOOR,
                             ylabel="Norm. intersection area", size=SIZE_IN["area"],
                             out_png=out_dir / PNG["area"], decade_step=2, xfont=REGION_XFONT),
        "inside": render_inside(stats["tg_inside_pct"], r_terms, size=SIZE_IN["inside"],
                                out_png=out_dir / PNG["inside"], xfont=REGION_XFONT),
    }
    meta = {
        "run_ids": run_ids,
        "population": {
            "error": "every scored TG a method answers (status.solved_mask), seen sites",
            "region": "TGs whose replayed region is non-empty (report-mtl-replay), seen sites",
        },
        "boxes": "hinges p25/p75, whiskers p5/p95, no fliers",
        "units": {"error_over_D": "d(pred, TG) / D; drawn x 1e3", "area_over_D2": "geodesic area / D^2"},
        "absent_from_region_panels": {"S-P": "answers at a VP, no region",
                                      "SPO": "a density over the map, no multilaterated region"},
        **stats,
    }
    (out_dir / JSON_NAME).write_text(json.dumps(meta, indent=1) + "\n")
    written["json"] = out_dir / JSON_NAME
    return written
