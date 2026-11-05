"""How much room a TG's cell leaves around it: ray-cast margins, direction-averaged and per error.

Cells are unbounded (every point on Earth has a nearest seed), so a cell has no
finite size to report. What the cell axis *does* decide is how far an answer
can move from the TG before its nearest seed changes. This module measures that
along rays.

## The margin along a ray

For a TG at t with seed s, `m(θ)` is the great-circle distance from t, along
the great circle leaving t at initial bearing θ, to the first point whose
nearest seed (great-circle, the rule `classify.score_rung` scores with) is not
s, capped at the footprint span D. Great-circle Voronoi cells are convex, so a
ray leaves the cell once: the margin is found by bisection, and an answer at
distance e along θ is correct **iff** e < m(θ). The cap loses nothing, because
every error is at most D: a ray that leaves beyond D behaves like one at D.

## The direction-averaged margin

    m̄ = E_θ~U[0,2π) [ min{m(θ), D} ]  ≈  (1/24) Σ_k min{m(15°·k), D}

the expected distance an answer can stray from the TG in a random direction
before it leaves the TG's cell. Since m̄ = ∫₀^D P_θ(m(θ) > e) de, m̄/D is the
probability that an error of uniformly random size in [0, D] and random
direction stays in the cell. It describes the cell, not a method; the margin
along a method's own error direction is `margin_err_dir`.

A TG whose own location's nearest seed is not its seed (complete linkage can
place a site in a seed whose cell does not hold it) has m = 0 everywhere and is
flagged `start_outside`.

Everything is written over D. No coordinate is written.
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

from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules.geodesy import pairwise_km  # noqa: E402
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths  # noqa: E402
from scripts.libs.cbg.rtt_model import EARTH_RADIUS_KM  # noqa: E402

KIND = "cell-margin"
LOC_NAME = "cell_margin_locations.csv"
TGS_NAME = "cell_margin_tgs.csv"
REPORT_NAME = "cell_margin.report.json"
BOX_NAME = "cell_margin_box.png"

STEP_DEG = 15.0
#: Bisection tolerance, as a share of D.
TOL = 1e-4
COMBOS = ("shortest_ping", "million_scale_cbg", "vanilla_cbg", "octant_cbg_hull", "octant_cbg_spl", "spotter_cbg")
SIDES = ("has-X", "no-X")


# ---- spherical primitives ------------------------------------------------------------


def destination(lat, lon, bearing_deg, dist_km):
    """Point at `dist_km` along the great circle leaving (lat, lon) at `bearing_deg`. Vectorised."""
    p1, l1 = np.radians(lat), np.radians(lon)
    th, d = np.radians(bearing_deg), np.asarray(dist_km, float) / EARTH_RADIUS_KM
    p2 = np.arcsin(np.clip(np.sin(p1) * np.cos(d) + np.cos(p1) * np.sin(d) * np.cos(th), -1, 1))
    l2 = l1 + np.arctan2(np.sin(th) * np.sin(d) * np.cos(p1), np.cos(d) - np.sin(p1) * np.sin(p2))
    return np.degrees(p2), (np.degrees(l2) + 540.0) % 360.0 - 180.0


def initial_bearing(lat1, lon1, lat2, lon2):
    """Initial great-circle bearing from 1 to 2, degrees. Vectorised."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def nearest_seed(lat, lon, seeds: pd.DataFrame) -> np.ndarray:
    d = pairwise_km(np.atleast_1d(lat), np.atleast_1d(lon), seeds.seed_lat.to_numpy(), seeds.seed_lon.to_numpy())
    return seeds.seed_id.to_numpy()[d.argmin(axis=1)]


def margins(lat, lon, bearing_deg, own_seed, seeds: pd.DataFrame, D_km: float, *, tol: float = TOL) -> np.ndarray:
    """`min{m(θ), D}` in km for each (origin, bearing) pair. All inputs broadcast to one length.

    0 where the origin itself is outside `own_seed`'s cell; D where the ray is
    still inside at D (open); otherwise the bisected exit distance.
    """
    lat, lon, th, own = np.broadcast_arrays(np.asarray(lat, float), np.asarray(lon, float),
                                            np.asarray(bearing_deg, float), np.asarray(own_seed))
    lat, lon, th, own = (a.ravel() for a in (lat, lon, th, own))
    out = np.full(lat.shape, float(D_km))
    start_in = nearest_seed(lat, lon, seeds) == own
    out[~start_in] = 0.0
    far = destination(lat, lon, th, np.full(lat.shape, D_km))
    exits = start_in & (nearest_seed(far[0], far[1], seeds) != own)
    if exits.any():
        lo = np.zeros(exits.sum()); hi = np.full(exits.sum(), float(D_km))
        la, lo_, t, o = lat[exits], lon[exits], th[exits], own[exits]
        for _ in range(int(math.ceil(math.log2(1.0 / tol))) + 1):
            mid = (lo + hi) / 2
            p = destination(la, lo_, t, mid)
            inside = nearest_seed(p[0], p[1], seeds) == o
            lo = np.where(inside, mid, lo)
            hi = np.where(inside, hi, mid)
        out[exits] = (lo + hi) / 2
    return out


# ---- one run ---------------------------------------------------------------------------


def _classify(run: RunPaths, analysis_root: Path | None) -> pd.DataFrame:
    frames = []
    for combo in COMBOS:
        p = run.analysis_dir("classify", root=analysis_root) / "healpix-128" / f"{combo}_tgs.parquet"
        if not p.exists():
            raise MissingArtifactError(f"{p} missing; run `classify --run-id {run.run_id}` first")
        frames.append(pd.read_parquet(p).assign(combo=combo))
    return pd.concat(frames, ignore_index=True)


def build_run(run: RunPaths, *, D_km: float, base: RunPaths, analysis_root: Path | None = None,
              step_deg: float = STEP_DEG) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(locations, tgs)` for one run; has-X from `base`'s `sp_pni_cells_tgs.csv`."""
    from scripts.analysis.v5.modules.loso_delta import attach_has_x
    from scripts.analysis.v5.modules.map_mtl import load_rung

    seeds = load_rung(run, 128, analysis_root=analysis_root).seeds
    cl = _classify(run, analysis_root)
    cl, src = attach_has_x(cl, base, analysis_root)
    if src is None:
        raise MissingArtifactError(f"{base.run_id}: no sp_pni_cells_tgs.csv for has-X labels")

    # -- per distinct TG location: the 24 rays ---------------------------------------
    loc = (cl.drop_duplicates("tg_id")
             .groupby(["tg_lat", "tg_lon", "tg_seed_id"], as_index=False)
             .agg(site_id=("site_id", "first"), has_x=("has_x", "first"), n_tgs=("tg_id", "size")))
    bearings = np.arange(0.0, 360.0, step_deg)
    m = margins(loc.tg_lat.to_numpy()[:, None], loc.tg_lon.to_numpy()[:, None], bearings[None, :],
                loc.tg_seed_id.to_numpy()[:, None], seeds, D_km).reshape(len(loc), len(bearings)) / D_km
    for k, b in enumerate(bearings):
        loc[f"m_{int(b):03d}"] = m[:, k]
    loc["mbar"] = m.mean(axis=1)
    loc["median"] = np.median(m, axis=1)
    loc["open_share"] = (m >= 1.0 - 1e-12).mean(axis=1)
    loc["start_outside"] = (m == 0.0).all(axis=1)
    loc.insert(0, "run_id", run.run_id)
    loc.insert(1, "loc_id", np.arange(len(loc)))

    # -- per (method, TG): the margin along its own error direction --------------------
    t = cl[cl.pred_lat.notna() & cl.cell_label.isin(["correct", "wrong"])].copy()
    th = initial_bearing(t.tg_lat.to_numpy(), t.tg_lon.to_numpy(), t.pred_lat.to_numpy(), t.pred_lon.to_numpy())
    t["margin_err_dir"] = margins(t.tg_lat.to_numpy(), t.tg_lon.to_numpy(), th, t.tg_seed_id.to_numpy(),
                                  seeds, D_km) / D_km
    t["error"] = t.pred_dist_to_tg_km / D_km
    t["error_over_margin"] = np.where(t.margin_err_dir > 0, t.error / t.margin_err_dir, np.inf)
    within = t.error < t.margin_err_dir
    near_boundary = (t.error - t.margin_err_dir).abs() <= 2 * TOL
    t["consistent"] = (within == t.cell_label.eq("correct")) | near_boundary
    t = t.merge(loc[["tg_lat", "tg_lon", "loc_id", "mbar", "start_outside"]], on=["tg_lat", "tg_lon"], how="left")
    keep = ["run_id", "combo", "tg_id", "site_id", "has_x", "status", "cell_label", "loc_id", "mbar",
            "start_outside", "margin_err_dir", "error", "error_over_margin", "consistent"]
    t["run_id"] = run.run_id
    loc_out = loc.drop(columns=["tg_lat", "tg_lon", "tg_seed_id"])
    return loc_out, t[keep]


# ---- pooled report -----------------------------------------------------------------------


def _site_boot(df: pd.DataFrame, stat, key: str, n: int = 2000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(key)]
    vals = [stat(pd.concat([groups[i] for i in rng.integers(0, len(groups), len(groups))]))
            for _ in range(n)]
    return [round(float(stat(df)), 4), round(float(np.nanpercentile(vals, 2.5)), 4),
            round(float(np.nanpercentile(vals, 97.5)), 4)]


def summarize(locs: pd.DataFrame, tgs: pd.DataFrame) -> dict:
    """1c by side (one value per location, site bootstrap) and 3d per method; shares and ratios only."""
    L = locs[~locs.start_outside].assign(site=lambda d: d.run_id + ":" + d.site_id.astype(str))
    side = lambda d, s: d[d.has_x == s]  # noqa: E731
    rep: dict = {"start_outside_pct": round(float(locs.start_outside.mean()) * 100, 2), "by_side": {}}
    for s in SIDES:
        d = side(L, s)
        rep["by_side"][s] = {
            "locations_pct": round(len(d) / len(L) * 100, 2),
            "mbar_over_D": {"p25": float(d.mbar.quantile(.25)), "p50": float(d.mbar.median()),
                            "p75": float(d.mbar.quantile(.75)), "mean": float(d.mbar.mean())},
            "mbar_mean_ci": _site_boot(d, lambda x: x.mbar.mean(), "site"),
            "at_cap_pct": round(float((d.mbar >= 1.0 - 1e-9).mean()) * 100, 2),
            "open_share_mean_pct": round(float(d.open_share.mean()) * 100, 2),
            "bg_median_over_rays_p50": float(d["median"].median()),
            "bg_mbar_p50_closed_only": float(d.loc[d.open_share == 0, "mbar"].median())
            if (d.open_share == 0).any() else math.nan,
            "bg_mbar_p50_tg_weighted": float(np.median(np.repeat(d.mbar.to_numpy(), d.n_tgs.to_numpy()))),
        }
    rep["noX_over_hasX_mbar_p50_ratio_ci"] = _site_boot(
        L, lambda x: side(x, "no-X").mbar.median() / side(x, "has-X").mbar.median(), "site")
    closed = L[L.open_share == 0]
    if closed.has_x.nunique() == 2:
        rep["bg_noX_over_hasX_closed_only_ratio_ci"] = _site_boot(
            closed, lambda x: side(x, "no-X").mbar.median() / side(x, "has-X").mbar.median(), "site")
    rep["consistency_pct_by_combo"] = (tgs.groupby("combo").consistent.mean() * 100).round(3).to_dict()
    nx = tgs[(tgs.has_x == "no-X") & ~tgs.start_outside]
    rep["noX_error_over_margin_p50_by_combo"] = nx.groupby("combo").error_over_margin.median().round(3).to_dict()
    rep["noX_margin_err_dir_p50_by_combo"] = nx.groupby("combo").margin_err_dir.median().round(4).to_dict()
    rep["noX_correct_error_over_margin_p50_by_combo"] = (
        nx[nx.cell_label == "correct"].groupby("combo").error_over_margin.median().round(3).to_dict())
    return rep


def render_box(locs: pd.DataFrame, out_png: Path) -> Path:
    """One box per side of m̄ / D, one point per location; dashed line at 1, cap share above each box."""
    from scripts.analysis.v5.modules.figure_pni_gap import CLUSTER_HUES
    from scripts.analysis.v5.modules.mapping import INK, INK_2

    L = locs[~locs.start_outside]
    hue = {"has-X": CLUSTER_HUES[0], "no-X": CLUSTER_HUES[1]}
    fig, ax = plt.subplots(figsize=(1.7, 1.6))
    rng = np.random.default_rng(0)
    for i, s in enumerate(SIDES):
        v = L.loc[L.has_x == s, "mbar"].to_numpy()
        art = ax.boxplot([v], positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                         medianprops={"color": INK, "lw": 0.9},
                         whiskerprops={"color": INK_2, "lw": 0.6}, capprops={"color": INK_2, "lw": 0.6})
        art["boxes"][0].set(facecolor=hue[s], alpha=0.45, edgecolor=hue[s], linewidth=0.6)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=2.5, color=hue[s], alpha=0.7, lw=0, zorder=3)
        ax.text(i, 1.04, f"{(v >= 1 - 1e-9).mean() * 100:.0f}%", ha="center", va="bottom", fontsize=6, color=INK)
    ax.axhline(1.0, color=INK_2, lw=0.5, ls="--", zorder=0)
    ax.text(1.85, 1.005, "unbounded", fontsize=5.5, color=INK_2, va="bottom", ha="right")
    ax.set_xticks([0, 1], ["has-$X$", "no-$X$"], fontsize=6)
    ax.set_xlim(-0.6, 1.9); ax.set_ylim(0, 1.15)
    ax.set_ylabel(r"direction-averaged margin $\bar m / D$", fontsize=6.5)
    ax.tick_params(axis="both", labelsize=6, length=2, pad=1.5)
    ax.grid(axis="y", alpha=0.25, lw=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_png


def build(seen: list[RunPaths], pairs: dict[str, RunPaths], *, D_km: dict[str, float],
          analysis_root: Path | None = None, step_deg: float = STEP_DEG,
          out_dir: Path | None = None) -> dict[str, Path]:
    """Per-run CSVs for every run (seen and LOSO twins); pooled report + box over the seen runs."""
    locs, tgs = [], []
    for base in seen:
        for regime, run in (("seen", base), ("unseen", pairs.get(base.run_id))):
            if run is None:
                continue
            loc, t = build_run(run, D_km=D_km[run.run_id], base=base, analysis_root=analysis_root,
                               step_deg=step_deg)
            out = out_dir / run.run_id if out_dir is not None else run.analysis_dir(KIND, root=analysis_root)
            out.mkdir(parents=True, exist_ok=True)
            loc.to_csv(out / LOC_NAME, index=False)
            t.to_csv(out / TGS_NAME, index=False)
            tgs.append(t.assign(regime=regime))
            if regime == "seen":
                locs.append(loc)
    L, T = pd.concat(locs, ignore_index=True), pd.concat(tgs, ignore_index=True)
    run_ids = sorted({r.run_id for r in seen} | {r.run_id for r in pairs.values()})
    out_dir = out_dir if out_dir is not None else cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)
    rep = {"run_ids": run_ids, "step_deg": step_deg, "seen": summarize(L, T[T.regime == "seen"])}
    if (T.regime == "unseen").any():
        rep["unseen_3d"] = {k: v for k, v in summarize(L, T[T.regime == "unseen"]).items()
                            if k.startswith("noX_") or k.startswith("consistency")}
    (out_dir / REPORT_NAME).write_text(json.dumps(rep, indent=1) + "\n")
    return {"report": out_dir / REPORT_NAME, "box": render_box(L, out_dir / BOX_NAME)}
