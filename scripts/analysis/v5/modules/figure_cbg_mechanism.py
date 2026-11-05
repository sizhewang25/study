"""Mechanism example of each CBG method's three phases (appendix, CBG implementation details).

One TG, one representative VP, five methods (SOI, VAN, OCT-H, OCT-S, SPO), three
panels per method:

(a) LTD: the representative VP's training pairs (SPO: pooled over VPs) in RTT / R_max
    vs distance / D, the 2/3 c reference line, the method's model, and the constraint
    the VP's RTT to the TG yields (dashed crossing lines, red dots on the bounds). The
    TG's true (RTT, distance) is the orange square.
(b) MTL: the TG's neighbourhood in a TG-centred equidistant projection over D, with no
    ticks and a scale bar. Every participant's disk (solid) or annulus (solid outer,
    dotted inner) is dimmed except the representative VP's, and the region is filled.
    SPO draws each VP's ring density exp(-z^2/2) instead, black where it is high.
(c) EST: the same map with nothing highlighted, plus the estimate as a red star.

## The example is re-run on K VPs

The map draws only the K lowest-RTT VPs of the TG. Drawing them against the
benchmark's region would show constraints that did not form it, so MTL and EST are
re-run on those K VPs with each method's own fitted LTD (the TG's fold), MTL spec and
CTR (`run.json`; the Monte Carlo medoid seeded with the TG's recorded seed). Only the
MTL's participants are drawn, so a disk the inclusion filter drops is not shown. With
K = every VP, the same code reproduces the recorded prediction (`pipeline_gate`).

## Picking the example

`pick_candidates` ranks (TG, VP) pairs on a site-stratified sample of the TGs every
method answers: all five K-VP pipelines must succeed and the VP must take part in all
four geometric MTLs. The score is the share of each region's boundary that the VP's
ring forms, summed over the geometric methods (the VP visibly binds), plus a bonus for
regions that are neither specks nor larger than the frame. The contact sheet lets the
user choose; `build_example` then draws the chosen pair.

No coordinate, km or ms reaches a figure: map axes carry no ticks, and LTD axes are
normalized by R_max and D.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules import map_mtl as MM  # noqa: E402
from scripts.analysis.v5.modules import mtl_replay as R  # noqa: E402
from scripts.analysis.v5.modules.methods import method_label  # noqa: E402
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths  # noqa: E402

KIND = "cbg-mechanism"
JSON_NAME = "cbg_mechanism.json"
PNG = "cbg_mech.{term}.{phase}.png"
CANDIDATES_CSV = "candidates.csv"
CANDIDATES_PNG = "candidates.png"

METHODS = ("million_scale_cbg", "vanilla_cbg", "octant_cbg_hull", "octant_cbg_spl", "spotter_cbg")
GEOMETRIC = METHODS[:4]
SPO = "spotter_cbg"
PHASES = ("ltd", "mtl", "est")
#: Table 3 (`tab:cbg-variants`) names, for the subcaptions and the JSON.
PHASE_NAMES = {
    "million_scale_cbg": ("Speed-of-Internet bound", "Disk intersection", "Geometric centroid"),
    "vanilla_cbg": ("Bestline bound", "Disk intersection", "Geometric centroid"),
    "octant_cbg_hull": ("Convex-hull bounds", "Annulus intersection with weighting", "Monte Carlo medoid"),
    "octant_cbg_spl": ("Spline bounds", "Annulus intersection with weighting", "Monte Carlo medoid"),
    "spotter_cbg": ("Normal distribution", "Joint Gaussian density", "Density argmax"),
}

DEFAULT_K = 6
#: 2/3 c in km per ms of RTT (one way at 200 km/ms, halved for the round trip).
SOI_KM_PER_MS = 100.0

# Fig. 5 (figs/sp-schematic.tex) colours.
VP_FILL = "#5B9BD5"
VP_DARK = "#1F4E79"
#: The TG is a yellow star; the estimate a red triangle.
TG_FILL = "#FFD400"
TG_EDGE = "#6b5a00"
TG_MARKER = "*"
STAR = "#E03C31"
EST_MARKER = "^"
EST_EDGE = "#7a1a14"
#: LTD training pairs: solid fluorescent green, under a translucent band.
BAND = "#c8c8c8"
BAND_ALPHA = 0.55
INK = "#000000"
DIM = "#9a9a9a"
DIM_ALPHA = 0.55
REGION_FILL = "#d4d4d4"
SCATTER = "#00C410"

PANEL_IN = (2.2, 1.9)
FONT = 7.0
TICK_FONT = 6.0
#: Frame padding around its content, as a share of the larger side.
FRAME_PAD = 0.15
#: Density raster resolution over the frame, per side.
DENSITY_PX = 400
#: Boundary tolerance for "the VP's ring forms this region edge", as a share of the frame side.
BIND_TOL = 0.01
SEGMENT_DEG = 0.02
#: LTD x-axis span, as a multiple of the representative VP's RTT to the TG.
LTD_ZOOM = 3.5
#: SPO's other rings are drawn at this share of black (they overlap into a grey wash otherwise).
SPO_DIM = 0.3


# ---- the example pipeline ----------------------------------------------------------------


@dataclass
class MethodExample:
    """One method's K-VP pipeline on one TG, in lon/lat."""

    combo: str
    fold: int
    participants: list[dict]            # vp_id, vp_lat, vp_lon, rtt_ms, upper_km, lower_km, mu_km, sigma_km
    region: Any = None                  # shapely lon/lat geometry (geometric methods)
    estimate: tuple[float, float] | None = None   # (lat, lon)
    mtl_success: bool = False
    #: The K VPs the MTL dropped (its inclusion filter, or a failed LTD): drawn hollow.
    filtered: list[dict] = field(default_factory=list)

    @property
    def term(self) -> str:
        return method_label(self.combo)

    @property
    def vp_ids(self) -> list[str]:
        return [p["vp_id"] for p in self.participants]


@dataclass
class RunCache:
    """Per-run inputs read once: edges, fitted LTDs, specs, recorded rows."""

    run: RunPaths
    edges: pd.DataFrame
    folds: dict[str, int]
    _models: dict = field(default_factory=dict)
    _cfg: dict = field(default_factory=dict)
    _rows: dict = field(default_factory=dict)

    @classmethod
    def load(cls, run: RunPaths, analysis_root: Path | None = None) -> "RunCache":
        from scripts.analysis.v5.modules import edges as EG
        from scripts.analysis.v5.modules.figure_sp_interconnect import load_edges

        scored = R._scored(run, METHODS[0], analysis_root)
        return cls(run, load_edges(EG.resolve_source_csv(run, None)),
                   dict(zip(scored.tg_id.astype(str), scored.fold.astype(int))))

    def cfg(self, combo: str, fold: int) -> dict:
        key = (combo, fold)
        if key not in self._cfg:
            self._cfg[key] = json.loads((self.run.combo_dir(combo, f"fold_{fold}") / "run.json").read_text())
        return self._cfg[key]

    def model(self, combo: str, fold: int):
        from scripts.analysis.v5.modules.figure_ltd_model import load_model

        key = (combo, fold)
        if key not in self._models:
            self._models[key] = load_model(self.run.combo_dir(combo, f"fold_{fold}"), self.cfg(combo, fold))[0]
        return self._models[key]

    def rows(self, combo: str, fold: int) -> pd.DataFrame:
        key = (combo, fold)
        if key not in self._rows:
            p = self.run.combo_dir(combo, f"fold_{fold}") / "targets.parquet"
            self._rows[key] = pd.read_parquet(
                p, columns=["target_id", "pred_lat", "pred_lon", "status", "seed", "ltd_predictions"]
            ).set_index("target_id")
        return self._rows[key]

    def observations(self, tg_id: str) -> pd.DataFrame:
        """The TG's (vp_id, vp_lat, vp_lon, rtt_ms, d_km), lowest RTT first."""
        e = self.edges[self.edges.tg_id == tg_id]
        if e.empty:
            raise MissingArtifactError(f"{self.run.run_id}: no edges for {tg_id}")
        return e.sort_values(["rtt_ms", "vp_id"]).reset_index(drop=True)


def _obs(rows: pd.DataFrame) -> list:
    from scripts.framework.v2.types import Coord, Latency, VpId

    return [(VpId(str(r.vp_id)), Coord(lat=float(r.vp_lat), lon=float(r.vp_lon)), Latency(float(r.rtt_ms)))
            for r in rows.itertuples(index=False)]


def run_method(cache: RunCache, combo: str, tg_id: str, obs_rows: pd.DataFrame) -> MethodExample:
    """LTD -> MTL -> CTR of `combo` on `obs_rows`, with the TG's fold's model and specs."""
    import scripts.framework.v2  # noqa: F401  (populates the registries)
    from scripts.framework.v2.registry import CTR_REGISTRY, MTL_REGISTRY

    fold = cache.folds[tg_id]
    cfg = cache.cfg(combo, fold)
    ltd = cache.model(combo, fold)
    results = [r for r in ltd.predict_all(_obs(obs_rows)) if r.success]
    mtl = MTL_REGISTRY[cfg["mtl"]](**(cfg.get("mtl_kwargs") or {}))
    m = mtl.multilaterate(results)
    used = set(map(str, m.participating_vp_ids or ()))
    parts = []
    for r in results:
        if str(r.vp_id) not in used:
            continue
        d = r.tg_distance
        parts.append({"vp_id": str(r.vp_id), "vp_lat": r.vp_coord.lat, "vp_lon": r.vp_coord.lon,
                      "rtt_ms": float(r.latency), "upper_km": float(d.upper_km),
                      "lower_km": float(d.lower_km or 0.0),
                      "mu_km": None if d.mu_km is None else float(d.mu_km),
                      "sigma_km": None if d.sigma_km is None else float(d.sigma_km)})
    dropped = [{"vp_id": str(r.vp_id), "vp_lat": float(r.vp_lat), "vp_lon": float(r.vp_lon)}
               for r in obs_rows.itertuples(index=False) if str(r.vp_id) not in used]
    ex = MethodExample(combo=combo, fold=fold, participants=parts, mtl_success=bool(m.success),
                       filtered=dropped)
    if not m.success:
        return ex
    if combo != SPO and hasattr(m.intersection, "geom_type"):
        ex.region = m.intersection
    ctr = CTR_REGISTRY[cfg["ctr"]](**(cfg.get("ctr_kwargs") or {}))
    seed = cache.rows(combo, fold).loc[tg_id, "seed"]
    if hasattr(ctr, "rng") and seed is not None and not pd.isna(seed):
        ctr.rng = np.random.default_rng(int(seed))
    c = ctr.select_centroid(m)
    if c.success and c.tg_coord is not None:
        ex.estimate = (float(c.tg_coord.lat), float(c.tg_coord.lon))
    return ex


def run_example(cache: RunCache, tg_id: str, k: int = DEFAULT_K) -> dict[str, MethodExample]:
    """Every method's pipeline on the TG's K lowest-RTT VPs."""
    obs = cache.observations(tg_id).head(k)
    return {c: run_method(cache, c, tg_id, obs) for c in METHODS}


def pipeline_gate(cache: RunCache, tg_id: str) -> dict[str, float]:
    """km between the all-VP rebuild and the recorded prediction, per method (0 = exact)."""
    obs = cache.observations(tg_id).set_index("vp_id")
    out = {}
    for combo in METHODS:
        row = cache.rows(combo, cache.folds[tg_id]).loc[tg_id]
        order = [str(p["vp_id"]) for p in row.ltd_predictions]
        ex = run_method(cache, combo, tg_id, obs.loc[order].reset_index())
        if ex.estimate is None or pd.isna(row.pred_lat):
            out[combo] = 0.0 if (ex.estimate is None) == pd.isna(row.pred_lat) else math.inf
            continue
        out[combo] = float(R._haversine_km(ex.estimate[0], ex.estimate[1], row.pred_lat, row.pred_lon))
    return out


# ---- projection --------------------------------------------------------------------------


class Frame:
    """TG-centred azimuthal equidistant plane, in units of D."""

    def __init__(self, tg_lat: float, tg_lon: float, D_km: float):
        from pyproj import Transformer

        crs = f"+proj=aeqd +lat_0={tg_lat} +lon_0={tg_lon} +datum=WGS84 +units=km"
        self._fwd = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self._inv = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        self.D = float(D_km)
        self.tg = (tg_lat, tg_lon)
        self.extent: tuple[float, float, float, float] | None = None

    def xy(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._fwd.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x) / self.D, np.asarray(y) / self.D

    def latlon(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self._inv.transform(np.asarray(x, float) * self.D, np.asarray(y, float) * self.D)
        return np.asarray(lat), np.asarray(lon)


def _xy_geom(frame: Frame, g):
    """A lon/lat shapely geometry, clipped to the valid box, densified and projected."""
    import shapely.ops
    from shapely import segmentize
    from shapely.geometry import box

    g = g.intersection(box(-180.0, -89.9, 180.0, 89.9))
    if g.is_empty:
        return g
    g = segmentize(g, SEGMENT_DEG)

    def f(lon, lat, z=None):
        x, y = frame.xy(lat, lon)
        return x, y

    return shapely.ops.transform(f, g)


def ring_lines(frame: Frame, combo_mtl: str, p: dict, n_pts: int = 64) -> tuple[list, list]:
    """The MTL's own ring for one participant, projected: (outer lines, inner lines)."""
    poly = R.ring_polygon(combo_mtl, p, n_pts)
    if poly is None:
        return [], []
    g = _xy_geom(frame, poly)
    polys = list(g.geoms) if g.geom_type == "MultiPolygon" else [g] if g.geom_type == "Polygon" else []
    outer = [np.asarray(q.exterior.coords) for q in polys]
    inner = [np.asarray(i.coords) for q in polys for i in q.interiors]
    return outer, inner


def fit_extent(frame: Frame, examples: dict[str, MethodExample], vp_id: str) -> tuple[float, float, float, float]:
    """Square extent around TG, the VP, every estimate and VAN's / OCT-H's regions, padded."""
    xs, ys = [0.0], [0.0]
    vp = next(p for ex in examples.values() for p in ex.participants if p["vp_id"] == vp_id)
    x, y = frame.xy(vp["vp_lat"], vp["vp_lon"]); xs.append(float(x)); ys.append(float(y))
    for ex in examples.values():
        if ex.estimate is not None:
            x, y = frame.xy(*ex.estimate); xs.append(float(x)); ys.append(float(y))
        if ex.combo in ("vanilla_cbg", "octant_cbg_hull") and ex.region is not None and not ex.region.is_empty:
            b = _xy_geom(frame, ex.region).bounds
            xs += [b[0], b[2]]; ys += [b[1], b[3]]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * (1 + 2 * FRAME_PAD)
    half = max(half, 1e-3)
    return cx - half, cx + half, cy - half, cy + half


# ---- candidates ----------------------------------------------------------------------------


def _bind_share(frame: Frame, ex: MethodExample, p: dict, tol: float) -> float:
    """Share of the region's boundary length lying within `tol` of the VP's ring."""
    from shapely.geometry import LineString, MultiLineString

    if ex.region is None or ex.region.is_empty:
        return 0.0
    mtl = "planar_circle" if ex.combo in ("million_scale_cbg", "vanilla_cbg") else "planar_annulus_weighted"
    outer, inner = ring_lines(frame, mtl, p)
    lines = [LineString(c) for c in outer + inner if len(c) > 1]
    if not lines:
        return 0.0
    rb = _xy_geom(frame, ex.region).boundary
    if rb.length == 0:
        return 0.0
    return float(rb.intersection(MultiLineString(lines).buffer(tol)).length / rb.length)


def _bearing_gap_deg(frame: Frame, parts: list[dict]) -> float:
    """Largest empty angular sector (degrees) around the TG among the VPs' bearings."""
    ang = sorted(float(np.degrees(np.arctan2(*frame.xy(p["vp_lat"], p["vp_lon"])[::-1]))) % 360 for p in parts)
    if len(ang) < 2:
        return 360.0
    gaps = np.diff(ang + [ang[0] + 360])
    return float(gaps.max())


def score_example(cache: RunCache, tg_id: str, examples: dict[str, MethodExample], D_km: float) -> list[dict]:
    """One row per VP in all four geometric MTLs, scored on how well the example illustrates.

    Favoured: each geometric region is formed by two or more VPs (each VP's share of
    the region boundary, `_bind_share`), the representative VP forms a visible part of
    it but not all of it, the VPs surround the TG (small largest bearing gap), the
    regions are neither specks nor larger than the frame, and VAN's region holds the TG.
    """
    from shapely.geometry import Point

    if not all(ex.mtl_success and ex.estimate is not None for ex in examples.values()):
        return []
    common = set.intersection(*(set(examples[c].vp_ids) for c in GEOMETRIC))
    if not common or len(examples["million_scale_cbg"].participants) < 2:
        return []
    lat, lon = (float(v) for v in cache.observations(tg_id).iloc[0][["tg_lat", "tg_lon"]])
    frame = Frame(lat, lon, D_km)
    gap = _bearing_gap_deg(frame, examples["octant_cbg_hull"].participants)
    van = examples["vanilla_cbg"].region
    van_holds = bool(van is not None and van.covers(Point(lon, lat)))
    rows = []
    for vp_id in sorted(common):
        ext = fit_extent(frame, examples, vp_id)
        side = ext[1] - ext[0]
        multi, rep_ok, size_ok, rep_share = 0, 0, 0, []
        for c in GEOMETRIC:
            ex = examples[c]
            shares = {p["vp_id"]: _bind_share(frame, ex, p, BIND_TOL * side) for p in ex.participants}
            multi += int(sum(v >= 0.1 for v in shares.values()) >= 2)
            rep_share.append(shares[vp_id])
            rep_ok += int(0.15 <= shares[vp_id] <= 0.6)
            if ex.region is not None and not ex.region.is_empty:
                share = _xy_geom(frame, ex.region).area / side**2
                size_ok += int(0.02 <= share <= 0.6)
        p = next(q for q in examples["vanilla_cbg"].participants if q["vp_id"] == vp_id)
        d = float(R._haversine_km(lat, lon, p["vp_lat"], p["vp_lon"]))
        score = multi + rep_ok + 0.5 * size_ok + (1 - gap / 360) * 2 + 0.5 * van_holds
        rows.append({"tg_id": tg_id, "vp_id": vp_id, "multi": multi, "rep_ok": rep_ok, "size_ok": size_ok,
                     "rep_share": "/".join(f"{v:.2f}" for v in rep_share), "bearing_gap": round(gap),
                     "van_holds_tg": van_holds, "vp_over_frame": round(d / D_km / side, 3),
                     "n_soi": len(examples["million_scale_cbg"].participants), "score": round(score, 3)})
    return rows


_POOL_CACHE: dict[str, RunCache] = {}


def _candidate_task(args: tuple) -> list[dict]:
    run_id, tg_id, k, D_km = args
    cache = _POOL_CACHE[run_id]
    try:
        return score_example(cache, tg_id, run_example(cache, tg_id, k), D_km)
    except Exception:  # noqa: BLE001  (a TG the pipeline cannot run is not a candidate)
        return []


def answered_by_all(run: RunPaths, analysis_root: Path | None = None) -> pd.DataFrame:
    """Scored TGs every method answers (VAN FALLBACK is not an answer), with site, fold and side."""
    from scripts.analysis.v5.modules.loso_delta import attach_has_x
    from scripts.analysis.v5.modules.status import solved_mask

    keep = None
    for combo in METHODS:
        p = run.analysis_dir("classify", root=analysis_root) / "healpix-128" / f"{combo}_tgs.parquet"
        c = pd.read_parquet(p)
        ok = set(c.loc[solved_mask(c), "tg_id"].astype(str))
        keep = ok if keep is None else keep & ok
    scored = R._scored(run, METHODS[0], analysis_root)
    scored = scored[scored.tg_id.astype(str).isin(keep)].copy()
    scored, _ = attach_has_x(scored, run, analysis_root)
    return scored


def pick_candidates(runs: list[RunPaths], *, D_km: float, k: int = DEFAULT_K, frac: float = 0.25,
                    workers: int = 1, analysis_root: Path | None = None,
                    progress: Callable[[str], None] | None = None) -> pd.DataFrame:
    """Ranked (run, TG, VP) candidates over a site-stratified sample of each run."""
    rows = []
    for run in runs:
        scored = answered_by_all(run, analysis_root)
        sample = R.sample_tgs(scored, limit=None, frac=frac)
        _POOL_CACHE[run.run_id] = RunCache.load(run, analysis_root)
        tasks = [(run.run_id, str(t), k, D_km) for t in sample.tg_id]
        if progress:
            progress(f"  {run.run_id}: {len(tasks)} TG(s) on {workers} worker(s)")
        if workers > 1:
            from multiprocessing import Pool

            with Pool(processes=workers) as pool:
                outs = pool.map(_candidate_task, tasks, chunksize=4)
        else:
            outs = [_candidate_task(t) for t in tasks]
        side = dict(zip(scored.tg_id.astype(str), scored.has_x))
        site = dict(zip(scored.tg_id.astype(str), scored.site_id))
        for out in outs:
            for r in out:
                rows.append({"run_id": run.run_id, "site_id": site.get(r["tg_id"]), "side": side.get(r["tg_id"]), **r})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("score", ascending=False).reset_index(drop=True)


# ---- rendering -----------------------------------------------------------------------------


def _style(ax, *, ticks: bool) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if not ticks:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.5); s.set_color("#666666")
    ax.tick_params(labelsize=TICK_FONT, length=2, width=0.5, pad=1.5)


def _nice(x: float) -> float:
    e = math.floor(math.log10(x))
    for m in (5, 2, 1):
        if m * 10**e <= x:
            return m * 10**e
    return 10**e


def _scale_bar(ax, ext) -> None:
    side = ext[1] - ext[0]
    L = _nice(side / 4)
    x0, y0 = ext[0] + 0.06 * side, ext[2] + 0.06 * side
    ax.plot([x0, x0 + L], [y0, y0], color=INK, lw=1.2, solid_capstyle="butt", zorder=9)
    ax.text(x0 + L / 2, y0 + 0.025 * side, f"${L:g}D$", ha="center", va="bottom", fontsize=FONT - 0.5, zorder=9)


def _poly_patch(g, **kw) -> PathPatch | None:
    polys = list(g.geoms) if g.geom_type in ("MultiPolygon", "GeometryCollection") else [g]
    verts, codes = [], []
    for q in polys:
        if q.geom_type != "Polygon" or q.is_empty:
            continue
        for ring in [q.exterior, *q.interiors]:
            c = np.asarray(ring.coords)
            verts.extend(c.tolist())
            codes.extend([MplPath.MOVETO] + [MplPath.LINETO] * (len(c) - 2) + [MplPath.CLOSEPOLY])
    if not verts:
        return None
    return PathPatch(MplPath(verts, codes), **kw)


def _markers(ax, frame: Frame, ex: MethodExample, vp_id: str, tg_xy=(0.0, 0.0)) -> None:
    for p in ex.filtered:
        x, y = frame.xy(p["vp_lat"], p["vp_lon"])
        ax.scatter([x], [y], s=15, marker="o", facecolor="white", edgecolor=VP_DARK, linewidth=0.6, zorder=6,
                   clip_on=True)
    for p in ex.participants:
        x, y = frame.xy(p["vp_lat"], p["vp_lon"])
        rep = p["vp_id"] == vp_id
        ax.scatter([x], [y], s=26 if rep else 16, marker="o", facecolor=STAR if rep else VP_FILL,
                   edgecolor=EST_EDGE if rep else VP_DARK, linewidth=0.5, zorder=7 if rep else 6, clip_on=True)
    ax.scatter([tg_xy[0]], [tg_xy[1]], s=48, marker=TG_MARKER, facecolor=TG_FILL, edgecolor=TG_EDGE, linewidth=0.5,
               zorder=8)


def _density_field(frame: Frame, ext, parts: list[dict]) -> list[np.ndarray]:
    """Per-VP exp(-z^2/2) over a DENSITY_PX raster of the frame."""
    xs = np.linspace(ext[0], ext[1], DENSITY_PX)
    ys = np.linspace(ext[2], ext[3], DENSITY_PX)
    X, Y = np.meshgrid(xs, ys)
    lat, lon = frame.latlon(X.ravel(), Y.ravel())
    fields = []
    for p in parts:
        s = R._haversine_km(p["vp_lat"], p["vp_lon"], lat, lon)
        z = (s - p["mu_km"]) / p["sigma_km"]
        fields.append(np.exp(-0.5 * z * z).reshape(X.shape))
    return fields


def draw_map(ax, frame: Frame, ex: MethodExample, vp_id: str, ext, *, highlight: bool, star: bool) -> None:
    """Panel (b) (highlight) or (c) (star): constraints, region, VPs, TG, scale bar."""
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    if ex.combo == SPO:
        fields = _density_field(frame, ext, ex.participants)
        others = [f for p, f in zip(ex.participants, fields) if not (highlight and p["vp_id"] == vp_id)]
        img = SPO_DIM * np.max(others, axis=0) if others else np.zeros((DENSITY_PX, DENSITY_PX))
        ax.imshow(img, extent=ext, origin="lower", cmap="Greys", vmin=0, vmax=1, interpolation="bilinear",
                  zorder=1)
        rep = [f for p, f in zip(ex.participants, fields) if highlight and p["vp_id"] == vp_id]
        if rep:
            # The representative VP's ring in LTD's red, opaque where its density is high.
            from matplotlib.colors import to_rgb

            rgba = np.zeros((*rep[0].shape, 4))
            rgba[..., :3] = to_rgb(STAR)
            rgba[..., 3] = 0.85 * rep[0]
            ax.imshow(rgba, extent=ext, origin="lower", interpolation="bilinear", zorder=1.5)
    else:
        mtl = "planar_circle" if ex.combo in ("million_scale_cbg", "vanilla_cbg") else "planar_annulus_weighted"
        if ex.region is not None and not ex.region.is_empty:
            patch = _poly_patch(_xy_geom(frame, ex.region), facecolor=REGION_FILL, edgecolor=INK, linewidth=0.6,
                                zorder=2)
            if patch is not None:
                ax.add_patch(patch)
        for p in sorted(ex.participants, key=lambda q: q["vp_id"] == vp_id):
            rep = highlight and p["vp_id"] == vp_id
            # The representative VP's constraint in LTD's red, so (a) and (b) read as one bound.
            color, lw, alpha = (STAR, 1.1, 1.0) if rep else (DIM, 0.6, DIM_ALPHA)
            outer, inner = ring_lines(frame, mtl, p)
            for c in outer:
                ax.plot(c[:, 0], c[:, 1], color=color, lw=lw, alpha=alpha, zorder=4 if rep else 3)
            for c in inner:
                ax.plot(c[:, 0], c[:, 1], color=color, lw=lw, alpha=alpha, ls=(0, (1, 1.2)), zorder=4 if rep else 3)
    _markers(ax, frame, ex, vp_id if highlight else "", )
    if star and ex.estimate is not None:
        x, y = frame.xy(*ex.estimate)
        ax.scatter([x], [y], s=34, marker=EST_MARKER, facecolor=STAR, edgecolor=EST_EDGE, linewidth=0.4, zorder=10)
    _scale_bar(ax, ext)
    _style(ax, ticks=False)


def ltd_curves(model, p: dict, rtts: np.ndarray) -> dict[str, np.ndarray]:
    """lower / upper / mu / sigma km of `model` for VP `p` over `rtts` (NaN where it declines)."""
    from scripts.framework.v2.types import Coord, Latency, VpId

    out = {k: np.full(len(rtts), np.nan) for k in ("lower", "upper", "mu", "sigma")}
    coord, vid = Coord(lat=p["vp_lat"], lon=p["vp_lon"]), VpId(p["vp_id"])
    for i, r in enumerate(rtts):
        try:
            res = model.predict(vid, coord, Latency(float(r)))
        except Exception:  # noqa: BLE001
            continue
        if not res.success or res.tg_distance is None:
            continue
        d = res.tg_distance
        out["lower"][i] = float(d.lower_km or 0.0); out["upper"][i] = float(d.upper_km)
        if d.mu_km is not None:
            out["mu"][i] = float(d.mu_km); out["sigma"][i] = float(d.sigma_km)
    return out


@dataclass
class LtdData:
    scatter: np.ndarray        # (n, 2) rtt_ms, km
    curves: dict[str, np.ndarray]
    grid: np.ndarray           # rtt_ms
    center: np.ndarray | None  # OCT-S spline centre, km
    cutoff_ms: float | None


def ltd_data(cache: RunCache, combo: str, fold: int, p: dict, samples: pd.DataFrame, xmax_ms: float) -> LtdData:
    from scripts.analysis.v5.modules.figure_ltd_model import center_for_vp, scatter_by_vp

    model = cache.model(combo, fold)
    if combo == SPO:
        pts = np.array([xy for v in scatter_by_vp(samples, max_points_per_vp=10**9).values() for xy in v])
        pts = pts[pts[:, 0] <= xmax_ms]
        pts = pts[:: max(1, len(pts) // 1500)]
    else:
        pts = np.array(scatter_by_vp(samples[samples.vp_id.astype(str) == p["vp_id"]],
                                     max_points_per_vp=10**9).get(p["vp_id"], [])).reshape(-1, 2)
    grid = np.linspace(0.02, xmax_ms, 300)
    center = None
    if combo == "octant_cbg_spl":
        c = center_for_vp(model, p["vp_id"], grid)
        center = np.array([np.nan if v[1] is None else v[1] for v in c]) if c else None
    cut = MM.ltd_cutoffs(model)
    cut = cut.get(p["vp_id"]) if isinstance(cut, dict) else cut
    return LtdData(pts, ltd_curves(model, p, grid), grid, center, cut)


def draw_ltd(ax, combo: str, data: LtdData, p: dict, d_true_km: float, *, D_km: float, R_ms: float,
             lim: tuple[float, float]) -> None:
    """Panel (a): scatter, 2/3 c, the model, and the VP's constraint at its RTT to the TG."""
    fx, fy = 1.0 / R_ms, 1.0 / D_km
    if len(data.scatter):
        ax.scatter(data.scatter[:, 0] * fx, data.scatter[:, 1] * fy, s=3.2, color=SCATTER, alpha=1.0, lw=0,
                   zorder=1, rasterized=True)
    g = data.grid * fx
    soi = data.grid * SOI_KM_PER_MS * fy
    c = data.curves
    if combo == "million_scale_cbg":
        ax.plot(g, soi, color=INK, lw=1.0, zorder=3)
    else:
        ax.plot(g, soi, color=INK, lw=0.6, ls=(0, (4, 2)), zorder=2)
        if combo == SPO:
            ax.fill_between(g, (c["mu"] - c["sigma"]) * fy, (c["mu"] + c["sigma"]) * fy, color=BAND, alpha=BAND_ALPHA, lw=0,
                            zorder=1.5)
            ax.plot(g, c["mu"] * fy, color=INK, lw=1.0, zorder=3)
            ax.plot(g, (c["mu"] - c["sigma"]) * fy, color=INK, lw=0.5, ls=(0, (2, 1.5)), zorder=3)
            ax.plot(g, (c["mu"] + c["sigma"]) * fy, color=INK, lw=0.5, ls=(0, (2, 1.5)), zorder=3)
        else:
            annulus = combo.startswith("octant")
            if annulus:
                ax.fill_between(g, c["lower"] * fy, c["upper"] * fy, color=BAND, alpha=BAND_ALPHA, lw=0,
                            zorder=1.5)
            ax.plot(g, c["upper"] * fy, color=INK, lw=1.0, zorder=3)
            if annulus:
                ax.plot(g, c["lower"] * fy, color=INK, lw=1.0, ls=(0, (1, 1.2)), zorder=3)
            if data.center is not None:
                ax.plot(g, data.center * fy, color=INK, lw=0.5, ls=(0, (3, 1, 1, 1)), zorder=3)
    if data.cutoff_ms:
        ax.axvline(data.cutoff_ms * fx, color="#777777", lw=0.5, ls=(0, (1, 1.5)), zorder=2)

    r = p["rtt_ms"] * fx
    if combo == SPO:
        bounds = [p["mu_km"]]
    elif combo.startswith("octant"):
        bounds = [p["lower_km"], p["upper_km"]]
    else:
        bounds = [p["upper_km"]]
    top = max(bounds) * fy
    ax.plot([r, r], [0, top], color=STAR, lw=0.7, ls=(0, (3, 2)), zorder=5)
    for b in bounds:
        ax.plot([0, r], [b * fy, b * fy], color=STAR, lw=0.7, ls=(0, (3, 2)), zorder=5)
    ax.scatter([r] * len(bounds), [b * fy for b in bounds], s=14, color=STAR, zorder=6, lw=0)
    if combo == SPO:
        ax.errorbar([r], [p["mu_km"] * fy], yerr=[[p["sigma_km"] * fy], [p["sigma_km"] * fy]], fmt="none",
                    ecolor=STAR, elinewidth=0.7, capsize=2, capthick=0.7, zorder=6)
    ax.scatter([r], [d_true_km * fy], s=40, marker=TG_MARKER, facecolor=TG_FILL, edgecolor=TG_EDGE, linewidth=0.5,
               zorder=7)
    ax.set_xlim(0, lim[0]); ax.set_ylim(0, lim[1])
    ax.set_xlabel("Norm. RTT", fontsize=FONT, labelpad=1)
    ax.set_ylabel("Norm. distance", fontsize=FONT, labelpad=1)
    _style(ax, ticks=True)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400)
    plt.close(fig)
    return path


def _fig(legend: bool = False):
    """LTD panel; with `legend`, a strip of LEGEND_IN below the axes holds the key."""
    w, h = PANEL_IN
    L = LEGEND_IN if legend else 0.0
    H = h + L
    fig = plt.figure(figsize=(w, H))
    return fig, fig.add_axes((0.17, (L + 0.16 * h) / H, 0.80, 0.80 * h / H))


def _fig_map(legend: bool = False):
    side = PANEL_IN[1]
    L = LEGEND_IN if legend else 0.0
    H = side + L
    fig = plt.figure(figsize=(side, H))
    return fig, fig.add_axes((0.02, (L + 0.02 * side) / H, 0.96, 0.96 * side / H))


# ---- legends ---------------------------------------------------------------------------

LEGEND_IN = 0.52
LEGEND_FONT = 5.6


def _key(fig, handles: list, ncol: int = 3) -> None:
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=LEGEND_FONT, frameon=False,
               handlelength=1.8, handletextpad=0.4, columnspacing=0.9, borderaxespad=0.15, labelspacing=0.35)


def _line(label, color=INK, ls="-", lw=1.0, marker=None, ms=3.5, mfc=None, mec=None, alpha=1.0):
    from matplotlib.lines import Line2D

    return Line2D([], [], color=color, ls=ls, lw=lw, marker=marker, markersize=ms, alpha=alpha,
                  markerfacecolor=mfc or color, markeredgecolor=mec or color, label=label)


def _mark(label, marker, fc, ec, ms=4.0):
    return _line(label, ls="none", marker=marker, ms=ms, mfc=fc, mec=ec, color=fc)


def ltd_legend(combo: str, data: LtdData, xmax_ms: float) -> list:
    h = [_mark("Pooled pairs" if combo == SPO else "Training pairs", "o", SCATTER, SCATTER, 2.5)]
    if combo == "million_scale_cbg":
        h.append(_line(r"$\frac{2}{3}c$ bound"))
    else:
        h.append(_line(r"$\frac{2}{3}c$", lw=0.6, ls=(0, (4, 2))))
        if combo == SPO:
            h += [_line(r"$\mu$"), _line(r"$\mu\pm\sigma$", lw=0.5, ls=(0, (2, 1.5)))]
        elif combo == "vanilla_cbg":
            h.append(_line("Bestline"))
        else:
            word = "hull" if combo == "octant_cbg_hull" else "bound"
            h += [_line(f"Upper {word}"), _line(f"Lower {word}", ls=(0, (1, 1.2)))]
            if data.center is not None:
                h.append(_line("Spline", lw=0.5, ls=(0, (3, 1, 1, 1))))
    if data.cutoff_ms and data.cutoff_ms <= xmax_ms:
        h.append(_line("Cutoff", color="#777777", lw=0.5, ls=(0, (1, 1.5))))
    h += [_line("VP's bound", color=STAR, lw=0.7, ls=(0, (3, 2)), marker="o", ms=3),
          _mark("True dist.", TG_MARKER, TG_FILL, TG_EDGE, 6.0)]
    return h


def map_legend(combo: str, *, highlight: bool, star: bool, filtered: bool = False) -> list:
    from matplotlib.patches import Patch

    shape = "disk" if combo in ("million_scale_cbg", "vanilla_cbg") else "annulus"
    h = [_mark("Rep. VP", "o", STAR, EST_EDGE) if highlight else _mark("VP", "o", VP_FILL, VP_DARK)]
    if highlight:
        h.append(_mark("Other VPs", "o", VP_FILL, VP_DARK))
    if filtered:
        h.append(_mark("Filtered VP", "o", "white", VP_DARK))
    h.append(_mark("TG", TG_MARKER, TG_FILL, TG_EDGE, 6.0))
    if combo == SPO:
        if highlight:
            h.append(Patch(facecolor=STAR, edgecolor="none", alpha=0.85, label="Rep. VP's ring"))
        h.append(Patch(facecolor="#b3b3b3", edgecolor="none", label="Other rings" if highlight else "VP rings"))
    else:
        if highlight:
            h.append(_line(f"Rep. VP's {shape}", color=STAR, lw=1.1))
        plural = "disks" if shape == "disk" else "annuli"
        h.append(_line(f"Other {plural}" if highlight else plural.capitalize(), color=DIM, lw=0.6, alpha=DIM_ALPHA))
        h.append(Patch(facecolor=REGION_FILL, edgecolor=INK, linewidth=0.6, label="Intersection"))
    if star:
        h.append(_mark("Estimate", EST_MARKER, STAR, EST_EDGE, 4.5))
    return h


def render(cache: RunCache, tg_id: str, vp_id: str, examples: dict[str, MethodExample], *, D_km: float,
           R_ms: float, out_dir: Path, inputs_root: Path) -> dict[str, Path]:
    """The 15 PNGs (three per method) plus the extent they share."""
    from scripts.analysis.v5.modules.figure_ltd_model import load_fit_samples

    obs = cache.observations(tg_id)
    tg_lat, tg_lon = float(obs.tg_lat.iloc[0]), float(obs.tg_lon.iloc[0])
    d_true = float(obs.set_index("vp_id").loc[vp_id, "d_km"])
    frame = Frame(tg_lat, tg_lon, D_km)
    ext = fit_extent(frame, examples, vp_id)
    fold = cache.folds[tg_id]
    samples, _ = load_fit_samples(cache.run, f"fold_{fold}", inputs_root=inputs_root,
                                  expect_n=cache.cfg(METHODS[0], fold).get("n_fit_samples"))
    datas = {}
    for ex in examples.values():
        if vp_id not in ex.vp_ids:
            raise ValueError(f"{vp_id} does not take part in {ex.term}'s MTL for {tg_id}")
    rtt = next(q["rtt_ms"] for q in examples["vanilla_cbg"].participants if q["vp_id"] == vp_id)
    xmax_ms = LTD_ZOOM * rtt
    for combo, ex in examples.items():
        p = next(q for q in ex.participants if q["vp_id"] == vp_id)
        datas[combo] = (ltd_data(cache, combo, fold, p, samples, xmax_ms), p)
    tops = [d_true]
    for combo, (d, _) in datas.items():
        if combo == "million_scale_cbg":
            continue
        cur = d.curves["mu"] + d.curves["sigma"] if combo == SPO else d.curves["upper"]
        tops.append(float(np.nanmax(cur)))
    xmax, ymax = xmax_ms / R_ms, max(tops) / D_km * 1.1
    written = {}
    for combo, ex in examples.items():
        data, p = datas[combo]
        term = ex.term
        fig, ax = _fig(legend=True)
        draw_ltd(ax, combo, data, p, d_true, D_km=D_km, R_ms=R_ms, lim=(xmax, ymax))
        _key(fig, ltd_legend(combo, data, xmax_ms))
        written[f"{term}.ltd"] = _save(fig, out_dir / PNG.format(term=term, phase="ltd"))
        for phase, hl, st in (("mtl", True, False), ("est", False, True)):
            fig, ax = _fig_map(legend=True)
            draw_map(ax, frame, ex, vp_id, ext, highlight=hl, star=st)
            _key(fig, map_legend(combo, highlight=hl, star=st, filtered=bool(ex.filtered)), ncol=2)
            written[f"{term}.{phase}"] = _save(fig, out_dir / PNG.format(term=term, phase=phase))
    return written


def render_candidates(caches: dict[str, RunCache], cands: pd.DataFrame, *, D_km: float, k: int,
                      out_png: Path, top: int = 6) -> Path:
    """Contact sheet: one row per top candidate, one EST map per method (internal, not for the paper)."""
    top_rows = cands.drop_duplicates(["run_id", "site_id"]).head(top)
    fig, axes = plt.subplots(len(top_rows), len(METHODS), figsize=(1.6 * len(METHODS), 1.7 * len(top_rows)),
                             squeeze=False)
    for i, r in enumerate(top_rows.itertuples(index=False)):
        cache = caches[r.run_id]
        ex = run_example(cache, r.tg_id, k)
        obs = cache.observations(r.tg_id)
        frame = Frame(float(obs.tg_lat.iloc[0]), float(obs.tg_lon.iloc[0]), D_km)
        ext = fit_extent(frame, ex, r.vp_id)
        for j, combo in enumerate(METHODS):
            ax = axes[i, j]
            draw_map(ax, frame, ex[combo], r.vp_id, ext, highlight=True, star=True)
            if i == 0:
                ax.set_title(method_label(combo), fontsize=FONT)
        axes[i, 0].set_ylabel(f"#{i} {r.run_id}\n{r.tg_id} {r.vp_id}\n{r.side} s={r.score} gap={r.bearing_gap}", fontsize=5)
    fig.tight_layout()
    return _save(fig, out_png)


def summarize_example(frame: Frame, examples: dict[str, MethodExample], D_km: float) -> dict:
    from shapely.geometry import Point

    out = {}
    for combo, ex in examples.items():
        row = {"phases": dict(zip(PHASES, PHASE_NAMES[combo])), "n_participants": len(ex.participants)}
        if ex.region is not None and not ex.region.is_empty:
            row["area_over_D2"] = R.geodesic_area_km2(ex.region) / D_km**2
            row["tg_inside"] = bool(ex.region.covers(Point(frame.tg[1], frame.tg[0])))
        if ex.estimate is not None:
            row["error_over_D"] = float(R._haversine_km(*frame.tg, *ex.estimate) / D_km)
        out[method_label(combo)] = row
    return out


def build_example(run: RunPaths, tg_id: str, vp_id: str, *, D_km: float, R_ms: float, k: int = DEFAULT_K,
                  inputs_root: Path, analysis_root: Path | None = None, gate: bool = True) -> dict[str, Path]:
    cache = RunCache.load(run, analysis_root)
    if tg_id not in cache.folds:
        raise ValueError(f"{tg_id} is not a scored TG of {run.run_id}")
    if gate:
        diffs = pipeline_gate(cache, tg_id)
        bad = {c: d for c, d in diffs.items() if not d < 1e-3}
        if bad:
            raise ValueError(f"pipeline rebuild does not reproduce the recorded prediction: {bad}")
    examples = run_example(cache, tg_id, k)
    out_dir = run.analysis_dir(KIND, root=analysis_root)
    written = render(cache, tg_id, vp_id, examples, D_km=D_km, R_ms=R_ms, out_dir=out_dir,
                     inputs_root=inputs_root)
    obs = cache.observations(tg_id)
    frame = Frame(float(obs.tg_lat.iloc[0]), float(obs.tg_lon.iloc[0]), D_km)
    meta = {"run_id": run.run_id, "tg_id": tg_id, "vp_id": vp_id, "k": k, "fold": cache.folds[tg_id],
            "pipeline_gate_km": pipeline_gate(cache, tg_id) if gate else None,
            "note": "MTL and EST re-run on the TG's K lowest-RTT VPs; not the benchmark's recorded prediction",
            "methods": summarize_example(frame, examples, D_km)}
    (out_dir / JSON_NAME).write_text(json.dumps(meta, indent=1) + "\n")
    written["json"] = out_dir / JSON_NAME
    return written


def build_candidates(runs: list[RunPaths], *, D_km: float, k: int = DEFAULT_K, top: int = 6, frac: float = 0.25,
                     workers: int = 1, analysis_root: Path | None = None,
                     progress: Callable[[str], None] | None = None) -> dict[str, Path]:
    cands = pick_candidates(runs, D_km=D_km, k=k, frac=frac, workers=workers, analysis_root=analysis_root,
                            progress=progress)
    if cands.empty:
        raise ValueError("no candidate (TG, VP) pair: every sampled TG failed a filter")
    out_dir = cross.cross_dir(sorted(r.run_id for r in runs), analysis_root=analysis_root, kind=KIND)
    cands.to_csv(out_dir / CANDIDATES_CSV, index=False)
    png = render_candidates({r.run_id: _POOL_CACHE[r.run_id] for r in runs}, cands, D_km=D_km, k=k,
                            out_png=out_dir / CANDIDATES_PNG, top=top)
    return {"csv": out_dir / CANDIDATES_CSV, "png": png}
