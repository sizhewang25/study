"""Replay each method's MTL on its stored participants: region area, truth inclusion, participants.

The benchmark persists `mtl_participants` (every constraint the MTL kept, with
its bounds and VP coordinate), `mtl_intersection_kind` and the point -- never
the feasible region. This module rebuilds the region with the run's own MTL
class and settings (`map_mtl.load_mtl_specs`, `map_mtl.participants_of`) and
measures it, so the §5 claims about how tight and how well placed each
method's region is rest on the geometry the method actually used.

## Variants

* The four geometric paper combos (`vanilla_cbg`, `million_scale_cbg`,
  `octant_cbg_hull`, `octant_cbg_spl`) replay each fold's own spec.
* `spotter_1sigma` -- **SPO's 1σ region**. `spotter_cbg` answers with a density
  surface, which has no feasible set. Its stored bands are exactly
  [µ − σ, µ + σ] (k = 1, clipped at 0 and at 2/3 c), so the region here is
  those bands, from every measuring VP, through Octant's own MTL (the same
  fold's `octant_cbg_hull` spec: the single face with the highest total
  1/RTT² weight), with the circle filter **off** because SPO's density uses
  every VP. It is a comparable region, not a bound on SPO's answer.
* `spotter_1sigma_filtered` -- the same with the filter on: the sensitivity
  arm, since OCT runs the filter and SPO does not.

## Per-TG metrics (`mtl_replay_tgs.parquet`, per run)

`replay_success`, `replay_kind` (`runner._intersection_kind`) and whether they
match the recorded `mtl_success` / `mtl_intersection_kind`; `area_over_D2`
(geodesic area of the lon/lat polygon over D²); `contains_tg`;
`tg_to_region_over_D` and `pred_to_region_over_D` (0 inside); the participant
count, its share of the TG's measuring VPs and its 1 / 2 / >=3 bin; the share
of participants whose ring excludes the TG by great-circle geometry
(`excl_true_share`) and by the MTL's own planar polygon (`excl_impl_share`);
and the median distance of those excluders over the TG's nearest-VP distance.
No coordinate is written.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import map_mtl as MM
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths

KIND = "mtl-replay"
TGS_NAME = "mtl_replay_tgs.parquet"
REPORT_NAME = "mtl_replay.report.json"

SPO_1SIGMA = "spotter_1sigma"
SPO_1SIGMA_FILTERED = "spotter_1sigma_filtered"


@dataclass(frozen=True)
class Variant:
    """Whose participants, whose MTL spec, and which spec keys to override."""

    participants_combo: str
    spec_combo: str
    overrides: tuple[tuple[str, Any], ...] = ()
    synthetic: bool = False  # no recorded region to match against


VARIANTS: dict[str, Variant] = {
    "vanilla_cbg": Variant("vanilla_cbg", "vanilla_cbg"),
    "million_scale_cbg": Variant("million_scale_cbg", "million_scale_cbg"),
    "octant_cbg_hull": Variant("octant_cbg_hull", "octant_cbg_hull"),
    "octant_cbg_spl": Variant("octant_cbg_spl", "octant_cbg_spl"),
    SPO_1SIGMA: Variant("spotter_cbg", "octant_cbg_hull", (("enable_circle_filter", False),), True),
    SPO_1SIGMA_FILTERED: Variant("spotter_cbg", "octant_cbg_hull", (("enable_circle_filter", True),), True),
}
PAPER_VARIANTS = ("vanilla_cbg", "million_scale_cbg", "octant_cbg_hull", "octant_cbg_spl", SPO_1SIGMA)

#: Boundary sampling step (degrees) for the TG / point-to-region distance.
BOUNDARY_STEP_DEG = 0.05
#: A point counts as on its region below this distance over D.
ON_REGION_TOL = 1e-3
_EARTH_R_KM = 6371.0088


# ---- specs ---------------------------------------------------------------------------


def override_spec(spec: tuple[str, str], overrides: tuple[tuple[str, Any], ...]) -> tuple[str, str]:
    """`(mtl_name, kwargs_json)` with `overrides` applied. Pure."""
    name, kwargs_json = spec
    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    kwargs.update(dict(overrides))
    return name, json.dumps(kwargs, sort_keys=True)


def variant_specs(run: RunPaths, variant: str) -> dict[int, tuple[str, str]]:
    """`{fold: (mtl_name, kwargs_json)}` the variant replays with."""
    v = VARIANTS[variant]
    return {f: override_spec(s, v.overrides) for f, s in MM.load_mtl_specs(run, v.spec_combo).items()}


# ---- geometry ------------------------------------------------------------------------


def _haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * _EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def ring_polygon(mtl_name: str, p: dict, n_pts: int):
    """The planar polygon the MTL itself draws for one participant, or None if unknown.

    `planar_circle` draws the outer disk only; the planar annulus MTLs draw
    outer minus inner. Both live in (lon, lat) degree space.
    """
    if mtl_name == "planar_circle":
        from scripts.framework.v2.mtl.planar_circle import _circle_to_planar_polygon

        return _circle_to_planar_polygon(p["vp_lat"], p["vp_lon"], p["upper_km"], n_pts)
    if mtl_name in ("planar_annulus", "planar_annulus_weighted"):
        from scripts.libs.octant.octant_geolocation import _annulus_to_shapely

        return _annulus_to_shapely(p["vp_lat"], p["vp_lon"], p["lower_km"], p["upper_km"], n_pts)
    return None


def geodesic_area_km2(region) -> float:
    """Geodesic area of a lon/lat Polygon / MultiPolygon on WGS84, km².

    Clipped to the valid lon/lat box first: a large planar 64-gon disk (SOI's
    2/3 c bound from a far VP) overshoots |lat| > 90, where pyproj returns NaN.
    """
    from pyproj import Geod
    from shapely.geometry import box

    region = region.intersection(box(-180.0, -90.0, 180.0, 90.0))
    if region.is_empty:
        return 0.0
    area, _ = Geod(ellps="WGS84").geometry_area_perimeter(region)
    return abs(area) / 1e6


def distance_to_region_km(region, lat: float, lon: float) -> float:
    """0 when (lat, lon) is in the region, else the great-circle km to its sampled boundary."""
    from shapely import segmentize
    from shapely.geometry import Point

    if region.covers(Point(lon, lat)):
        return 0.0
    xy = np.asarray(segmentize(region.boundary, BOUNDARY_STEP_DEG).coords
                    if region.boundary.geom_type == "LineString"
                    else [c for g in segmentize(region.boundary, BOUNDARY_STEP_DEG).geoms for c in g.coords])
    return float(_haversine_km(lat, lon, xy[:, 1], xy[:, 0]).min())


def tg_metrics(
    mtl_name: str,
    kwargs_json: str,
    participants: list[dict],
    *,
    tg: tuple[float, float],
    pred: tuple[float, float] | None,
    d_near_km: float,
    D_km: float,
) -> dict:
    """Replay one TG's MTL and measure its region. Pure up to the framework import; worker-safe."""
    import scripts.framework.v2  # noqa: F401  (populates the registries)
    from scripts.benchmark.v2.runner import _intersection_kind
    from scripts.framework.v2.registry import MTL_REGISTRY

    kwargs = json.loads(kwargs_json) if kwargs_json else {}
    mtl = MTL_REGISTRY[mtl_name](**kwargs)
    res = mtl.multilaterate(MM.ltd_results(participants)) if participants else None
    success = bool(res is not None and res.success)
    region = res.intersection if success else None
    planar = region is not None and hasattr(region, "geom_type")

    used = list(res.participating_vp_ids) if res is not None and res.participating_vp_ids else None
    kept = [p for p in participants if used is None or p["vp_id"] in set(used)]

    lat, lon = tg
    d = _haversine_km(lat, lon, np.array([p["vp_lat"] for p in kept]), np.array([p["vp_lon"] for p in kept])) \
        if kept else np.array([])
    up = np.array([p["upper_km"] for p in kept]); lo = np.array([p["lower_km"] for p in kept])
    excl_true = (d > up) | (d < lo)
    n_pts = int(kwargs.get("n_pts", 64))
    from shapely.geometry import Point

    pt = Point(lon, lat)
    polys = [ring_polygon(mtl_name, p, n_pts) for p in kept]
    excl_impl = np.array([(g is not None) and (not g.covers(pt)) for g in polys], dtype=bool) \
        if polys else np.array([], dtype=bool)
    excluder = excl_true | excl_impl
    far = d[excluder] / max(d_near_km, 1.0) if excluder.any() else np.array([])

    out = {
        "replay_success": success,
        "replay_kind": _intersection_kind(res) if res is not None else "none",
        "n_participants": len(kept),
        "excl_true_share": float(excl_true.mean()) if len(kept) else math.nan,
        "excl_impl_share": float(excl_impl.mean()) if len(kept) else math.nan,
        "any_excluder": bool(excluder.any()),
        "excluder_far_ratio_median": float(np.median(far)) if far.size else math.nan,
        "area_over_D2": math.nan,
        "contains_tg": False,
        "tg_to_region_over_D": math.nan,
        "pred_to_region_over_D": math.nan,
    }
    if planar and not region.is_empty:
        out["area_over_D2"] = geodesic_area_km2(region) / D_km**2
        out["tg_to_region_over_D"] = distance_to_region_km(region, lat, lon) / D_km
        out["contains_tg"] = out["tg_to_region_over_D"] == 0.0
        if pred is not None and all(np.isfinite(pred)):
            out["pred_to_region_over_D"] = distance_to_region_km(region, pred[0], pred[1]) / D_km
    return out


def _task(args: tuple) -> dict:
    mtl_name, kwargs_json, participants, kw = args
    return tg_metrics(mtl_name, kwargs_json, participants, **kw)


# ---- inputs --------------------------------------------------------------------------


def _scored(run: RunPaths, combo: str, analysis_root: Path | None) -> pd.DataFrame:
    """The scored TGs (after the min-VP floor) with their site, from `classify`."""
    path = run.analysis_dir("classify", root=analysis_root) / "healpix-128" / f"{combo}_tgs.parquet"
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `classify --run-id {run.run_id}` first")
    return pd.read_parquet(path, columns=["tg_id", "site_id", "fold"])


def _targets(run: RunPaths, combo: str) -> pd.DataFrame:
    frames = []
    for fold in run.fold_ids:
        p = run.combo_dir(combo, fold) / "targets.parquet"
        if not p.exists():
            continue
        t = pd.read_parquet(p, columns=["target_id", "target_lat", "target_lon", "pred_lat", "pred_lon",
                                        "n_obs", "mtl_success", "mtl_intersection_kind", "mtl_participants"])
        frames.append(t.assign(fold=int(fold.split("_")[1])))
    if not frames:
        raise MissingArtifactError(f"{run.run_id}: no targets.parquet for {combo}")
    return pd.concat(frames, ignore_index=True).rename(columns={"target_id": "tg_id"})


def _nearest_vp_km(run: RunPaths) -> pd.Series:
    from scripts.analysis.v5.modules import edges as EG
    from scripts.analysis.v5.modules.figure_sp_interconnect import load_edges

    e = load_edges(EG.resolve_source_csv(run, None))
    return e.groupby("tg_id").d_km.min()


def sample_tgs(scored: pd.DataFrame, *, limit: int | None, frac: float, seed: int = 0) -> pd.DataFrame:
    """`limit` first TGs (by id), or a site-stratified `frac` of every site's TGs (at least one each)."""
    s = scored.sort_values("tg_id")
    if limit:
        return s.head(limit)
    if frac >= 1.0:
        return s
    rng = np.random.default_rng(seed)
    parts = [g.iloc[np.sort(rng.choice(len(g), max(1, math.ceil(frac * len(g))), replace=False))]
             for _, g in s.groupby("site_id")]
    return pd.concat(parts).sort_values("tg_id")


# ---- one run -------------------------------------------------------------------------


def replay_run(
    run: RunPaths,
    variants: tuple[str, ...] = PAPER_VARIANTS,
    *,
    D_km: float,
    workers: int = 1,
    limit: int | None = None,
    frac: float = 1.0,
    analysis_root: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Replay `variants` on `run` and (re)write its `mtl_replay_tgs.parquet` rows for them."""
    out_dir = run.analysis_dir(KIND, root=analysis_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    d_near = _nearest_vp_km(run)
    rows = []
    for variant in variants:
        v = VARIANTS[variant]
        specs = variant_specs(run, variant)
        if not specs:
            raise MissingArtifactError(f"{run.run_id}: no run.json MTL spec for {v.spec_combo}")
        MM._invalidate_stale_cache(cache_dir, variant, specs, progress=progress)
        scored = sample_tgs(_scored(run, v.participants_combo, analysis_root), limit=limit, frac=frac)
        tg = _targets(run, v.participants_combo).merge(scored[["tg_id", "site_id"]], on="tg_id", how="inner")
        tasks, done = [], {}
        for r in tg.itertuples(index=False):
            cached = MM._read_region_cache(cache_dir, variant, r.tg_id)
            if cached:
                done[r.tg_id] = cached
                continue
            name, kw = specs[int(r.fold)]
            tasks.append((r.tg_id, (name, kw, MM.participants_of(r.mtl_participants), dict(
                tg=(float(r.target_lat), float(r.target_lon)),
                pred=(float(r.pred_lat), float(r.pred_lon)) if pd.notna(r.pred_lat) else None,
                d_near_km=float(d_near.get(r.tg_id, np.nan)), D_km=D_km))))
        if progress:
            progress(f"  {run.run_id} {variant}: {len(tasks)} to replay on {workers} worker(s), {len(done)} cached")
        if tasks:
            if workers > 1:
                from multiprocessing import Pool

                with Pool(processes=workers) as pool:
                    outs = pool.map(_task, [t[1] for t in tasks], chunksize=1)
            else:
                outs = [_task(t[1]) for t in tasks]
            for (tid, _), m in zip(tasks, outs):
                MM._write_region_cache(cache_dir, variant, tid, m)
                done[tid] = m
        for r in tg.itertuples(index=False):
            m = dict(done[r.tg_id])
            n_obs = int(r.n_obs) if pd.notna(r.n_obs) else 0
            m.update(
                variant=variant, run_id=run.run_id, tg_id=r.tg_id, fold=int(r.fold),
                participant_share=m["n_participants"] / n_obs if n_obs else math.nan,
                participant_bin=("0" if m["n_participants"] == 0 else "1" if m["n_participants"] == 1
                                 else "2" if m["n_participants"] == 2 else ">=3"),
                recorded_success=None if v.synthetic else bool(r.mtl_success),
                recorded_kind=None if v.synthetic else r.mtl_intersection_kind,
            )
            m["replay_matches"] = (None if v.synthetic else
                                   bool(m["replay_success"] == m["recorded_success"]
                                        and m["replay_kind"] == m["recorded_kind"]))
            rows.append(m)
    new = pd.DataFrame(rows)
    path = out_dir / TGS_NAME
    if path.exists():
        old = pd.read_parquet(path)
        new = pd.concat([old[~old.variant.isin(variants)], new], ignore_index=True)
    new.to_parquet(path, index=False)
    return path


# ---- SPO: the density vs its 1σ region -----------------------------------------------

DENSITY_NAME = "spo_density_check.parquet"
HDR_LEVELS = (0.5, 0.9)
_LTD_CACHE: dict[str, Any] = {}


def _spo_ltd(combo_dir: str):
    if combo_dir not in _LTD_CACHE:
        from scripts.benchmark.v2.checkpoint import load_ltd_checkpoint

        _LTD_CACHE[combo_dir] = load_ltd_checkpoint(Path(combo_dir))
    return _LTD_CACHE[combo_dir]


def density_task(args: tuple) -> dict:
    """Rebuild SPO's density from its fold's fitted LTD and compare it with SPO's 1σ region.

    Returns the argmax-vs-recorded check (distance / D), the HDR cell sets'
    area over D², the share of HDR cells whose centre lies in the 1σ region,
    and the share of the 1σ region's area those cells cover.
    """
    import scripts.framework.v2  # noqa: F401
    from shapely.geometry import Point

    from scripts.framework.v2.registry import MTL_REGISTRY
    from scripts.framework.v2.types import Coord, Latency, VpId

    combo_dir, dens_name, dens_kw, ring_name, ring_kw, participants, pred, D_km = args
    ltd = _spo_ltd(combo_dir)
    results = [ltd.predict(VpId(p["vp_id"]), Coord(lat=p["vp_lat"], lon=p["vp_lon"]), Latency(p["rtt_ms"]))
               for p in participants if p["rtt_ms"]]
    results = [r for r in results if r.success]
    band_err = max((abs(r.tg_distance.upper_km - p["upper_km"]) + abs(r.tg_distance.lower_km - p["lower_km"])
                    for r, p in zip(results, participants)), default=math.nan)
    dens = MTL_REGISTRY[dens_name](**json.loads(dens_kw)).multilaterate(results)
    out = {"band_rebuild_max_err_km": float(band_err), "density_success": bool(dens.success)}
    if not dens.success or dens.density is None:
        return out
    cells = dens.density.cells
    logp = np.asarray(dens.density.log_density, float)
    lat = np.array([c.lat for c in cells]); lon = np.array([c.lon for c in cells])
    k = int(np.argmax(logp))
    out["argmax_to_pred_over_D"] = float(_haversine_km(lat[k], lon[k], pred[0], pred[1]) / D_km)
    mass = np.exp(logp - logp.max()); mass /= mass.sum()
    order = np.argsort(-mass); cum = np.cumsum(mass[order])
    mtl = MTL_REGISTRY[dens_name](**json.loads(dens_kw))
    cell_km2 = float(getattr(mtl, "cell_area_km2", math.nan)) if dens.density.resolution else math.nan
    ring = MTL_REGISTRY[ring_name](**json.loads(ring_kw)).multilaterate(MM.ltd_results(participants))
    region = ring.intersection if ring.success else None
    if region is not None:
        out["argmax_in_1sigma"] = bool(region.covers(Point(lon[k], lat[k])))
        area_1s = geodesic_area_km2(region)
    for level in HDR_LEVELS:
        idx = order[: int(np.searchsorted(cum, level) + 1)]
        tag = f"hdr{int(level * 100)}"
        out[f"{tag}_n_cells_share_of_evaluated"] = len(idx) / len(cells)
        out[f"{tag}_area_over_D2"] = len(idx) * cell_km2 / D_km**2
        if region is not None:
            inside = np.array([region.covers(Point(lon[i], lat[i])) for i in idx])
            out[f"{tag}_cells_in_1sigma"] = float(inside.mean())
            out[f"{tag}_covers_1sigma_area"] = float(inside.sum() * cell_km2 / area_1s) if area_1s else math.nan
    return out


def density_check(run: RunPaths, *, D_km: float, frac: float, workers: int = 1,
                  analysis_root: Path | None = None, progress: Callable[[str], None] | None = None) -> Path:
    """`spo_density_check.parquet` for a site-stratified `frac` of `run`'s scored TGs."""
    dens_specs = MM.load_mtl_specs(run, "spotter_cbg")
    ring_specs = variant_specs(run, SPO_1SIGMA)
    scored = sample_tgs(_scored(run, "spotter_cbg", analysis_root), limit=None, frac=frac)
    tg = _targets(run, "spotter_cbg").merge(scored[["tg_id"]], on="tg_id")
    tasks = []
    for r in tg.itertuples(index=False):
        f = int(r.fold)
        tasks.append((r.tg_id, (str(run.combo_dir("spotter_cbg", f"fold_{f}")), *dens_specs[f], *ring_specs[f],
                                MM.participants_of(r.mtl_participants), (float(r.pred_lat), float(r.pred_lon)), D_km)))
    if progress:
        progress(f"  {run.run_id} SPO density check: {len(tasks)} TG(s) on {workers} worker(s)")
    if workers > 1:
        from multiprocessing import Pool

        with Pool(processes=workers) as pool:
            outs = pool.map(density_task, [t[1] for t in tasks], chunksize=1)
    else:
        outs = [density_task(t[1]) for t in tasks]
    df = pd.DataFrame([{"run_id": run.run_id, "tg_id": tid, **o} for (tid, _), o in zip(tasks, outs)])
    out = run.analysis_dir(KIND, root=analysis_root)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / DENSITY_NAME, index=False)
    return out / DENSITY_NAME


# ---- pooled report -------------------------------------------------------------------


def _pct(s: pd.Series) -> float:
    s = s.dropna()
    return round(float(s.mean()) * 100, 2) if len(s) else math.nan


def summarize(df: pd.DataFrame) -> dict:
    """Per (regime, variant, side): shares and normalized geometry. Percentages / ratios only."""
    out: dict[str, Any] = {}
    for (regime, variant, side), g in df.groupby(["regime", "variant", "has_x"], dropna=False):
        ok = g[g.replay_success]
        bins = (g.participant_bin.value_counts(normalize=True) * 100).round(2).to_dict()
        out.setdefault(regime, {}).setdefault(variant, {})[str(side)] = {
            "tg_share_of_side_pct": None,  # filled by caller when sampling
            "empty_pct": round(100 - _pct(g.replay_success), 2),
            "contains_tg_pct": _pct(g.contains_tg.astype(float)),
            "contains_tg_of_answered_pct": _pct(ok.contains_tg.astype(float)),
            "area_over_D2": {q: float(ok.area_over_D2.quantile(p)) if len(ok) else math.nan
                             for q, p in (("p25", .25), ("p50", .5), ("p75", .75))},
            "tg_to_region_over_D_p50_outside": float(ok.loc[~ok.contains_tg, "tg_to_region_over_D"].median())
            if (~ok.contains_tg).any() else math.nan,
            "pred_off_region_pct": _pct((ok.pred_to_region_over_D > ON_REGION_TOL).astype(float)),
            "participant_bins_pct": bins,
            "participant_share_p50": float(g.participant_share.median()),
            "excl_true_share_p50": float(g.excl_true_share.median()),
            "excl_impl_share_p50": float(g.excl_impl_share.median()),
            "any_excluder_pct": _pct(g.any_excluder.astype(float)),
            "excluder_far_ratio_p50": float(g.excluder_far_ratio_median.median()),
            "replay_match_pct": _pct(g.replay_matches.astype(float)) if g.replay_matches.notna().any() else None,
        }
    return out


def report(
    seen: list[RunPaths],
    pairs: dict[str, RunPaths],
    *,
    analysis_root: Path | None = None,
) -> Path:
    """Pool every run's `mtl_replay_tgs.parquet`; has-X from each base run's `sp_pni_cells_tgs.csv`."""
    from scripts.analysis.v5.modules.loso_delta import attach_has_x

    frames = []
    for base in seen:
        for regime, run in (("seen", base), ("unseen", pairs.get(base.run_id))):
            if run is None:
                continue
            p = run.analysis_dir(KIND, root=analysis_root) / TGS_NAME
            if not p.exists():
                continue
            df, src = attach_has_x(pd.read_parquet(p), base, analysis_root)
            if src is None:
                raise MissingArtifactError(f"{base.run_id}: no sp_pni_cells_tgs.csv for has-X labels")
            frames.append(df.assign(regime=regime))
    if not frames:
        raise MissingArtifactError("no mtl_replay_tgs.parquet; run report-mtl-replay first")
    df = pd.concat(frames, ignore_index=True)
    run_ids = sorted({r.run_id for r in seen} | {r.run_id for r in pairs.values()})
    out_dir = cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)
    rep = {"run_ids": run_ids, "by_regime_variant_side": summarize(df),
           "gate_replay_match_pct": {v: _pct(g.replay_matches.astype(float))
                                     for v, g in df[df.replay_matches.notna()].groupby("variant")},
           "note": "areas over D^2 and distances over D; SPO's 1σ region is a comparable region, "
                   "not a bound on SPO's density answer"}
    dens = [run.analysis_dir(KIND, root=analysis_root) / DENSITY_NAME for run in seen]
    dens = [pd.read_parquet(p) for p in dens if p.exists()]
    if dens:
        dd = pd.concat([attach_has_x(d, b, analysis_root)[0] for d, b in zip(dens, seen)], ignore_index=True)
        rep["spo_density_check_seen"] = {
            str(side): {c: (float(g[c].median()) if g[c].dtype.kind == "f" else _pct(g[c].astype(float)))
                        for c in g.columns if c not in ("run_id", "tg_id", "has_x")}
            for side, g in dd.groupby("has_x")
        }
        rep["spo_density_check_seen"]["argmax_matches_pred_pct"] = _pct(
            (dd.argmax_to_pred_over_D <= ON_REGION_TOL).astype(float))
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(rep, indent=1, default=lambda x: None if x is None else float(x)) + "\n")
    df.drop(columns=[c for c in df.columns if c.endswith("_lat") or c.endswith("_lon")], errors="ignore") \
      .to_parquet(out_dir / "mtl_replay_tgs.pooled.parquet", index=False)
    return path
