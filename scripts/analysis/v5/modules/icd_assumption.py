"""How far the interconnect-distance assumption holds. A side report for the ICD runs.

`distance: interconnect_distance` (`GenericCSVSource`) fits every LTD against
the routing distance `d(VP, X*) + d(X*, TG)`, X* the interconnect nearest the
TG. X* is picked from the TG's location alone. Two things can go wrong, and
this report measures both, per run:

* **S-P disagrees with X\\***. The S-P VP sits at the interconnect the TG's
  traffic actually crosses (`figure_sp_interconnect`), so X_sp, the
  interconnect nearest the S-P VP, is the evidence against X*. Reported on
  every TG and again on the TGs where S-P is credible (S-P VP within
  `QUOTED_RADIUS_KM` of an interconnect and `rho >= LOW_RHO`), so a
  disagreement there counts against X* rather than against an uninformative S-P.
* **The routing distance breaks the propagation floor**: `d > FLOOR_KM_PER_MS
  x RTT`, a round trip at 2/3 c. A pair that breaks it under `d_route` but not
  under `d_air` teaches the fit that an RTT covers more distance than physics
  allows. Reported over every (VP, TG) edge, split by whether S-P agrees with
  X* on that TG, and on the S-P VP's own edge through X* against through X_sp.
* **The same floor and stretch with every edge routed through X_sp**, the
  path `distance: sp_interconnect_distance` fits on, so the two routed modes
  read side by side whichever one the run used (`fit_distance`).

The edge set is every (VP, TG) pair at its minimum RTT (`load_edges`): the fit
universe pooled over folds, since each TG is a fit target in all folds but its
own. Shares are therefore fold-independent.

Percentages only, never absolute TG or site counts. Distances are km, since
nothing here is drawn. Site shares use `sites.site_key`, so ~20 replicas of
one site don't count as 20 observations.

Command: `report-icd-assumption`. Writes beside the run's `pni-gap` output.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import figure_sp_interconnect as F
from scripts.analysis.v5.modules import pni_gap as P
from scripts.analysis.v5.modules import sites as S
from scripts.analysis.v5.modules.geodesy import haversine_km
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths

REPORT_NAME = "icd_assumption.report.json"
CSV_NAME = "icd_assumption_tgs.csv"

#: The floor and the credibility thresholds are `figure_sp_interconnect`'s, so
#: the two reports agree on what "violates" and "at an interconnect" mean.
FLOOR_KM_PER_MS = F.FLOOR_KM_PER_MS
CREDIBLE_RADIUS_KM = F.QUOTED_RADIUS_KM
LOW_RHO = F.LOW_RHO

#: Per-TG columns written to the CSV. No coordinate, VP or interconnect id.
CSV_COLUMNS = (
    "run_id", "tg_id", S.SITE_KEY_COL, "same_interconnect", "sp_credible",
    "x_separation_km", "d_via_tg_km", "d_via_sp_km", "rtt_sp_ms",
    "sp_violates_via_tg", "sp_violates_via_sp",
    "edge_violates_air_pct", "edge_violates_route_pct", "edge_stretch_p50",
)


# -- frames -------------------------------------------------------------------


def route_edges(edges: pd.DataFrame, pnis: pd.DataFrame) -> pd.DataFrame:
    """`edges` plus `d_route_km` through each TG's nearest interconnect, the
    floor flags under both distances, and `stretch = d_route / d_air`.

    `d_route_km` is the distance `GenericCSVSource` bakes into fit samples.
    """
    out = edges.copy()
    tg_x = F._to_interconnects(out.tg_lat, out.tg_lon, pnis)
    k = tg_x.argmin(axis=1)
    rows = np.arange(len(out))
    vp_x = haversine_km(out.vp_lat.to_numpy(float), out.vp_lon.to_numpy(float),
                        pnis.pni_lat.to_numpy()[k], pnis.pni_lon.to_numpy()[k])
    out["d_route_km"] = vp_x + tg_x[rows, k]
    floor = FLOOR_KM_PER_MS * out.rtt_ms
    out["violates_air"] = out.d_km > floor
    out["violates_route"] = out.d_route_km > floor
    with np.errstate(divide="ignore", invalid="ignore"):
        out["stretch"] = np.where(out.d_km > 0, out.d_route_km / out.d_km, np.nan)
    return out


def route_edges_via_sp(routed: pd.DataFrame, tgs: pd.DataFrame, pnis: pd.DataFrame) -> pd.DataFrame:
    """`routed` plus the same path through X_sp, the interconnect nearest each
    TG's S-P VP: `d_route_sp_km`, `violates_route_sp`, `stretch_sp`.

    `d_route_sp_km` is the distance `sp_interconnect_distance` bakes into fit
    samples (the source picks X_sp with the same tie-break).
    """
    out = routed.copy()
    k = out.tg_id.map(tgs.set_index("tg_id").sp_interconnect_index).to_numpy(dtype=int)
    x_lat, x_lon = pnis.pni_lat.to_numpy()[k], pnis.pni_lon.to_numpy()[k]
    out["d_route_sp_km"] = (
        haversine_km(out.vp_lat.to_numpy(float), out.vp_lon.to_numpy(float), x_lat, x_lon)
        + haversine_km(out.tg_lat.to_numpy(float), out.tg_lon.to_numpy(float), x_lat, x_lon)
    )
    out["violates_route_sp"] = out.d_route_sp_km > FLOOR_KM_PER_MS * out.rtt_ms
    with np.errstate(divide="ignore", invalid="ignore"):
        out["stretch_sp"] = np.where(out.d_km > 0, out.d_route_sp_km / out.d_km, np.nan)
    return out


def tg_frame(routed: pd.DataFrame, pnis: pd.DataFrame, *, run_id: str) -> pd.DataFrame:
    """`figure_sp_interconnect.tg_frame` plus X* vs X_sp and the per-TG edge shares.

    `routed` is `route_edges(edges, pnis)`.
    """
    tgs = F.tg_frame(routed, pnis, run_id=run_id)
    x_lat, x_lon = pnis.pni_lat.to_numpy(), pnis.pni_lon.to_numpy()

    sp = tgs[["tg_id", "sp_vp_id"]].merge(
        routed[["tg_id", "vp_id", "vp_lat", "vp_lon"]],
        left_on=["tg_id", "sp_vp_id"], right_on=["tg_id", "vp_id"], how="left", validate="1:1",
    )
    k_tg = F._to_interconnects(tgs.tg_lat, tgs.tg_lon, pnis).argmin(axis=1)
    k_sp = F._to_interconnects(sp.vp_lat, sp.vp_lon, pnis).argmin(axis=1)
    tgs["x_separation_km"] = haversine_km(x_lat[k_tg], x_lon[k_tg], x_lat[k_sp], x_lon[k_sp])
    # Kept for `route_edges_via_sp`; not written to the CSV (no interconnect ids).
    tgs["sp_interconnect_index"] = k_sp
    tgs["sp_credible"] = tgs[f"sp_at_{CREDIBLE_RADIUS_KM:g}km"] & (tgs.rho >= LOW_RHO)
    tgs["sp_violates_via_tg"] = tgs.d_via_tg_km > FLOOR_KM_PER_MS * tgs.rtt_sp_ms
    tgs["sp_violates_via_sp"] = tgs.d_via_sp_km > FLOOR_KM_PER_MS * tgs.rtt_sp_ms

    by_tg = routed.groupby("tg_id")
    per_tg = pd.DataFrame({
        "edge_violates_air_pct": 100.0 * by_tg.violates_air.mean(),
        "edge_violates_route_pct": 100.0 * by_tg.violates_route.mean(),
        "edge_stretch_p50": by_tg.stretch.median(),
    })
    return tgs.merge(per_tg, left_on="tg_id", right_index=True, how="left", validate="1:1")


# -- the report ---------------------------------------------------------------


def _stats(s: pd.Series) -> dict:
    s = s.dropna()
    return {"p50": F._q(s, 0.50), "p90": F._q(s, 0.90), "max": float(s.max()) if len(s) else float("nan")}


def _site_shares(tgs: pd.DataFrame, flag: str) -> dict:
    """Share of sites where any / every replica carries `flag`."""
    by_site = tgs.groupby(S.SITE_KEY_COL)[flag]
    return {"any_pct": F._pct(by_site.any()), "all_pct": F._pct(by_site.all())}


def _agreement(tgs: pd.DataFrame, universe: int) -> dict:
    off = tgs[~tgs.same_interconnect]
    off_flag = tgs.assign(disagree=~tgs.same_interconnect)
    return {
        "share_of_tgs_pct": 100.0 * len(tgs) / universe if universe else float("nan"),
        "disagree_tgs_pct": F._pct(~tgs.same_interconnect),
        "disagree_sites": _site_shares(off_flag, "disagree"),
        # Over the disagreeing TGs: how far apart the two interconnects are,
        # and how much longer the S-P VP's path through X* is than through X_sp.
        "x_separation_km": _stats(off.x_separation_km),
        "route_overstatement_km": _stats(off.d_via_tg_km - off.d_via_sp_km),
    }


def _floor(e: pd.DataFrame) -> dict:
    return {
        "air_pct": F._pct(e.violates_air),
        "route_pct": F._pct(e.violates_route),
        # Broken only by the routing distance: what ICD adds to the fit.
        "route_only_pct": F._pct(e.violates_route & ~e.violates_air),
    }


def _stretch(s: pd.Series) -> dict:
    s = s.dropna()
    return {**{f"p{q:g}": F._q(s, q / 100) for q in (50, 90, 99)}, "max": float(s.max())}


def report(routed: pd.DataFrame, tgs: pd.DataFrame, meta: dict) -> dict:
    """Every number of the side report, as one JSON-ready dict.

    `routed` carries both routed paths: `route_edges` then `route_edges_via_sp`.
    """
    same = routed.tg_id.map(tgs.set_index("tg_id").same_interconnect).to_numpy(dtype=bool)
    credible = tgs[tgs.sp_credible]
    return {
        **meta,
        "units": {
            "agreement": "per TG, and per site (site_key)",
            "floor_violations.edges": "per (VP, TG) min-RTT edge = the fit universe pooled over folds",
            "floor_violations.sp_edge": "per TG, the S-P VP's edge only",
            "stretch": "per (VP, TG) edge with d_air > 0",
            "*_via_sp_interconnect": "the same, every edge routed through its TG's X_sp",
        },
        "thresholds": {
            "floor_km_per_ms": FLOOR_KM_PER_MS,
            "credible_radius_km": CREDIBLE_RADIUS_KM,
            "credible_min_rho": LOW_RHO,
        },
        "agreement": {
            "all": _agreement(tgs, len(tgs)),
            "sp_credible": _agreement(credible, len(tgs)),
        },
        "floor_violations": {
            "edges": _floor(routed),
            "edges_where_sp_agrees": _floor(routed[same]),
            "edges_where_sp_disagrees": _floor(routed[~same]),
            "sp_edge": {
                "via_tg_nearest_pct": F._pct(tgs.sp_violates_via_tg),
                "via_sp_nearest_pct": F._pct(tgs.sp_violates_via_sp),
                "via_tg_nearest_sites": _site_shares(tgs, "sp_violates_via_tg"),
            },
            "edges_via_sp_interconnect": {
                "air_pct": F._pct(routed.violates_air),
                "route_pct": F._pct(routed.violates_route_sp),
                "route_only_pct": F._pct(routed.violates_route_sp & ~routed.violates_air),
            },
        },
        "stretch": _stretch(routed.stretch),
        "stretch_via_sp_interconnect": _stretch(routed.stretch_sp),
    }


# -- runs ---------------------------------------------------------------------


def _fit_distance(run: RunPaths, inputs_root: Path) -> tuple[str, str | None]:
    """`(distance mode, interconnect_csv_path)` the run's LTDs were fit with.

    Read off the per-fold inputs manifests, the record of what was actually
    materialized, rather than the config: v5 reaches a config only through
    `target_space.json`, and a run without one must not read as air distance.
    Manifests written before the `distance` key existed are air distance.
    """
    manifests = sorted((inputs_root / run.source / run.run_id / run.setup).glob("*/manifest.json"))
    if not manifests:
        raise MissingArtifactError(
            f"{run.run_id}: no materialized inputs under {inputs_root}; "
            f"cannot tell which distance its LTDs were fit with."
        )
    seen = set()
    for m in manifests:
        rec = json.loads(m.read_text())
        seen.add((rec.get("distance", "air_distance"), rec.get("interconnect_csv_path")))
    if len(seen) > 1:
        raise ValueError(f"{run.run_id}: folds were materialized with different distances {sorted(seen)}")
    return seen.pop()


def build(
    run: RunPaths,
    pni_csv: Path,
    *,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
    inputs_root: Path | None = None,
) -> Path:
    """Write the report and the per-TG CSV for one run; return the report path.

    Refuses an ICD run whose baked interconnect list is not `pni_csv`: the
    report would then describe a different X* than the one the LTDs fit.
    """
    from scripts.analysis.v5.modules import edges as E
    from scripts.benchmark.v2.inputs import DEFAULT_INPUTS_ROOT

    mode, baked = _fit_distance(run, inputs_root or DEFAULT_INPUTS_ROOT)
    if baked is not None and Path(baked).resolve() != Path(pni_csv).resolve():
        raise ValueError(
            f"{run.run_id} fits through {baked} but the report was given {pni_csv}; "
            f"pass --pni-csv {baked}."
        )
    csv = E.resolve_source_csv(run, source_csv)
    pnis = P.load_pnis(pni_csv)
    routed = route_edges(F.load_edges(csv), pnis)
    tgs = tg_frame(routed, pnis, run_id=run.run_id)
    routed = route_edges_via_sp(routed, tgs, pnis)
    meta = {
        "run_id": run.run_id,
        "source_csv": str(csv),
        "source_csv_sha256": P.sha256_file(Path(csv)),
        "pni_csv": str(pni_csv),
        "pni_csv_sha256": P.sha256_file(Path(pni_csv)),
        "fit_distance": mode,
    }
    out = report(routed, tgs, meta)

    out_dir = P.output_dir(run.run_id, pni_csv, analysis_root=analysis_root)
    tgs.sort_values("tg_id")[list(CSV_COLUMNS)].to_csv(out_dir / CSV_NAME, index=False)
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps(out, indent=2) + "\n")
    return path
