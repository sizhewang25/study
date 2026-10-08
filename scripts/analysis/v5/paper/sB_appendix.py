"""Appendices: HEALPix granularities (A) and the geometric-centroid ablation (B).

* `tab:healpix-levels` -- pixel count, area and width per level, from
  `4 pi R^2 / (12 N_side^2)` with R = 6,371 km (arithmetic, no artifact).
* **Appendix B** -- OCT-H / OCT-S against their geometric-centroid twins,
  seen and unseen: accuracy, flipped sites, displacement, the seen->unseen
  drop of each arm, and the EST runtime (`report-variant-delta`).
"""

from __future__ import annotations

import math

import pandas as pd

from scripts.analysis.v5.paper import fmt
from scripts.analysis.v5.paper.core import Context, Table, from_frame, network

SECTION = "A-B"
TITLE = "Appendices"

EARTH_RADIUS_KM = 6371.0
LEVELS = range(3, 9)


def healpix_levels() -> Table:
    rows = []
    for level in LEVELS:
        nside = 2 ** level
        npix = 12 * nside ** 2
        area = 4 * math.pi * EARTH_RADIUS_KM ** 2 / npix
        rows.append({"level": level, "nside": nside, "npix": npix, "area_km2": area, "width_km": math.sqrt(area)})
    return from_frame(
        "tab:healpix-levels", f"HEALPix granularities (R = {EARTH_RADIUS_KM:,.0f} km)", "arithmetic",
        pd.DataFrame(rows),
        [("Level", lambda r: str(int(r.level))), ("N_side", lambda r: str(int(r.nside))),
         ("N_pix", lambda r: f"{int(r.npix):,}"), ("Pixel area km²", lambda r: f"{round(r.area_km2):,}"),
         ("Pixel width km", lambda r: fmt.dp(r.width_km, 1))],
    )


def build(ctx: Context) -> tuple[list[Table], dict[str, str]]:
    v_dir = ctx.all_dir("variant-delta")
    v = pd.read_csv(v_dir / "variant_delta.csv")
    v["net"] = [network(s) for s in v.scope]
    v = v.assign(_pooled_last=(v.net == "All").astype(int)).sort_values(
        ["original_label", "regime", "_pooled_last", "net"]).drop(columns="_pooled_last")
    tables = [healpix_levels(), from_frame(
        "appendix B: accuracy", "Original vs geometric-centroid EST, per regime and network",
        "VD variant_delta.csv", v,
        [("Method", lambda r: r.original_label), ("Regime", lambda r: r.regime), ("Network", lambda r: r.net),
         ("Acc original", lambda r: fmt.frac(r.acc_orig)), ("Acc GEO", lambda r: fmt.frac(r.acc_var)),
         ("Δ", lambda r: fmt.dp(r.d_acc_pp, 1, signed=True)),
         ("Sites with a flip", lambda r: fmt.pct1(r.sites_flip_pct)),
         ("Median orig ×10⁻³", lambda r: fmt.e3(r.p50_orig_norm_e3)), ("Median GEO ×10⁻³", lambda r: fmt.e3(r.p50_var_norm_e3)),
         ("Displacement p50 ×10⁻³", lambda r: fmt.dp(r.disp_p50_norm_e3, 2))],
        note="Δ in pp, one decimal. Displacement: distance between the two arms' predictions, both answered.",
    )]
    hx_path = v_dir / "variant_delta.by_has_x.csv"
    if hx_path.exists():
        h = pd.read_csv(hx_path)
        h = h[h.scope == "pooled"].sort_values(["original_label", "regime", "has_x"])
        tables.append(from_frame(
            "appendix B: has-X / no-X", "Accuracy change within has-X and no-X, pooled",
            "VD variant_delta.by_has_x.csv (pooled)", h,
            [("Method", lambda r: r.original_label), ("Regime", lambda r: r.regime), ("Side", lambda r: r.has_x),
             ("Δ", lambda r: fmt.dp(r.d_acc_pp, 1, signed=True)),
             ("Correct→wrong", lambda r: str(int(r.n_correct_to_wrong))),
             ("Wrong→correct", lambda r: str(int(r.n_wrong_to_correct)))],
            note="Flip counts in TGs, for inspection.",
        ))
    d = pd.read_csv(v_dir / "variant_delta.drop.csv")
    d["net"] = [network(s) for s in d.scope]
    tables.append(from_frame(
        "appendix B: seen → unseen drop", "Each arm's change from seen to unseen sites",
        "VD variant_delta.drop.csv", d.sort_values(["original_label", "net"]),
        [("Method", lambda r: r.original_label), ("Network", lambda r: r.net),
         ("Drop original", lambda r: fmt.pp(r.drop_orig_pp, signed=True)), ("Drop GEO", lambda r: fmt.pp(r.drop_var_pp, signed=True)),
         ("Difference", lambda r: fmt.pp(r.did_acc_pp, signed=True)),
         ("Sites where the drop differs", lambda r: fmt.pct1(r.sites_drop_differs_pct))],
    ))
    t = pd.read_csv(v_dir / "variant_delta.timing.csv")
    tables.append(from_frame(
        "appendix B: runtime", "EST phase and pipeline runtime per TG, both arms", "VD variant_delta.timing.csv",
        t.sort_values(["original_label", "regime"]),
        [("Method", lambda r: r.original_label), ("Regime", lambda r: r.regime),
         ("EST p50 original", lambda r: fmt.runtime(r.ctr_p50_ms_orig)),
         ("EST p50 GEO", lambda r: f"{fmt.dp(r.ctr_p50_ms_var, 2)} ms"),
         ("Pipeline p50 original", lambda r: fmt.runtime(r.total_p50_ms_orig)),
         ("Pipeline p50 GEO", lambda r: fmt.runtime(r.total_p50_ms_var))],
    ))
    return tables, {"VD": ctx.source(v_dir)}
