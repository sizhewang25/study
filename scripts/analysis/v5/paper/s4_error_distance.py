"""§4 Evaluation on error distance: the data behind its table and figures.

* `tab:err-dist-percentiles` / `fig:cdf-err-dist-pooled-mesh` -- each method's
  normalized error at p5..p99 and its answered share (`plot-error-cdf`).
* `fig:cdf-vp-proximity` -- d_geo, d_sp and Delta_VP percentiles, the zero-gap
  share, and the shares at the configured thresholds (`plot-vp-distance-cdf`).
* `fig:scatter-vp_pni_proximity` -- the Ward clusters: shares, rho, and the
  ranges of D_X-TG and Delta_VP (`plot-pni-gap`).
* `fig:boxplot-vp_pni_proximity_rtt` -- each cluster's normalized minimum RTT
  (`plot-pni-cluster-rtt`).
* `sp-mechanism` / `sp-taxonomy` -- §4.2's two reasoning chains and the cause
  of every S-P prediction, in percentages, correlations and ratios only
  (`report-sp-pni-cells`, `plot-sp-baseline`, `plot-sp-interconnect`,
  `report-sp-taxonomy`).

Distances are normalized x10^-3 (`fmt.e3`); shares are whole percent.
"""

from __future__ import annotations

import json

import pandas as pd

from scripts.analysis.v5.paper import fmt
from scripts.analysis.v5.paper.core import Context, Table, from_frame, method_rank, network

SECTION = "4"
TITLE = "Evaluation on Error Distance"

#: The paper table's percentiles.
PCTS = (5, 25, 50, 75, 95, 99)
#: The VP-distance figure's series, as the paper names them.
SERIES = {"d_geo": "d_geo", "d_sp": "d_sp", "gap": "Δ_VP"}


def _norm(km: float, bounds: dict) -> float:
    return (km - bounds["min"]) / (bounds["max"] - bounds["min"]) * 1000


def build(ctx: Context) -> tuple[list[Table], dict[str, str]]:
    e_dir, v_dir, p_dir = ctx.seen_dir("classify"), ctx.seen_dir("vp-distance-cdf"), ctx.seen_dir("pni-gap")
    tables: list[Table] = []

    # -- the percentile table ------------------------------------------------------
    e = pd.read_csv(e_dir / "error_cdf.pooled.norm.cut.csv")
    e = e.assign(_o=method_rank(e.method_label)).sort_values("_o")
    e["answered"] = e.n_solved / e.n_tgs
    tables.append(from_frame(
        "tab:err-dist-percentiles", "Normalized error distance (×10⁻³) by percentile; answered share",
        "E error_cdf.pooled.norm.cut.csv", e,
        [("Method", lambda r: r.method_label),
         *[(f"p{p}", lambda r, p=p: fmt.e3(r[f"pred_dist_to_tg_norm_e3_p{p}"])) for p in PCTS],
         ("Answered", lambda r: fmt.frac(r.answered))],
        note="`--` marks a percentile above the method's answered share (undefined).",
    ))

    # -- d_geo, d_sp, Delta_VP ----------------------------------------------------
    v = pd.read_csv(v_dir / "vp_distance_cdf.norm.csv")
    tables.append(from_frame(
        "fig:cdf-vp-proximity", "d_geo, d_sp and Δ_VP over all TGs (normalized ×10⁻³)",
        "V vp_distance_cdf.norm.csv", v,
        [("Series", lambda r: SERIES[r.series]),
         *[(f"p{p}", lambda r, p=p: fmt.e3(r[f"p{p}_norm_e3"])) for p in (5, 25, 50, 75, 90, 95)],
         ("Max", lambda r: fmt.e3(r.max_norm_e3)),
         ("Zero", lambda r: fmt.pct(r.zero_share_pct))],
        note="Zero = share of TGs at exactly 0 (Δ_VP: nearest VP is the lowest-RTT VP).",
    ))
    shares = v_dir / "vp_distance_cdf.norm.shares.csv"
    if shares.exists():
        s = pd.read_csv(shares)
        tables.append(from_frame(
            "fig:cdf-vp-proximity (thresholds)", "Share of TGs at or below / above each threshold",
            "V vp_distance_cdf.norm.shares.csv", s,
            [("Series", lambda r: SERIES[r.series]), ("Threshold ×10⁻³", lambda r: fmt.e3(r.threshold)),
             ("≤", lambda r: fmt.pct(r.share_le_pct)), (">", lambda r: fmt.pct(r.share_gt_pct))],
        ))

    # -- the clusters ---------------------------------------------------------------
    pm = json.loads((p_dir / "pni_gap.manifest.json").read_text())
    bounds = pm["dist_norm_km"]
    c = pd.DataFrame(pm["clusters"])
    for col in ("d_pni_min_km", "d_pni_max_km", "gap_min_km", "gap_max_km"):
        c[col.replace("_km", "_norm_e3")] = _norm(c[col], bounds)
    overall = pd.DataFrame([{
        "cluster": "All", "tgs_pct": 100.0, "sites_pct": 100.0,
        "rho_tgs": pm["spearman"]["rho_tgs"], "rho_points": pm["spearman"]["rho_points"],
    }])
    rows = pd.concat([c, overall], ignore_index=True)
    tables.append(from_frame(
        "fig:scatter-vp_pni_proximity",
        f"Clusters of Δ_VP against D_X-TG ({pm['clustering']['method']}, k = {pm['clustering']['k']})",
        "P pni_gap.manifest.json", rows,
        [("Cluster", lambda r: f"C{r.cluster}" if r.cluster != "All" else "All"),
         ("TGs", lambda r: fmt.pct(r.tgs_pct)), ("Sites", lambda r: fmt.pct(r.sites_pct)),
         ("ρ (TGs)", lambda r: fmt.rho(r.rho_tgs)), ("ρ (points)", lambda r: fmt.rho(r.rho_points)),
         ("D_X-TG ×10⁻³", lambda r: _range(r, "d_pni")), ("Δ_VP ×10⁻³", lambda r: _range(r, "gap"))],
        note="ρ over TGs weights each (site, Δ_VP) point by its replicas; over points counts each once.",
    ))

    rtt = pd.read_csv(p_dir / "pni_cluster_rtt.csv")
    tables.append(from_frame(
        "fig:boxplot-vp_pni_proximity_rtt", "Normalized minimum RTT of each cluster's TGs",
        "P pni_cluster_rtt.csv", rtt,
        [("Cluster", lambda r: f"C{int(r.cluster)}"), ("TGs", lambda r: fmt.pct(r.tgs_pct)),
         *[(f"p{p}", lambda r, p=p: fmt.dp(r[f"p{p}_norm"], 3)) for p in (5, 25, 50, 75, 95)]],
    ))

    tables.extend(_sp_tables(p_dir))

    sources = {"E": ctx.source(e_dir), "V": ctx.source(v_dir), "P": ctx.source(p_dir)}
    return tables, sources


def _sp_tables(p_dir) -> list[Table]:
    """§4.2: S-P answers at the ingress point, and is correct only when the TG's cell holds it.

    Percentages, correlations and ratios only: no distance, RTT or count.
    """
    cells = json.loads((p_dir / "sp_pni_cells.report.json").read_text())
    base = json.loads((p_dir / "sp_baseline.manifest.json").read_text())
    inter = json.loads((p_dir / "sp_interconnect.report.json").read_text())
    tax = json.loads((p_dir / "sp_taxonomy.report.json").read_text())

    a, sc = cells["shares"]["all"], base["scatter"]
    acc = {r["side"]: r for r in base["accuracy"] if r["run_id"] == "all"}
    sig = tax["rtt_signal"]
    rows = [
        ("1", "S-P lands in the cell of the TG's nearest interconnect (vs a random VP)",
         f"{fmt.pct1(a['pred_in_x_cell_pct'])} vs {fmt.pct1(a['random_in_x_cell_pct'])}", "P sp_pni_cells (shares.all)"),
        ("1", "S-P VP's nearest interconnect is the TG's nearest", fmt.pct1(a["same_interconnect_pct"]),
         "P sp_pni_cells (shares.all)"),
        ("1", "S-P misses the cell of X: all / traffic enters elsewhere (B+D) / RTT not discriminative (A+C)",
         f"{fmt.pct1(tax['misses_x_cell']['pct'])} / {fmt.pct1(tax['misses_x_cell']['x_in_ne_x_pct'])} / "
         f"{fmt.pct1(tax['misses_x_cell']['rtt_not_discriminative_pct'])}", "P sp_taxonomy (misses_x_cell)"),
        ("1", "Of the enters-elsewhere TGs, those no path through X can explain (B)",
         fmt.pct(tax["misses_x_cell"]["proven_share_of_x_in_ne_x_pct"]), "P sp_taxonomy (misses_x_cell)"),
        ("1", "TGs whose lowest RTT is below the propagation bound through the nearest interconnect",
         fmt.pct1(inter["path_fit"]["via_tg_nearest_interconnect"]["tgs_below_floor_pct"]),
         "P sp_interconnect (path_fit)"),
        ("1", "S-P answers at the nearest VP location: all / has-X / no-X",
         " / ".join(fmt.pct1(sig[k]["nearest_vp_is_lowest_pct"]) for k in ("all", "has-X", "no-X")),
         "P sp_taxonomy (rtt_signal)"),
        ("1", "Lowest RTT at the nearest VP location over the TG's lowest RTT, median: all / has-X / no-X",
         " / ".join(fmt.times(sig[k]["nearest_over_lowest_rtt_median"]) for k in ("all", "has-X", "no-X")),
         "P sp_taxonomy (rtt_signal)"),
        ("1", "Fig. 5: ρ(S-P error, D_TG-nearest VP) / ρ(S-P error, D_X-TG)",
         f"{fmt.dp(sc['spearman_d_geo_d_sp'], 2)} / {fmt.dp(sc['spearman_d_x_d_sp'], 2)}", "P sp_baseline (scatter)"),
        ("1", "Median of S-P error / D_TG-nearest VP, and of S-P error / D_X-TG (D_X-TG = 0 counts as above)",
         f"{fmt.times(sc['median_d_sp_over_d_geo'])} / {fmt.times(sc['median_d_sp_over_d_x'])}",
         "P sp_baseline (scatter)"),
        ("2", "S-P lands in a cell holding an interconnect (vs a random VP)",
         f"{fmt.pct1(a['pred_in_interconnect_cell_pct'])} vs {fmt.pct1(a['random_in_interconnect_cell_pct'])}",
         "P sp_pni_cells (shares.all)"),
        ("2", "TGs labelled has-X / no-X", f"{fmt.pct(acc['has-X']['tgs_pct'])} / {fmt.pct(acc['no-X']['tgs_pct'])}",
         "P sp_baseline (accuracy)"),
        ("2", "S-P correct: has-X / no-X",
         f"{fmt.pct1(acc['has-X']['correct_pct'])} / {fmt.pct1(acc['no-X']['correct_pct'])}", "P sp_baseline (accuracy)"),
        ("2", "The has-X label predicts S-P's outcome", fmt.pct1(cells["rule"]["all"]["agree_pct"]),
         "P sp_pni_cells (rule)"),
        ("2", "Wrong no-X answers in a cell adjacent to the TG's", fmt.pct(acc["no-X"]["wrong_adjacent_pct"]),
         "P sp_baseline (accuracy)"),
        ("2", "Median distance to the nearest VP, no-X over has-X", fmt.times(sc["median_d_geo_no_x_over_has_x"]),
         "P sp_baseline (scatter)"),
    ]
    mech = pd.DataFrame(rows, columns=["chain", "claim", "value", "source"])
    out = [from_frame(
        "sp-mechanism", "§4.2: S-P answers at the ingress point (1) and is correct only when the TG's cell holds it (2)",
        "P sp_pni_cells / sp_baseline / sp_interconnect / sp_taxonomy", mech,
        [("Chain", lambda r: r.chain), ("Claim", lambda r: r.claim), ("Value", lambda r: r.value),
         ("Source", lambda r: r.source)],
        note="Percentages, correlations and ratios only.",
    )]

    t = []
    for side in ("has-X", "no-X"):
        for code, c in tax["sides"][side]["causes"].items():
            if c["n_tgs"]:
                t.append({"side": side, "code": code, **c})
    inflated = base["has_x_misses"].get("inflated RTT", {}) or {}
    band = tax["rho_band"]
    out.append(from_frame(
        "sp-taxonomy", "S-P causes per side (first match wins)", "P sp_taxonomy (sides)", pd.DataFrame(t),
        [("Side", lambda r: r.side), ("Cause", lambda r: f"{r.code} {r.cause}"),
         ("Of side", lambda r: fmt.pct1(r.side_pct)), ("Of all", lambda r: fmt.pct1(r.all_pct)),
         ("Networks", lambda r: ", ".join(network(x) for x in r.runs))],
        note=(f"A = per-TG ρ(RTT, D_VP-TG) < {band['low_rho']}: A's largest ρ {fmt.dp(band['a_rho_max'], 2)}, "
              f"the smallest of a correct has-X TG {fmt.dp(band['correct_has_x_rho_min'], 2)} "
              f"({'separated' if band['separated'] else 'NOT separated'}). Has-X A TGs' min RTT is "
              f"≥ {fmt.times(inflated.get('min_rtt_over_correct_max'))} the largest of any correct has-X TG."),
    ))
    return out


def _range(r: pd.Series, what: str) -> str:
    lo, hi = r.get(f"{what}_min_norm_e3"), r.get(f"{what}_max_norm_e3")
    if not fmt.defined(lo):
        return ""
    return f"{fmt.e3(lo)}{fmt.DASH}{fmt.e3(hi)}"
