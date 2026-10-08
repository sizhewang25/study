"""§4 Evaluation on error distance: the data behind its table and figures.

* `tab:err-dist-percentiles` / `fig:cdf-err-dist-pooled-mesh` -- each method's
  normalized error at p5..p99 and its answered share (`plot-error-cdf`).
* `fig:cdf-vp-proximity` -- d_geo, d_sp and Delta_VP percentiles, the zero-gap
  share, and the shares at the configured thresholds (`plot-vp-distance-cdf`).
* `fig:scatter-vp_pni_proximity` -- the Ward clusters: shares, rho, and the
  ranges of D_X-TG and Delta_VP (`plot-pni-gap`).
* `fig:boxplot-vp_pni_proximity_rtt` -- each cluster's normalized minimum RTT
  (`plot-pni-cluster-rtt`).

Distances are normalized x10^-3 (`fmt.e3`); shares are whole percent.
"""

from __future__ import annotations

import json

import pandas as pd

from scripts.analysis.v5.paper import fmt
from scripts.analysis.v5.paper.core import Context, Table, from_frame, method_rank

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

    sources = {"E": ctx.source(e_dir), "V": ctx.source(v_dir), "P": ctx.source(p_dir)}
    return tables, sources


def _range(r: pd.Series, what: str) -> str:
    lo, hi = r.get(f"{what}_min_norm_e3"), r.get(f"{what}_max_norm_e3")
    if not fmt.defined(lo):
        return ""
    return f"{fmt.e3(lo)}{fmt.DASH}{fmt.e3(hi)}"
