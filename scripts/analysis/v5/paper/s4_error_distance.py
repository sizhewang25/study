"""§4 Evaluation on error distance: E.1-E.33 of the task inventory.

Reads three pooled artifact folders of the group's seen runs:

* **E** -- `classify/<g>.seen/error_cdf.pooled.norm.cut.{csv,ratios.csv}`
  (`plot-error-cdf`): the percentile table and every ratio between methods.
* **V** -- `vp-distance-cdf/<g>.seen/vp_distance_cdf.norm.{csv,shares.csv,manifest.json}`
  (`plot-vp-distance-cdf`): d_geo, d_sp, Delta_VP.
* **P** -- `pni-gap/<g>.seen/{pni_gap.manifest.json,pni_cluster_rtt.csv}`
  (`plot-pni-gap`, `plot-pni-cluster-rtt`): the clusters and their RTTs.

The table's percentiles are the paper's (`PCTS`); "every percentile" means
those. "The next method" at a percentile is the best non-Octant one. Claim
sources are plain strings; the report quotes them.
"""

from __future__ import annotations

import json
import math

import pandas as pd

from scripts.analysis.v5.paper import fmt
from scripts.analysis.v5.paper.core import Claim, Context, Table
from scripts.libs.cbg.rtt_model import SPEED_OF_LIGHT_KM_MS, SPEED_RATIO

SECTION = "4"
TITLE = "Evaluation on Error Distance"

#: The paper table's columns, in order.
PCTS = (5, 25, 50, 75, 95, 99)
#: The paper table's rows, in order.
ROWS = ("S-P", "SOI", "VAN", "OCT-H", "OCT-S", "SPO")
OCT = ("OCT-H", "OCT-S")
#: A ratio within this of 1 reads as "tied".
TIE = 0.05

SUB_41 = "§4.1 Achievable error distances"
SUB_42 = "§4.2 What limits Shortest Ping"

E_CSV = "error_cdf.pooled.norm.cut.csv"
E_RATIOS = "error_cdf.pooled.norm.cut.ratios.csv"
V_CSV = "vp_distance_cdf.norm.csv"
V_SHARES = "vp_distance_cdf.norm.shares.csv"
V_MANIFEST = "vp_distance_cdf.norm.manifest.json"
P_MANIFEST = "pni_gap.manifest.json"
P_RTT = "pni_cluster_rtt.csv"


def _col(p: int) -> str:
    return f"pred_dist_to_tg_norm_e3_p{p}"


class _Data:
    """The section's artifacts, loaded once."""

    def __init__(self, ctx: Context):
        self.e_dir = ctx.seen_dir("classify")
        self.v_dir = ctx.seen_dir("vp-distance-cdf")
        self.p_dir = ctx.seen_dir("pni-gap")
        self.e = pd.read_csv(self.e_dir / E_CSV).set_index("method_label")
        r = pd.read_csv(self.e_dir / E_RATIOS)
        self.r = r.set_index(["percentile", "method_label", "reference_label"])
        self.v = pd.read_csv(self.v_dir / V_CSV).set_index("series")
        self.vs = pd.read_csv(self.v_dir / V_SHARES)
        self.vm = json.loads((self.v_dir / V_MANIFEST).read_text())
        self.pm = json.loads((self.p_dir / P_MANIFEST).read_text())
        self.prtt = pd.read_csv(self.p_dir / P_RTT).set_index("cluster")
        self.clusters = {c["cluster"]: c for c in self.pm["clusters"]}

    def p(self, method: str, pct: int) -> float:
        return float(self.e.loc[method, _col(pct)])

    def ratio(self, method: str, ref: str, pct, *, inv: bool = False) -> float:
        key = pct if isinstance(pct, str) else f"p{pct}"
        return float(self.r.loc[(key, method, ref), "ratio_inv" if inv else "ratio"])

    def share(self, series: str, threshold: float, column: str) -> float | None:
        rows = self.vs[(self.vs.series == series) & (self.vs.threshold.sub(threshold).abs() < 1e-9)]
        return float(rows.iloc[0][column]) if len(rows) else None

    def next_best(self, pct: int) -> str:
        """The non-Octant method with the smallest error at `pct`."""
        rest = [m for m in ROWS if m not in OCT and fmt.defined(self.p(m, pct))]
        return min(rest, key=lambda m: self.p(m, pct))


def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(n, f"{n}th")


# ---- §4.1 ---------------------------------------------------------------------


def percentile_table(d: _Data) -> Table:
    """E.1-E.3: Table err-dist-percentiles, cell for cell, best per column in bold."""
    best = {}
    for p in PCTS:
        vals = {m: d.p(m, p) for m in ROWS if fmt.defined(d.p(m, p))}
        best[p] = min(vals, key=vals.get)
    rows = [
        "| Method | " + " | ".join(f"p{p}" for p in PCTS) + " | Answered |",
        "|---" * (len(PCTS) + 2) + "|",
    ]
    for m in ROWS:
        cells = [fmt.e3(d.p(m, p)) for p in PCTS]
        cells = [f"**{c}**" if best[p] == m else c for p, c in zip(PCTS, cells)]
        answered = fmt.pct(100 * d.e.loc[m, "n_solved"] / d.e.loc[m, "n_tgs"])
        rows.append(f"| {m} | " + " | ".join(cells) + f" | {answered} |")
    bold = ", ".join(f"p{p}: {best[p]}" for p in PCTS)
    return Table(
        id="E.1–E.3", sub=SUB_41, title=f"Table err-dist-percentiles (bold = best: {bold})",
        markdown="\n".join(rows), source=f"E {E_CSV}",
    )


def claims_41(d: _Data) -> list[Claim]:
    out: list[Claim] = []
    sp = "S-P"

    van = d.e.loc["VAN"]
    unans = 100 * (1 - van.n_solved / van.n_tgs)
    out.append(Claim(
        "E.4", SUB_41, f"VAN leaves {fmt.pct(unans)} of the TGs unanswered, so its curve ends at "
        f"{fmt.pct(100 - unans)}.", f"E {E_CSV} n_solved/n_tgs", {"unanswered_pct": unans},
    ))

    order = sorted(ROWS, key=lambda m: d.p(m, 50))
    listing = ", ".join(f"{m} ({fmt.e3(d.p(m, 50))})" for m in order)
    out.append(Claim(
        "E.5", SUB_41, f"By median, the methods rank {listing}.", f"E {E_CSV} p50",
        {m: d.p(m, 50) for m in order},
    ))

    for cid, m in (("E.6", "OCT-H"), ("E.7", "OCT-S")):
        ahead = [d.ratio(m, sp, p, inv=True) for p in PCTS]
        r = fmt.rng(min(ahead), max(ahead), fmt.num1, fmt.TIMES)
        out.append(Claim(
            cid, SUB_41, f"{m} is ahead of S-P at every percentile, by {r}.",
            f"E {E_RATIOS} ratio_inv vs S-P", {f"p{p}": a for p, a in zip(PCTS, ahead)},
            note="" if min(ahead) > 1 else "Not ahead at every percentile.",
        ))

    nxt = d.next_best(25)
    a_h, a_s = d.ratio("OCT-H", nxt, 25, inv=True), d.ratio("OCT-S", nxt, 25, inv=True)
    both = fmt.rng(min(a_h, a_s), max(a_h, a_s), fmt.num1, fmt.TIMES)
    out.append(Claim(
        "E.8", SUB_41, f"At p25, the two OCT methods are {both} ahead of the next method ({nxt}): "
        f"OCT-H {fmt.times(a_h)}, OCT-S {fmt.times(a_s)}.",
        f"E {E_RATIOS} ratio_inv vs {nxt}", {"OCT-H": a_h, "OCT-S": a_s, "next": nxt},
    ))

    margins = {p: (d.next_best(p), d.ratio("OCT-H", d.next_best(p), p, inv=True)) for p in (50, 75, 95)}
    parts = [f"{fmt.times(v)} at p{p} (vs {n})" for p, (n, v) in margins.items()]
    out.append(Claim(
        "E.9", SUB_41, "OCT-H's margin over the next method narrows to " + ", ".join(parts) + ".",
        f"E {E_RATIOS} ratio_inv", {f"p{p}": v for p, (_, v) in margins.items()},
    ))

    within = max(abs(d.ratio("OCT-S", "OCT-H", p) - 1) for p in (5, 25, 50)) * 100
    out.append(Claim(
        "E.10", SUB_41, f"Through p50, OCT-S is within {fmt.pct(within)} of OCT-H.",
        f"E {E_RATIOS} OCT-S/OCT-H", {"within_pct": within},
    ))

    tail = [d.ratio("OCT-S", "OCT-H", p) for p in (75, 95, 99)]
    r = fmt.rng(min(tail), max(tail), fmt.num1, fmt.TIMES)
    out.append(Claim(
        "E.11", SUB_41, f"From p75 on, OCT-S's error is {r} OCT-H's.",
        f"E {E_RATIOS} OCT-S/OCT-H", {f"p{p}": t for p, t in zip((75, 95, 99), tail)},
    ))

    soi = {p: d.ratio("SOI", sp, p, inv=True) for p in PCTS}
    same = all(abs(soi[p] - 1) < 1e-9 for p in (5, 25))
    mid = fmt.rng(min(soi[50], soi[75]), max(soi[50], soi[75]), fmt.num1, fmt.TIMES)
    p95 = "tied" if abs(soi[95] - 1) < TIE else f"at {fmt.times(soi[95])}"
    out.append(Claim(
        "E.12", SUB_41, f"SOI is {'identical to' if same else 'not identical to'} S-P through p25, "
        f"ahead by {mid} at p50 and p75, {p95} at p95 ({fmt.times(soi[95])}), and ahead by "
        f"{fmt.times(soi[99])} at p99.",
        f"E {E_RATIOS} ratio_inv vs S-P", {f"p{p}": v for p, v in soi.items()},
        note=f"Tied = within {fmt.pct(100 * TIE)} of 1.",
    ))

    van_behind = [d.ratio("VAN", sp, p) for p in PCTS if fmt.defined(d.p("VAN", p))]
    r = fmt.rng(min(van_behind), max(van_behind), fmt.num1, fmt.TIMES)
    out.append(Claim(
        "E.13", SUB_41, f"VAN is behind S-P at every percentile it defines, by {r}.",
        f"E {E_RATIOS} VAN/S-P", {"ratios": van_behind},
    ))

    behind = [p for p in PCTS if d.p("SPO", p) > d.p(sp, p)]
    ahead = [p for p in PCTS if p not in behind]
    out.append(Claim(
        "E.14", SUB_41, "SPO is behind S-P at " + ", ".join(f"p{p}" for p in behind)
        + (("; ahead at " + ", ".join(f"p{p}" for p in ahead)) if ahead else "") + ".",
        f"E {E_CSV}", {"behind": behind},
    ))

    best_p5 = min((m for m in ROWS if m != "SPO"), key=lambda m: d.p(m, 5))
    over = d.p("SPO", 5) / d.p(best_p5, 5)
    out.append(Claim(
        "E.15", SUB_41, f"SPO's p5 is {fmt.times(over)} that of the best method ({best_p5}), "
        f"i.e. {fmt.dp(math.log10(over), 1)} orders of magnitude.",
        f"E {E_CSV} p5", {"ratio": over, "best": best_p5},
    ))

    s_spo, s_oct = d.ratio("SPO", "SPO", "p50/p5"), d.ratio("OCT-H", "OCT-H", "p50/p5")
    out.append(Claim(
        "E.16", SUB_41, f"SPO's median is {fmt.times(s_spo)} its p5, against "
        f"{fmt.dp(s_oct, 0)}{fmt.TIMES} for OCT-H.",
        f"E {E_RATIOS} p50/p5", {"SPO": s_spo, "OCT-H": s_oct},
        note="A spread of 10× or more is printed whole.",
    ))

    p95_all = {m: d.p(m, 95) for m in ROWS if fmt.defined(d.p(m, 95))}
    rank = sorted(p95_all, key=p95_all.get, reverse=True).index("SPO") + 1
    out.append(Claim(
        "E.17", SUB_41, f"SPO's p95 ({fmt.e3(p95_all['SPO'])}) is the "
        f"{'' if rank == 1 else _ordinal(rank) + ' '}largest of the {len(p95_all)} defined p95 values.",
        f"E {E_CSV} p95", {"rank_from_worst": rank},
    ))

    sp_oct = d.ratio(sp, "OCT-H", 50)
    out.append(Claim(
        "E.18", SUB_41, f"S-P's median error is {fmt.times(sp_oct)} OCT-H's.",
        f"E {E_RATIOS} S-P/OCT-H p50", {"ratio": sp_oct},
    ))
    return out


# ---- §4.2 ---------------------------------------------------------------------


def claims_42(d: _Data) -> list[Claim]:
    out: list[Claim] = []
    geo, spv, gap = (float(d.v.loc[s, "p50_norm_e3"]) for s in ("d_geo", "d_sp", "gap"))

    le34 = d.share("d_geo", 3.4, "share_le_pct")
    out.append(Claim(
        "E.19", SUB_42, f"Half of the TGs have a VP within a normalized distance of {fmt.e3(geo)}"
        + (f" ({fmt.pct(le34)} are within 3.4)" if le34 is not None else "") + ".",
        f"V {V_CSV} d_geo p50; {V_SHARES}", {"d_geo_p50": geo, "share_le_3.4": le34},
    ))

    e_sp = d.p("S-P", 50)
    out.append(Claim(
        "E.20", SUB_42, f"S-P's median error is {fmt.e3(spv)}.",
        f"V {V_CSV} d_sp p50", {"d_sp_p50": spv, "error_cdf_sp_p50": e_sp},
        note="" if fmt.e3(spv) == fmt.e3(e_sp) else f"Differs from the error CDF's {fmt.e3(e_sp)}.",
    ))

    ratio = d.vm["median_ratio"]["d_sp_over_d_geo"]
    out.append(Claim(
        "E.21", SUB_42, f"S-P's median error is {fmt.times(ratio)} the d_geo median.",
        f"V {V_MANIFEST} median_ratio", {"ratio": ratio},
    ))

    zero = float(d.v.loc["gap", "zero_share_pct"])
    out.append(Claim(
        "E.22", SUB_42, f"{fmt.pct(zero)} of the TGs have zero divergence.",
        f"V {V_CSV} gap zero_share_pct", {"zero_pct": zero},
    ))

    gt_half = d.share("gap", 22.8, "share_gt_pct")
    out.append(Claim(
        "E.23", SUB_42, f"Δ_VP exceeds {fmt.e3(gap)} (its median) for half of the TGs"
        + (f" (it exceeds 22.8 for {fmt.pct(gt_half)})" if gt_half is not None else "") + ".",
        f"V {V_CSV} gap p50; {V_SHARES}", {"gap_p50": gap, "share_gt_22.8": gt_half},
    ))

    gt228 = d.share("gap", 228, "share_gt_pct")
    out.append(Claim(
        "E.24", SUB_42, f"Δ_VP exceeds 228 for {fmt.pct(gt228)} of the TGs.",
        f"V {V_SHARES} gap share_gt_pct", {"share_gt_228": gt228},
        note="" if gt228 is not None else "No share at 228 in the artifact; run with --share-at 228.",
    ))

    gmax = float(d.v.loc["gap", "max_norm_e3"])
    out.append(Claim(
        "E.25", SUB_42, f"The largest Δ_VP is {fmt.e3(gmax)}, {fmt.pct(gmax / 10)} of the maximum distance.",
        f"V {V_CSV} gap max_norm_e3", {"gap_max": gmax},
    ))

    cl = d.pm["clustering"]
    out.append(Claim(
        "E.26", SUB_42, f"The TGs are grouped by {cl['method'].capitalize()}'s clustering into "
        f"{cl['k']} clusters ({cl['k_source']}).", f"P {P_MANIFEST} clustering",
        {"method": cl["method"], "k": cl["k"]},
    ))

    rho = d.pm["spearman"]["rho_tgs"]
    out.append(Claim(
        "E.27", SUB_42, f"Over all TGs, Spearman's ρ = {fmt.rho(rho)}.",
        f"P {P_MANIFEST} spearman.rho_tgs", {"rho": rho},
    ))

    c1, c2, c3 = (d.clusters[i] for i in (1, 2, 3))
    out.append(Claim(
        "E.28", SUB_42, f"C2 (ρ = {fmt.rho(c2['rho_tgs'])}) follows the expectation.",
        f"P {P_MANIFEST} clusters[2]", {"rho": c2["rho_tgs"]},
    ))
    out.append(Claim(
        "E.29", SUB_42, f"C3 holds {fmt.pct(c3['tgs_pct'])} of the TGs, ρ = {fmt.rho(c3['rho_tgs'])}.",
        f"P {P_MANIFEST} clusters[3]", {"tgs_pct": c3["tgs_pct"], "rho": c3["rho_tgs"]},
    ))

    lo, hi = d.pm["dist_norm_km"]["min"], d.pm["dist_norm_km"]["max"]
    d_x = (c3["d_pni_max_km"] - lo) / (hi - lo) * 1000
    out.append(Claim(
        "E.30", SUB_42, f"Every C3 TG has an interconnect within a normalized distance of {fmt.e3(d_x)}.",
        f"P {P_MANIFEST} clusters[3].d_pni_max_km / D", {"d_pni_max_norm_e3": d_x},
    ))

    c3_rtt = float(d.prtt.loc[3, "p50_norm"])
    out.append(Claim(
        "E.31", SUB_42, f"C3's median normalized minimum RTT is {fmt.dp(c3_rtt, 2)}.",
        f"P {P_RTT} cluster 3 p50_norm", {"p50_norm": c3_rtt},
    ))
    out.append(Claim(
        "E.32", SUB_42, f"C1 (ρ = {fmt.rho(c1['rho_tgs'])}) shows a much weaker correlation.",
        f"P {P_MANIFEST} clusters[1]", {"rho": c1["rho_tgs"]},
    ))

    km = 0.5 * SPEED_OF_LIGHT_KM_MS * SPEED_RATIO   # 1 ms RTT -> one-way 0.5 ms
    out.append(Claim(
        "E.33", SUB_42, f"Under the speed of Internet, 1 ms of RTT corresponds to {fmt.dp(km, 0)} km.",
        "constant: rtt_model.SPEED_OF_LIGHT_KM_MS × SPEED_RATIO / 2", {"km": km},
    ))
    return out


# ---- entry point ----------------------------------------------------------------


def build(ctx: Context) -> tuple[list[Claim], list[Table], dict[str, str]]:
    d = _Data(ctx)
    root = d.e_dir.parents[1]
    sources = {
        "E": str(d.e_dir.relative_to(root)) + "/",
        "V": str(d.v_dir.relative_to(root)) + "/",
        "P": str(d.p_dir.relative_to(root)) + "/",
    }
    return [*claims_41(d), *claims_42(d)], [percentile_table(d)], sources
