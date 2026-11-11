"""Slugs and claims of the mesh group (pro-paper): §1, §3, §6, §7, §8 and App. D.

Keys are topical, not sectional, so one number printed in several places (an
accuracy in §1, §6.1, Table x-cell and §8) is one slug. Every getter returns
the unrounded value; percent kinds return percent (0-100).

Where a number used to come from a one-off (`paper_stats.py`, notes), the
getter is the port and `source` names the artifact it now reads.
"""

from __future__ import annotations

import math

import pandas as pd

from scripts.analysis.v5.modules.paths import resolve_run
from scripts.analysis.v5.paper.core import network
from scripts.analysis.v5.paper_values.common import (
    CBG, LIGHT, METHODS, OCTH, OCTH_GEO, SP, TERM, Ctx, KEY_TO_TERM, combo,
)
from scripts.analysis.v5.paper_values.registry import put

S1, S3, S61, S62, S63, S64, S65, S7, S8, SD = (
    "§1", "§3", "§6.1", "§6.2", "§6.3", "§6.4", "§6.5", "§7", "§8", "App. D")

BARS = "classify/<seen>/outcome_bars.unbounded.pooled.healpix-128.csv"
ECDF = "classify/<seen>/error_cdf.pooled.norm.cut.csv"
TAX = "pni-gap/<seen>/sp_taxonomy.report.json"
BASE = "pni-gap/<seen>/sp_baseline.manifest.json"
CELLS = "pni-gap/<seen>/sp_pni_cells.report.json"
HASX = "loso-delta/<all>/loso_delta.by_has_x.csv (pooled, acc_base)"
XSIDE = "x-side-regions/<seen>/x_side_regions.json"
REPLAY = "mtl-replay/<all>/mtl_replay.report.json (seen)"
STEP = "step-heading/<seen>/step_heading.json"
COST = "cost/<seen>/cost_box.pooled.heap.csv (pipeline)"
EXT = "cost/<seen>/cost_box.pooled.heap.extrapolation.csv (all)"
PARETO = "pareto/<all>/pareto.csv"
PAIRS = "pareto/<all>/pareto.pairs.csv"
VDELTA = "variant-delta/<all>/variant_delta.csv (pooled)"


def _ok(holds: bool, detail: str) -> tuple[bool, str]:
    return bool(holds), detail


# ---- §1 / §6.1: accuracy and error distance ------------------------------------------

for m in METHODS:
    put(f"acc.seen.{m}", "pct", BARS, lambda c, m=m: c.acc(m), section=S61,
        note=f"{KEY_TO_TERM[m]} region classification accuracy, pooled, seen sites")
    for p in (25, 50, 75):
        put(f"err.p{p}.{m}", "e3", ECDF, lambda c, m=m, p=p: c.err(m, p), section=S61,
            note=f"{KEY_TO_TERM[m]} normalized error distance p{p} (x1e-3), answered TGs")

put("unans.seen.van", "pct", BARS, lambda c: c.unanswered("van"), section=S61)
put("err.van.answered", "pct", ECDF,
    lambda c: float(c.ecdf.loc[combo("van"), "n_plotted"] / c.ecdf.loc[combo("van"), "n_tgs"] * 100),
    section=S61, note="where VAN's CDF stops (the p-number)")
for p, f in ((50, "x1"), (25, "x1"), (75, "x0")):
    put(f"err.ratio.p{p}.s-p-over-oct-h", f, ECDF, lambda c, p=p: c.err("s-p", p) / c.err("oct-h", p),
        section=S61)
put("err.ratio.p50.spo-over-s-p", "x0", ECDF, lambda c: c.err("spo", 50) / c.err("s-p", 50), section=S61)
put("err.ratio.p50.van-over-s-p", "x1", ECDF, lambda c: c.err("van", 50) / c.err("s-p", 50), section=S61)
for a, b in (("soi", "s-p"), ("spo", "s-p"), ("oct-h", "s-p"), ("oct-h", "spo")):
    put(f"acc.seen.diff.{a}-minus-{b}", "pp", BARS, lambda c, a=a, b=b: c.acc(a) - c.acc(b), section=S61)


def _rank(c: Ctx, value) -> list[str]:
    return [KEY_TO_TERM[m] for m in sorted(METHODS, key=value)]


# ---- §3: the location filter -----------------------------------------------------------

def _filter(c: Ctx, net: str) -> float:
    """Share removed by the location filter in one network (`AS-B`, by its dataset label) or `All`."""
    d = c.csv("dataset", "dataset.csv", pool="all")
    d = d.assign(net=["All" if r == "all" else network(lbl) for r, lbl in zip(d.run_id, d.dataset)]).set_index("net")
    return float(d.loc[net, "est_removed_tgs"] / d.loc[net, "est_tgs_before_filter"] * 100)


put("data.filter.removed.all", "pct", "dataset/<all>/dataset.csv (est_removed_tgs / est_tgs_before_filter)",
    lambda c: _filter(c, "All"), section=S3)
put("data.filter.removed.as-b", "pct", "dataset/<all>/dataset.csv", lambda c: _filter(c, "AS-B"),
    section=S3)


# ---- §6.2: S-P -----------------------------------------------------------------------

def _tax(c: Ctx) -> dict:
    return c.json("pni-gap", "sp_taxonomy.report.json")


def _base(c: Ctx) -> dict:
    return c.json("pni-gap", "sp_baseline.manifest.json")


def _cells(c: Ctx) -> dict:
    return c.json("pni-gap", "sp_pni_cells.report.json")["shares"]["all"]


put("sp.nearest-vp-is-lowest", "pct", f"{TAX} (rtt_signal.all.nearest_vp_is_lowest_pct)",
    lambda c: _tax(c)["rtt_signal"]["all"]["nearest_vp_is_lowest_pct"], section=S62)
put("sp.rho.err-vs-dx", "f2", f"{BASE} (scatter.spearman_d_x_d_sp)",
    lambda c: _base(c)["scatter"]["spearman_d_x_d_sp"], section=S62)
put("sp.rho.err-vs-dvp", "f2", f"{BASE} (scatter.spearman_d_geo_d_sp)",
    lambda c: _base(c)["scatter"]["spearman_d_geo_d_sp"], section=S62)
put("sp.err-over-dx.p50", "x1", f"{BASE} (scatter.median_d_sp_over_d_x)",
    lambda c: _base(c)["scatter"]["median_d_sp_over_d_x"], section=S62)
put("sp.err-over-dvp.p50", "x1", f"{BASE} (scatter.median_d_sp_over_d_geo)",
    lambda c: _base(c)["scatter"]["median_d_sp_over_d_geo"], section=S62)
put("sp.pred-in-x-cell", "pct", f"{CELLS} (shares.all.pred_in_x_cell_pct)",
    lambda c: _cells(c)["pred_in_x_cell_pct"], section=S62)
put("sp.random-in-x-cell", "pct", f"{CELLS} (shares.all.random_in_x_cell_pct)",
    lambda c: _cells(c)["random_in_x_cell_pct"], section=S62)
put("sp.pred-not-in-x-cell", "pct", f"{TAX} (misses_x_cell.pct)",
    lambda c: _tax(c)["misses_x_cell"]["pct"], section=S62)
put("x.share.has-x", "pct", f"{TAX} (sides.has-X.tgs_pct)", lambda c: _tax(c)["sides"]["has-X"]["tgs_pct"],
    section=S62)
put("x.share.no-x", "pct", f"{TAX} (sides.no-X.tgs_pct)", lambda c: _tax(c)["sides"]["no-X"]["tgs_pct"],
    section=S62)


def _acc_side(c: Ctx, side: str) -> pd.DataFrame:
    return [r for r in _base(c)["accuracy"] if r["run_id"] == "all" and r["side"] == side][0]


put("sp.nox.wrong-adjacent", "pct", f"{BASE} (accuracy, all, no-X, wrong_adjacent_pct)",
    lambda c: _acc_side(c, "no-X")["wrong_adjacent_pct"], section=S62)
put("sp.hasx.wrong", "pct", f"{TAX} (100 - sides.has-X correct)",
    lambda c: 100 - _tax(c)["sides"]["has-X"]["causes"]["0"]["side_pct"], section=S62)
put("sp.hasx.near-tie", "pct", f"{TAX} (sides.has-X.causes.C.side_pct)",
    lambda c: _tax(c)["sides"]["has-X"]["causes"]["C"]["side_pct"], section=S62)
put("sp.hasx.inflated", "pct", f"{TAX} (sides.has-X.causes.A.side_pct)",
    lambda c: _tax(c)["sides"]["has-X"]["causes"]["A"]["side_pct"], section=S62)
put("sp.inflated.min-rtt-ratio", "x1", f"{BASE} (has_x_misses.inflated RTT.min_rtt_over_correct_max)",
    lambda c: _base(c)["has_x_misses"]["inflated RTT"]["min_rtt_over_correct_max"], section=S62)
put("sp.inflated.rho-max", "f2", f"{TAX} (rho_band.a_rho_max)", lambda c: _tax(c)["rho_band"]["a_rho_max"],
    section=S62)
put("sp.correct.rho-min", "f2", f"{TAX} (rho_band.correct_has_x_rho_min)",
    lambda c: _tax(c)["rho_band"]["correct_has_x_rho_min"], section=S62)


# ---- Table x-cell and §6.3 -----------------------------------------------------------------

def _hx(c: Ctx) -> pd.DataFrame:
    h = c.csv("loso-delta", "loso_delta.by_has_x.csv", pool="all")
    return h[h.scope == "pooled"].set_index(["method", "has_x"])


for m in METHODS:
    for side, sk in (("has-X", "has-x"), ("no-X", "no-x")):
        put(f"x.acc.{sk}.{m}", "pct", HASX,
            lambda c, m=m, side=side: float(_hx(c).loc[(combo(m), side), "acc_base"] * 100), section=S63,
            note=f"{KEY_TO_TERM[m]} accuracy on {side} TGs, seen sites")
put("x.unans.has-x.van", "pct", f"{HASX} (unanswered_base)",
    lambda c: float(_hx(c).loc[(combo("van"), "has-X"), "unanswered_base"] * 100), section=S64)
put("x.acc.has-x.gain-max", "pp1", HASX,
    lambda c: max(float(_hx(c).loc[(combo(m), "has-X"), "acc_base"]) for m in CBG) * 100
    - float(_hx(c).loc[(SP, "has-X"), "acc_base"]) * 100,
    section=S64, note="the most any CBG method gains over S-P on has-X TGs, from unrounded shares")
put("sp.hasx.headroom", "pp", HASX, lambda c: 100 - float(_hx(c).loc[(SP, "has-X"), "acc_base"]) * 100,
    section=S64, note="what a CBG method can gain at most on has-X")


def _xrtt(c: Ctx) -> dict:
    x = c.csv("pni-gap", "x_cell_rtt.csv")
    x = x.assign(net=[network(r) for r in x.run_id])
    return {r.side: r for r in x[x.net == "All"].itertuples()}


put("x.min-rtt.p50-ratio", "x1bare", "pni-gap/<seen>/x_cell_rtt.csv (All, p50_norm no-X / has-X)",
    lambda c: _xrtt(c)["no-X"].p50_norm / _xrtt(c)["has-X"].p50_norm, section=S63)
put("x.rho.p10.no-x", "f2", f"{TAX} (routing.no-X.per_tg_rho_p10)",
    lambda c: _tax(c)["routing"]["no-X"]["per_tg_rho_p10"], section=S63)
put("x.rho.p10.has-x", "f2", f"{TAX} (routing.has-X.per_tg_rho_p10)",
    lambda c: _tax(c)["routing"]["has-X"]["per_tg_rho_p10"], section=S63)


# ---- §6.4 / §6.5: regions, disks, steps ------------------------------------------------------

def _xs(c: Ctx) -> dict:
    return c.json("x-side-regions", "x_side_regions.json")


def _mr(c: Ctx, m: str, side: str) -> dict:
    return c.json("mtl-replay", "mtl_replay.report.json", pool="all")["by_regime_variant_side"]["seen"][combo(m)][side]


def _xerr(c: Ctx, m: str, side: str) -> float:
    return _xs(c)["error_over_D"][KEY_TO_TERM[m]][side]["p50"]


def _area(c: Ctx, m: str, side: str) -> float:
    return _xs(c)["area_over_D2"][KEY_TO_TERM[m]][side]["p50"]


def _inside(c: Ctx, m: str, side: str) -> float:
    return _xs(c)["tg_inside_pct"][KEY_TO_TERM[m]][side]


put("soi.hasx.single-disk", "pct", f"{REPLAY} (participant_bins_pct['1'])",
    lambda c: _mr(c, "soi", "has-X")["participant_bins_pct"]["1"], section=S64)
for m in ("oct-h", "oct-s"):
    put(f"{m}.hasx.excludes-tg", "pct", f"{XSIDE} (100 - tg_inside_pct)",
        lambda c, m=m: 100 - _inside(c, m, "has-X"), section=S64)
put("oct-h.hasx.err-ratio", "x1", f"{XSIDE} (error_over_D p50, S-P / OCT-H, has-X)",
    lambda c: _xerr(c, "s-p", "has-X") / _xerr(c, "oct-h", "has-X"), section=S64)
put("van.hasx.ge3-disks", "pct", f"{REPLAY} (participant_bins_pct['>=3'])",
    lambda c: _mr(c, "van", "has-X")["participant_bins_pct"][">=3"], section=S64)
put("spo.hasx.err-ratio", "x1", f"{XSIDE} (error_over_D p50, SPO / S-P, has-X)",
    lambda c: _xerr(c, "spo", "has-X") / _xerr(c, "s-p", "has-X"), section=S64)


put("van.nox.contains-tg", "pct", f"{XSIDE} (tg_inside_pct, VAN, no-X)", lambda c: _inside(c, "van", "no-X"),
    section=S65)
put("soi.area.nox-over-hasx", "x0", f"{XSIDE} (area_over_D2 p50, SOI no-X / has-X)",
    lambda c: _area(c, "soi", "no-X") / _area(c, "soi", "has-X"), section=S65)
put("soi.nox.le2-disks", "pct", f"{REPLAY} (participant_bins_pct '1' + '2', SOI no-X)",
    lambda c: _mr(c, "soi", "no-X")["participant_bins_pct"].get("1", 0)
    + _mr(c, "soi", "no-X")["participant_bins_pct"].get("2", 0), section=S65)


def _sh(c: Ctx, m: str) -> dict:
    return c.json("step-heading", "step_heading.json")["by_method"][KEY_TO_TERM[m]]


for m in CBG:
    put(f"step.r.p50.{m}", "f2", f"{STEP} (step_median_ci[0])", lambda c, m=m: _sh(c, m)["step_median_ci"][0],
        section=S65)
put("step.theta.p50.spo", "int", f"{STEP} (heading_median_ci_deg[0])",
    lambda c: _sh(c, "spo")["heading_median_ci_deg"][0], section=S65)
put("step.gap.van.correct", "pct", f"{STEP} (gap_closed_median_correct_vs_wrong[0])",
    lambda c: _sh(c, "van")["gap_closed_median_correct_vs_wrong"][0] * 100, section=S65)
put("step.gap.van.wrong", "pct", f"{STEP} (gap_closed_median_correct_vs_wrong[1])",
    lambda c: _sh(c, "van")["gap_closed_median_correct_vs_wrong"][1] * 100, section=S65)


def _headings(c: Ctx) -> pd.DataFrame:
    """Per-TG step/heading with labels; the correct-vs-wrong split is not in step_heading.json."""
    from scripts.analysis.v5.modules import figure_step_heading as SH

    return c.memo("headings", lambda: SH.load([resolve_run(r, c.outputs_root) for r in c.runs("seen")],
                                               analysis_root=c.analysis_root))


def _theta(c: Ctx, m: str, correct: bool) -> float:
    g = _headings(c)
    g = g[g.term == KEY_TO_TERM[m]]
    ok = g.cell_label.eq("correct")
    return float(g.heading[ok if correct else ~ok].median())


for m in ("oct-h", "oct-s"):
    for label, flag in (("correct", True), ("wrong", False)):
        put(f"step.theta.{m}.{label}", "int",
            "figure_step_heading.load (classify + sp_pni_cells per seen run); ported from paper_stats.py",
            lambda c, m=m, flag=flag: _theta(c, m, flag), section=S65)


# ---- §7: overhead ----------------------------------------------------------------------------

def _pipe(c: Ctx) -> pd.DataFrame:
    b = c.csv("cost", "cost_box.pooled.heap.csv")
    return b[b.stage == "pipeline"].set_index(["method", "channel"])


def _rt(c: Ctx, cid: str, stat: str) -> float:
    return float(_pipe(c).loc[(cid, "runtime"), stat])


def _heap(c: Ctx, cid: str, stat: str) -> float:
    return float(_pipe(c).loc[(cid, "memory_heap"), stat])


def _ext(c: Ctx) -> pd.DataFrame:
    e = c.csv("cost", "cost_box.pooled.heap.extrapolation.csv")
    return e[e.scope == "all"].set_index("method")


for m in LIGHT:
    put(f"rt.p50.{m}", "ms", COST, lambda c, m=m: _rt(c, combo(m), "p50"), section=S7,
        note=f"{KEY_TO_TERM[m]} median runtime per TG, ms")
put("rt.p50.oct-h", "s", COST, lambda c: _rt(c, OCTH, "p50"), section=S7)
put("rt.p95.oct-h", "s", COST, lambda c: _rt(c, OCTH, "p95"), section=S7)
put("rt.mean.oct-h", "s", COST, lambda c: _rt(c, OCTH, "mean"), section=S7)
put("rt.mean-over-p50.oct-h", "x1", f"{EXT} (mean_over_p50)",
    lambda c: float(_ext(c).loc[OCTH, "mean_over_p50"]), section=S7)
put("rt.p50.oct-h-geo", "s", COST, lambda c: _rt(c, OCTH_GEO, "p50"), section=S7)
put("heap.p50.oct-h", "sig2", COST, lambda c: _heap(c, OCTH, "p50"), section=S7, note="MB")
put("heap.p50.oct-h-geo", "sig2", COST, lambda c: _heap(c, OCTH_GEO, "p50"), section=S7, note="MB")
for m in (*LIGHT, "oct-h"):
    put(f"batch.core-hours.{m}", "sig2", f"{EXT} (core_hours, 1M TGs)",
        lambda c, m=m: float(_ext(c).loc[combo(m), "core_hours"]), section=S7)
put("batch.wall-days.oct-h", "f1", f"{EXT} (wall_days, 32 cores)", lambda c: float(_ext(c).loc[OCTH, "wall_days"]),
    section=S7)
put("batch.mem-gb", "sig2", f"{COST} (32 x the largest peak heap of any method)",
    lambda c: 32 * max(_heap(c, combo(m), "max") for m in (*LIGHT, "oct-h")) / 1024, section=S7)


# ---- §8: Pareto and SOTA ---------------------------------------------------------------------

def _pairs(c: Ctx) -> pd.DataFrame:
    return c.csv("pareto", "pareto.pairs.csv", pool="all")


def _ahead_every_net(c: Ctx, regime: str) -> list[str]:
    p = _pairs(c)
    out = []
    for m in CBG:
        row = p[(p.regime == regime) & (p.metric == "cell") & (p.a == SP) & (p.b == combo(m))]
        if bool(row["consistent"].iloc[0]) and c.pa(regime, m, "acc") > c.pa(regime, "s-p", "acc"):
            out.append(KEY_TO_TERM[m])
    return out


for m in METHODS:
    put(f"acc.unseen.{m}", "pct", f"{PARETO} (unseen, cell, acc)", lambda c, m=m: c.pa("unseen", m, "acc") * 100,
        section=S8)
for m in CBG:
    put(f"loso.drop.{m}", "pp", f"{PARETO} (seen acc - unseen acc)",
        lambda c, m=m: (c.pa("seen", m, "acc") - c.pa("unseen", m, "acc")) * 100, section=S8)
for regime in ("seen", "unseen"):
    for m in CBG:
        fmt_ = "s" if m in ("oct-h", "oct-s") else "ms"
        put(f"pareto.{regime}.rt.{m}", fmt_, f"{PARETO} ({regime}, runtime_p50_ms)",
            lambda c, m=m, regime=regime: c.pa(regime, m, "runtime_p50_ms"), section=S8)
put("pareto.seen.rt-ratio.oct-h-over-spo", "x0", PARETO,
    lambda c: c.pa("seen", "oct-h", "runtime_p50_ms") / c.pa("seen", "spo", "runtime_p50_ms"), section=S8)
put("pareto.unseen.spo.spread", "pp", f"{PARETO} (unseen, SPO, spread_pp)", lambda c: c.pa("unseen", "spo", "spread_pp"),
    section=S8)
put("pareto.unseen.spo.acc-min", "pct", f"{PARETO} (acc_min)", lambda c: c.pa("unseen", "spo", "acc_min") * 100,
    section=S8)
put("pareto.unseen.spo.acc-max", "pct", f"{PARETO} (acc_max)", lambda c: c.pa("unseen", "spo", "acc_max") * 100,
    section=S8)


def _frontier(c: Ctx, regime: str) -> list[str]:
    return [KEY_TO_TERM[m] for m in CBG if bool(c.pareto.loc[(regime, "cell", combo(m)), "on_frontier"])]


# ---- Appendix D: point estimation of Octant ----------------------------------------------------

def _vd(c: Ctx) -> pd.DataFrame:
    v = c.csv("variant-delta", "variant_delta.csv", pool="all")
    return v[v.scope == "pooled"].set_index(["original", "regime"])


def _vdrop(c: Ctx) -> pd.DataFrame:
    d = c.csv("variant-delta", "variant_delta.drop.csv", pool="all")
    return d[d.scope == "pooled"].set_index("original")


for m in ("oct-h", "oct-s"):
    for regime in ("seen", "unseen"):
        put(f"appd.acc-delta.{m}.{regime}", "pp1s", f"{VDELTA} (d_acc_pp)",
            lambda c, m=m, regime=regime: float(_vd(c).loc[(combo(m), regime), "d_acc_pp"]), section=SD)
put("appd.sites-flip.max", "pct", f"{VDELTA} (max sites_flip_pct)",
    lambda c: float(_vd(c).sites_flip_pct.max()), section=SD)
put("appd.drop-diff.oct-h", "pp1s", "variant-delta/<all>/variant_delta.drop.csv (pooled, did_acc_pp)",
    lambda c: float(_vdrop(c).loc[OCTH, "did_acc_pp"]), section=SD)
put("appd.drop-differs-sites.oct-h", "pct", "variant-delta/<all>/variant_delta.drop.csv (sites_drop_differs_pct)",
    lambda c: float(_vdrop(c).loc[OCTH, "sites_drop_differs_pct"]), section=SD)
put("appd.est-ratio.oct-h.seen", "xsig1", "variant-delta/<all>/variant_delta.timing.csv (ctr_p50_ratio)",
    lambda c: float(c.csv("variant-delta", "variant_delta.timing.csv", pool="all")
                    .set_index(["original", "regime"]).loc[(OCTH, "seen"), "ctr_p50_ratio"]), section=SD)
