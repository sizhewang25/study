"""§5 Evaluation on region classification: the data behind its tables and figures.

Seen-site regime unless a table says otherwise. Shares are of all TGs (or of
the group / site set the column names), whole percent.

* `fig:stackbar-cls-acc-overall` -- correct / wrong / unanswered per method (`plot-outcome-bars`).
* `tab:x-cell` -- has-X / no-X / overall accuracy per method and network, the
  group shares, and VAN's unanswered share within has-X (`report-loso-delta`,
  `plot-x-cell-rtt`).
* `fig:boxplot-rtt-of-x-cell-by-asn` -- normalized minimum RTT per network and
  side, and the sites across the split (`plot-x-cell-rtt`).
* **S-P by side** -- S-P's correct share and where its wrong answers land
  (`report-sp-pni-cells`).
* `fig:upset-cell-correct`, `tab:cell-correct-sites` and the nesting cohorts
  (`plot-correct-upset`).
* `fig:stackbar-bounded-pooled` -- predictions by pixel distance, bounded accuracy (`plot-outcome-bars`).
* `fig:spo-exclusive-pixel`, `fig:spo-replica-spread`, `fig:spo-peripherality`,
  and each network's peripheral-cell share (`plot-exclusive-error`,
  `plot-stability`, `plot-peripherality`, `build-answer-space`).
* `tab:seen-unseen` and its per-network view (`report-loso-delta`).
"""

from __future__ import annotations

import json

import pandas as pd

from scripts.analysis.v5.modules.methods import method_label
from scripts.analysis.v5.paper import fmt
from scripts.analysis.v5.paper.core import METHOD_ORDER, Context, Table, from_frame, method_rank, network

SECTION = "5"
TITLE = "Evaluation on Region Classification"

NETWORKS = ("All", "AS-A", "AS-B", "AS-C")
SIDES = ("has-X", "no-X")


def _ordered(frame: pd.DataFrame, col: str = "method_label") -> pd.DataFrame:
    return frame.assign(_o=method_rank(frame[col])).sort_values("_o").drop(columns="_o")


def _scope_net(scope: str) -> str:
    return "All" if scope == "pooled" else network(scope.split("->")[0])


def build(ctx: Context) -> tuple[list[Table], dict[str, str]]:
    c_dir, p_dir, l_dir = ctx.seen_dir("classify"), ctx.seen_dir("pni-gap"), ctx.all_dir("loso-delta")
    x_dir, s_dir, pe_dir = (ctx.seen_dir(k) for k in ("exclusive-error", "stability", "peripherality"))
    pa_dir = ctx.all_dir("pareto")
    tables: list[Table] = []

    # -- overall accuracy -----------------------------------------------------------
    ub = _ordered(pd.read_csv(c_dir / "outcome_bars.unbounded.pooled.healpix-128.csv")
                  .assign(method_label=lambda f: f.method.map(method_label)))
    tables.append(from_frame(
        "fig:stackbar-cls-acc-overall", "Region classification, pooled", "C outcome_bars.unbounded.pooled.healpix-128.csv", ub,
        [("Method", lambda r: r.method_label), ("Correct", lambda r: fmt.frac(r.share_n_cell_correct)),
         ("Wrong", lambda r: fmt.frac(r.share_n_cell_wrong)), ("Unanswered", lambda r: fmt.frac(r.share_n_cell_unanswered))],
    ))

    # -- Table x-cell -------------------------------------------------------------
    hx = pd.read_csv(l_dir / "loso_delta.by_has_x.csv")
    hx["net"] = hx.scope.map(_scope_net)
    ld = pd.read_csv(l_dir / "loso_delta.csv")
    ld["net"] = ld.scope.map(_scope_net)
    xr = pd.read_csv(p_dir / "x_cell_rtt.csv")
    xr["net"] = [network(r) for r in xr.run_id]
    share = {(r.net, r.side): r.tgs_pct for r in xr.itertuples()}
    cols = ["Method"] + [f"{n} {k}" for n in NETWORKS for k in (*SIDES, "Acc")]
    rows = [["(group share)"] + [x for n in NETWORKS for x in
                                (fmt.pct(share[(n, "has-X")]), fmt.pct(share[(n, "no-X")]), "")]]
    raw = []
    for m in METHOD_ORDER:
        row = [m]
        for n in NETWORKS:
            g = hx[(hx.net == n) & (hx.method_label == m)].set_index("has_x")
            acc = ld[(ld.net == n) & (ld.method_label == m)].acc_base.iloc[0]
            cell = fmt.frac(g.loc["has-X", "acc_base"])
            if m == "VAN":
                cell += f" ({fmt.frac(g.loc['has-X', 'unanswered_base'], sign=False)})"
            row += [cell, fmt.frac(g.loc["no-X", "acc_base"]), fmt.frac(acc)]
            raw.append({"method": m, "network": n, "has_x": g.loc["has-X", "acc_base"],
                        "no_x": g.loc["no-X", "acc_base"], "acc": acc,
                        "has_x_unanswered": g.loc["has-X", "unanswered_base"]})
        rows.append(row)
    tables.append(Table(
        "tab:x-cell", "Accuracy within has-X / no-X and overall, per network (seen sites)",
        "L loso_delta.by_has_x.csv (acc_base), loso_delta.csv (acc_base); P x_cell_rtt.csv (tgs_pct)",
        cols, rows, raw, note="VAN's has-X cells carry its unanswered share within has-X in parentheses.",
    ))

    # -- RTT of has-X / no-X ---------------------------------------------------------
    split = json.loads((p_dir / "x_cell_rtt.manifest.json").read_text())["rtt_norm_ms"]["split_norm"]
    tables.append(from_frame(
        "fig:boxplot-rtt-of-x-cell-by-asn", "Normalized minimum RTT per network and side",
        "P x_cell_rtt.csv", xr,
        [("Network", lambda r: r.net), ("Side", lambda r: r.side), ("TGs", lambda r: fmt.pct(r.tgs_pct)),
         ("Sites", lambda r: fmt.pct(r.sites_pct)),
         *[(f"p{p}", lambda r, p=p: fmt.dp(r[f"p{p}_norm"], 3)) for p in (5, 25, 50, 75, 95)],
         (f"≤ split ({fmt.dp(split, 3)})", lambda r: fmt.pct(r.le_split_pct))],
    ))
    ex = pd.read_csv(p_dir / "x_cell_rtt.exceptions.csv")
    ex["net"] = [network(r) for r in ex.run_id]
    tables.append(from_frame(
        "fig:boxplot-rtt-of-x-cell-by-asn (exceptions)", "Sites with TGs on the wrong side of the split",
        "P x_cell_rtt.exceptions.csv", ex,
        [("Network", lambda r: r.net), ("Side", lambda r: r.side), ("Site", lambda r: str(r.site)),
         ("Cluster", lambda r: f"C{r.clusters}"), ("TGs across / site", lambda r: f"{r.n_tgs_across}/{r.n_tgs_site}"),
         ("Normalized RTT", lambda r: f"{fmt.dp(r.min_rtt_norm, 3)}{fmt.DASH}{fmt.dp(r.max_rtt_norm, 3)}")],
        note="Site = index among the network's sites (no coordinate). Counts for inspection only.",
    ))

    # -- S-P by side ----------------------------------------------------------------
    rep = json.loads((p_dir / "sp_pni_cells.report.json").read_text())
    sides = pd.DataFrame([{"side": s, **{k: v for k, v in rep["sides"][s].items() if not isinstance(v, dict)},
                           "misses": "; ".join(f"{network(r)} {n}" for r, n in rep["sides"][s]["misses_by_run"].items()),
                           "tie_correct": rep["sides"][s]["n_near_tie_median"]["correct"],
                           "tie_wrong": rep["sides"][s]["n_near_tie_median"]["wrong"]} for s in SIDES])
    tables.append(from_frame(
        "S-P by side", "Shortest Ping on has-X / no-X TGs",
        "P sp_pni_cells.report.json (sides)", sides,
        [("Side", lambda r: r.side), ("Correct", lambda r: fmt.pct(r.correct_pct)),
         ("Wrong in an interconnect cell", lambda r: fmt.pct(r.wrong_in_interconnect_cell_pct)),
         ("Misses by network (TGs)", lambda r: r.misses),
         ("VPs within 1 ms, median: correct / wrong", lambda r: f"{fmt.dp(r.tie_correct, 0)} / {fmt.dp(r.tie_wrong, 0)}")],
        note=f"All TGs: S-P's prediction lands in an interconnect cell for "
             f"{fmt.pct1(rep['shares']['all']['pred_in_interconnect_cell_pct'])}.",
    ))

    # -- correct sets -----------------------------------------------------------------
    st = _ordered(pd.read_csv(c_dir / "correct_upset.pooled.cell.sites.csv"))
    tables.append(from_frame(
        "tab:cell-correct-sites", "Share of sites at which a method is correct on ≥1, >50%, all TGs",
        "C correct_upset.pooled.cell.sites.csv", st,
        [("Method", lambda r: r.method_label), ("≥ 1", lambda r: fmt.pct(100 * r.n_sites_any / r.n_sites)),
         ("> 50%", lambda r: fmt.pct(100 * r.n_sites_majority / r.n_sites)),
         ("All", lambda r: fmt.pct(100 * r.n_sites_all / r.n_sites))],
    ))
    n_sites = int(st.n_sites.iloc[0])
    it = pd.read_csv(c_dir / "correct_upset.pooled.cell.intersections.csv")
    tables.append(from_frame(
        "fig:upset-cell-correct", "Exact sets of correct methods", "C correct_upset.pooled.cell.intersections.csv", it,
        [("Correct methods", lambda r: r.members), ("TGs", lambda r: fmt.frac1(r.share)),
         ("Sites", lambda r: fmt.pct(100 * r.n_sites / n_sites))],
        note="One decimal: several combinations hold under 1% of TGs.",
    ))
    ne = pd.read_csv(c_dir / "correct_upset.pooled.cell.nesting.csv")
    tables.append(from_frame(
        "fig:upset-cell-correct (cohorts)", "Cohorts against S-P, single-method cohorts, all-or-nothing sites",
        "C correct_upset.pooled.cell.nesting.csv", ne,
        [("Kind", lambda r: r.kind), ("Method", lambda r: r.method_label),
         ("TGs", lambda r: fmt.frac1(r.share_of_tgs)),
         ("Of reference", lambda r: fmt.frac1(r.share_of_reference)),
         ("Sites", lambda r: fmt.frac(r.share_of_sites)), ("has-X", lambda r: fmt.frac(r.has_x_share))],
        note="misses_baseline: S-P correct, method wrong (reference = S-P's correct TGs). "
             "adds_over_baseline: method correct, S-P wrong. only: the method (or family) is the whole "
             "correct set. all_or_nothing_sites: sites where the method is right on all or none.",
    ))

    # -- bounded -----------------------------------------------------------------------
    bd = _ordered(pd.read_csv(c_dir / "outcome_bars.pooled.healpix-128.csv")
                  .assign(method_label=lambda f: f.method.map(method_label)))
    bd["wrong"] = bd.share_n_cell_wrong
    tables.append(from_frame(
        "fig:stackbar-bounded-pooled", "Predictions by pixel distance; bounded accuracy (≤ 2 pixels)",
        "C outcome_bars.pooled.healpix-128.csv", bd,
        [("Method", lambda r: r.method_label),
         ("Correct, own pixel", lambda r: fmt.frac(r.share_n_ring0_cell_correct)),
         ("1 px", lambda r: fmt.frac(r.share_n_ring1_cell_correct)),
         ("2 px", lambda r: fmt.frac(r.share_n_ring2_cell_correct)),
         ("> 2 px", lambda r: fmt.frac(r.share_n_beyond_cell_correct)),
         ("Wrong", lambda r: fmt.frac(r.wrong)), ("Unanswered", lambda r: fmt.frac(r.share_n_cell_unanswered)),
         ("Bounded acc.", lambda r: fmt.frac(r.accuracy_bounded)),
         ("Of correct: ≤ 1 px", lambda r: fmt.frac(r.of_correct_le_ring1)),
         ("Of correct: > 2 px", lambda r: fmt.frac(r.of_correct_beyond))],
    ))
    bds = pd.read_csv(pa_dir / "pareto.datasets.csv")
    bds = _ordered(bds[(bds.regime == "seen") & (bds.metric == "bounded")])
    wide = bds.assign(net=[network(d) for d in bds.dataset]).pivot_table(
        index="method_label", columns="net", values="acc", sort=False).reset_index()
    wide = _ordered(wide)
    tables.append(from_frame(
        "fig:stackbar-bounded-pooled (per network)", "Bounded accuracy per network (seen sites)",
        "PA pareto.datasets.csv (seen, bounded)", wide,
        [("Method", lambda r: r.method_label), *[(n, lambda r, n=n: fmt.frac(r[n])) for n in NETWORKS[1:]]],
    ))

    # -- SPO vs OCT-H -------------------------------------------------------------------
    xm = json.loads((x_dir / "exclusive_error_cdf.spo_vs_octh.manifest.json").read_text())
    who = {"only_a": f"{method_label(xm['method_a'])} correct, {method_label(xm['method_b'])} not",
           "only_b": f"{method_label(xm['method_b'])} correct, {method_label(xm['method_a'])} not"}
    xs = pd.DataFrame(xm["stats"])
    tables.append(from_frame(
        "fig:spo-exclusive-pixel", "Pixel distance on the TGs only one method classifies correctly",
        "XE exclusive_error_cdf.spo_vs_octh.manifest.json (stats)", xs,
        [("Cohort", lambda r: who.get(r.cohort, r.cohort)), ("TGs", lambda r: fmt.frac(r.share_of_all)),
         ("p50 px", lambda r: fmt.dp(r.offset_p50, 0)), ("p90 px", lambda r: fmt.dp(r.offset_p90, 0)),
         ("Max px", lambda r: str(r.offset_max)),
         *[(f"> {k} px", lambda r, k=k: fmt.frac(r[f"share_of_cohort_beyond_{k}"])) for k in (2, 5, 10, 20)]],
        note="> k px: share of the cohort.",
    ))
    sm = json.loads((s_dir / "paired_std_grid_offset.spo_vs_octh.manifest.json").read_text())
    ss = _ordered(pd.DataFrame(sm["stats"]))
    tables.append(from_frame(
        "fig:spo-replica-spread", "Std. dev. of pixel distance across a site's replicas, per site",
        "ST paired_std_grid_offset.spo_vs_octh.manifest.json (stats)", ss,
        [("Method", lambda r: r.method_label),
         ("Sites with zero spread", lambda r: fmt.pct(100 * r.n_units_zero_spread / r.n_units)),
         *[(f"p{p}", lambda r, p=p: fmt.dp(r[f"spread_p{p}"], 1)) for p in (5, 25, 50, 75, 95)],
         ("Max", lambda r: fmt.dp(r.spread_max, 1))],
    ))
    pm = json.loads((pe_dir / "peripherality.SPO-vs-OCT-H.manifest.json").read_text())
    name = {"a_wins": f"{method_label(pm['method_a'])} more accurate",
            "b_wins": f"{method_label(pm['method_b'])} more accurate", "tied": "tied"}
    ps = pd.DataFrame(pm["stats"])
    tables.append(from_frame(
        "fig:spo-peripherality", "Distance from a site's seed to the seed centroid (normalized over sites)",
        "PE peripherality.SPO-vs-OCT-H.manifest.json (stats)", ps,
        [("Sites", lambda r: name.get(r.category, r.category)),
         ("Share of sites", lambda r: fmt.pct(100 * r.n_sites / ps.n_sites.sum())),
         *[(k, lambda r, k=k: fmt.dp(r[f"{k}_norm"], 2)) for k in ("min", "p25", "p50", "p75", "max")]],
    ))
    per = []
    for run in ctx.runs("seen"):
        meta = json.loads((ctx.run_dir(run, "answer-space", "healpix-128") / "meta.json").read_text())
        per.append({"net": network(run), "share": meta["peripheral_seed_share"]})
    tables.append(from_frame(
        "peripheral cells", "Share of each network's cells that are peripheral (unbounded)",
        "<run>/answer-space/healpix-128/meta.json (peripheral_seed_share)", pd.DataFrame(per),
        [("Network", lambda r: r.net), ("Peripheral cells", lambda r: fmt.frac(r.share))],
    ))

    # -- seen vs unseen ------------------------------------------------------------------
    pooled = _ordered(ld[ld.net == "All"]).copy()
    nox = hx[(hx.net == "All") & (hx.has_x == "no-X")].set_index("method_label")
    pooled["nox_base"] = pooled.method_label.map(nox.acc_base)
    pooled["nox_loso"] = pooled.method_label.map(nox.acc_loso)
    tables.append(from_frame(
        "tab:seen-unseen", "Seen vs unseen sites, pooled", "L loso_delta.csv, loso_delta.by_has_x.csv (pooled)", pooled,
        [("Method", lambda r: r.method_label), ("Acc seen", lambda r: fmt.frac(r.acc_base)),
         ("Acc unseen", lambda r: fmt.frac(r.acc_loso)), ("Δ", lambda r: fmt.pp(100 * r.d_acc, signed=True, unit=False)),
         ("Sites ↓", lambda r: fmt.pct(r.sites_drop_pct)), ("Sites ↑", lambda r: fmt.pct(r.sites_rise_pct)),
         ("No-X seen", lambda r: fmt.frac(r.nox_base)), ("No-X unseen", lambda r: fmt.frac(r.nox_loso)),
         ("Median seen", lambda r: fmt.e3(r.p50_base_norm_e3)), ("Median unseen", lambda r: fmt.e3(r.p50_loso_norm_e3)),
         ("Unans. seen", lambda r: fmt.frac(r.unanswered_base)), ("Unans. unseen", lambda r: fmt.frac(r.unanswered_loso)),
         ("Correct→wrong", lambda r: fmt.pct(100 * r.n_correct_to_wrong / r.n_tgs)),
         ("Wrong→correct", lambda r: fmt.pct(100 * r.n_wrong_to_correct / r.n_tgs))],
        note="Δ in pp. Median error ×10⁻³, unanswered ranked last.",
    ))
    tables.append(from_frame(
        "tab:seen-unseen (pixels)", "Own pixel and bounded accuracy, seen vs unseen, pooled",
        "L loso_delta.csv (pooled)", pooled,
        [("Method", lambda r: r.method_label), ("Own pixel seen", lambda r: fmt.frac(r.own_pixel_base)),
         ("Own pixel unseen", lambda r: fmt.frac(r.own_pixel_loso)),
         ("Bounded seen", lambda r: fmt.frac(r.acc_bounded_base)), ("Bounded unseen", lambda r: fmt.frac(r.acc_bounded_loso))],
    ))
    per_net = _ordered(ld[ld.net != "All"]).copy()
    tables.append(from_frame(
        "tab:seen-unseen (per network)", "Seen vs unseen sites per network",
        "L loso_delta.csv (per pair)", per_net.sort_values(["net"], kind="stable"),
        [("Network", lambda r: r.net), ("Method", lambda r: r.method_label),
         ("Acc seen", lambda r: fmt.frac(r.acc_base)), ("Acc unseen", lambda r: fmt.frac(r.acc_loso)),
         ("Own pixel seen", lambda r: fmt.frac(r.own_pixel_base)), ("Own pixel unseen", lambda r: fmt.frac(r.own_pixel_loso)),
         ("Median seen", lambda r: fmt.e3(r.p50_base_norm_e3)), ("Median unseen", lambda r: fmt.e3(r.p50_loso_norm_e3))],
    ))
    nox_tab = hx[hx.net == "All"].copy()
    tables.append(from_frame(
        "tab:seen-unseen (has-X / no-X)", "Seen vs unseen accuracy within has-X and no-X, pooled",
        "L loso_delta.by_has_x.csv (pooled)", _ordered(nox_tab).sort_values("has_x", kind="stable"),
        [("Side", lambda r: r.has_x), ("Method", lambda r: r.method_label),
         ("Acc seen", lambda r: fmt.frac(r.acc_base)), ("Acc unseen", lambda r: fmt.frac(r.acc_loso)),
         ("Unans. seen", lambda r: fmt.frac(r.unanswered_base)), ("Unans. unseen", lambda r: fmt.frac(r.unanswered_loso))],
    ))

    sources = {"C": ctx.source(c_dir), "P": ctx.source(p_dir), "L": ctx.source(l_dir),
               "PA": ctx.source(pa_dir), "XE": ctx.source(x_dir), "ST": ctx.source(s_dir),
               "PE": ctx.source(pe_dir)}
    return tables, sources
