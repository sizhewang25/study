"""§7 State of the art under operator criteria: the data behind Table sota and the Pareto panels.

* `tab:sota` -- pooled accuracy (worst-best network) under the four criteria,
  whether the method beats S-P in every network, and the median runtime per
  regime (`plot-pareto`).
* `fig:pareto` (per network) -- every network's accuracy under each criterion.
* `fig:pareto` (frontier) -- each criterion's frontier, cheapest first, and
  every method's lead over S-P and best-worst spread, in pp.
"""

from __future__ import annotations

import pandas as pd

from scripts.analysis.v5.modules.status import SHORTEST_PING
from scripts.analysis.v5.paper import fmt
from scripts.analysis.v5.paper.core import METHOD_ORDER, Context, Table, from_frame, method_rank, network

SECTION = "7"
TITLE = "State of the Art under Operator Criteria"

CRITERIA = (("seen", "cell"), ("seen", "bounded"), ("unseen", "cell"), ("unseen", "bounded"))
NAME = {"cell": "unbounded", "bounded": "bounded"}


def build(ctx: Context) -> tuple[list[Table], dict[str, str]]:
    pa_dir = ctx.all_dir("pareto")
    pa = pd.read_csv(pa_dir / "pareto.csv")
    pairs = pd.read_csv(pa_dir / "pareto.pairs.csv")
    ds = pd.read_csv(pa_dir / "pareto.datasets.csv")
    idx = pa.set_index(["regime", "metric", "method_label"])

    def beats_sp(regime, metric, method) -> bool:
        sp_vs = pairs[(pairs.regime == regime) & (pairs.metric == metric)
                      & (((pairs.a == SHORTEST_PING) & (pairs.b == method)) | ((pairs.a == method) & (pairs.b == SHORTEST_PING)))]
        if sp_vs.empty:
            return False
        r = sp_vs.iloc[0]
        wins = r.n_a_wins if r.a == method else r.n_b_wins
        return int(wins) == int(r.n_datasets)

    label_to_id = dict(zip(pa.method_label, pa.method))
    cols = ["Method", *[f"{g} {NAME[m]}" for g, m in CRITERIA], "Runtime seen", "Runtime unseen"]
    rows, raw = [], []
    for m in METHOD_ORDER:
        row = [m]
        rec = {"method": m}
        for g, metric in CRITERIA:
            r = idx.loc[(g, metric, m)]
            dag = "†" if m != "S-P" and beats_sp(g, metric, label_to_id[m]) else ""
            row.append(f"{fmt.frac(r.acc, sign=False)} ({fmt.frac(r.acc_min, sign=False)}{fmt.DASH}"
                       f"{fmt.frac(r.acc_max, sign=False)}){dag}")
            rec.update({f"{g}_{metric}": r.acc, f"{g}_{metric}_min": r.acc_min,
                        f"{g}_{metric}_max": r.acc_max, f"{g}_{metric}_beats_sp_everywhere": bool(dag)})
        for g in ("seen", "unseen"):
            rt = idx.loc[(g, "cell", m), "runtime_p50_ms"]
            row.append(fmt.runtime(rt) if fmt.defined(rt) else "--")
            rec[f"{g}_runtime_p50_ms"] = rt
        rows.append(row)
        raw.append(rec)
    tables = [Table(
        "tab:sota", "Accuracy (%) under the four criteria, pooled (worst–best network); median runtime per TG",
        "PA pareto.csv, pareto.pairs.csv", cols, rows, raw,
        note="† = more accurate than S-P in every network (pareto.pairs.csv `consistent`, S-P losing all).",
    )]

    ds = ds.assign(net=[network(d) for d in ds.dataset])
    wide = ds.pivot_table(index="method_label", columns=["regime", "metric", "net"], values="acc", sort=False)
    nets = sorted(ds.net.unique())
    per = []
    for m in METHOD_ORDER:
        per.append({"method_label": m, **{f"{g} {NAME[mt]} {n}": wide.loc[m, (g, mt, n)]
                                          for g, mt in CRITERIA for n in nets}})
    per = pd.DataFrame(per)
    tables.append(from_frame(
        "fig:pareto (per network)", "Accuracy per network under each criterion", "PA pareto.datasets.csv", per,
        [("Method", lambda r: r.method_label),
         *[(f"{g[0].upper()}{NAME[mt][0].upper()} {n}", lambda r, c=f"{g} {NAME[mt]} {n}": fmt.frac(r[c], sign=False))
           for g, mt in CRITERIA for n in nets]],
        note="Column = regime (S seen / U unseen) + criterion (U unbounded / B bounded) + network.",
    ))

    pa = pa.assign(_o=method_rank(pa.method_label), crit=[f"{g} {NAME[m]}" for g, m in zip(pa.regime, pa.metric)])
    pa["_c"] = [CRITERIA.index((g, m)) for g, m in zip(pa.regime, pa.metric)]
    pa = pa.sort_values(["_c", "_o"])
    tables.append(from_frame(
        "fig:pareto (frontier)", "Frontier membership, lead over S-P and spread across networks",
        "PA pareto.csv", pa,
        [("Criterion", lambda r: r.crit), ("Method", lambda r: r.method_label),
         ("Accuracy", lambda r: fmt.frac(r.acc)), ("Lead over S-P", lambda r: fmt.pp(r.lead_vs_sp_pp, signed=True)),
         ("Worst–best spread", lambda r: fmt.pp(r.spread_pp)),
         ("On frontier", lambda r: "yes" if r.on_frontier else ""),
         ("Runtime p50", lambda r: fmt.runtime(r.runtime_p50_ms) if fmt.defined(r.runtime_p50_ms) else "--")],
    ))
    return tables, {"PA": ctx.source(pa_dir)}
