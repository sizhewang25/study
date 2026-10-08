"""§6 Evaluation on overhead: the data behind the cost boxes and the batch budget.

* `fig:boxplot-speed-memory` -- runtime per TG (phases summed) and peak heap
  per TG (max over phases), per method, plus each phase's median runtime
  (`plot-cost-box`).
* **batch budget** -- the mean runtime extrapolated to `extrapolate_tgs` TGs on
  `cores` workers, per network and pooled, with the memory those workers need
  (`plot-cost-box --extrapolate-tgs --cores`).
"""

from __future__ import annotations

import pandas as pd

from scripts.analysis.v5.paper import fmt
from scripts.analysis.v5.paper.core import Context, Table, from_frame, method_rank, network

SECTION = "6"
TITLE = "Evaluation on Overhead"


def _label(r) -> str:
    return r.method_label if not (isinstance(r.overlay_on, str) and r.overlay_on) else f"{r.method_label} (overlay)"


def build(ctx: Context) -> tuple[list[Table], dict[str, str]]:
    k_dir = ctx.seen_dir("cost")
    k = pd.read_csv(k_dir / "cost_box.pooled.heap.csv")
    k = k.assign(_o=method_rank(k.method_label)).sort_values(["_o", "method_label"], kind="stable")
    pipe = k[k.stage == "pipeline"]
    rt, heap = pipe[pipe.channel == "runtime"], pipe[pipe.channel == "memory_heap"].set_index("method")
    rt = rt.assign(heap_p50=rt.method.map(heap.p50), heap_p95=rt.method.map(heap.p95), heap_max=rt.method.map(heap["max"]))
    tables = [from_frame(
        "fig:boxplot-speed-memory", "Runtime and peak heap per TG, pooled (seen sites)",
        "K cost_box.pooled.heap.csv (stage = pipeline)", rt,
        [("Method", _label), *[(f"Runtime p{p}", lambda r, p=p: fmt.runtime(r[f"p{p}"])) for p in (5, 25, 50, 75, 95)],
         ("Runtime mean", lambda r: fmt.runtime(r["mean"])), ("Runtime max", lambda r: fmt.runtime(r["max"])),
         ("Heap p50 MB", lambda r: fmt.dp(r.heap_p50, 2)), ("Heap p95 MB", lambda r: fmt.dp(r.heap_p95, 2)),
         ("Heap max MB", lambda r: fmt.dp(r.heap_max, 2))],
        note="Runtime sums the three phases per TG; heap is the largest of the three. Every evaluated TG.",
    )]

    # pivot_table drops index rows holding NaN, and a slot method's overlay_on is empty
    stages = k[(k.channel == "runtime") & (k.stage != "pipeline")].assign(overlay_on=lambda f: f.overlay_on.fillna(""))
    wide = stages.pivot_table(index=["_o", "method_label", "overlay_on"], columns="stage", values="p50",
                              sort=False).reset_index().sort_values(["_o", "method_label"])
    heap_st = k[(k.channel == "memory_heap") & (k.stage != "pipeline")].pivot_table(
        index="method_label", columns="stage", values="p50")
    for s in ("ltd", "mtl", "ctr"):
        wide[f"heap_{s}"] = wide.method_label.map(heap_st[s])
    tables.append(from_frame(
        "fig:boxplot-speed-memory (phases)", "Median runtime and heap of each phase",
        "K cost_box.pooled.heap.csv (stage = ltd, mtl, ctr)", wide,
        [("Method", _label), ("LTD", lambda r: fmt.runtime(r.ltd)), ("MTL", lambda r: fmt.runtime(r.mtl)),
         ("EST", lambda r: fmt.runtime(r.ctr)),
         ("Heap LTD MB", lambda r: fmt.dp(r.heap_ltd, 2)), ("Heap MTL MB", lambda r: fmt.dp(r.heap_mtl, 2)),
         ("Heap EST MB", lambda r: fmt.dp(r.heap_ctr, 2))],
        note="EST is the paper's name for the CTR phase.",
    ))

    x = pd.read_csv(k_dir / "cost_box.pooled.heap.extrapolation.csv")
    x["net"] = [network(s) for s in x.scope]
    x = x.assign(_o=method_rank(x.method_label)).sort_values(["_o", "method_label", "net"], kind="stable")
    n, cores = int(x.extrapolate_tgs.iloc[0]), int(x.cores.iloc[0])
    tables.append(from_frame(
        "batch budget", f"{n:,} TGs on {cores} cores, from the mean runtime per TG",
        "K cost_box.pooled.heap.extrapolation.csv", x,
        [("Method", _label), ("Network", lambda r: r.net), ("Mean runtime", lambda r: fmt.runtime(r.runtime_mean_ms)),
         ("Core-hours", lambda r: fmt.dp(r.core_hours, 0)), ("Wall clock", lambda r: _wall(r.wall_hours)),
         ("MTL share of mean", lambda r: fmt.frac(r.mtl_share_of_mean)),
         (f"{cores} × peak heap MB", lambda r: fmt.dp(r.peak_per_tg_x_cores_mb, 0)),
         (f"{cores} × process RSS MB", lambda r: fmt.dp(r.process_rss_x_cores_mb, 0))],
        note="An extrapolation: mean × N / cores, contention-free. Process RSS is the largest per-fold "
             "`run_peak_rss_bytes`, what a machine must hold per worker.",
    ))
    return tables, {"K": ctx.source(k_dir)}


def _wall(hours: float) -> str:
    if hours < 1:
        return f"{fmt.dp(hours * 60, 0)} min"
    if hours < 48:
        return f"{fmt.dp(hours, 1)} h"
    return f"{fmt.dp(hours / 24, 1)} days"
