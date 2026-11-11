"""Slugs and claims of the traffic-filtered group (pro-tf): App. E, §8's TF run-in, §1.

Built on the TF machine only: the TF runs are not on the mesh machine. Every
comparison with the mesh is a derived slug (`derived.py`), evaluated at
render time from both groups' values files.
"""

from __future__ import annotations

import pandas as pd

from scripts.analysis.v5.modules.paths import resolve_run
from scripts.analysis.v5.paper_values.common import CBG, METHODS, SP, Ctx, KEY_TO_TERM, combo
from scripts.analysis.v5.paper_values.registry import TF, put

SE, S8, S1 = "App. E", "§8", "§1"
BARS = "classify/<pro-tf>/outcome_bars.unbounded.pooled.healpix-128.csv"
ECDF = "classify/<pro-tf>/error_cdf.pooled.norm.cut.csv"
PARETO = "pareto/<pro-tf>/pareto.csv"

for m in METHODS:
    put(f"tf.acc.{m}", "pct", BARS, lambda c, m=m: c.acc(m), group=TF, section=SE)
    put(f"tf.err.p50.{m}", "sig2", ECDF, lambda c, m=m: c.err(m, 50), group=TF, section=SE)
for m in CBG:
    put(f"tf.pareto.rt.{m}", "ms", f"{PARETO} (seen, runtime_p50_ms)", lambda c, m=m: c.pa("seen", m, "runtime_p50_ms"),
        group=TF, section=S8, note="read by claim.tf-runtime-drop-order only")
put("tf.unans.van", "pct", BARS, lambda c: c.unanswered("van"), group=TF, section=SE)
put("tf.acc.best-cbg", "pct", BARS, lambda c: max(c.acc(m) for m in CBG), group=TF, section=S1,
    note="the most accurate CBG method on the TF datasets")
put("tf.err.ratio.p50.s-p-over-oct-h", "x0", ECDF, lambda c: c.err("s-p", 50) / c.err("oct-h", 50),
    group=TF, section=SE)
put("tf.err.ratio.p50.spo-over-s-p", "x1", ECDF, lambda c: c.err("spo", 50) / c.err("s-p", 50),
    group=TF, section=SE)


def _tgs(c: Ctx) -> pd.DataFrame:
    """Per TG: VP count (n_obs) and has-X, over the TF seen runs (ported from paper_stats.tf)."""
    from scripts.analysis.v5.modules.loso_delta import attach_has_x

    def load():
        frames = []
        for rid in c.runs("seen"):
            run = resolve_run(rid, c.outputs_root)
            p = run.analysis_dir("classify", root=c.analysis_root) / "healpix-128" / f"{SP}_tgs.parquet"
            t, _ = attach_has_x(pd.read_parquet(p, columns=["tg_id", "n_obs"]), run, c.analysis_root)
            frames.append(t.assign(run_id=rid))
        tg = pd.concat(frames, ignore_index=True)
        if tg.has_x.isna().any():
            raise FileNotFoundError("a TF run lacks sp_pni_cells_tgs.csv: run report-sp-pni-cells on it")
        return tg
    return c.memo("tf-tgs", load)


SRC_TGS = "<run>/classify/healpix-128/shortest_ping_tgs.parquet (n_obs) + <run>/pni-gap/*/sp_pni_cells_tgs.csv"
put("tf.x.share.has-x", "pct", SRC_TGS, lambda c: float((_tgs(c).has_x == "has-X").mean() * 100), group=TF,
    section=SE)
put("tf.vps.one", "pct", SRC_TGS, lambda c: float((_tgs(c).n_obs == 1).mean() * 100), group=TF, section=SE)
put("tf.vps.two", "pct", SRC_TGS, lambda c: float((_tgs(c).n_obs == 2).mean() * 100), group=TF, section=SE)
put("tf.nox.max-vps", "int", SRC_TGS, lambda c: float(_tgs(c).loc[_tgs(c).has_x == "no-X", "n_obs"].max()),
    group=TF, section=SE)


def _ahead(c: Ctx, m: str) -> bool:
    p = c.csv("pareto", "pareto.pairs.csv", pool="all")
    row = p[(p.regime == "seen") & (p.metric == "cell") & (p.a == SP) & (p.b == combo(m))]
    return bool(row["consistent"].iloc[0]) and c.pa("seen", m, "acc") > c.pa("seen", "s-p", "acc")
