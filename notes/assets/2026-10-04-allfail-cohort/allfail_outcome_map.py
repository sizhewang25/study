"""Outcome maps (v5 figure_outcome_map.render) restricted to the all-fail cohort:
TGs where none of S-P, SOI, VAN, OCT-H, SPO is cell-correct."""
import sys
from dataclasses import replace
from pathlib import Path
import pandas as pd
from scripts.analysis.v5.cli import resolve_run, DEFAULT_OUTPUTS_ROOT
from scripts.analysis.v5.modules import figure_outcome_map as F

S = Path(sys.argv[1])
cohort = pd.read_csv(S / "gt_check.csv")
cohort = cohort[cohort.af]
keep = {r: set(g.tg_id) for r, g in cohort.groupby("run")}
runs = [resolve_run(r, DEFAULT_OUTPUTS_ROOT) for r in ["pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh"]]
data = F.load(runs)
frames = {(rid, m): f[f.tg_id.isin(keep[rid])].reset_index(drop=True) for (rid, m), f in data.frames.items()}
for (rid, m), f in frames.items():
    assert len(f) == len(keep[rid]), (rid, m, len(f))
data = replace(data, frames=frames)

tag = f"ALL-FAIL cohort only (no S-P/SOI/VAN/OCT-H/SPO cell-correct; n={len(cohort)} TGs, {cohort.site.nunique()} sites)"
F._SUBJECT = {k: f"{tag}\n{v}" for k, v in F._SUBJECT.items()}
out = S / "outcome-map-allfail"
out.mkdir(exist_ok=True)
for c in F.COHORTS:
    table, _ = F.build_csv(data, c, F.DEFAULT_EXTENT)
    table.to_csv(out / f"outcome_map.allfail.{c}.csv", index=False)
    png, counts = F.render(data, out / f"outcome_map.allfail.{c}.png", cohort=c)
    print(png, {(x["method_label"], x["dataset"]): x["n_cohort_tgs"] for x in counts})
