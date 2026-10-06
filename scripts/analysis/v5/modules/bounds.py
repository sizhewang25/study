"""The normalization bounds, computed and checked -- `report-bounds`.

Every distance axis is drawn as d / D and every RTT axis as r / R_max, with D
and R_max fixed over the pooled datasets so a normalized value means the same
in every figure. The values are *declared* (`analysis.common.dist_norm_km` /
`rtt_norm_ms`, in a run config or its group file) because the figures must not
compute them per call. This command computes them from the data and checks the
declarations, so a declared bound is never an unverified number.

* **D** -- the largest great-circle distance between any two distinct VP or
  site coordinates of the runs' canonical CSVs, pooled (`footprint.footprint_span`).
* **R_max** -- the largest per-(VP, TG) minimum RTT, pooled: the RTT the
  figures draw (`figure_sp_interconnect.load_edges` takes the minimum per pair).

Pass every run the figures pool. A LOSO twin reads its mesh's CSV, so adding
it changes nothing. Each run's resolved declaration (`labels`, which merges
its config with its groups) must equal the computed value at the declared
precision (`TOL`), and the runs must agree with each other. Any mismatch is
listed in the report and fails the command.

Command: `report-bounds`. Writes `_cross/bounds/<n>-runs-<hash>/bounds.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import edges as EDGES
from scripts.analysis.v5.modules import footprint
from scripts.analysis.v5.modules.figure_sp_interconnect import load_edges
from scripts.analysis.v5.modules.labels import (
    DIST_NORM_PATH,
    RTT_NORM_PATH,
    dataset_label,
    declared_dist_norm_km,
    declared_rtt_norm_ms,
)
from scripts.analysis.v5.modules.paths import RunPaths

KIND = "bounds"
REPORT_NAME = "bounds.json"

#: Declared bounds are written to 3 dp; a computed value within half a unit of
#: the last place matches.
TOL = 0.0005


def max_rtt(runs: list[RunPaths]) -> dict:
    """The largest per-pair minimum RTT over the runs' edge CSVs, and where it is."""
    per_run, seen = {}, {}
    for run in runs:
        csv = EDGES.resolve_source_csv(run)
        if csv not in seen:
            e = load_edges(csv)
            seen[csv] = {"max_ms": float(e["rtt_ms"].max()), "n_pairs": int(len(e))}
        per_run[run.run_id] = {**seen[csv], "source_csv": str(csv)}
    top = max(per_run, key=lambda r: per_run[r]["max_ms"])
    return {
        "max_ms": round(per_run[top]["max_ms"], 3),
        "held_by": top,
        "held_by_label": dataset_label(top),
        "per_run": per_run,
        "definition": "largest per-(VP, TG) minimum RTT over the runs' canonical CSVs, pooled",
    }


def _check(name: str, computed: float, declared: dict[str, tuple[float, float] | None]) -> list[str]:
    problems = []
    for run_id, bounds in declared.items():
        if bounds is None:
            problems.append(f"{run_id}: declares no analysis.common.{name}")
        elif bounds[0] != 0.0:
            problems.append(f"{run_id}: analysis.common.{name}.min is {bounds[0]}, expected 0")
        elif abs(bounds[1] - computed) > TOL:
            problems.append(f"{run_id}: analysis.common.{name}.max is {bounds[1]}, computed {computed}")
    return problems


def build(runs: list[RunPaths], *, analysis_root: Path | None = None,
          outputs_root: Path | None = None) -> tuple[Path, list[str]]:
    """Compute both bounds, check every run's declaration; `(report path, problems)`."""
    if not runs:
        raise ValueError("pass at least one --run-id")
    kw = {} if outputs_root is None else {"root": outputs_root}
    span = footprint.footprint_span(runs)
    rtt = max_rtt(runs)
    dist_decl = {r.run_id: declared_dist_norm_km(r.run_id, **kw) for r in runs}
    rtt_decl = {r.run_id: declared_rtt_norm_ms(r.run_id, **kw) for r in runs}
    problems = (_check(DIST_NORM_PATH[-1], span["span_km"], dist_decl)
                + _check(RTT_NORM_PATH[-1], rtt["max_ms"], rtt_decl))
    out_dir = cross.cross_dir([r.run_id for r in runs], analysis_root=analysis_root, kind=KIND)
    path = out_dir / REPORT_NAME
    path.write_text(json.dumps({
        "kind": KIND,
        "run_ids": [r.run_id for r in runs],
        "computed": {"dist_norm_km_max": span["span_km"], "rtt_norm_ms_max": rtt["max_ms"]},
        "declared": {
            "dist_norm_km": {k: list(v) if v else None for k, v in dist_decl.items()},
            "rtt_norm_ms": {k: list(v) if v else None for k, v in rtt_decl.items()},
        },
        "tolerance": TOL,
        "ok": not problems,
        "problems": problems,
        "footprint": span,
        "rtt": rtt,
        "note": "both maxima are confidential in the paper; the figures print only normalized values",
    }, indent=2) + "\n")
    return path, problems
