"""§3 Methodology: the datasets, folds, answer space and normalization bounds.

* **datasets** -- per network and pooled: VPs, TGs, sites, replicas per site,
  seeds and the share of sites sharing one, the share of TGs not measured by
  every VP, the filter-removal estimate, the K-fold and LOSO folds, and the
  largest error against D (`report-dataset`).
* **bounds** -- D and the largest per-pair RTT, computed and checked against
  the declarations (`report-bounds`).

Absolute counts are printed for inspection; the paper states shares only.
"""

from __future__ import annotations

import json

import pandas as pd

from scripts.analysis.v5.paper import fmt
from scripts.analysis.v5.paper.core import Context, Table, from_frame, network

SECTION = "3"
TITLE = "Methodology"


def _i(v) -> str:
    return "" if not fmt.defined(v) else str(int(v))


def build(ctx: Context) -> tuple[list[Table], dict[str, str]]:
    d_dir, b_dir = ctx.all_dir("dataset"), ctx.seen_dir("bounds")
    d = pd.read_csv(d_dir / "dataset.csv")
    d["net"] = [network(r) for r in d.run_id]
    tables = [from_frame(
        "datasets", "Mesh-ping datasets, per network and pooled", "D dataset.csv", d,
        [("Network", lambda r: r.net), ("VPs", lambda r: _i(r.n_vps)), ("TGs", lambda r: _i(r.n_tgs)),
         ("Sites", lambda r: _i(r.n_sites)),
         ("Replicas/site", lambda r: f"{_i(r.replicas_min)}{fmt.DASH}{_i(r.replicas_max)}"),
         ("Seeds", lambda r: _i(r.n_seeds)), ("Sites sharing a seed", lambda r: fmt.pct(r.sites_merged_pct)),
         ("TGs not measured by every VP", lambda r: fmt.pct(r.partial_mesh_pct)),
         ("Fewest VPs on a TG", lambda r: _i(r.vps_per_tg_min)),
         ("Removed by filter (est.)", lambda r: fmt.pct1(r.est_removed_pct))],
        note="Removed by filter is an estimate: it assumes every site started with the replica cap "
             "(the pre-filter CSVs do not survive).",
    ), from_frame(
        "folds", "Training/test folds and the largest error", "D dataset.csv", d,
        [("Network", lambda r: r.net), ("K-fold folds", lambda r: _i(r.kfold_n_folds)),
         ("Fold sizes", lambda r: f"{_i(r.kfold_fold_size_min)}{fmt.DASH}{_i(r.kfold_fold_size_max)}"),
         ("Sites spanning every fold", lambda r: fmt.pct(r.kfold_sites_all_folds_pct)),
         ("LOSO folds", lambda r: _i(r.get("loso_n_folds"))),
         ("Sites per LOSO fold (max)", lambda r: _i(r.get("loso_sites_per_fold_max"))),
         ("Max error / D", lambda r: fmt.dp(r.max_error_over_d, 2))],
    )]

    b = json.loads((b_dir / "bounds.json").read_text())
    rows = pd.DataFrame([
        {"bound": "D (dist_norm_km.max)", "computed": b["computed"]["dist_norm_km_max"], "unit": "km",
         "declared": _declared(b, "dist_norm_km")},
        {"bound": "R_max (rtt_norm_ms.max)", "computed": b["computed"]["rtt_norm_ms_max"], "unit": "ms",
         "declared": _declared(b, "rtt_norm_ms"), "held_by": network(b["rtt"]["held_by"])},
    ])
    tables.append(from_frame(
        "bounds", "Normalization bounds (confidential in the paper)", "B bounds.json", rows,
        [("Bound", lambda r: r.bound), ("Computed", lambda r: f"{fmt.dp(r.computed, 3)} {r.unit}"),
         ("Declared", lambda r: r.declared),
         ("Held by", lambda r: r.held_by if isinstance(r.get("held_by"), str) else "")],
        note=("All declarations match." if b["ok"] else "MISMATCH: " + "; ".join(b["problems"])),
    ))
    return tables, {"D": ctx.source(d_dir), "B": ctx.source(b_dir)}


def _declared(b: dict, key: str) -> str:
    vals = {tuple(v) if v else None for v in b["declared"][key].values()}
    if len(vals) != 1:
        return "differs across runs"
    (v,) = vals
    return "none" if v is None else fmt.dp(v[1], 3)
