"""Per-TG cause evidence for S-P's wrong cells, pooled over the three meshes.

Run from the cbg-framework root with the .venv python.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import pni_gap as P
from scripts.analysis.v5.modules import sp_pni_cells as SPC
from scripts.analysis.v5.modules.figure_sp_interconnect import load_edges
from scripts.analysis.v5.modules.geodesy import haversine_km
from scripts.analysis.v5.modules.labels import declared_pni_csv
from scripts.analysis.v5.modules.map_answer_space import load_rung
from scripts.analysis.v5.cli import resolve_run

OUT = Path(sys.argv[1])
from scripts.analysis.v5.modules.paths import DEFAULT_OUTPUTS_ROOT as ROOT
from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules.labels import load_group
cross.use_group(load_group("pro-paper"))
RUNS = ["pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh"]
FLOOR = 100.0  # km per ms of RTT (round trip at 2/3 c)
TIE = 1.0      # ms

rows = []
for rid in RUNS:
    run = resolve_run(rid, ROOT)
    pni_csv = declared_pni_csv(rid, ROOT)
    out_dir = P.pooled_output_dir(RUNS)
    clusters, meta = P.read_clusters(out_dir, run_ids=RUNS)
    rec = {r["run_id"]: r for r in meta.get("runs", [])}
    csv = P.checked_source_csv(run, pni_csv, rec.get(rid, {}), None)
    pnis = P.load_pnis(pni_csv)
    seeds = load_rung(run, SPC.SOURCE_NSIDE).seeds
    edges = load_edges(csv)
    sp = SPC.load_sp_frame(run)
    tgs = SPC.tg_frame(sp, seeds, pnis, run_id=rid)

    # each VP's cell and its nearest interconnect
    vps = edges.drop_duplicates("vp_id")[["vp_id", "vp_lat", "vp_lon"]]
    vseed, _ = SPC.nearest_seed(vps.vp_lat, vps.vp_lon, seeds)
    vp_seed = dict(zip(vps.vp_id, vseed))
    e = edges.merge(tgs[["tg_id", "tg_seed_id", "x_seed_id"]], on="tg_id")
    e["vp_seed"] = e.vp_id.map(vp_seed)
    lo = e.groupby("tg_id").rtt_ms.transform("min")
    e["tied"] = e.rtt_ms <= lo + TIE
    g = e.groupby("tg_id")
    feat = pd.DataFrame({
        "min_rtt_ms": g.rtt_ms.min(),
        "n_vps": g.size(),
        "n_vps_own_cell": g.apply(lambda x: (x.vp_seed == x.tg_seed_id).sum(), include_groups=False),
        "own_cell_best_rtt_ms": g.apply(lambda x: x.rtt_ms[x.vp_seed == x.tg_seed_id].min()
                                        if (x.vp_seed == x.tg_seed_id).any() else np.nan,
                                        include_groups=False),
        "n_tied": g.tied.sum(),
        "tied_cells": g.apply(lambda x: x.vp_seed[x.tied].nunique(), include_groups=False),
        "tie_has_own_cell_vp": g.apply(lambda x: (x.tied & (x.vp_seed == x.tg_seed_id)).any(),
                                       include_groups=False),
        "rho": g.apply(lambda x: pd.Series(x.d_km.values).corr(pd.Series(x.rtt_ms.values),
                                                                method="spearman"),
                       include_groups=False),
    }).reset_index()

    # paths: S-P VP -> X (TG's nearest) -> TG, and S-P VP -> K (its own nearest) -> TG
    tx = haversine_km(tgs.tg_lat.values[:, None], tgs.tg_lon.values[:, None],
                      pnis.pni_lat.values[None, :], pnis.pni_lon.values[None, :])
    vx = haversine_km(tgs.pred_lat.values[:, None], tgs.pred_lon.values[:, None],
                      pnis.pni_lat.values[None, :], pnis.pni_lon.values[None, :])
    r = np.arange(len(tgs))
    kx, kk = tx.argmin(1), vx.argmin(1)
    tgs["d_via_x_km"] = vx[r, kx] + tx[r, kx]
    tgs["d_via_k_km"] = vx[r, kk] + tx[r, kk]
    # best interconnect path that fits the RTT: min over all interconnects of VP->I->TG
    tgs["d_via_best_km"] = (vx + tx).min(1)
    tgs = tgs.merge(feat, on="tg_id", how="left", validate="1:1")
    rows.append(tgs)

t = pd.concat(rows, ignore_index=True)
t["has_x"] = t.tg_cell_holds_x
# 1) X_in != nearest X: RTT is impossible through X (below the 2/3c floor)
t["below_floor_via_x"] = t.min_rtt_ms < t.d_via_x_km / FLOOR
# stretch of the S-P RTT over its own (via K) path, and over the via-X path
t["stretch_via_k"] = t.min_rtt_ms / (t.d_via_k_km / FLOOR)
t["stretch_via_x"] = t.min_rtt_ms / (t.d_via_x_km / FLOOR)
t["excess_via_k_ms"] = t.min_rtt_ms - t.d_via_k_km / FLOOR
t.to_csv(OUT, index=False)
print("wrote", OUT, len(t))
