"""Per TG, over every VP that measured it: which interconnect's path explains the RTTs."""
import sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from scripts.analysis.v5.modules import pni_gap as P, sp_pni_cells as SPC, cross
from scripts.analysis.v5.modules.labels import load_group, declared_pni_csv
from scripts.analysis.v5.modules.paths import DEFAULT_OUTPUTS_ROOT as ROOT
from scripts.analysis.v5.modules.figure_sp_interconnect import load_edges
from scripts.analysis.v5.modules.map_answer_space import load_rung
from scripts.analysis.v5.modules.geodesy import haversine_km
from scripts.analysis.v5.cli import resolve_run
cross.use_group(load_group("pro-paper"))
RUNS = ["pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh"]
FLOOR = 100.0
_, meta = P.read_clusters(P.pooled_output_dir(RUNS), run_ids=RUNS)
rec = {r["run_id"]: r for r in meta["runs"]}
out = []
for rid in RUNS:
    run = resolve_run(rid, ROOT); pc = declared_pni_csv(rid, ROOT)
    e = load_edges(P.checked_source_csv(run, pc, rec[rid], None))
    pnis = P.load_pnis(pc).reset_index(drop=True)
    seeds = load_rung(run, SPC.SOURCE_NSIDE).seeds
    pseed, _ = SPC.nearest_seed(pnis.pni_lat, pnis.pni_lon, seeds)
    for tg, x in e.groupby("tg_id", sort=False):
        vi = haversine_km(x.vp_lat.values[:, None], x.vp_lon.values[:, None],
                          pnis.pni_lat.values[None, :], pnis.pni_lon.values[None, :])
        ti = haversine_km(x.tg_lat.values[:1, None], x.tg_lon.values[:1, None],
                          pnis.pni_lat.values[None, :], pnis.pni_lon.values[None, :])[0]
        path = vi + ti[None, :]                       # VP -> I -> TG, km
        rtt = x.rtt_ms.values
        feasible = (rtt[:, None] >= path / FLOOR).all(0)   # no VP below the floor via I
        rho = np.array([spearmanr(path[:, j], rtt).statistic for j in range(len(pnis))])
        xi = int(ti.argmin())
        bi = int(np.nanargmax(rho))
        floor_x = path[:, xi].min() / FLOOR               # best-case RTT through X
        out.append(dict(run_id=rid, tg_id=tg, x_idx=xi, best_idx=bi,
                        x_feasible=bool(feasible[xi]), n_feasible=int(feasible.sum()),
                        rho_x=rho[xi], rho_best=rho[bi],
                        rho_tg=spearmanr(x.d_km.values, rtt).statistic,
                        best_seed=pseed[bi], x_seed_chk=pseed[xi],
                        d_x_best_km=float(haversine_km(pnis.pni_lat[xi], pnis.pni_lon[xi],
                                                       pnis.pni_lat[bi], pnis.pni_lon[bi])),
                        excess_over_x_floor_ms=rtt.min() - floor_x,
                        direct_feasible=bool((rtt >= x.d_km.values / FLOOR).all()),
                        n_vps_break_x=int((rtt < path[:, xi] / FLOOR).sum()),
                        max_break_x_ms=float((path[:, xi] / FLOOR - rtt).max())))
pd.DataFrame(out).to_csv(sys.argv[1], index=False)
print("wrote", len(out))
