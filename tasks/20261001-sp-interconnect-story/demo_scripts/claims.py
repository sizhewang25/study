import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scripts.analysis.v5.modules.paths import resolve_run
from scripts.analysis.v5.modules import figure_vp_proximity as P, edges
from scripts.libs.canonical.schema import load_canonical_csv
run = resolve_run("as01-260728-260802-mesh")
src = load_canonical_csv(edges.resolve_source_csv(run, None))
print("rows per (tg,vp):", src.groupby(["target_id","vp_id"]).size().describe()[["min","50%","max"]].tolist())
e = src.groupby(["target_id", "vp_id"], as_index=False).agg(rtt=("rtt_ms", "min"), vp_lat=("vp_lat", "first"),
    vp_lon=("vp_lon", "first"), tg_lat=("target_lat", "first"), tg_lon=("target_lon", "first"))
e["d"] = P.haversine_km(e.vp_lat.values, e.vp_lon.values, e.tg_lat.values, e.tg_lon.values)
g = e.groupby("target_id")
geo = e.loc[g.d.idxmin()].set_index("target_id"); sp = e.loc[g.rtt.idxmin()].set_index("target_id")
t = pd.DataFrame({"lat": geo.tg_lat, "lon": geo.tg_lon, "d_geo": geo.d, "rtt_geo": geo.rtt, "d_sp": sp.d, "rtt_sp": sp.rtt})
t["gap"] = t.d_sp - t.d_geo
pni = pd.read_csv("datasets/pni/as01-us-pni.approx.csv")
t["d_pni"] = P.haversine_km(t.lat.values[:, None], t.lon.values[:, None], pni.pni_lat.values[None], pni.pni_lon.values[None]).min(1)
t["cl"] = np.select([t.gap > 3000, (t.gap > 400) & (t.d_pni > 500)], ["A", "B"], "C")
# propagation RTT at 2/3 c: 1 ms RTT = 100 km one-way
t["prop_geo"] = t.d_geo / 100; t["infl_geo"] = t.rtt_geo - t.prop_geo          # excess RTT on nearest VP
t["dprop"] = t.gap / 100                                                        # propagation advantage of nearest VP
t["drtt"] = t.rtt_geo - t.rtt_sp                                                # measured RTT disadvantage
nz = t[t.gap > 0]
print("\nnonzero-gap TGs by cluster:\n", nz.groupby("cl").agg(n=("gap","size"), sites=("lat","nunique"),
   rtt_geo_med=("rtt_geo","median"), infl_geo_med=("infl_geo","median"),
   dprop_med=("dprop","median"), drtt_med=("drtt","median"), drtt_max=("drtt","max")).round(2))
# site-level view (one row per site) for cluster C nonzero gap
s = nz[nz.cl=="C"].groupby(["lat","lon"]).agg(n=("gap","size"), d_geo=("d_geo","median"), rtt_geo=("rtt_geo","median"),
    d_sp=("d_sp","median"), rtt_sp=("rtt_sp","median"), gap=("gap","median"), dprop=("dprop","median"), drtt=("drtt","median"))
print("\ncluster C nonzero-gap sites:\n", s.round(2).to_string())
# zero-gap sites: nearest VP RTT
z = t[t.gap == 0].groupby(["lat","lon"]).agg(n=("gap","size"), d_geo=("d_geo","median"), rtt_geo=("rtt_geo","median"))
print("\nzero-gap sites:\n", z.round(2).to_string())
# claim 1 vs common-mode: in A, how do ALL VPs' RTT-minus-propagation look? correlation of rtt with distance
for c in "ABC":
    ee = e[e.target_id.isin(t.index[t.cl == c])]
    r = ee.groupby("target_id").apply(lambda x: np.corrcoef(x.d, x.rtt)[0, 1]).median()
    print(c, "median per-TG corr(dist, rtt) =", round(r, 2), " median per-TG min(rtt - d/100) =",
          round(ee.assign(x=ee.rtt - ee.d/100).groupby("target_id").x.min().median(), 1))
# claim-3 check: each VP's floor = its min RTT to any target in the mesh (upper bound on its access delay)
fl = e.groupby("vp_id").rtt.min()
t["geo_vp"] = geo.vp_id; t["sp_vp"] = sp.vp_id
t["dfloor"] = t.geo_vp.map(fl) - t.sp_vp.map(fl)
t["floor_geo"] = t.geo_vp.map(fl); t["floor_sp"] = t.sp_vp.map(fl)
nz = t[t.gap > 0]
print("\nVP floors (ms): median", fl.median().round(2), "IQR", fl.quantile([.25,.75]).round(2).tolist(), "n VPs", len(fl))
print(nz.groupby("cl").agg(floor_geo=("floor_geo","median"), floor_sp=("floor_sp","median"), dfloor=("dfloor","median"),
      drtt=("drtt","median"), dprop=("dprop","median"), share_drtt_gt_dprop=("drtt", lambda x: (x > nz.loc[x.index,"dprop"]).mean()),
      share_dfloor_ge_half_drtt=("dfloor", lambda x: (x >= 0.5*nz.loc[x.index,"drtt"]).mean())).round(2))
sc = nz[nz.cl=="C"].groupby(["lat","lon"]).agg(dprop=("dprop","median"), drtt=("drtt","median"), dfloor=("dfloor","median"), floor_geo=("floor_geo","median"), floor_sp=("floor_sp","median"))
print(sc.round(2).to_string())
bsites = t[t.cl=="B"].groupby(["lat","lon"]).agg(n=("gap","size"), d_pni=("d_pni","first"), d_geo=("d_geo","median"), rtt_geo=("rtt_geo","median"), d_sp=("d_sp","median"), rtt_sp=("rtt_sp","median"))
print("\nB sites\n", bsites.round(1).to_string())
# leave-site-out floor: VP's min RTT to targets at OTHER sites
e["site"] = list(zip(e.tg_lat.round(4), e.tg_lon.round(4)))
vs = e.groupby(["vp_id", "site"]).rtt.min().rename("m").reset_index()
def lso(vp, site):
    x = vs[(vs.vp_id == vp) & (vs.site != site)]
    return x.m.min()
t["site"] = list(zip(t.lat.round(4), t.lon.round(4)))
nzC = t[(t.gap > 0) & (t.cl == "C")].drop_duplicates(["site", "geo_vp", "sp_vp"]).copy()
nzC["lso_geo"] = [lso(v, s) for v, s in zip(nzC.geo_vp, nzC.site)]
nzC["lso_sp"] = [lso(v, s) for v, s in zip(nzC.sp_vp, nzC.site)]
nzC["dlso"] = nzC.lso_geo - nzC.lso_sp
print("\nC leave-site-out (one row per site x VP pair)\n", nzC[["site","d_geo","d_sp","dprop","drtt","dlso","lso_geo","lso_sp"]].round(2).to_string(index=False))
print("corr(drtt, dlso) =", np.corrcoef(nzC.drtt, nzC.dlso)[0,1].round(2), " sign agree:", (np.sign(nzC.dlso) == np.sign(nzC.drtt)).mean().round(2))
