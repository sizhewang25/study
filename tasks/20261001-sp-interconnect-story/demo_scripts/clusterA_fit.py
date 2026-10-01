import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scripts.analysis.v5.modules.paths import resolve_run
from scripts.analysis.v5.modules import figure_vp_proximity as P, edges
from scripts.libs.canonical.schema import load_canonical_csv
src = load_canonical_csv(edges.resolve_source_csv(resolve_run("as01-260728-260802-mesh"), None))
A = set(pd.read_csv(f"{sys.argv[1]}/pni_gap_large.csv").target_id)
src["site"] = src.target_lat.round(3).astype(str) + "," + src.target_lon.round(3).astype(str)
src["A"] = src.target_id.isin(A)
def hv(la1, lo1, la2, lo2):
    la1, lo1, la2, lo2 = map(np.radians, (la1, lo1, la2, lo2))
    h = np.sin((la2-la1)/2)**2 + np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*6371*np.arcsin(np.sqrt(h))
glat, glon = np.meshgrid(np.arange(-55, 71, 1.0), np.arange(-180, 180, 1.0), indexing="ij")
G = np.c_[glat.ravel(), glon.ravel()]
def fit(x, T):
    r = x.rtt_ms.values; vl, vo = x.vp_lat.values, x.vp_lon.values
    D = hv(vl[None], vo[None], G[:, :1], G[:, 1:])                    # grid x VP
    dT = hv(vl, vo, T[0], T[1]); dYT = hv(G[:, 0], G[:, 1], T[0], T[1])
    out = {"flat": r.std()}
    res = r[None] - D / 100; sd = np.where(res.min(1) >= 0, res.std(1), np.inf); i = sd.argmin()           # responder at X (2/3c, round trip)
    out["unicast@X"] = sd[i]; out["X"] = tuple(G[i]); out["a_X"] = res[i].mean()
    res = r[None] - (dT[None] + D + dYT[:, None]) / 200; sd = np.where(res.min(1) >= 0, res.std(1), np.inf); j = sd.argmin()   # one direction via Y
    out["asym-via-Y"] = sd[j]; out["Y"] = tuple(G[j]); out["a_Y"] = res[j].mean()
    res = r - dT / 100; out["unicast@GT"] = res.std()
    return out
rows = []
for (site, isA), x in src.groupby(["site", "A"]):
    if not (isA or site.startswith("37.363")): continue
    x = x.groupby("vp_id", as_index=False).agg(rtt_ms=("rtt_ms", "median"), vp_lat=("vp_lat", "first"), vp_lon=("vp_lon", "first"))
    T = tuple(map(float, site.split(",")))
    o = fit(x, T); o.update(site=site, cluster="A" if isA else "C", n_vp=len(x), rtt_min=x.rtt_ms.min()); rows.append(o)
pd.set_option("display.width", 250)
print(pd.DataFrame(rows)[["site", "cluster", "n_vp", "rtt_min", "flat", "unicast@GT", "unicast@X", "X", "a_X", "asym-via-Y", "Y", "a_Y"]].round(1).to_string(index=False))
