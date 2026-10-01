import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from scripts.analysis.v5.modules.paths import resolve_run
from scripts.analysis.v5.modules import figure_vp_proximity as P, edges
from scripts.libs.canonical.schema import load_canonical_csv
src = load_canonical_csv(edges.resolve_source_csv(resolve_run("as01-260728-260802-mesh"), None))
A = pd.read_csv(f"{sys.argv[1]}/pni_gap_large.csv").target_id
e = src[src.target_id.isin(A)].copy()
e["site"] = e.target_lat.round(2).astype(str) + "," + e.target_lon.round(2).astype(str)
e["d_tg"] = P.haversine_km(e.vp_lat.values, e.vp_lon.values, e.target_lat.values, e.target_lon.values)
NYC = (40.71, -74.01)   # where the winning VP sits
e["d_nyc"] = P.haversine_km(e.vp_lat.values, e.vp_lon.values, np.full(len(e), NYC[0]), np.full(len(e), NYC[1]))
e["tg_nyc"] = P.haversine_km(e.target_lat.values, e.target_lon.values, np.full(len(e), NYC[0]), np.full(len(e), NYC[1]))
for s, x in e.groupby("site"):
    # hypothesis: one direction of every path goes via NYC -> RTT ~ (d_tg + d_nyc + tg_nyc)/200 ms at 2/3 c
    pred = (x.d_tg + x.d_nyc + x.tg_nyc) / 200
    print(f"{s}: n_pairs={len(x)}  corr(rtt,d_tg)={np.corrcoef(x.rtt_ms, x.d_tg)[0,1]:+.2f}  corr(rtt,d_vp->NYC)={np.corrcoef(x.rtt_ms, x.d_nyc)[0,1]:+.2f}"
          f"  corr(rtt, via-NYC pred)={np.corrcoef(x.rtt_ms, pred)[0,1]:+.2f}  median rtt-pred={np.median(x.rtt_ms-pred):.1f} ms  IQR={np.subtract(*np.percentile(x.rtt_ms-pred,[75,25])):.1f}"
          f"  per-VP RTT spread across the site's IPs (median)={x.groupby('vp_id').rtt_ms.agg(lambda r: r.max()-r.min()).median():.2f} ms")
