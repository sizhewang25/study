"""RTT plausibility of labels on the OCT-only cohort, independent of OCT:
consensus = medoid of S-P/SOI/VAN/SPO predictions; compare RTT from VPs near it
with the 2/3c floor back to the label, and check label feasibility."""
import sys
import numpy as np, pandas as pd
R = 6371.0
def hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, (a1, o1, a2, o2))
    h = np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin((o2-o1)/2)**2
    return 2*R*np.arcsin(np.sqrt(h))
OUT = sys.argv[1]
coh = pd.read_csv(f"{OUT}/lso_oct_cohort.csv")
M = ["shortest_ping", "million_scale_cbg", "vanilla_cbg", "spotter_cbg"]
rows = []
for run, g in coh.groupby("run"):
    b = f"outputs/analysis/v5/{run}/classify/healpix-128"
    fr = {m: pd.read_parquet(f"{b}/{m}_tgs.parquet").set_index("tg_id") for m in M}
    e = pd.read_csv(f"datasets/final/as{run[6:8]}-20260728-20260802.mainland.sanitized.csv")
    pni = pd.read_csv(f"datasets/pni/as{run[6:8]}-us-pni.approx.csv")
    for r in g.itertuples():
        P = [(fr[m].loc[r.tg_id, "pred_lat"], fr[m].loc[r.tg_id, "pred_lon"]) for m in M
             if fr[m].loc[r.tg_id, "cell_label"] != "unanswered"]
        la, lo = np.array(P).T
        D = hav(la[:, None], lo[:, None], la[None], lo[None]); k = D.sum(1).argmin()
        cl, co = la[k], lo[k]
        ee = e[e.target_id == r.tg_id]
        dL = hav(ee.vp_lat.values, ee.vp_lon.values, r.tg_lat, r.tg_lon)
        dP = hav(ee.vp_lat.values, ee.vp_lon.values, cl, co)
        nearP = ee.rtt_ms.values[dP <= 50]
        rows.append(dict(run=run, tg_id=r.tg_id, cohort=r.cohort, tg_lat=r.tg_lat, tg_lon=r.tg_lon,
                         cons_lat=cl, cons_lon=co, d_label_cons=hav(r.tg_lat, r.tg_lon, cl, co),
                         d_cons_pni=hav(cl, co, pni.pni_lat.values, pni.pni_lon.values).min(),
                         min_rtt=ee.rtt_ms.min(), label_minslack=(ee.rtt_ms.values - dL/100).min(),
                         n_vp_nearP=len(nearP), rtt_nearP=nearP.min() if len(nearP) else np.nan,
                         floor_LP=hav(r.tg_lat, r.tg_lon, cl, co)/100))
pd.DataFrame(rows).to_csv(f"{OUT}/octonly_label_check.csv", index=False)
