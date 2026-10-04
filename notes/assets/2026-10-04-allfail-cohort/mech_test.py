"""Mechanism tests on the all-fail cohort.
OCT-H: full-calibration replay vs leave-site-out calibration (memorization).
SPO:   sum z^2 at SPO answer / label / consensus under the fold's pooled mu,sigma."""
import pickle, sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np, pandas as pd
from scripts.framework.v2.model import CBGModel
from scripts.framework.v2.ltd.base import FitSample
from scripts.framework.v2.ltd.bounded_spline import BoundedSplineLTD
from scripts.framework.v2.types import Coord
from scripts.framework.v2.registry import MTL_REGISTRY, CTR_REGISTRY

S = Path(sys.argv[1]); B = Path("outputs/benchmark/v2")
R = 6371.0
def hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, (a1, o1, a2, o2))
    h = np.sin((a2-a1)/2)**2 + np.cos(a1)*np.cos(a2)*np.sin((o2-o1)/2)**2
    return 2*R*np.arcsin(np.sqrt(h))

coh = pd.read_csv(S/"gt_check.csv"); coh = coh[coh.af]
OCT = dict(ltd="bounded_spline", mtl="planar_annulus_weighted", ctr="monte_carlo_medoid",
           ltd_kwargs=dict(fit_spline=False, cutoff_min_points=5, bin_size_ms=5.0),
           mtl_kwargs=dict(n_pts=64, enable_circle_filter=True, weight_mode="inv_power", weight_k=2, highest_weight_only=True),
           ctr_kwargs=dict(n_samples=1024))

def annulus_share(ltd, obs, pt):
    """share of VPs whose [inner,outer] at the observed RTT admits distance to pt"""
    hit = n = 0
    for vid, c, rtt in obs:
        r = ltd.predict(vid, c, rtt)
        if not r.success: continue
        n += 1; d = hav(c.lat, c.lon, *pt)
        hit += r.tg_distance.lower_km <= d <= r.tg_distance.upper_km
    return hit / n if n else np.nan

def job(args):
    run, fold, site, tgs = args
    r = run[6:8]
    root = B/run/"generic_csv/anchors_to_probes"
    e = pd.read_csv(f"datasets/final/as{r}-20260728-20260802.mainland.sanitized.csv")
    folds = pd.read_parquet(f"outputs/analysis/v5/{run}/classify/healpix-128/shortest_ping_tgs.parquet")[["tg_id","fold","tg_lat","tg_lon"]]
    fmap = dict(zip(folds.tg_id, folds.fold))
    e["fold"] = e.target_id.map(fmap)
    tr = e[e.fold != fold]
    lat0, lon0 = map(float, site.split("|")[1].split(","))
    same = (tr.target_lat.round(3) == lat0) & (tr.target_lon.round(3) == lon0)
    def samples(df):
        return [FitSample(v, Coord(a, b), Coord(c, d), float(t)) for v, a, b, c, d, t in
                zip(df.vp_id, df.vp_lat, df.vp_lon, df.target_lat, df.target_lon, df.rtt_ms)]
    full = pickle.load(open(root/f"fold_{fold}/octant_cbg_hull/fit_checkpoint.pkl", "rb"))
    lso = BoundedSplineLTD(**OCT["ltd_kwargs"]); lso.fit(samples(tr[~same]))
    spo = pickle.load(open(root/f"fold_{fold}/spotter_cbg/fit_checkpoint.pkl", "rb"))
    pub = {c: pd.read_parquet(root/f"fold_{fold}/{c}/targets.parquet").set_index("target_id")
           for c in ["octant_cbg_hull", "spotter_cbg"]}
    out = []
    for t in tgs:
        row = coh[(coh.run == run) & (coh.tg_id == t)].iloc[0]
        ee = e[e.target_id == t]
        obs = [(v, Coord(a, b), float(x)) for v, a, b, x in zip(ee.vp_id, ee.vp_lat, ee.vp_lon, ee.rtt_ms)]
        L = (row.tg_lat, row.tg_lon); P = (row.med_lat, row.med_lon)
        seed = int(pub["octant_cbg_hull"].loc[t, "seed"])
        res = {"run": run, "fold": fold, "site": site, "tg_id": t,
               "n_same_site_train": int(same.sum() // max(1, ee.vp_id.nunique()))}
        for name, ltd in [("full", full), ("lso", lso)]:
            m = CBGModel.from_config(**OCT); m.ltd = ltd
            m.ctr.rng = np.random.default_rng(seed)
            g = m.geolocate(obs)
            res[f"{name}_status"] = g.status.name
            res[f"{name}_lat"] = g.coord.lat if g.coord else np.nan
            res[f"{name}_lon"] = g.coord.lon if g.coord else np.nan
            res[f"{name}_admit_label"] = annulus_share(ltd, obs, L)
            res[f"{name}_admit_cons"] = annulus_share(ltd, obs, P)
        res["pub_lat"], res["pub_lon"] = pub["octant_cbg_hull"].loc[t, ["pred_lat","pred_lon"]]
        # SPO: z^2 decomposition
        mu, sg, vc = [], [], []
        for v, c, x in obs:
            rr = spo.predict(v, c, x)
            if rr.success and rr.tg_distance is not None and rr.tg_distance.has_distribution:
                mu.append(rr.tg_distance.mu_km); sg.append(rr.tg_distance.sigma_km); vc.append((c.lat, c.lon, x))
        mu, sg = np.array(mu), np.array(sg); vc = np.array(vc)
        sp = pub["spotter_cbg"].loc[t, ["pred_lat","pred_lon"]].astype(float).values
        def z2(pt):
            z = (hav(vc[:,0], vc[:,1], *pt) - mu) / sg
            return z*z
        zs, zl, zc = z2(sp), z2(L), z2(P)
        o = np.argsort(vc[:,2])
        res.update(spo_lat=sp[0], spo_lon=sp[1],
                   spo_sumz2_at_spo=zs.sum(), spo_sumz2_at_label=zl.sum(), spo_sumz2_at_cons=zc.sum(),
                   spo_mu_minrtt=mu[o[0]], spo_sigma_minrtt=sg[o[0]], spo_mu_median=np.median(mu),
                   spo_w_top10_lowrtt=(1/sg[o[:max(1,len(o)//10)]]**2).sum()/(1/sg**2).sum(),
                   spo_top10pct_share_label=np.sort(zl)[::-1][:max(1,len(zl)//10)].sum()/zl.sum(),
                   spo_median_absz_at_spo=np.median(np.sqrt(zs)), spo_median_absz_at_label=np.median(np.sqrt(zl)),
                   spo_dist_to_nearest_vp=hav(vc[:,0],vc[:,1],*sp).min())
        out.append(res)
    return out

if __name__ == "__main__":
    jobs = []
    folds = {r: pd.read_parquet(f"outputs/analysis/v5/{r}/classify/healpix-128/shortest_ping_tgs.parquet").set_index("tg_id").fold
             for r in coh.run.unique()}
    coh["fold"] = [folds[r][t] for r, t in zip(coh.run, coh.tg_id)]
    for (run, fold, site), g in coh.groupby(["run", "fold", "site"]):
        jobs.append((run, int(fold), site, list(g.tg_id)))
    print(len(jobs), "jobs", flush=True)
    with Pool(6) as p:
        rows = [r for part in p.imap_unordered(job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(S/"mech_test.csv", index=False)
    print("done", len(rows))
