"""Leave-site-out (LSO) test for OCT-H and OCT-S: does an OCT-only cell success
survive when the target's own site (its IP replicas) is removed from the
calibration? Replays each fold's checkpoint first (must match the published
prediction), then refits the same LTD without same-site rows.

Cohorts: octonly = OCT-H or OCT-S cell-correct and none of S-P/SOI/VAN/SPO;
control = a stratified sample of OCT-H-correct TGs where another method is also correct."""
import json, pickle, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pandas as pd
from scripts.analysis.v5.cli import resolve_run, DEFAULT_OUTPUTS_ROOT
from scripts.analysis.v5.modules.map_answer_space import load_rung
from scripts.analysis.v5.modules.geodesy import pairwise_km
from scripts.framework.v2.model import CBGModel
from scripts.framework.v2.ltd.base import FitSample
from scripts.framework.v2.registry import LTD_REGISTRY
from scripts.framework.v2.types import Coord

OUT = Path(sys.argv[1])
B = Path("outputs/benchmark/v2")
RUNS = ["pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh"]
M = {"shortest_ping": "S-P", "million_scale_cbg": "SOI", "vanilla_cbg": "VAN",
     "octant_cbg_hull": "OCT-H", "octant_cbg_spl": "OCT-S", "spotter_cbg": "SPO"}
OCT = ["octant_cbg_hull", "octant_cbg_spl"]


def cohort() -> pd.DataFrame:
    rows = []
    for run in RUNS:
        b = f"outputs/analysis/v5/{run}/classify/healpix-128"
        fr = {m: pd.read_parquet(f"{b}/{m}_tgs.parquet").set_index("tg_id") for m in M}
        c = pd.DataFrame({t: fr[m].cell_label == "correct" for m, t in M.items()})
        oth = c[["S-P", "SOI", "VAN", "SPO"]].any(axis=1)
        base = fr["shortest_ping"][["fold", "tg_lat", "tg_lon", "tg_seed_id"]].copy()
        base["run"] = run
        base["octonly"] = (c["OCT-H"] | c["OCT-S"]) & ~oth
        base["ctrl_pool"] = c["OCT-H"] & oth
        for t in ["OCT-H", "OCT-S"]:
            base[f"pub_{t}_correct"] = c[t]
        rows.append(base.reset_index())
    d = pd.concat(rows, ignore_index=True)
    ctrl = d[d.ctrl_pool].groupby("run", group_keys=False).apply(lambda g: g.sample(40, random_state=0))
    d["cohort"] = np.where(d.octonly, "octonly", np.where(d.index.isin(ctrl.index), "control", ""))
    return d[d.cohort != ""]


def job(args):
    run, fold, site, recs = args
    root = B / run / "generic_csv/anchors_to_probes"
    e = pd.read_csv(f"datasets/final/as{run[6:8]}-20260728-20260802.mainland.sanitized.csv")
    folds = pd.read_parquet(f"outputs/analysis/v5/{run}/classify/healpix-128/shortest_ping_tgs.parquet")
    e["fold"] = e.target_id.map(dict(zip(folds.tg_id, folds.fold)))
    tr = e[e.fold != fold]
    lat0, lon0 = site
    same = (tr.target_lat.round(3) == lat0) & (tr.target_lon.round(3) == lon0)
    lso_samples = [FitSample(v, Coord(a, b), Coord(c, d), float(t)) for v, a, b, c, d, t in
                   zip(tr.vp_id[~same], tr.vp_lat[~same], tr.vp_lon[~same],
                       tr.target_lat[~same], tr.target_lon[~same], tr.rtt_ms[~same])]
    seeds = load_rung(resolve_run(run, DEFAULT_OUTPUTS_ROOT), 128).seeds
    out = []
    for combo in OCT:
        cfg = json.loads((root / f"fold_{fold}/{combo}/run.json").read_text())
        full = pickle.load(open(root / f"fold_{fold}/{combo}/fit_checkpoint.pkl", "rb"))
        lso = LTD_REGISTRY[cfg["ltd"]](**cfg["ltd_kwargs"]); lso.fit(lso_samples)
        pub = pd.read_parquet(root / f"fold_{fold}/{combo}/targets.parquet").set_index("target_id")
        for r in recs:
            ee = e[e.target_id == r["tg_id"]]
            obs = [(v, Coord(a, b), float(x)) for v, a, b, x in zip(ee.vp_id, ee.vp_lat, ee.vp_lon, ee.rtt_ms)]
            res = {"run": run, "tg_id": r["tg_id"], "cohort": r["cohort"], "combo": M[combo], "fold": fold,
                   "site_lat": lat0, "site_lon": lon0, "n_same_site_train": int(same.sum() // max(1, ee.vp_id.nunique())),
                   "pub_correct": bool(r[f"pub_{M[combo]}_correct"])}
            for name, ltd in [("full", full), ("lso", lso)]:
                m = CBGModel.from_config(cfg["ltd"], cfg["mtl"], cfg["ctr"], ltd_kwargs=cfg["ltd_kwargs"],
                                         mtl_kwargs=cfg["mtl_kwargs"], ctr_kwargs=cfg["ctr_kwargs"])
                m.ltd = ltd
                m.ctr.rng = np.random.default_rng(int(pub.loc[r["tg_id"], "seed"]))
                g = m.geolocate(obs)
                plat, plon = (g.coord.lat, g.coord.lon) if g.coord else (np.nan, np.nan)
                ok = g.status.name == "SUCCESS"
                pseed = seeds.seed_id.to_numpy()[pairwise_km([plat], [plon], seeds.seed_lat, seeds.seed_lon).argmin(axis=1)][0] if ok else -1
                res.update({f"{name}_status": g.status.name, f"{name}_lat": plat, f"{name}_lon": plon,
                            f"{name}_correct": bool(ok and pseed == r["tg_seed_id"]),
                            f"{name}_err_km": float(pairwise_km([plat], [plon], [r["tg_lat"]], [r["tg_lon"]])[0, 0]) if ok else np.nan})
            res["replay_dev_km"] = float(pairwise_km([res["full_lat"]], [res["full_lon"]],
                                                     [pub.loc[r["tg_id"], "pred_lat"]], [pub.loc[r["tg_id"], "pred_lon"]])[0, 0])
            out.append(res)
    return out


if __name__ == "__main__":
    d = cohort()
    d.to_csv(OUT / "lso_oct_cohort.csv", index=False)
    print(d.cohort.value_counts().to_dict(), flush=True)
    jobs = [(run, int(f), (round(la, 3), round(lo, 3)), g.to_dict("records"))
            for (run, f, la, lo), g in d.assign(la=d.tg_lat.round(3), lo=d.tg_lon.round(3)).groupby(["run", "fold", "la", "lo"])]
    print(len(jobs), "jobs", flush=True)
    with Pool(6) as p:
        rows = [r for part in p.imap_unordered(job, jobs) for r in part]
    pd.DataFrame(rows).to_csv(OUT / "lso_oct.csv", index=False)
    print("done", len(rows), flush=True)
