"""Audit a LOSO run's materialized splits against its edge CSV.

Per fold: the eval set is exactly one site (all of its TGs), the fit samples
are exactly the CSV rows of every other site, and the held-out site is absent
from fit. Across folds: the eval sets partition the TGs and the sites.
(`fit_samples.probe_id` is coordinate-keyed, so fit is audited by site and by
row count, not by target id.)"""
import glob, re, sys
import pandas as pd
import yaml

run = sys.argv[1]
cfg = yaml.safe_load(open(f"configs/{run}.yaml"))
csv = pd.read_csv(cfg["benchmark"]["source_kwargs"]["csv_path"])
csv = csv[csv.rtt_ms > 0]
csv["site"] = list(zip(csv.target_lat.round(6), csv.target_lon.round(6)))
rows_per_site = csv.groupby("site").size()
tgs_per_site = csv.groupby("site").target_id.agg(set)
root = f"inputs/benchmark/v2/generic_csv/{run}/anchors_to_probes"
folds = sorted(glob.glob(f"{root}/fold_*"), key=lambda p: int(p.rsplit("_", 1)[1]))
held, problems = [], []
for f in folds:
    ev = pd.read_parquet(f"{f}/eval_observations.parquet", columns=["target_id", "target_lat", "target_lon"])
    fit = pd.read_parquet(f"{f}/fit_samples.parquet", columns=["probe_lat", "probe_lon"])
    ev_sites = set(zip(ev.target_lat.round(6), ev.target_lon.round(6)))
    fit_sites = set(zip(fit.probe_lat.round(6), fit.probe_lon.round(6)))
    if len(ev_sites) != 1:
        problems.append((f, f"eval spans {len(ev_sites)} sites")); continue
    (s,) = ev_sites
    held.append(s)
    if set(ev.target_id) != tgs_per_site[s]: problems.append((f, "eval is not the whole site"))
    if s in fit_sites: problems.append((f, "held-out site in fit"))
    if fit_sites | {s} != set(rows_per_site.index): problems.append((f, "fit misses a site"))
    if len(fit) != rows_per_site.drop(s).sum(): problems.append((f, f"fit rows {len(fit)} != {rows_per_site.drop(s).sum()}"))
ok_partition = sorted(held) == sorted(rows_per_site.index)
print(f"{run}: {len(folds)} folds / {len(rows_per_site)} sites / {csv.target_id.nunique()} TGs; "
      f"each site held out once: {ok_partition}; problems: {problems or 'none'}")
