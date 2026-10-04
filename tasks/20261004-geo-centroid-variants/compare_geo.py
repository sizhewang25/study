"""GEO vs original: paired, site-clustered comparison of the CTR swap.

OCT-H-GEO / OCT-S-GEO differ from OCT-H / OCT-S only in the CTR step
(`geometric_centroid` instead of `monte_carlo_medoid`). This scores both arms
in memory with the v5 scorer (`classify.load_method_frame` + `score_method`
against the run's existing nside-128 answer space), so labels, denominators and
the solved mask are exactly v5's. It writes nothing under `outputs/`.

Run from the repo root:
    PATH=$PWD/.venv/bin:$PATH python tasks/20261004-geo-centroid-variants/compare_geo.py

CSV tables land in tasks/20261004-geo-centroid-variants/tables/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.v5.modules import classify as C  # noqa: E402
from scripts.analysis.v5.modules import sites as SITES  # noqa: E402
from scripts.analysis.v5.modules.answer_space import load_answer_space  # noqa: E402
from scripts.analysis.v5.modules.geodesy import elementwise_km  # noqa: E402
from scripts.analysis.v5.modules.loso_delta import (  # noqa: E402
    BOOT_SEED, N_BOOT, _quantile, site_bootstrap,
)
from scripts.analysis.v5.modules.paths import (  # noqa: E402
    DEFAULT_ANALYSIS_ROOT, DEFAULT_OUTPUTS_ROOT, RunPaths,
)
from scripts.analysis.v5.modules.status import solved_mask  # noqa: E402

NSIDE = 128
NETS = {"pro-as01": "AS-A", "pro-as02": "AS-B", "pro-as03": "AS-C"}
REGIMES = {"mesh": "K-fold", "loso": "LOSO"}
PAIRS = {"OCT-H": ("octant_cbg_hull", "octant_cbg_hull_geo"),
         "OCT-S": ("octant_cbg_spl", "octant_cbg_spl_geo")}
TIMING = ("ltd_ms", "mtl_ms", "ctr_ms")
OUT = Path(__file__).resolve().parent / "tables"
RUNJSON_IGNORE = {
    "combo_id", "ctr", "ctr_kwargs", "started_at", "fit_ms", "fit_alloc_peak_bytes",
    "fit_heap_peak_bytes", "fit_rss_peak_bytes", "run_baseline_rss_bytes",
    "rss_after_inputs_bytes", "rss_after_fit_bytes", "run_peak_rss_bytes",
}


def run_paths(run_id: str) -> RunPaths:
    return RunPaths(run_id=run_id, root=DEFAULT_OUTPUTS_ROOT, source="generic_csv",
                    setup="anchors_to_probes")


def space_for(run_id: str):
    # Built by path rather than RunPaths.answer_space_dir, which mkdirs.
    return load_answer_space(DEFAULT_ANALYSIS_ROOT / run_id / "answer-space" / f"healpix-{NSIDE}")


def stored_tgs(run_id: str, method: str) -> pd.DataFrame:
    p = DEFAULT_ANALYSIS_ROOT / run_id / "classify" / f"healpix-{NSIDE}" / C.TGS_PARQUET.format(method=method)
    return pd.read_parquet(p)


def has_x_map(net: str) -> pd.Series:
    found = sorted((DEFAULT_ANALYSIS_ROOT / f"{net}-mesh" / "pni-gap").glob("*/sp_pni_cells_tgs.csv"))
    assert len(found) == 1, found
    c = pd.read_csv(found[0], usecols=["tg_id", "tg_cell_holds_x"]).drop_duplicates("tg_id")
    return pd.Series(np.where(c["tg_cell_holds_x"].astype(bool), "has-X", "no-X"), index=c["tg_id"])


# ---- invariants ----------------------------------------------------------------


def check_folds_and_config(rp: RunPaths, orig: str, geo: str) -> dict:
    folds = rp.fold_ids
    info = {"n_folds": len(folds), "missing": [], "tg_set_mismatch": [], "config_diff": []}
    for f in folds:
        po, pg = rp.combo_dir(orig, f), rp.combo_dir(geo, f)
        for p in (po, pg):
            if not (p / "targets.parquet").exists() or not (p / "run.json").exists():
                info["missing"].append(str(p))
        if info["missing"]:
            continue
        to = pd.read_parquet(po / "targets.parquet", columns=["target_id"])["target_id"]
        tg = pd.read_parquet(pg / "targets.parquet", columns=["target_id"])["target_id"]
        if set(to) != set(tg) or len(to) != len(tg):
            info["tg_set_mismatch"].append(f)
        jo = {k: v for k, v in json.loads((po / "run.json").read_text()).items() if k not in RUNJSON_IGNORE}
        jg = {k: v for k, v in json.loads((pg / "run.json").read_text()).items() if k not in RUNJSON_IGNORE}
        diff = sorted(k for k in set(jo) | set(jg) if jo.get(k) != jg.get(k))
        if diff:
            info["config_diff"].append((f, diff))
    return info


def sanity_reproduce(run_id: str, method: str, scored: pd.DataFrame) -> dict:
    st = stored_tgs(run_id, method).set_index("tg_id")
    me = scored.set_index("tg_id")
    assert set(st.index) == set(me.index), f"{run_id} {method}: tg sets differ from stored"
    me = me.loc[st.index]
    out = {}
    for col in ("cell_label", "status", C.GRID_OFFSET, "tg_seed_id", "pred_seed_id"):
        out[col] = int((me[col].astype(str).to_numpy() != st[col].astype(str).to_numpy()).sum())
    for col in ("pred_dist_to_tg_km", "pred_lat", "pred_lon"):
        a, b = me[col].to_numpy(float), st[col].to_numpy(float)
        same = (np.isnan(a) & np.isnan(b)) | (a == b)
        out[col] = int((~same).sum())
    return out


# ---- per-run paired frame --------------------------------------------------------


def paired_frame(net: str, regime: str, pair: str, space, hx: pd.Series, inv: dict) -> pd.DataFrame:
    run_id = f"{net}-{regime}"
    rp = run_paths(run_id)
    orig, geo = PAIRS[pair]
    inv_key = f"{run_id}/{pair}"
    inv[inv_key] = check_folds_and_config(rp, orig, geo)

    so = C.score_method(C.load_method_frame(rp, orig, columns=TIMING), space)
    sg = C.score_method(C.load_method_frame(rp, geo, columns=TIMING), space)
    inv[inv_key]["reproduce_stored_" + orig] = sanity_reproduce(run_id, orig, so)
    for s in (so, sg):
        assert not s["tg_id"].duplicated().any()
    so, sg = so.set_index("tg_id"), sg.set_index("tg_id")
    assert set(so.index) == set(sg.index)
    sg = sg.loc[so.index]
    assert (so["fold"].to_numpy() == sg["fold"].to_numpy()).all()

    st_o, st_g = so["status"].astype(str).to_numpy(), sg["status"].astype(str).to_numpy()
    inv[inv_key]["n_tgs"] = len(so)
    inv[inv_key]["status_mismatch"] = int((st_o != st_g).sum())
    inv[inv_key]["status_counts_orig"] = pd.Series(st_o).value_counts().to_dict()
    inv[inv_key]["status_counts_geo"] = pd.Series(st_g).value_counts().to_dict()

    f = pd.DataFrame({"tg_id": so.index, "tg_lat": so["tg_lat"].to_numpy(),
                      "tg_lon": so["tg_lon"].to_numpy(), "fold": so["fold"].to_numpy()})
    f["net"], f["network"], f["regime"], f["pair"], f["run_id"] = net, NETS[net], REGIMES[regime], pair, run_id
    # Site keyed on the mesh run id so a site is the same key in both regimes.
    f["site_key"] = SITES.site_key(f, run_id=f"{net}-mesh").to_numpy()
    f["has_x"] = f["tg_id"].map(hx).to_numpy()
    for tag, s in (("base", so), ("loso", sg)):  # base = original, loso = GEO (site_bootstrap's names)
        solved = (solved_mask(s) & s["pred_lat"].notna()).to_numpy()
        lab = s["cell_label"].to_numpy()
        off = s[C.GRID_OFFSET].to_numpy()
        f[f"{tag}_status"] = s["status"].astype(str).to_numpy()
        f[f"{tag}_solved"] = solved
        f[f"{tag}_correct"] = solved & (lab == "correct")
        f[f"{tag}_r1"] = solved & (lab == "correct") & (off >= 0) & (off <= 1)
        f[f"{tag}_r2"] = solved & (lab == "correct") & (off >= 0) & (off <= 2)
        f[f"{tag}_err_km"] = np.where(solved, s["pred_dist_to_tg_km"].to_numpy(float), np.nan)
        f[f"{tag}_pred_lat"] = s["pred_lat"].to_numpy(float)
        f[f"{tag}_pred_lon"] = s["pred_lon"].to_numpy(float)
        for c in TIMING:
            f[f"{tag}_{c}"] = s[c].to_numpy(float)
    both = f["base_pred_lat"].notna() & f["loso_pred_lat"].notna()
    disp = np.full(len(f), np.nan)
    disp[both.to_numpy()] = elementwise_km(
        f.loc[both, "base_pred_lat"].to_numpy(), f.loc[both, "base_pred_lon"].to_numpy(),
        f.loc[both, "loso_pred_lat"].to_numpy(), f.loc[both, "loso_pred_lon"].to_numpy())
    f["disp_km"] = disp
    f["disp_solved_km"] = np.where(f["base_solved"] & f["loso_solved"], disp, np.nan)
    return f.reset_index(drop=True)


# ---- statistics ----------------------------------------------------------------


def boot_extra(g: pd.DataFrame, n_boot=N_BOOT, seed=BOOT_SEED) -> dict:
    """Site-clustered CI for the ring-bounded deltas (same resampling as site_bootstrap)."""
    _, inv = np.unique(g["site_key"].to_numpy(), return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(inv.max() + 1)]
    arr = {k: g[k].to_numpy(float) for k in ("base_r1", "loso_r1", "base_r2", "loso_r2")}
    rng = np.random.default_rng(seed)
    d1, d2 = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        d1[b] = arr["loso_r1"][idx].mean() - arr["base_r1"][idx].mean()
        d2[b] = arr["loso_r2"][idx].mean() - arr["base_r2"][idx].mean()
    return {"d_r1_ci_lo": np.percentile(d1, 2.5), "d_r1_ci_hi": np.percentile(d1, 97.5),
            "d_r2_ci_lo": np.percentile(d2, 2.5), "d_r2_ci_hi": np.percentile(d2, 97.5)}


def stats(g: pd.DataFrame) -> dict:
    bc, gc = g["base_correct"].to_numpy(bool), g["loso_correct"].to_numpy(bool)
    be, ge = g["base_err_km"].to_numpy(float), g["loso_err_km"].to_numpy(float)
    n = len(g)
    r = {"n_tgs": n, "n_sites": g["site_key"].nunique(),
         "unans_orig": float((~g["base_solved"]).mean()), "unans_geo": float((~g["loso_solved"]).mean()),
         "acc_orig": bc.mean(), "acc_geo": gc.mean(), "d_acc": gc.mean() - bc.mean(),
         "r1_orig": g["base_r1"].mean(), "r1_geo": g["loso_r1"].mean(),
         "r2_orig": g["base_r2"].mean(), "r2_geo": g["loso_r2"].mean()}
    r["d_r1"], r["d_r2"] = r["r1_geo"] - r["r1_orig"], r["r2_geo"] - r["r2_orig"]
    for q in (25, 50, 75, 90):
        r[f"p{q}_orig"], r[f"p{q}_geo"] = _quantile(be, q / 100), _quantile(ge, q / 100)
        r[f"d_p{q}"] = r[f"p{q}_geo"] - r[f"p{q}_orig"]
    c2w, w2c = bc & ~gc, ~bc & gc
    r.update({"n_c2w": int(c2w.sum()), "n_w2c": int(w2c.sum()),
              "sh_c2w": c2w.mean(), "sh_w2c": w2c.mean(),
              "sites_c2w": g.loc[c2w, "site_key"].nunique(), "sites_w2c": g.loc[w2c, "site_key"].nunique(),
              "sites_any_flip": g.loc[c2w | w2c, "site_key"].nunique()})
    d = g["disp_solved_km"].to_numpy(float)
    r["disp_p50"], r["disp_p90"] = _quantile(d, 0.5), _quantile(d, 0.9)
    r["disp_share_lt1km"] = float((d[np.isfinite(d)] < 1).mean())
    return r


def summarize(long: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in long.groupby(by, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(by, key))
        row.update(stats(g))
        row.update(site_bootstrap(g))
        row.update(boot_extra(g))
        rows.append(row)
    return pd.DataFrame(rows)


def drop_table(long: pd.DataFrame, n_boot=N_BOOT, seed=BOOT_SEED) -> pd.DataFrame:
    """K-fold -> LOSO drop per arm and the diff-in-diff (GEO drop - orig drop), site CI."""
    rows = []
    for pair, g in long.groupby("pair", sort=False):
        m = g[g["regime"] == "K-fold"].set_index(["net", "tg_id"])
        l = g[g["regime"] == "LOSO"].set_index(["net", "tg_id"]).loc[m.index]
        keys = m["site_key"].to_numpy()
        assert (keys == l["site_key"].to_numpy()).all()
        A = {"mo_c": m["base_correct"], "mg_c": m["loso_correct"], "lo_c": l["base_correct"], "lg_c": l["loso_correct"],
             "mo_e": m["base_err_km"], "mg_e": m["loso_err_km"], "lo_e": l["base_err_km"], "lg_e": l["loso_err_km"]}
        A = {k: v.to_numpy(float) for k, v in A.items()}

        def calc(idx):
            acc = {k: A[k][idx].mean() for k in ("mo_c", "mg_c", "lo_c", "lg_c")}
            p50 = {k: _quantile(A[k][idx], 0.5) for k in ("mo_e", "mg_e", "lo_e", "lg_e")}
            drop_acc_o, drop_acc_g = acc["lo_c"] - acc["mo_c"], acc["lg_c"] - acc["mg_c"]
            drop_p50_o, drop_p50_g = p50["lo_e"] - p50["mo_e"], p50["lg_e"] - p50["mg_e"]
            return [drop_acc_o, drop_acc_g, drop_acc_g - drop_acc_o, drop_p50_o, drop_p50_g, drop_p50_g - drop_p50_o,
                    acc["mo_c"], acc["lo_c"], acc["mg_c"], acc["lg_c"], p50["mo_e"], p50["lo_e"], p50["mg_e"], p50["lg_e"]]

        allidx = np.arange(len(keys))
        point = calc(allidx)
        _, inv = np.unique(keys, return_inverse=True)
        members = [np.flatnonzero(inv == i) for i in range(inv.max() + 1)]
        rng = np.random.default_rng(seed)
        B = np.array([calc(np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))]))
                      for _ in range(n_boot)])
        names = ["drop_acc_orig", "drop_acc_geo", "did_acc", "drop_p50_orig", "drop_p50_geo", "did_p50",
                 "acc_kfold_orig", "acc_loso_orig", "acc_kfold_geo", "acc_loso_geo",
                 "p50_kfold_orig", "p50_loso_orig", "p50_kfold_geo", "p50_loso_geo"]
        row = {"pair": pair, "n_tgs": len(keys), "n_sites": len(members)}
        for i, nm in enumerate(names):
            row[nm] = point[i]
            if i < 6:
                row[nm + "_ci_lo"], row[nm + "_ci_hi"] = np.nanpercentile(B[:, i], 2.5), np.nanpercentile(B[:, i], 97.5)
        rows.append(row)
    return pd.DataFrame(rows)


def timing_table(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pair, regime), g in long.groupby(["pair", "regime"], sort=False):
        for tag, arm in (("base", pair), ("loso", pair + "-GEO")):
            ctr = g[f"{tag}_ctr_ms"].to_numpy(float)
            tot = (g[f"{tag}_ltd_ms"] + g[f"{tag}_mtl_ms"] + g[f"{tag}_ctr_ms"]).to_numpy(float)
            rows.append({"regime": regime, "method": arm, "n": len(g),
                         "ctr_p50_ms": _quantile(ctr, .5), "ctr_p95_ms": _quantile(ctr, .95),
                         "total_p50_ms": _quantile(tot, .5), "total_p95_ms": _quantile(tot, .95)})
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    inv: dict = {}
    frames = []
    for net in NETS:
        hx = has_x_map(net)
        for regime in REGIMES:
            space = space_for(f"{net}-{regime}")
            for pair in PAIRS:
                frames.append(paired_frame(net, regime, pair, space, hx, inv))
    long = pd.concat(frames, ignore_index=True)
    long["has_x"] = long["has_x"].fillna("unknown")

    (OUT / "invariants.json").write_text(json.dumps(inv, indent=2, default=str) + "\n")
    per_run = summarize(long, ["pair", "regime", "network"])
    pooled = summarize(long, ["pair", "regime"])
    by_x = summarize(long, ["pair", "regime", "has_x"])
    drop = drop_table(long)
    timing = timing_table(long)
    for name, df in (("per_run", per_run), ("pooled", pooled), ("by_has_x", by_x),
                     ("kfold_to_loso_drop", drop), ("timing", timing)):
        df.to_csv(OUT / f"{name}.csv", index=False)

    # Invariant digest
    bad = {k: v for k, v in inv.items()
           if v["missing"] or v["tg_set_mismatch"] or v["config_diff"] or v["status_mismatch"]
           or any(v[f"reproduce_stored_{PAIRS[k.split('/')[1]][0]}"].values())}
    print("folds:", {k: v["n_folds"] for k, v in inv.items()})
    print("invariant failures:", json.dumps(bad, indent=1, default=str) if bad else "none")
    pd.set_option("display.width", 250, "display.max_columns", 80)
    for name, df in (("pooled", pooled), ("per_run", per_run), ("by_has_x", by_x),
                     ("drop", drop), ("timing", timing)):
        print(f"\n== {name}\n", df.round(4).to_string())


if __name__ == "__main__":
    main()
