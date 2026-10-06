"""A one-phase variant against its original, under both regimes -- `report-variant-delta`.

`octant_cbg_hull_geo` / `octant_cbg_spl_geo` are OCT-H / OCT-S with one change:
the EST phase takes the geometric centroid instead of the Monte Carlo medoid.
Calibration and multilateration are the same code, so any difference between
the two arms is the swapped phase alone. The paper's appendix reports that
difference; this module is where its numbers come from (it replaces the
one-off `tasks/20261004-geo-centroid-variants/compare_geo.py`).

## Scored in memory, never written into `classify/`

The variant combos are not in any config's `combo_ids`, so `classify` never
scores them and no figure draws them. This module scores both arms of every
pair itself, with the v5 scorer (`classify.load_method_frame` +
`score_method`) against the run's own answer space, so labels, denominators
and the solved mask are exactly `classify`'s. It reads the original's stored
`*_tgs.parquet` only to check that its in-memory scoring reproduces it.

## Pairing and what is refused

`--pair BASE:LOSO` names a seen-site run and its unseen-site twin; both are
compared. Per run and variant, the two arms must cover the same TGs in the
same folds, and their `run.json` must agree on everything but the EST phase
and the run's own bookkeeping (`RUNJSON_IGNORE`). A difference is refused.

## The numbers

Per `(variant, regime, scope)` -- scope each network and `pooled` -- and again
split by has-X / no-X when the base run has `report-sp-pni-cells` output:

* accuracy of both arms (unanswered = wrong) and their difference in pp;
* correct->wrong and wrong->correct TGs, and the sites where any TG flips
  (`sites_flip_pct`, "the accuracy changes at no more than 5% of the sites");
* the median error of both arms over every TG, unanswered ranked last;
* how far the variant moves the prediction (`disp_p50_km`, max), and
  normalized when the configs declare `analysis.common.dist_norm_km`.

`variant_delta.drop.csv`: per variant and scope, each arm's change from seen to
unseen sites, their difference (`did_acc_pp`), and the sites whose drop differs
between the arms. `variant_delta.timing.csv`: per variant and regime, the EST
phase and pipeline runtime of both arms and the EST ratio.

No bootstrap, as in `report-loso-delta`: with ~20 replicas per site the
effective sample is the sites, and the site counts are written beside every
share instead.

Command: `report-variant-delta`. Needs `build-answer-space` on both runs of
every pair (and `classify` for the reproduction check). Writes
`_cross/variant-delta/<n>-runs-<hash>/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import dist_norm as DN
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules import sites as SITES
from scripts.analysis.v5.modules.geodesy import elementwise_km
from scripts.analysis.v5.modules.loso_delta import _roster_median, attach_has_x
from scripts.analysis.v5.modules.map_answer_space import load_rung
from scripts.analysis.v5.modules.methods import method_label
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import solved_mask

KIND = "variant-delta"
SOURCE_NSIDE = G.NSIDE_LADDER[0]
POOLED = "pooled"
SEEN, UNSEEN = "seen", "unseen"

#: `original -> variant`, the paper's ablation.
DEFAULT_VARIANTS: dict[str, str] = {
    "octant_cbg_hull": "octant_cbg_hull_geo",
    "octant_cbg_spl": "octant_cbg_spl_geo",
}

#: The per-phase runtime columns read beside the scoring columns.
TIMING = ("ltd_ms", "mtl_ms", "ctr_ms")

#: `run.json` keys allowed to differ between the arms: the EST phase itself and
#: the run's own bookkeeping (timestamps, process memory, the fit's timing).
RUNJSON_IGNORE = frozenset({
    "combo_id", "ctr", "ctr_kwargs", "started_at", "fit_ms", "fit_alloc_peak_bytes",
    "fit_heap_peak_bytes", "fit_rss_peak_bytes", "run_baseline_rss_bytes",
    "rss_after_inputs_bytes", "rss_after_fit_bytes", "run_peak_rss_bytes",
})

#: Columns of the stored parquet the reproduction check compares.
_REPRODUCE = ("cell_label", "status", C.GRID_OFFSET, "tg_seed_id", "pred_seed_id",
              "pred_dist_to_tg_km")

NAMES = {
    "summary": "variant_delta.csv",
    "by_has_x": "variant_delta.by_has_x.csv",
    "drop": "variant_delta.drop.csv",
    "timing": "variant_delta.timing.csv",
    "manifest": "variant_delta.manifest.json",
}


def parse_variant(spec: str) -> tuple[str, str]:
    """`"ORIGINAL=VARIANT"` -> `(original, variant)`."""
    orig, sep, var = spec.partition("=")
    if not sep or not orig or not var or orig == var:
        raise ValueError(f"--variant takes ORIGINAL=VARIANT (two combo ids), got {spec!r}")
    return orig, var


# ---- invariants ----------------------------------------------------------------


def check_arms(run: RunPaths, orig: str, var: str) -> dict:
    """Same TGs per fold and the same `run.json` bar `RUNJSON_IGNORE`; raises otherwise."""
    folds = run.fold_ids
    for fold in folds:
        po, pv = run.combo_dir(orig, fold), run.combo_dir(var, fold)
        for p in (po, pv):
            if not (p / "targets.parquet").exists() or not (p / "run.json").exists():
                raise MissingArtifactError(f"{p} has no targets.parquet/run.json; run the benchmark combo")
        to = pd.read_parquet(po / "targets.parquet", columns=["target_id"])["target_id"]
        tv = pd.read_parquet(pv / "targets.parquet", columns=["target_id"])["target_id"]
        if sorted(to) != sorted(tv):
            raise ValueError(f"{run.run_id}/{fold}: {orig} and {var} hold different TGs")
        jo = {k: v for k, v in json.loads((po / "run.json").read_text()).items() if k not in RUNJSON_IGNORE}
        jv = {k: v for k, v in json.loads((pv / "run.json").read_text()).items() if k not in RUNJSON_IGNORE}
        diff = sorted(k for k in set(jo) | set(jv) if jo.get(k) != jv.get(k))
        if diff:
            raise ValueError(
                f"{run.run_id}/{fold}: {orig} and {var} differ beyond the EST phase in {diff}; "
                f"the comparison would not isolate the swapped phase."
            )
    return {"n_folds": len(folds)}


def reproduces_stored(run: RunPaths, method: str, scored: pd.DataFrame, nside: int,
                      root: Path | None) -> dict[str, int]:
    """Per column, the TGs where the in-memory scoring differs from `classify`'s parquet.

    Empty dict when `classify` has not scored the run (nothing to compare).
    """
    path = run.classify_dir(nside, root=root) / C.TGS_PARQUET.format(method=method)
    if not path.exists():
        return {}
    stored = pd.read_parquet(path, columns=["tg_id", *_REPRODUCE]).set_index("tg_id")
    mine = scored.set_index("tg_id").loc[stored.index]
    out = {}
    for col in _REPRODUCE:
        a, b = mine[col].to_numpy(), stored[col].to_numpy()
        if a.dtype.kind == "f" or b.dtype.kind == "f":
            same = (np.isnan(a.astype(float)) & np.isnan(b.astype(float))) | (a.astype(float) == b.astype(float))
        else:
            same = a.astype(str) == b.astype(str)
        out[col] = int((~same).sum())
    return out


# ---- the per-TG frame -------------------------------------------------------------


def paired_frame(run: RunPaths, base: RunPaths, regime: str, orig: str, var: str, *,
                 nside: int, root: Path | None) -> tuple[pd.DataFrame, dict]:
    """One row per TG of `run`: both arms' verdicts, errors, timings, and the displacement.

    Sites are keyed on `base`'s run id, so a site is the same key in both regimes.
    """
    invariants = check_arms(run, orig, var)
    space = load_rung(run, nside, analysis_root=root)
    so = C.score_method(C.load_method_frame(run, orig, columns=TIMING), space)
    sv = C.score_method(C.load_method_frame(run, var, columns=TIMING), space)
    invariants["reproduce_stored"] = reproduces_stored(run, orig, so, nside, root)
    if any(invariants["reproduce_stored"].values()):
        raise ValueError(
            f"{run.run_id}/{orig}: scoring in memory does not reproduce classify's parquet "
            f"({invariants['reproduce_stored']}); re-run `classify --run-id {run.run_id}`."
        )
    so, sv = so.set_index("tg_id"), sv.set_index("tg_id")
    if so.index.duplicated().any() or set(so.index) != set(sv.index):
        raise ValueError(f"{run.run_id}: {orig} and {var} score different TG sets")
    sv = sv.loc[so.index]
    if not (so["fold"].to_numpy() == sv["fold"].to_numpy()).all():
        raise ValueError(f"{run.run_id}: {orig} and {var} put a TG in different folds")
    f = pd.DataFrame({
        "base_run": base.run_id, "run_id": run.run_id, "regime": regime,
        "original": orig, "variant": var, "tg_id": so.index,
        "tg_lat": so["tg_lat"].to_numpy(), "tg_lon": so["tg_lon"].to_numpy(),
    })
    f["site_key"] = SITES.site_key(f, run_id=base.run_id).to_numpy()
    for tag, s in (("orig", so), ("var", sv)):
        solved = (solved_mask(s) & s["pred_lat"].notna()).to_numpy()
        f[f"{tag}_solved"] = solved
        f[f"{tag}_correct"] = solved & (s["cell_label"] == "correct").to_numpy()
        f[f"{tag}_err_km"] = np.where(solved, s["pred_dist_to_tg_km"].to_numpy(float), np.nan)
        f[f"{tag}_pred_lat"] = s["pred_lat"].to_numpy(float)
        f[f"{tag}_pred_lon"] = s["pred_lon"].to_numpy(float)
        for c in TIMING:
            f[f"{tag}_{c}"] = s[c].to_numpy(float)
    both = (f["orig_solved"] & f["var_solved"]).to_numpy()
    disp = np.full(len(f), np.nan)
    disp[both] = elementwise_km(
        f.loc[both, "orig_pred_lat"].to_numpy(), f.loc[both, "orig_pred_lon"].to_numpy(),
        f.loc[both, "var_pred_lat"].to_numpy(), f.loc[both, "var_pred_lon"].to_numpy())
    f["disp_km"] = disp
    invariants.update({
        "n_tgs": int(len(f)),
        "n_status_differs": int((so["status"].astype(str).to_numpy() != sv["status"].astype(str).to_numpy()).sum()),
    })
    return f, invariants


# ---- statistics ----------------------------------------------------------------


def _norm(km: float, bounds: tuple[float, float] | None, *, difference: bool = False) -> float:
    """`km` in the paper's normalized unit: a distance (`dist_norm.distance`) or,
    for the displacement between two predictions, a difference."""
    if bounds is None or not np.isfinite(km):
        return float("nan")
    f = DN.difference if difference else DN.distance
    return float(np.asarray(f(km, bounds)).item())


def stats(g: pd.DataFrame, bounds: tuple[float, float] | None = None) -> dict:
    """One `(variant, regime, scope)` block's numbers."""
    oc, vc = g["orig_correct"].to_numpy(bool), g["var_correct"].to_numpy(bool)
    n, n_sites = len(g), g["site_key"].nunique()
    c2w, w2c = oc & ~vc, ~oc & vc
    flip_sites = g.loc[c2w | w2c, "site_key"].nunique()
    disp = g["disp_km"].to_numpy(float)
    disp = disp[np.isfinite(disp)]
    p50o = _roster_median(g["orig_err_km"].to_numpy(float), g["orig_solved"].to_numpy(bool))
    p50v = _roster_median(g["var_err_km"].to_numpy(float), g["var_solved"].to_numpy(bool))
    d50 = float(np.median(disp)) if disp.size else float("nan")
    return {
        "n_tgs": n, "n_sites": n_sites,
        "acc_orig": float(oc.mean()), "acc_var": float(vc.mean()),
        "d_acc_pp": 100 * float(vc.mean() - oc.mean()),
        "unanswered_orig": float((~g["orig_solved"]).mean()),
        "unanswered_var": float((~g["var_solved"]).mean()),
        "n_correct_to_wrong": int(c2w.sum()), "n_wrong_to_correct": int(w2c.sum()),
        "n_sites_flip": int(flip_sites),
        "sites_flip_pct": 100 * flip_sites / n_sites if n_sites else float("nan"),
        "p50_orig_km": p50o, "p50_var_km": p50v,
        "p50_orig_norm_e3": _norm(p50o, bounds), "p50_var_norm_e3": _norm(p50v, bounds),
        "disp_p50_km": d50, "disp_max_km": float(disp.max()) if disp.size else float("nan"),
        "disp_p50_norm_e3": _norm(d50, bounds, difference=True),
    }


def summary_table(long: pd.DataFrame, by: list[str], bounds) -> pd.DataFrame:
    """One row per `by` group, scoped per network and pooled."""
    rows = []
    for scope, scoped in [*long.groupby("base_run", sort=False), (POOLED, long)]:
        for key, g in scoped.groupby(by, sort=False, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            row = {"scope": scope, **dict(zip(by, key))}
            row["original_label"] = method_label(row.get("original", ""))
            row.update(stats(g, bounds))
            rows.append(row)
    return pd.DataFrame(rows)


def drop_table(long: pd.DataFrame) -> pd.DataFrame:
    """Per variant and scope: each arm's seen -> unseen change and their difference."""
    rows = []
    for scope, scoped in [*long.groupby("base_run", sort=False), (POOLED, long)]:
        for (orig, var), g in scoped.groupby(["original", "variant"], sort=False):
            key = ["base_run", "tg_id"]
            seen = g[g["regime"] == SEEN].set_index(key)
            unseen = g[g["regime"] == UNSEEN].set_index(key)
            if seen.empty or unseen.empty:
                continue
            unseen = unseen.loc[seen.index]
            acc = {f"{tag}_{reg}": float(fr[f"{tag}_correct"].mean())
                   for reg, fr in ((SEEN, seen), (UNSEEN, unseen)) for tag in ("orig", "var")}
            per_site = {}
            for tag in ("orig", "var"):
                s = seen.groupby("site_key")[f"{tag}_correct"].mean()
                u = unseen.groupby("site_key")[f"{tag}_correct"].mean()
                per_site[tag] = (u - s).sort_index()
            differs = (per_site["orig"] - per_site["var"]).abs() > 1e-9
            rows.append({
                "scope": scope, "original": orig, "original_label": method_label(orig), "variant": var,
                "n_tgs": int(len(seen)), "n_sites": int(len(differs)),
                "acc_seen_orig": acc["orig_seen"], "acc_unseen_orig": acc["orig_unseen"],
                "acc_seen_var": acc["var_seen"], "acc_unseen_var": acc["var_unseen"],
                "drop_orig_pp": 100 * (acc["orig_unseen"] - acc["orig_seen"]),
                "drop_var_pp": 100 * (acc["var_unseen"] - acc["var_seen"]),
                "did_acc_pp": 100 * ((acc["var_unseen"] - acc["var_seen"])
                                     - (acc["orig_unseen"] - acc["orig_seen"])),
                "n_sites_drop_differs": int(differs.sum()),
                "sites_drop_differs_pct": 100 * float(differs.mean()) if len(differs) else float("nan"),
            })
    return pd.DataFrame(rows)


def timing_table(long: pd.DataFrame) -> pd.DataFrame:
    """Per variant and regime, pooled: EST and pipeline runtime of both arms."""
    rows = []
    for (orig, var, regime), g in long.groupby(["original", "variant", "regime"], sort=False):
        row = {"original": orig, "original_label": method_label(orig), "variant": var,
               "regime": regime, "n_tgs": int(len(g))}
        for tag in ("orig", "var"):
            ctr = g[f"{tag}_ctr_ms"].to_numpy(float)
            total = sum(g[f"{tag}_{c}"].to_numpy(float) for c in TIMING)
            row.update({
                f"ctr_p50_ms_{tag}": float(np.nanmedian(ctr)),
                f"ctr_mean_ms_{tag}": float(np.nanmean(ctr)),
                f"total_p50_ms_{tag}": float(np.nanmedian(total)),
                f"total_mean_ms_{tag}": float(np.nanmean(total)),
            })
        row["ctr_p50_ratio"] = row["ctr_p50_ms_orig"] / row["ctr_p50_ms_var"] if row["ctr_p50_ms_var"] > 0 else float("nan")
        row["ctr_mean_ratio"] = row["ctr_mean_ms_orig"] / row["ctr_mean_ms_var"] if row["ctr_mean_ms_var"] > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---- artifacts ---------------------------------------------------------------


def build(
    pairs: list[tuple[RunPaths, RunPaths]], *,
    variants: dict[str, str] | None = None, nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
    dist_norm_km: dict[str, tuple[float, float] | None] | None = None,
) -> dict[str, Path]:
    """Every table and the manifest for these pairs. Returns `kind -> path`."""
    if not pairs:
        raise ValueError("pass at least one --pair BASE:LOSO")
    nside = G.validate_nside(nside)
    variants = dict(variants or DEFAULT_VARIANTS)
    run_ids = [r.run_id for pair in pairs for r in pair]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"a run appears in more than one pair: {run_ids}")
    bounds = DN.common_bounds(run_ids, dist_norm_km)

    frames, invariants, has_x_sources = [], {}, {}
    for base, loso in pairs:
        pair_frames = []
        for run, regime in ((base, SEEN), (loso, UNSEEN)):
            for orig, var in variants.items():
                f, inv = paired_frame(run, base, regime, orig, var, nside=nside, root=analysis_root)
                invariants[f"{run.run_id}/{orig}={var}"] = inv
                pair_frames.append(f)
        pair_long, src = attach_has_x(pd.concat(pair_frames, ignore_index=True), base, analysis_root)
        has_x_sources[base.run_id] = src
        frames.append(pair_long)
    long = pd.concat(frames, ignore_index=True)

    by = ["original", "variant", "regime"]
    out_dir = cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)
    written = {k: out_dir / v for k, v in NAMES.items()}
    summary_table(long, by, bounds).to_csv(written["summary"], index=False)
    if all(s is not None for s in has_x_sources.values()):
        summary_table(long, [*by, "has_x"], bounds).to_csv(written["by_has_x"], index=False)
    else:
        written.pop("by_has_x")
    drop_table(long).to_csv(written["drop"], index=False)
    timing_table(long).to_csv(written["timing"], index=False)
    written["manifest"].write_text(json.dumps({
        "kind": KIND,
        "pairs": [{"seen": b.run_id, "unseen": l.run_id} for b, l in pairs],
        "variants": variants,
        "nside": nside,
        "tables": {k: p.name for k, p in written.items() if k != "manifest"},
        "invariants": invariants,
        "has_x_sources": has_x_sources,
        "dist_norm_km": DN.manifest_entry(bounds),
        "rules": {
            "correct": "cell_label == 'correct' and status.solved_mask (unanswered = wrong)",
            "median": "over every TG, unanswered ranked last (report-loso-delta's convention)",
            "displacement": "great-circle km between the two arms' predictions, both answered",
            "sites_flip": "sites with at least one TG whose verdict differs between the arms",
            "drop_differs": "sites whose seen->unseen accuracy change differs between the arms",
        },
    }, indent=2) + "\n")
    return written
