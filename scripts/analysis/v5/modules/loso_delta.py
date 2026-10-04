"""Seen site vs unseen site: each method's K-fold run against its LOSO twin.

The mesh runs split folds per IP with DistGeo, and ~20 IP replicas share every
site, so a test TG's own site is almost always in its fit set: a calibrated
method can recall the site's label rather than infer it from RTT. The LOSO
twin (`fold_by: site`, k = number of sites) holds every site out whole. This
report pairs the two runs **on the same TGs** and asks, per method, what the
held-out site costs.

## Pairing, and what is refused

`--pair BASE:LOSO` names a K-fold run and its leave-one-site-out twin. For
each method both runs' `classify` frames are joined on `tg_id`, and the pair is
refused unless:

* the two runs score the same TGs, at the same coordinates, in the same cell
  (`tg_seed_id`) -- otherwise the delta compares two answer spaces;
* the parameter-free methods (`PARAMETER_FREE`: S-P and SOI, which fit
  nothing) predict **identically** in both runs. They cannot change with the
  split, so a difference means the runs were not built from the same inputs.

A method missing from either run is refused rather than dropped, as in every
`_cross/` figure.

## The numbers

Per method, over every TG (the `classify` denominator: unanswered rows are
wrong, not excluded):

* cell accuracy in each run and its change, `d_acc = loso - base`;
* error p50/p90 on solved rows (`status.solved_mask`, the error-CDF
  convention), and the change in p50;
* the cell transitions -- correct->wrong (a seen-site win that needed the
  site) and wrong->correct.

`d_acc` and `d_p50_km` carry a **paired, site-clustered bootstrap** CI: sites
are resampled with replacement and every TG of a drawn site comes along in
both runs, because ~20 replicas at one site are closer to one observation than
to twenty. TG counts sit beside site counts in every table for the same
reason.

## Breakdowns

* `by_distance`: each site's distance to the nearest *other* site of its run
  -- the closest place the LOSO calibration still holds -- in `DISTANCE_BINS`.
  On the pro meshes the median is 200-350 km, so LOSO mostly tests an unseen
  metro, and a held-out site with a same-metro neighbour still has one.
* `by_has_x`: `report-sp-pni-cells`' `tg_cell_holds_x` -- has-X (the TG's
  cell holds its nearest interconnect) vs no-X -- when the base run carries
  exactly one `sp_pni_cells_tgs.csv`. The flag, not the cluster id: each run
  numbers its `plot-pni-gap` clusters independently, so a pooled "C1" would
  mix different clusters. Skipped, and said so in the manifest, otherwise.

## Pooling

Every pair's rows are stacked (`<base_run>::<tg_id>`) and summarised again: a
micro-average, so a dataset weighs by its TG count. Site keys carry the run id,
so two meshes' sites never merge.

Command: `report-loso-delta`. Needs `classify` on both runs of every pair.
Writes `_cross/loso-delta/<n>-runs-<hash>/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules import sites as SITES
from scripts.analysis.v5.modules.figure_outcome_map import scored_methods
from scripts.analysis.v5.modules.geodesy import pairwise_km
from scripts.analysis.v5.modules.methods import method_label, method_order
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths
from scripts.analysis.v5.modules.status import SHORTEST_PING, solved_mask

#: `_cross/<KIND>/<n>-runs-<hash>/`.
KIND = "loso-delta"

#: The rung whose `*_tgs.parquet` is read. v5 runs one; read off the ladder.
SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: Methods that fit nothing, so the split cannot move them. Required to be
#: identical across a pair: they are the check that both runs saw one input.
PARAMETER_FREE: tuple[str, ...] = (SHORTEST_PING, "million_scale_cbg")

#: Coordinates closer than this (degrees) are the same prediction. Parquet
#: round-trips floats exactly; this only absorbs a recomputation's last ulp.
COORD_TOL_DEG = 1e-9

#: Nearest-other-site distance bins (km), right-open. The last bin is open.
DISTANCE_BINS: tuple[float, ...] = (0.0, 50.0, 200.0, 400.0, np.inf)
DISTANCE_LABELS: tuple[str, ...] = ("<50 km", "50-200 km", "200-400 km", ">=400 km")

N_BOOT = 2000
BOOT_SEED = 0
CI = (2.5, 97.5)

#: Separator in the pooled index; occurs in neither a run id nor a TG id.
RUN_KEY_SEP = "::"
POOLED = "pooled"

NAMES = {
    "summary": "loso_delta.csv",
    "by_distance": "loso_delta.by_distance.csv",
    "by_has_x": "loso_delta.by_has_x.csv",
    "transitions": "loso_delta.transitions.csv",
    "membership": "loso_delta.membership.csv",
    "manifest": "loso_delta.manifest.json",
}

_TG_COLUMNS = (
    "tg_id", "tg_lat", "tg_lon", "tg_seed_id", "status", "pred_lat", "pred_lon",
    "pred_dist_to_tg_km", "cell_label",
)


# ---- loading -----------------------------------------------------------------


def parse_pair(spec: str) -> tuple[str, str]:
    """`"BASE:LOSO"` -> `(base, loso)`."""
    base, sep, loso = spec.partition(":")
    if not sep or not base or not loso or ":" in loso:
        raise ValueError(f"--pair takes BASE:LOSO (two run ids), got {spec!r}")
    if base == loso:
        raise ValueError(f"--pair {spec!r} pairs a run with itself")
    return base, loso


def _read(run: RunPaths, method: str, nside: int, root: Path | None) -> pd.DataFrame:
    path = run.classify_dir(nside, root=root) / C.TGS_PARQUET.format(method=method)
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `classify --run-id {run.run_id}` first")
    df = pd.read_parquet(path, columns=list(_TG_COLUMNS))
    if df["tg_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate tg_id")
    return df.set_index("tg_id")


def pair_methods(
    base: RunPaths, loso: RunPaths, methods: list[str] | None, nside: int,
    root: Path | None,
) -> list[str]:
    """Methods scored in both runs, `TERM_ORDER`; a requested one missing is refused."""
    have = {
        r.run_id: set(scored_methods(r, nside, analysis_root=root)) for r in (base, loso)
    }
    if methods:
        missing = {r: sorted(set(methods) - ms) for r, ms in have.items() if set(methods) - ms}
        if missing:
            raise ValueError(f"requested methods not scored: {missing}")
        have = {r: ms & set(methods) for r, ms in have.items()}
    common = cross.guard_common_methods(
        have, remedy="classify both runs of the pair, or pass --method."
    )
    return method_order(common)


def load_pair(
    base: RunPaths, loso: RunPaths, methods: list[str], *, nside: int = SOURCE_NSIDE,
    root: Path | None = None,
) -> pd.DataFrame:
    """One row per (method, TG): both runs' label, error and prediction.

    Refuses a pair whose TG set, coordinates or cells differ, and one whose
    parameter-free methods moved.
    """
    rows = []
    for method in methods:
        b, l = _read(base, method, nside, root), _read(loso, method, nside, root)
        if set(b.index) != set(l.index):
            only_b, only_l = sorted(set(b.index) - set(l.index)), sorted(set(l.index) - set(b.index))
            raise ValueError(
                f"{method}: {base.run_id} and {loso.run_id} score different TGs "
                f"({len(only_b)} only in base, e.g. {only_b[:3]}; {len(only_l)} only "
                f"in LOSO, e.g. {only_l[:3]}). A LOSO twin must reuse the base CSV."
            )
        l = l.loc[b.index]
        for col in ("tg_lat", "tg_lon", "tg_seed_id"):
            if not np.array_equal(b[col].to_numpy(), l[col].to_numpy()):
                raise ValueError(
                    f"{method}: {col} differs between {base.run_id} and {loso.run_id}; "
                    f"the two answer spaces are not the same, so a cell delta is meaningless."
                )
        frame = pd.DataFrame(
            {
                "base_run": base.run_id,
                "loso_run": loso.run_id,
                "method": method,
                "tg_id": b.index,
                "tg_lat": b["tg_lat"].to_numpy(),
                "tg_lon": b["tg_lon"].to_numpy(),
                "tg_seed_id": b["tg_seed_id"].to_numpy(),
            }
        )
        for tag, df in (("base", b), ("loso", l)):
            solved = solved_mask(df).to_numpy()
            frame[f"{tag}_status"] = df["status"].astype(str).to_numpy()
            frame[f"{tag}_label"] = df["cell_label"].astype(str).to_numpy()
            frame[f"{tag}_correct"] = (df["cell_label"] == "correct").to_numpy()
            frame[f"{tag}_solved"] = solved
            frame[f"{tag}_err_km"] = np.where(solved, df["pred_dist_to_tg_km"].to_numpy(float), np.nan)
            frame[f"{tag}_pred_lat"] = df["pred_lat"].to_numpy(float)
            frame[f"{tag}_pred_lon"] = df["pred_lon"].to_numpy(float)
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    guard_parameter_free(out)
    out["site_key"] = SITES.site_key(out, run_id=base.run_id).to_numpy()
    out["nearest_site_km"] = out["site_key"].map(nearest_other_site_km(out))
    return out


def guard_parameter_free(long: pd.DataFrame) -> None:
    """S-P and SOI must predict identically in both runs of a pair."""
    for method in PARAMETER_FREE:
        rows = long[long["method"] == method]
        if rows.empty:
            continue
        same_status = rows["base_status"].to_numpy() == rows["loso_status"].to_numpy()
        both_nan = rows[["base_pred_lat", "loso_pred_lat"]].isna().all(axis=1).to_numpy()
        d_lat = np.abs(rows["base_pred_lat"].to_numpy() - rows["loso_pred_lat"].to_numpy())
        d_lon = np.abs(rows["base_pred_lon"].to_numpy() - rows["loso_pred_lon"].to_numpy())
        same_pred = both_nan | ((d_lat <= COORD_TOL_DEG) & (d_lon <= COORD_TOL_DEG))
        bad = ~(same_status & same_pred)
        if bad.any():
            ex = rows.loc[bad, "tg_id"].head(3).tolist()
            raise ValueError(
                f"{method_label(method)} fits nothing, yet {int(bad.sum())} of its "
                f"predictions differ between {rows['base_run'].iat[0]} and "
                f"{rows['loso_run'].iat[0]} (e.g. {ex}). The runs did not see the "
                f"same input; rebuild the LOSO twin from the base config's CSV."
            )


def nearest_other_site_km(long: pd.DataFrame) -> pd.Series:
    """`site_key -> km` to the nearest other site of the same run."""
    sites = long.drop_duplicates("site_key")[["site_key", "tg_lat", "tg_lon"]]
    if len(sites) < 2:
        return pd.Series(np.inf, index=sites["site_key"].to_numpy())
    d = pairwise_km(sites["tg_lat"].to_numpy(), sites["tg_lon"].to_numpy())
    np.fill_diagonal(d, np.inf)
    return pd.Series(d.min(axis=1), index=sites["site_key"].to_numpy())


def distance_bin(km: pd.Series) -> pd.Series:
    return pd.cut(km, bins=list(DISTANCE_BINS), labels=list(DISTANCE_LABELS), right=False)


HAS_X, NO_X = "has-X", "no-X"


def attach_has_x(long: pd.DataFrame, base: RunPaths, root: Path | None) -> tuple[pd.DataFrame, str | None]:
    """Add `has_x` from the base run's one `sp_pni_cells_tgs.csv`, if it has one."""
    found = sorted(base.analysis_dir("pni-gap", root=root).glob("*/sp_pni_cells_tgs.csv"))
    if len(found) != 1:
        return long.assign(has_x=pd.NA), None
    c = pd.read_csv(found[0], usecols=["tg_id", "tg_cell_holds_x"]).drop_duplicates("tg_id")
    c["has_x"] = np.where(c["tg_cell_holds_x"].astype(bool), HAS_X, NO_X)
    return long.merge(c[["tg_id", "has_x"]], on="tg_id", how="left"), str(found[0])


# ---- the numbers -------------------------------------------------------------


def _quantile(values: np.ndarray, q: float) -> float:
    v = values[np.isfinite(values)]
    return float(np.quantile(v, q)) if v.size else float("nan")


def _stats(g: pd.DataFrame) -> dict:
    """One method's numbers over the rows of `g` (one pair, or pooled)."""
    be, le = g["base_err_km"].to_numpy(float), g["loso_err_km"].to_numpy(float)
    bc, lc = g["base_correct"].to_numpy(bool), g["loso_correct"].to_numpy(bool)
    n = len(g)
    return {
        "n_tgs": n,
        "n_sites": int(g["site_key"].nunique()),
        "acc_base": float(bc.mean()) if n else float("nan"),
        "acc_loso": float(lc.mean()) if n else float("nan"),
        "d_acc": float(lc.mean() - bc.mean()) if n else float("nan"),
        "p50_base_km": _quantile(be, 0.5),
        "p50_loso_km": _quantile(le, 0.5),
        "d_p50_km": _quantile(le, 0.5) - _quantile(be, 0.5),
        "p90_base_km": _quantile(be, 0.9),
        "p90_loso_km": _quantile(le, 0.9),
        "unanswered_base": float((~g["base_solved"].to_numpy(bool)).mean()) if n else float("nan"),
        "unanswered_loso": float((~g["loso_solved"].to_numpy(bool)).mean()) if n else float("nan"),
        "n_correct_to_wrong": int((bc & ~lc).sum()),
        "n_wrong_to_correct": int((~bc & lc).sum()),
        "n_sites_correct_to_wrong": int(g.loc[bc & ~lc, "site_key"].nunique()),
    }


def site_bootstrap(
    g: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> dict:
    """Paired, site-clustered CI for `d_acc` and `d_p50_km` over one method's rows.

    Each replicate draws sites with replacement; a drawn site contributes all
    of its TGs, in both runs, so the pairing survives the resampling.
    """
    keys = g["site_key"].to_numpy()
    sites, inv = np.unique(keys, return_inverse=True)
    if len(sites) < 2 or n_boot <= 0:
        return {f"{m}_ci_{s}": float("nan") for m in ("d_acc", "d_p50_km") for s in ("lo", "hi")}
    members = [np.flatnonzero(inv == i) for i in range(len(sites))]
    bc = g["base_correct"].to_numpy(float)
    lc = g["loso_correct"].to_numpy(float)
    be = g["base_err_km"].to_numpy(float)
    le = g["loso_err_km"].to_numpy(float)
    rng = np.random.default_rng(seed)
    d_acc = np.empty(n_boot)
    d_p50 = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(sites), len(sites))])
        d_acc[b] = lc[idx].mean() - bc[idx].mean()
        d_p50[b] = _quantile(le[idx], 0.5) - _quantile(be[idx], 0.5)
    lo, hi = CI
    return {
        "d_acc_ci_lo": float(np.percentile(d_acc, lo)),
        "d_acc_ci_hi": float(np.percentile(d_acc, hi)),
        "d_p50_km_ci_lo": float(np.nanpercentile(d_p50, lo)),
        "d_p50_km_ci_hi": float(np.nanpercentile(d_p50, hi)),
    }


def summary_table(
    long: pd.DataFrame, scope: str, *, by: str | None = None,
    n_boot: int = N_BOOT, seed: int = BOOT_SEED,
) -> pd.DataFrame:
    """One row per method (and per `by` value), with the bootstrap CI."""
    rows = []
    keys = ["method"] + ([by] if by else [])
    for key, g in long.groupby(keys, sort=False, observed=True, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = {"scope": scope, "method": key[0], "method_label": method_label(key[0])}
        if by:
            row[by] = key[1]
        row.update(_stats(g))
        row.update(site_bootstrap(g, n_boot=n_boot, seed=seed))
        rows.append(row)
    out = pd.DataFrame(rows)
    order = {m: i for i, m in enumerate(method_order(long["method"].unique()))}
    sort = ["_o"] + ([by] if by else [])
    return out.assign(_o=out["method"].map(order)).sort_values(sort).drop(columns="_o").reset_index(drop=True)


def transition_table(long: pd.DataFrame, scope: str) -> pd.DataFrame:
    """Per method, the 3x3 table of base label -> LOSO label (counts + sites)."""
    t = (
        long.groupby(["method", "base_label", "loso_label"], observed=True)
        .agg(n_tgs=("tg_id", "size"), n_sites=("site_key", "nunique"))
        .reset_index()
    )
    t.insert(0, "scope", scope)
    return t


def stack(longs: list[pd.DataFrame]) -> pd.DataFrame:
    """Every pair as one population, TGs keyed `<base_run>::<tg_id>`."""
    out = pd.concat(longs, ignore_index=True)
    out["tg_id"] = out["base_run"] + RUN_KEY_SEP + out["tg_id"].astype(str)
    return out


# ---- artifacts ---------------------------------------------------------------


def build(
    pairs: list[tuple[RunPaths, RunPaths]],
    *,
    methods: list[str] | None = None,
    nside: int = SOURCE_NSIDE,
    analysis_root: Path | None = None,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    source: str = "all",
) -> dict[str, Path]:
    """Every table and the manifest for these pairs. Returns `kind -> path`."""
    if not pairs:
        raise ValueError("pass at least one --pair BASE:LOSO")
    nside = G.validate_nside(nside)
    run_ids = [r.run_id for pair in pairs for r in pair]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"a run appears in more than one pair: {run_ids}")

    method_sets = [pair_methods(b, l, methods, nside, analysis_root) for b, l in pairs]
    if len({tuple(m) for m in method_sets}) > 1:
        raise ValueError(
            f"the pairs score different methods {dict(zip([b.run_id for b, _ in pairs], method_sets))}; "
            f"pass --method to pick a common set."
        )
    chosen = method_sets[0]

    longs, has_x_sources = [], {}
    for base, loso in pairs:
        long = load_pair(base, loso, chosen, nside=nside, root=analysis_root)
        long, src = attach_has_x(long, base, analysis_root)
        has_x_sources[base.run_id] = src
        longs.append(long)

    scoped = [(f"{b.run_id}->{l.run_id}", lg) for (b, l), lg in zip(pairs, longs)]
    if len(pairs) > 1:
        scoped.append((POOLED, stack(longs)))

    summary, by_dist, by_has_x, transitions = [], [], [], []
    have_has_x = all(s is not None for s in has_x_sources.values())
    for scope, lg in scoped:
        summary.append(summary_table(lg, scope, n_boot=n_boot, seed=seed))
        by_dist.append(summary_table(
            lg.assign(nearest_site_bin=distance_bin(lg["nearest_site_km"])), scope,
            by="nearest_site_bin", n_boot=n_boot, seed=seed,
        ))
        if have_has_x:
            by_has_x.append(summary_table(lg, scope, by="has_x", n_boot=n_boot, seed=seed))
        transitions.append(transition_table(lg, scope))

    out_dir = cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)
    written = {k: out_dir / v for k, v in NAMES.items()}
    pd.concat(summary, ignore_index=True).to_csv(written["summary"], index=False)
    pd.concat(by_dist, ignore_index=True).to_csv(written["by_distance"], index=False)
    if by_has_x:
        pd.concat(by_has_x, ignore_index=True).to_csv(written["by_has_x"], index=False)
    else:
        written.pop("by_has_x")
    pd.concat(transitions, ignore_index=True).to_csv(written["transitions"], index=False)
    member_cols = [
        "base_run", "loso_run", "method", "tg_id", "site_key", "nearest_site_km", "has_x",
        "base_label", "loso_label", "base_err_km", "loso_err_km", "base_status", "loso_status",
    ]
    pd.concat(longs, ignore_index=True)[member_cols].to_csv(written["membership"], index=False)
    written["manifest"].write_text(_manifest(
        pairs, chosen, source=source, nside=nside, n_boot=n_boot, seed=seed,
        has_x_sources=has_x_sources, names={k: p.name for k, p in written.items()},
    ))
    return written


def _manifest(pairs, methods, *, source, nside, n_boot, seed, has_x_sources, names) -> str:
    body = {
        "tables": {k: v for k, v in names.items() if k != "manifest"},
        "pairs": [{"base": b.run_id, "loso": l.run_id} for b, l in pairs],
        "methods": methods,
        "methods_source": source,
        "method_labels": {m: method_label(m) for m in methods},
        "nside": nside,
        "denominator": (
            "every TG classify scored; unanswered rows are wrong on the cell axis "
            "and excluded from the error percentiles (status.solved_mask)."
        ),
        "guards": {
            "same_tgs": "identical tg_id set, tg_lat/tg_lon and tg_seed_id in both runs",
            "parameter_free_identical": (
                f"{[method_label(m) for m in PARAMETER_FREE]} fit nothing, so their "
                f"status and prediction must match exactly across a pair "
                f"(tolerance {COORD_TOL_DEG} deg)."
            ),
        },
        "bootstrap": {
            "kind": "paired, site-clustered: sites resampled with replacement, every TG of a drawn site in both runs",
            "n_boot": n_boot, "seed": seed, "ci_percentiles": list(CI),
        },
        "distance_bins_km": {
            "edges": [e if np.isfinite(e) else "inf" for e in DISTANCE_BINS],
            "labels": list(DISTANCE_LABELS),
            "meaning": "each site's great-circle distance to the nearest other site of its run",
        },
        "has_x_sources": has_x_sources,
        "pooling": (
            f"micro: every pair's rows stacked (<base_run>{RUN_KEY_SEP}<tg_id>) and "
            "summarised again; site keys carry the run id."
        ) if len(pairs) > 1 else None,
        "site_caveat": (
            "~20 IP replicas share a site; n_sites beside every n_tgs is the "
            "effective sample size, and the CI resamples sites, not TGs."
        ),
    }
    return json.dumps(body, indent=2, default=str) + "\n"
