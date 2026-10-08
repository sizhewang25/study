"""The benchmark's materialized inputs: the VPs and observations a run consumed.

`inputs/benchmark/v2/<source>/<run_id>/<setup>/<fold>/` holds what the
benchmark's `materialize` stage handed each fold's combos:

* `vp_configs.parquet` -- the VP roster (`load_vps`).
* `eval_observations.parquet` -- every observation of every eval TG, one row
  per observation, **after** the source's eval-side filters (`min_obs`, the
  traffic-weighted flow mask, the presplit weight mask). That is what makes it
  the right population for the Shortest-Ping control (`shortest_ping`): S-P
  picks its VP from exactly the observations the CBG arms were scored on.

These replace the two sidecars v5 used to take from benchmark-side commands --
`materialize-target-space`'s `vps.csv` and `eval-source`'s
`eval_per_target.csv` -- so v5 runs on the benchmark tree, its inputs and the
configs alone. Verified identical on pro-as01-mesh (399/399 S-P VPs).

## The one filter the parquet does not carry

`ComboSpec.pair_weight_min` is applied by the runner *after* it reads the
parquet (`benchmark/v2/runner.py:_filter_by_weight`) and is recorded in each
combo's `run.json`. `eval_observations` reapplies it, per fold, and refuses a
fold whose combos disagree: S-P is one baseline per run, and there is no
single right population to pick it from when the CBG arms saw different ones.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths

VP_CONFIGS = "vp_configs.parquet"
EVAL_OBSERVATIONS = "eval_observations.parquet"

#: `vp_configs.parquet` -> v5 names, as `materialize-target-space` renamed
#: them for `vps.csv` (benchmark/v2/cli.py). `continent`, `region` and `city`
#: keep their names, as they did there.
_VP_RENAME = {"lat": "vp_lat", "lon": "vp_lon", "asn": "vp_asn", "country": "vp_country"}

#: What `shortest_ping` returns, one row per `(tg_id, fold)`.
SHORTEST_PING_COLUMNS = (
    "tg_id", "fold", "sping_vp_id", "sping_vp_lat", "sping_vp_lon",
    "min_rtt_ms", "min_inflation",
)


def default_root() -> Path:
    """`benchmark.v2.inputs.DEFAULT_INPUTS_ROOT`, imported lazily."""
    from scripts.benchmark.v2.inputs import DEFAULT_INPUTS_ROOT

    return Path(DEFAULT_INPUTS_ROOT)


def setup_dir(run: RunPaths, inputs_root: Path | None = None) -> Path:
    """`<inputs_root>/<source>/<run_id>/<setup>/`."""
    return Path(inputs_root or default_root()) / run.source / run.run_id / run.setup


def fold_dir(run: RunPaths, fold_id: str, inputs_root: Path | None = None) -> Path:
    """`<inputs_root>/<source>/<run_id>/<setup>/<fold>/`.

    Built from `RunPaths` rather than `benchmark.v2.inputs.inputs_dir_for`, which
    derives the same path from a live `DataSource` instance -- constructing one
    here would mean re-deriving the source kwargs this layer deliberately does
    not carry.
    """
    return setup_dir(run, inputs_root) / fold_id


def _fold_files(run: RunPaths, name: str, inputs_root: Path | None) -> list[tuple[str, Path]]:
    """`(fold_id, path)` for every fold of the run's output tree, all required.

    Keyed on the output tree's folds, not a glob of the inputs tree: a fold
    that was scored but whose inputs are gone would otherwise vanish from the
    population without a word.
    """
    folds = run.fold_ids
    if not folds:
        raise MissingArtifactError(f"{run.run_id}: no fold_* under {run.setup_dir}")
    out = [(f, fold_dir(run, f, inputs_root) / name) for f in folds]
    missing = [str(p) for _, p in out if not p.exists()]
    if missing:
        raise MissingArtifactError(
            f"{run.run_id}: {len(missing)} of {len(out)} folds have no {name} "
            f"(first: {missing[0]}); run the benchmark's materialize stage, or "
            f"pass --inputs-root"
        )
    return out


def load_vps(run: RunPaths, inputs_root: Path | None = None) -> pd.DataFrame:
    """The run's VP roster, unioned over folds: `vp_id, vp_lat, vp_lon, ...`.

    Raises if one `vp_id` sits at two coordinates in different folds -- the
    union would otherwise keep whichever fold came first.
    """
    frames = [pd.read_parquet(p) for _, p in _fold_files(run, VP_CONFIGS, inputs_root)]
    vps = pd.concat(frames, ignore_index=True).rename(columns=_VP_RENAME)
    vps["vp_id"] = vps["vp_id"].astype(str)
    coords = vps.drop_duplicates(["vp_id", "vp_lat", "vp_lon"])
    moved = coords["vp_id"].duplicated(keep=False)
    if moved.any():
        raise ValueError(
            f"{run.run_id}: {coords.loc[moved, 'vp_id'].nunique()} vp_id(s) carry "
            f"different coordinates in different folds, e.g. "
            f"{sorted(coords.loc[moved, 'vp_id'].unique())[:5]}"
        )
    return vps.drop_duplicates("vp_id").reset_index(drop=True)


def runner_pair_weight_min(run: RunPaths, fold_id: str) -> float | None:
    """The runner's `pair_weight_min` for one fold, read off its combos' `run.json`.

    None when unset. Raises when the fold's combos disagree, or when the fold
    has no `run.json` to read it from.
    """
    seen: dict[str, float | None] = {}
    fold = run.setup_dir / fold_id
    for combo in sorted(p for p in fold.iterdir() if p.is_dir()) if fold.is_dir() else ():
        rj = combo / "run.json"
        if rj.exists():
            seen[combo.name] = json.loads(rj.read_text()).get("pair_weight_min")
    if not seen:
        raise MissingArtifactError(f"{run.run_id}: no combo run.json under {fold}")
    values = set(seen.values())
    if len(values) > 1:
        raise ValueError(
            f"{run.run_id} {fold_id}: combos ran with different pair_weight_min "
            f"{seen}, so they were scored on different observations and there is "
            f"no one population to pick the Shortest-Ping VP from."
        )
    return values.pop()


def eval_observations(run: RunPaths, inputs_root: Path | None = None) -> pd.DataFrame:
    """Every observation the run's combos were scored on, with an int `fold`.

    Columns in benchmark names (`target_id`, `target_lat`, `vp_id`, `vp_lat`,
    ..., `rtt_ms`, `weight`), the shape `libs.canonical.pairs.build_pairs`
    takes. Row order within a fold is the parquet's, which is the source's
    `iter_eval_targets` order -- it decides RTT ties.
    """
    frames = []
    for fold_id, path in _fold_files(run, EVAL_OBSERVATIONS, inputs_root):
        obs = pd.read_parquet(path).rename(columns={"latency_ms": "rtt_ms"})
        thr = runner_pair_weight_min(run, fold_id)
        if thr is not None:
            # `_filter_by_weight`: obs below the threshold go; a TG left with
            # none goes with them.
            obs = obs[obs["weight"] >= thr]
        obs = obs.assign(fold=int(fold_id.split("_")[1]))
        frames.append(obs)
    out = pd.concat(frames, ignore_index=True)
    out["target_id"] = out["target_id"].astype(str)
    out["vp_id"] = out["vp_id"].astype(str)
    return out


def shortest_ping(run: RunPaths, inputs_root: Path | None = None) -> pd.DataFrame:
    """Per `(tg_id, fold)`: the min-RTT VP, the min RTT and the min inflation.

    The same reduction `benchmark.v2.eval_source.per_target_metrics` makes
    (`idxmin` over `build_pairs`' min-RTT-per-pair rows; ties go to the first
    row of the stable sort), over the observations the run actually scored
    rather than the raw CSV. `min_inflation` is NaN when every pair of the TG
    is colocated.
    """
    from scripts.libs.canonical.pairs import build_pairs

    obs = eval_observations(run, inputs_root)
    parts = []
    for fold, group in obs.groupby("fold", sort=True):
        pairs = build_pairs(group.drop(columns="fold"))
        best = pairs.loc[pairs.groupby("target_id")["rtt_ms"].idxmin()].set_index("target_id")
        g = pairs.groupby("target_id")
        parts.append(
            pd.DataFrame(
                {
                    "tg_id": best.index.to_numpy(),
                    "fold": np.full(len(best), int(fold)),
                    "sping_vp_id": best["vp_id"].to_numpy(),
                    "sping_vp_lat": best["vp_lat"].to_numpy(),
                    "sping_vp_lon": best["vp_lon"].to_numpy(),
                    "min_rtt_ms": best["rtt_ms"].to_numpy(),
                    "min_inflation": g["inflation"].min().reindex(best.index).to_numpy(),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=list(SHORTEST_PING_COLUMNS))
    return pd.concat(parts, ignore_index=True)[list(SHORTEST_PING_COLUMNS)]
