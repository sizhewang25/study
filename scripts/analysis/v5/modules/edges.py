"""The run's canonical `(vp_id, target_id, rtt_ms)` CSV, located. Ported from v4.

Only the VP-proximity figure opens the measurement table; every other v5 module
reads the benchmark's scored outputs. The CSV is the one the run's config
names, plus one refusal. The refusal is the point: on a traffic-weighted arm
filtering on the fly the config names the **mesh** superset, and silently
accepting it would measure VP distances over edges the weighted run never had.

The CSV speaks benchmark names (`target_id`, `target_lat`, ...). `load_min_rtt`
and the VP-proximity figure rename to `tg_*` at load time, the one place those
names appear in v5.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.analysis.v5.modules.paths import (
    REPO_ROOT,
    MissingArtifactError,
    RunPaths,
)


class MeshSupersetError(MissingArtifactError):
    """The only CSV on record describes a strictly larger edge set than this arm.

    A subclass, not a flag: absence is degradable, this is not. The file
    parses and every RTT in it is real, but they are measurements the weighted
    run never made.
    """


def _under_repo(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_source_csv(run: RunPaths, override: Path | None = None) -> Path:
    """Path to the run's canonical edge CSV.

    `override` if given, else the CSV the run's config names
    (`config_io.edge_csv_from_kwargs`: `weighted_csv_path`, else
    `mesh_csv_path`, else `csv_path`), repo-relative. The config is found by
    its `run_id` (`labels.config_path`), which raises when there is none.

    Refuses, before anything else, an arm whose only file is the mesh
    superset: a traffic-weighted run filtering on the fly holds no file of its
    own edges.
    """
    if override is not None:
        p = Path(override)
        if not p.exists():
            raise MissingArtifactError(f"--source-csv {p} does not exist")
        return p

    from scripts.analysis.v5.modules.labels import config_path
    from scripts.benchmark.v2.config_io import (
        edge_csv_from_kwargs,
        resolve_bench_config,
        source_kwargs,
    )

    cfg_path = config_path(run.run_id)
    csv, superset = edge_csv_from_kwargs(source_kwargs(resolve_bench_config(cfg_path)))
    if superset:
        raise MeshSupersetError(
            f"{run.run_id}: {cfg_path} names {csv} as a MESH SUPERSET of this "
            f"arm's edge set (the traffic filter runs on the fly, so no file "
            f"holds the pruned flows). Derive a weighted CSV via "
            f"scripts/processing/source/derive_traffic_weighted_cbg_data.smk and "
            f"point the config's `weighted_csv_path` at it, or pass --source-csv "
            f"explicitly to accept the mesh graph."
        )
    if csv is None:
        raise MissingArtifactError(
            f"{run.run_id}: {cfg_path} names no edge CSV (expected "
            f"benchmark.source_kwargs csv_path / mesh_csv_path / "
            f"weighted_csv_path). Pass --source-csv <path>."
        )
    p = _under_repo(str(csv))
    if not p.exists():
        raise MissingArtifactError(
            f"{run.run_id}: {cfg_path} names {csv}, which does not exist at {p}. "
            f"Pass --source-csv <path>."
        )
    return p


#: What `load_min_rtt` returns, in v5 names. `tg_id`, not `target_id`: the
#: canonical CSV speaks benchmark names and this is where they are left behind.
MIN_RTT_COLUMNS = ("tg_id", "vp_id", "rtt_ms")


def load_min_rtt(run: RunPaths, *, source_csv: Path | None = None) -> pd.DataFrame:
    """`(tg_id, vp_id, rtt_ms)`, one row per pair at its minimum RTT.

    Deduplicated to the minimum because a dataset can measure a pair
    repeatedly and a viewer shows one number per VP. Minimum rather than mean:
    it is what every LTD in the framework consumes, so the RTT drawn beside a
    VP is the RTT its constraint was built from, and the per-VP inflation here
    cannot disagree with `eval_per_target.csv`'s `min_inflation` about which
    edge it describes.

    Read through `load_canonical_csv`, which owns the schema, lowercases the
    header and drops NaN rows and `rtt_ms <= 0` exactly as the benchmark's own
    source does -- so these are the observations the benchmark ran on.

    Raises `MeshSupersetError` through `resolve_source_csv` on a weighted arm.
    That refusal is not degradable; see the class docstring.
    """
    from scripts.libs.canonical.schema import load_canonical_csv

    from scripts.analysis.v5.modules.answer_space import BENCHMARK_TG_COLUMNS

    path = Path(source_csv) if source_csv is not None else resolve_source_csv(run)
    df = load_canonical_csv(path).rename(columns=BENCHMARK_TG_COLUMNS)
    return (
        df.groupby(["tg_id", "vp_id"], as_index=False)["rtt_ms"]
        .min()
        .reset_index(drop=True)[list(MIN_RTT_COLUMNS)]
    )
