"""Reading a benchmark config without importing the benchmark.

Pure functions over the YAML: no DataSource, runner or typer import, so the
analysis side (`scripts.analysis.v5`) can resolve a run's edge CSV from its
config without dragging in the whole `cli.py` import graph. `cli.py` re-exports
these under its historical private names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

#: Top-level config keys the Snakefile forwards to the source as flags. They
#: live beside `benchmark:`, not inside `source_kwargs:`.
EVAL_WEIGHT_KEYS = ("eval_pair_weight_min", "eval_kept_traffic_fraction")


def resolve_bench_config(path: Path) -> dict:
    """Flatten a benchmark config: `benchmark:` wins key by key over the top level.

    Both accepted shapes -- unified (`configs/<run_id>.yaml`) and flat
    (`scripts/benchmark/v2/config/*.yaml`) -- collapse to one dict, exactly as
    `Snakefile`'s `bcfg` resolves them. Duplicated rather than imported because
    a Snakefile is not a module; `test_cli.py` pins the two against each other
    so a config that drives a run keeps driving the target space identically.
    """
    import yaml

    cfg = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(cfg, dict):
        raise TypeError(f"{path}: top level must be a mapping, got {type(cfg).__name__}")
    bench = cfg.get("benchmark") or {}
    if not isinstance(bench, dict):
        raise TypeError(
            f"{path}: `benchmark:` must be a mapping, got {type(bench).__name__}"
        )
    merged = {k: v for k, v in cfg.items() if k != "benchmark"}
    merged.update(bench)
    return merged


def source_kwargs(cfg: dict) -> dict:
    """`source_kwargs:` plus the top-level eval weight keys that are set.

    The dict half of `cli._source_kwargs_from_config`, without its check that
    the source class accepts the weight keys -- that check needs the class, and
    a reader only needs to know what the run was configured with.
    """
    kwargs = dict(cfg.get("source_kwargs") or {})
    for key in EVAL_WEIGHT_KEYS:
        if cfg.get(key) is not None:
            kwargs[key] = cfg[key]
    return kwargs


def edge_csv_from_kwargs(kwargs: dict) -> tuple[Optional[Path], bool]:
    """`(path, is_mesh_superset)` for the CSV holding this run's edge set.

    For `traffic_weighted_csv` in precomputed mode that is `weighted_csv_path`:
    the pruned flows *are* the graph the run evaluates. On-the-fly mode has no
    such file, so the mesh comes back with the flag set -- it is a strict
    superset of the evaluated edges, and `build-bipartite-graph` reading it
    would report the mesh's density as the weighted arm's.
    """
    weighted = kwargs.get("weighted_csv_path")
    if weighted:
        return Path(weighted), False
    mesh = kwargs.get("mesh_csv_path") or kwargs.get("csv_path")
    if not mesh:
        return None, False
    superset = (
        kwargs.get("eval_kept_traffic_fraction") is not None
        or kwargs.get("eval_pair_weight_min") is not None
    )
    return Path(mesh), superset
