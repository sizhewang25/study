"""The dataset a run belongs to, declared rather than parsed.

`cross.short_dataset` used to read the dataset off the run id by splitting on
the first hyphen -- `as01-260728-260802-mesh` -> `as01`. That assumes the shape
`<dataset>-<arm>`, which any prefix breaks: `pro-as01-mesh`, `pro-as02-mesh`
and `pro-as03-mesh` all read `pro`. The collapse was not only cosmetic. The
pooled octant-finetuning frame namespaced its sites on that name, so three
datasets' sites merged into one and the site-clustered bootstrap resampled
units that do not exist.

So the name is **declared**, in the config, under `analysis.common`:

    analysis:
      common:
        dataset_label: as01

## Why this module reads a config at all

`paths.combo_ids` states v5's rule: read the output tree, not a config, so
there is one source of truth. This is the documented exception, kept in one
module so it stays one exception. The label is a *display* name -- nothing in
the tree records it, and nothing can derive it, which is the whole point.

The run reaches its config through `target_space.json`'s `config` key, written
by the benchmark alongside the `csv` key that `edges.resolve_source_csv`
already reads the same way.

## Also: the PNI list

`analysis.common.pni_csv` is read here too, for the same reason: nothing in
the output tree records where an operator peers. `declared` walks any key
path; `declared_pni_csv` is the one other caller.

## Also: which methods a figure draws

`analysis.<command>.combo_ids` narrows one command's figure to the methods it
names -- a paper figure can drop OCT-S while the run still holds it. The tree
records what was *scored*; which of those a figure is *about* is an editorial
choice nothing can derive, so it is the third declared value. Read by
`declared_combo_ids`; `cli._methods_for` decides precedence and checks the
names against the tree.

## Never raises (except a malformed `combo_ids`)

Every break in the chain falls back to the run id, which is unique by
construction and so is always a correct-if-verbose label. The breaks are real,
not hypothetical: five runs have no `target_space.json` at all
(`as0{1,2,3}-260728-260802`, `as01-260728-260802-mesh-heapfix`,
`as7018_us_test01`), and `as01-materialization-test`'s manifest names a config
that has since been deleted. A missing label is not an error -- only the
`pro-*` configs declare one.

Identity is a separate concern and never comes from here: sites key on
`run_id` (`sites.site_key`), and the pooled cross directory keys on a hash of
the run ids. A declared label is free text, so two runs may well declare the
same one -- `cross.guard_distinct_labels` is what refuses to pool those.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

from scripts.analysis.v5.modules.paths import (
    DEFAULT_OUTPUTS_ROOT,
    REPO_ROOT,
    MissingArtifactError,
    resolve_run,
)

#: Where the label is declared, as a path through the config mapping.
LABEL_PATH = ("analysis", "common", "dataset_label")


def _under_repo(value: str) -> Path:
    """Manifest paths are repo-root-relative (the benchmark's `_repo_relative`)."""
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


def config_path(run_id: str, root: Path | str = DEFAULT_OUTPUTS_ROOT) -> Path | None:
    """The config that produced `run_id`, or None if it cannot be reached.

    Note the run id is **not** the config stem in general: the `-mesh-grafted`
    runs record the non-grafted config they were produced against, which is why
    this reads the manifest rather than guessing `configs/<run_id>.yaml`.
    """
    try:
        run = resolve_run(run_id, root)
    except (MissingArtifactError, OSError):
        return None
    return run_config_path(run)


def run_config_path(run) -> Path | None:
    """`config_path` for a run already resolved: its `target_space.json`'s `config`.

    Split out for `answer_space`, which holds a `RunPaths` and must read the
    config of *that* run -- resolving the id again against a default root could
    find a different tree.
    """
    if not run.target_space_json.exists():
        return None
    try:
        recorded = json.loads(run.target_space_json.read_text()).get("config")
    except (json.JSONDecodeError, OSError):
        return None
    if not recorded:
        return None
    p = _under_repo(recorded)
    return p if p.exists() else None


@lru_cache(maxsize=None)
def dataset_label(run_id: str, root: Path | str = DEFAULT_OUTPUTS_ROOT) -> str:
    """The run's declared `analysis.common.dataset_label`, else the run id.

    Memoized: a pooled figure asks for the same handful of run ids once per
    panel, per CSV row group and again for the manifest, and each miss is a
    directory scan plus a YAML parse.
    """
    node = declared(run_id, LABEL_PATH, root)
    return run_id if node is None else str(node)


def declared(run_id: str, key_path: tuple[str, ...], root: Path | str = DEFAULT_OUTPUTS_ROOT):
    """The scalar at `key_path` in the run's config, or None.

    None for every break in the chain: no config, an unparsable one, a missing
    key, or a non-scalar value. A non-scalar would be carried into a filename
    or a CSV column, so it is refused the same way an absent one is.
    """
    node = _node(run_id, key_path, root)
    if node is None or isinstance(node, (dict, list)):
        return None
    return node


def _node(run_id: str, key_path: tuple[str, ...], root: Path | str = DEFAULT_OUTPUTS_ROOT):
    """The raw value at `key_path` in the run's config, or None if any link breaks."""
    path = config_path(run_id, root)
    if path is None:
        return None
    try:
        node = yaml.safe_load(path.read_text())
    except (yaml.YAMLError, OSError):
        return None
    for key in key_path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


#: Where a run's operator PNI list is declared. Read by `plot-pni-gap` and
#: `plot-pni-cluster-rtt` when `--pni-csv` is not given.
PNI_CSV_PATH = ("analysis", "common", "pni_csv")


def declared_pni_csv(run_id: str, root: Path | str = DEFAULT_OUTPUTS_ROOT) -> Path | None:
    """The run's declared `analysis.common.pni_csv`, repo-relative resolved, or None.

    Returned whether or not the file exists: a config naming a missing PNI list
    is a broken config, and `pni_gap.load_pnis` says so, rather than this
    quietly reading it as "no PNI list declared".
    """
    value = declared(run_id, PNI_CSV_PATH, root)
    return None if value is None else _under_repo(str(value))


#: Where a run's RTT normalizer is declared: the `{min, max}` ms bounds every
#: RTT axis is min-max normalized with. Read by `plot-pni-cluster-rtt`.
RTT_NORM_PATH = ("analysis", "common", "rtt_norm_ms")

#: Where a run's distance normalizer is declared: the `{min, max}` km bounds
#: every distance axis is min-max normalized with. Read by `plot-error-cdf`.
DIST_NORM_PATH = ("analysis", "common", "dist_norm_km")


def declared_rtt_norm_ms(
    run_id: str, root: Path | str = DEFAULT_OUTPUTS_ROOT
) -> tuple[float, float] | None:
    """The run's declared `analysis.common.rtt_norm_ms` as `(min, max)`, or
    None for "plot in ms". Validated as `declared_dist_norm_km` is."""
    return _declared_bounds(run_id, RTT_NORM_PATH, "ms", root)


def declared_dist_norm_km(
    run_id: str, root: Path | str = DEFAULT_OUTPUTS_ROOT
) -> tuple[float, float] | None:
    """The run's declared `analysis.common.dist_norm_km` as `(min, max)`, or
    None for "plot in km".

    Both bounds are required and must satisfy 0 <= min < max. Anything else
    raises: falling back to km would print the absolute distances the
    normalizer exists to hide, and the figure would not say so.
    """
    return _declared_bounds(run_id, DIST_NORM_PATH, "km", root)


def _declared_bounds(
    run_id: str, key_path: tuple[str, ...], unit: str, root: Path | str
) -> tuple[float, float] | None:
    """`(min, max)` at `key_path`, None if absent; anything malformed raises."""
    value = _node(run_id, key_path, root)
    if value is None:
        return None
    key = ".".join(key_path)

    def number(v) -> bool:
        return not isinstance(v, bool) and isinstance(v, (int, float))

    if not isinstance(value, dict) or set(value) != {"min", "max"}:
        raise ValueError(
            f"{run_id}: {key} must be a mapping with exactly "
            f"`min` and `max` ({unit}), got {value!r}"
        )
    lo, hi = value["min"], value["max"]
    if not (number(lo) and number(hi) and 0 <= lo < hi):
        raise ValueError(
            f"{run_id}: {key} needs numbers with "
            f"0 <= min < max, got min={lo!r} max={hi!r}"
        )
    return float(lo), float(hi)


#: The key, inside a command's own `analysis.<command>:` block, naming the
#: methods that command's figures draw.
COMBO_IDS_KEY = "combo_ids"


def declared_combo_ids(
    run_id: str, command: str, root: Path | str = DEFAULT_OUTPUTS_ROOT
) -> list[str] | None:
    """The run's `analysis.<command>.combo_ids`, or None for "draw everything".

    None when nothing declares it -- no config, no block, an empty `{}`
    placeholder, or an explicit `null`. Unlike the scalars above, a value that
    IS there but malformed raises: falling back to every method would draw the
    very combo the author meant to leave out, and the figure would not say so.
    """
    value = _node(run_id, ("analysis", command, COMBO_IDS_KEY), root)
    if value is None:
        return None
    where = f"{run_id}: analysis.{command}.{COMBO_IDS_KEY}"
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of combo ids, got {value!r}")
    if not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{where} must hold combo id strings, got {value!r}")
    if len(set(value)) != len(value):
        raise ValueError(f"{where} names a combo twice: {value!r}")
    return list(value)


#: The key, inside `analysis.plot-cost-box:`, mapping a drawn method to the
#: variant drawn on its slot (`figure_cost_box`'s overlays).
OVERLAY_KEY = "overlay"


def declared_overlays(
    run_id: str, command: str, root: Path | str = DEFAULT_OUTPUTS_ROOT
) -> dict[str, str] | None:
    """The run's `analysis.<command>.overlay` as `{host: variant}`, or None.

    Malformed raises, as for `combo_ids`: an overlay silently dropped would
    leave the figure claiming a comparison it does not draw.
    """
    value = _node(run_id, ("analysis", command, OVERLAY_KEY), root)
    if value is None:
        return None
    where = f"{run_id}: analysis.{command}.{OVERLAY_KEY}"
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{where} must be a non-empty HOST: VARIANT mapping, got {value!r}")
    if not all(isinstance(k, str) and k and isinstance(v, str) and v for k, v in value.items()):
        raise ValueError(f"{where} must map combo ids to combo ids, got {value!r}")
    return dict(value)
