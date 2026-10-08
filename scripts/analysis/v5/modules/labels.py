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

The run reaches its config through the config's own `run_id` key: every
`configs/*.yaml` is indexed by the `run_id` it declares (`config_path`). The
benchmark's output tree is not consulted, so nothing written by
`materialize-target-space` is needed -- `edges.resolve_source_csv` reaches the
run's CSV through the same lookup.

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

## A run with no config raises

`MissingConfigError`, naming the run id and the directory searched. It used to
fall back -- to the run id as label, to km and ms for the normalizers, to every
method for a figure -- and a figure built that way looks finished. A missing
*key* in a config that exists is still not an error: only the `pro-*` configs
declare a label, and the run id stands in for it. Also raised: a malformed
`combo_ids` / `overlay`, and a conflict between a run and its groups.

## Group files: what a pooled set of runs declares once

A **group file**, `configs/groups/<group_id>.yaml`, names a set of runs that
are pooled together (`runs`, a list or a mapping of role -> list) and holds
the same `analysis:` shape as a run config:

* `analysis.common` -- values that belong to the set rather than to one run,
  such as the normalization bounds (D and the largest RTT, taken over the
  pooled datasets). These are read here: every `analysis.common` lookup
  resolves the key from the run's own config **and** from every group listing
  the run. One source is enough; several must agree exactly, or the lookup
  raises `ConflictingDeclarationError` naming each source, so a run config
  that kept a stale copy of a group value is caught rather than preferred.
* `analysis.<command>` -- that command's **CLI flags**, not config keys:
  `cli --group <id>` loads them as the command's defaults (`cli.group_default_map`).
  They are never read here, so a run config's `analysis.<command>.combo_ids`
  means what it always meant.

Membership is declared in the group file only, never in the run config.

Identity is a separate concern and never comes from here: sites key on
`run_id` (`sites.site_key`), and the pooled cross directory keys on a hash of
the run ids. A declared label is free text, so two runs may well declare the
same one -- `cross.guard_distinct_labels` is what refuses to pool those.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from scripts.analysis.v5.modules.paths import (
    DEFAULT_OUTPUTS_ROOT,
    REPO_ROOT,
    MissingArtifactError,
)

#: Where the label is declared, as a path through the config mapping.
LABEL_PATH = ("analysis", "common", "dataset_label")

#: Where run configs live: `configs/*.yaml`, each declaring its `run_id`.
CONFIGS_DIR = REPO_ROOT / "configs"

#: Where group files live: `configs/groups/<group_id>.yaml`.
GROUPS_DIR = REPO_ROOT / "configs" / "groups"

#: The only part of a group file merged into run lookups; the per-command
#: blocks are CLI flags (`cli.group_default_map`).
GROUP_SHARED = ("analysis", "common")


class ConflictingDeclarationError(ValueError):
    """A key declared by a run config and/or its groups with different values."""


def _groups(groups_dir: str) -> tuple[dict, ...]:
    """Every group file under `groups_dir`, parsed and checked.

    Memoized on the files and their mtimes, so an edited group file is re-read
    within one process.
    """
    files = sorted(Path(groups_dir).glob("*.yaml"))
    return _parse_groups(tuple((str(f), f.stat().st_mtime_ns) for f in files))


@lru_cache(maxsize=None)
def _parse_groups(files: tuple[tuple[str, int], ...]) -> tuple[dict, ...]:
    out = []
    for name, _ in files:
        path = Path(name)
        body = yaml.safe_load(path.read_text()) or {}
        gid = body.get("group_id")
        if gid != path.stem:
            raise ValueError(f"{path}: group_id {gid!r} must equal the file stem {path.stem!r}")
        members = group_members(body)
        if len(set(members)) != len(members):
            raise ValueError(f"{path}: a run is listed twice in `runs`")
        out.append({**body, "_path": str(path)})
    return tuple(out)


def group_members(group: dict, role: str | None = None) -> list[str]:
    """The run ids a group lists: all of them, or one role's (`runs.<role>`)."""
    runs = group.get("runs") or []
    if isinstance(runs, list):
        if role is not None:
            raise ValueError(f"group {group.get('group_id')!r} lists runs without roles; no {role!r}")
        return [str(r) for r in runs]
    if not isinstance(runs, dict):
        raise ValueError(f"group {group.get('group_id')!r}: `runs` must be a list or a role mapping")
    if role is not None:
        if role not in runs:
            raise ValueError(f"group {group.get('group_id')!r} has no role {role!r}; it has {sorted(runs)}")
        return [str(r) for r in runs[role] or []]
    return [str(r) for v in runs.values() for r in (v or [])]


def load_group(group_id: str, groups_dir: Path | str | None = None) -> dict:
    """The group file `<groups_dir>/<group_id>.yaml`, parsed; raises if absent."""
    for g in _groups(str(groups_dir or GROUPS_DIR)):
        if g["group_id"] == group_id:
            return g
    raise MissingArtifactError(f"no group {group_id!r} under {groups_dir or GROUPS_DIR}")


def groups_of(run_id: str, groups_dir: Path | str | None = None) -> list[dict]:
    """Every group listing `run_id`, in file order."""
    return [g for g in _groups(str(groups_dir or GROUPS_DIR)) if run_id in group_members(g)]


def _under_repo(value: str) -> Path:
    """Config-declared paths are repo-root-relative unless absolute."""
    p = Path(value)
    return p if p.is_absolute() else REPO_ROOT / p


class MissingConfigError(MissingArtifactError):
    """No config under `CONFIGS_DIR` declares the run id."""


def _run_id_of(body) -> str | None:
    """A config's `run_id`: top level, or under `benchmark:` (the flat shape)."""
    if not isinstance(body, dict):
        return None
    rid = body.get("run_id")
    if rid is None and isinstance(body.get("benchmark"), dict):
        rid = body["benchmark"].get("run_id")
    return None if rid is None else str(rid)


def _config_index(configs_dir: str) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """`({run_id: config paths}, unparsable paths)` over `configs_dir/*.yaml`.

    Memoized on the files and their mtimes, as `_groups` is. `groups/` is a
    subdirectory, so the flat glob never reaches it; `template.yaml` is skipped
    by name.
    """
    files = sorted(f for f in Path(configs_dir).glob("*.yaml") if f.name != "template.yaml")
    return _parse_config_index(tuple((str(f), f.stat().st_mtime_ns) for f in files))


@lru_cache(maxsize=None)
def _parse_config_index(
    files: tuple[tuple[str, int], ...],
) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    index: dict[str, list[str]] = {}
    broken: list[str] = []
    for name, _ in files:
        try:
            body = yaml.safe_load(Path(name).read_text())
        except (yaml.YAMLError, OSError):
            # One half-edited config must not break every other run's lookup;
            # it is named in the error only if the lookup comes up empty.
            broken.append(name)
            continue
        rid = _run_id_of(body)
        if rid is not None:
            index.setdefault(rid, []).append(name)
    return {k: tuple(v) for k, v in index.items()}, tuple(broken)


def config_path(run_id: str, root: Path | str = DEFAULT_OUTPUTS_ROOT) -> Path:
    """The config under `CONFIGS_DIR` whose `run_id` is `run_id`.

    Found by the key, not by guessing `configs/<run_id>.yaml`: the file name is
    a convention, the `run_id` is what the benchmark ran under. Raises
    `MissingConfigError` when no config declares it -- every value this module
    serves (labels, normalizers, the PNI list, figure methods) would otherwise
    fall back silently, and a figure in km with run-id labels looks finished.
    Raises when two configs declare it, rather than picking one.

    `root` is unused and kept so every caller's signature stays the same: the
    outputs tree no longer takes part in finding a config.
    """
    del root
    return find_config(run_id, CONFIGS_DIR)


def find_config(run_id: str, configs_dir: Path | str) -> Path:
    """`config_path` against an explicit directory. Same errors."""
    index, broken = _config_index(str(configs_dir))
    hits = index.get(str(run_id), ())
    if len(hits) > 1:
        raise MissingConfigError(
            f"{run_id}: several configs declare this run_id: {list(hits)}. Keep one."
        )
    if not hits:
        hint = f" Unparsable configs skipped: {list(broken)}." if broken else ""
        raise MissingConfigError(
            f"{run_id}: no config under {configs_dir} declares `run_id: {run_id}`. "
            f"Restore the config the run was produced with.{hint}"
        )
    return Path(hits[0])


def run_config_path(run) -> Path:
    """`config_path` for a run already resolved. Same lookup, keyed on its id."""
    return config_path(run.run_id)


@lru_cache(maxsize=None)
def dataset_label(run_id: str, root: Path | str = DEFAULT_OUTPUTS_ROOT) -> str:
    """The run's declared `analysis.common.dataset_label`, else the run id.

    Raises `MissingConfigError` for a run with no config at all.

    Memoized: a pooled figure asks for the same handful of run ids once per
    panel, per CSV row group and again for the manifest, and each miss is a
    directory scan plus a YAML parse.
    """
    node = declared(run_id, LABEL_PATH, root)
    return run_id if node is None else str(node)


def declared(run_id: str, key_path: tuple[str, ...], root: Path | str = DEFAULT_OUTPUTS_ROOT):
    """The scalar at `key_path` in the run's config or its groups, or None.

    None for a missing key or a non-scalar value. A non-scalar would be
    carried into a filename or a CSV column, so it is refused the same way an
    absent one is. Raises when the run has no config (`MissingConfigError`) and
    when two sources declare different values.
    """
    node = _node(run_id, key_path, root)
    if node is None or isinstance(node, (dict, list)):
        return None
    return node


def _walk(node, key_path: tuple[str, ...]):
    for key in key_path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _run_node(run_id: str, key_path: tuple[str, ...], root: Path | str = DEFAULT_OUTPUTS_ROOT):
    """The raw value at `key_path` in the run's own config, or None if the key
    is absent. A run with no config raises (`config_path`)."""
    path = config_path(run_id, root)
    return _walk(yaml.safe_load(path.read_text()), key_path)


def _node(run_id: str, key_path: tuple[str, ...], root: Path | str = DEFAULT_OUTPUTS_ROOT):
    """The value at `key_path` from the run's config (and, under
    `analysis.common`, its groups), or None.

    Every source that declares the key must declare the same value; otherwise
    `ConflictingDeclarationError`. An empty `{}` block counts as undeclared.
    """
    found = []
    own = _run_node(run_id, key_path, root)
    if own is not None and own != {}:
        found.append((str(config_path(run_id, root)), own))
    shared = tuple(key_path[:2]) == GROUP_SHARED  # per-command group blocks are CLI flags
    for g in groups_of(run_id) if shared else ():
        value = _walk(g, key_path)
        if value is not None and value != {}:
            found.append((g["_path"], value))
    if not found:
        return None
    first = found[0][1]
    if any(v != first for _, v in found[1:]):
        listing = "; ".join(f"{src}: {v!r}" for src, v in found)
        raise ConflictingDeclarationError(
            f"{run_id}: {'.'.join(key_path)} is declared differently by {listing}. "
            f"Declare it in one place (the group file for a pooled value)."
        )
    return first


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
