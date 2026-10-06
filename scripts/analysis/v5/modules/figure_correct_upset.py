"""Which methods classify each TG correctly: an UpSet plot of cell correctness.

The outcome bars are **unpaired**: two methods at 60% accuracy can be right on
the same 60% of the TGs or on disjoint ones. This figure is the pairing: on
every TG, the exact set of methods whose prediction falls in the TG's cell.

## Correct is classify's verdict, answered rows only

A method is correct on a TG when `cell_label == "correct"` **and** the row is
answered (`status.solved_mask`), the scoring rule of the paper and of
`figure_pareto.cell_correct`. `classify` also labels FALLBACK rows, which carry
the shortest-ping VP's coordinate, so reading the label alone would credit VAN
with S-P's answers on exactly the TGs VAN refused. `--metric bounded` further
requires the prediction within `figure_pareto.BOUND_PIXELS` pixels of the TG's
pixel.

## Unlike the champion UpSet, a TG can have no member

A champion is relative to the best error on the TG, so every TG has one. A
correct method is absolute, so the `(none)` combination -- TGs that every
method gets wrong -- is a real column, drawn like any other tie column (grey,
no black dots).

## Small combinations are lumped

Six methods give up to 64 combinations, and the pooled pro meshes hold 25,
most under 1% of the TGs. Combinations below `MIN_SHARE` are drawn as one
dashed column with no dots; the tables keep every combination. The set-size
bars are each method's accuracy over every TG, so they equal the outcome
bars' correct share.

Drawing and tables come from `figure_champion_upset`; only the mask differs.

## Nesting against the baseline, and families

The text reads more off the mask than the intersections show directly: how
many of S-P's correct TGs a method misses, how many it adds, and the cohort a
method family (Octant: OCT-H and OCT-S) alone gets right. `<stem>.nesting.csv`
holds them, one row per `(kind, method)`:

* `misses_baseline` -- S-P correct, the method wrong; `share_of_reference` is
  over S-P's correct TGs, `share_of_tgs` over all.
* `adds_over_baseline` -- the method correct, S-P wrong.
* `only` -- the method (or family) is the whole correct set: a non-empty
  subset of `FAMILIES[...]` for a family.
* `all_or_nothing_sites` -- sites where the method is right on all or none
  of the site's TGs, as a share of sites.

`any_cbg` is every method but S-P taken together. Every row carries its site
count and, when `report-sp-pni-cells` has written `sp_pni_cells_tgs.csv` for
the same runs, `has_x_share`: the share of the cohort whose own cell holds X.

Command: `plot-correct-upset`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules import figure_champion_upset as U
from scripts.analysis.v5.modules import figure_error_cdf as E
from scripts.analysis.v5.modules import figure_pareto as F
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.methods import (
    method_label,
    method_order,
    method_term_table,
    methods_source,
)
from scripts.analysis.v5.modules.paths import (
    CLASSIFY_KIND,
    DEFAULT_ANALYSIS_ROOT,
    MissingArtifactError,
    RunPaths,
)
from scripts.analysis.v5.modules.sp_pni_cells import CSV_NAME as SP_PNI_CSV
from scripts.analysis.v5.modules.sites import site_key
from scripts.analysis.v5.modules.status import SHORTEST_PING

PER_RUN = U.PER_RUN
POOLED = U.POOLED
LAYOUTS = U.LAYOUTS
SOURCE_NSIDE = F.SOURCE_NSIDE

#: `metric -> (scoring rule, set-size axis label)`.
METRICS = {
    "cell": (F.cell_correct, "Accuracy (%)"),
    "bounded": (F.bounded_correct, "Bounded accuracy (%)"),
}
DEFAULT_METRIC = "cell"

#: Exact combinations below this share of the TGs share one lumped column.
MIN_SHARE = 0.01

#: Text width of the paper, a little taller than the champion figure's rows.
FIGSIZE: tuple[float, float] = (5.45, 2.9)

#: Gap between the accuracy bars and the dot matrix, wide enough for the
#: method names that sit in it at this text-width canvas.
WSPACE = 0.42

#: Bar labels in whole percent, as the paper rounds accuracy: 0.1 pp is
#: under one TG pooled, and ~20 replicas per site make it less still.
PCT_DECIMALS = 0

STEM = "correct_upset"
KINDS = U.KINDS
_SUFFIX = U._SUFFIX

_READ = ("tg_id", "tg_lat", "tg_lon", "status", "cell_label", F.C.GRID_OFFSET)

SET_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "metric", "method", "method_label", "is_baseline",
    "n_tgs", "n_correct", "share", "n_only", "n_sites", "n_sites_correct",
)
INTERSECTION_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "metric", "members", "n_methods", "n_tgs", "share", "n_sites",
)
SITE_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "metric", "method", "method_label", "is_baseline",
    "n_sites", "n_sites_any", "n_sites_majority", "n_sites_all",
)


def validate_metric(metric: str) -> str:
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}; pick from {sorted(METRICS)}")
    return metric


def artifact_names(layout: str, metric: str = DEFAULT_METRIC) -> dict[str, str]:
    """`kind -> filename`: `correct_upset[.pooled].<metric>.<suffix>`."""
    parts = [STEM, metric]
    if layout == POOLED:
        parts.insert(1, "pooled")
    stem = ".".join(parts)
    return {kind: f"{stem}.{_SUFFIX[kind]}" for kind in KINDS}


# ---- loading ----------------------------------------------------------------


def correct_matrix(
    run: RunPaths,
    metric: str = DEFAULT_METRIC,
    *,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """One run's `tg_id x method` correctness, columns in `TERM_ORDER`, and its sites.

    Every method must cover the same TGs, as in `figure_error_cdf.error_matrix`.
    """
    rule = METRICS[validate_metric(metric)][0]
    cls_dir = run.classify_dir(SOURCE_NSIDE, root=analysis_root)
    chosen = list(methods) if methods else E.scored_methods(cls_dir)
    if not chosen:
        raise MissingArtifactError(
            f"{cls_dir} holds no *_tgs.parquet; run `classify --run-id {run.run_id}` first"
        )
    if len(chosen) > U.MAX_METHODS:
        raise ValueError(
            f"{run.run_id} scores {len(chosen)} methods; an UpSet over more than "
            f"{U.MAX_METHODS} is unreadable. Pick a subset with --method."
        )
    cols: dict[str, pd.Series] = {}
    sites: pd.Series | None = None
    for method in chosen:
        df = E._load_tgs(run, method, SOURCE_NSIDE, analysis_root=analysis_root, columns=_READ)
        if df["tg_id"].duplicated().any():
            raise ValueError(f"{run.run_id}/{method}: duplicate tg_id")
        cols[method] = pd.Series(rule(df), index=df["tg_id"].to_numpy())
        if sites is None:
            sites = site_key(df, run_id=run.run_id).set_axis(df["tg_id"].to_numpy())
    first = cols[chosen[0]].index.sort_values()
    odd = [m for m in chosen[1:] if not cols[m].index.sort_values().equals(first)]
    if odd:
        sizes = {m: int(len(cols[m])) for m in chosen}
        raise ValueError(
            f"{run.run_id}: methods cover different TG sets ({sizes}); re-run "
            f"`classify --run-id {run.run_id}`, or pin a consistent set with --method."
        )
    order = method_order(chosen)
    frame = pd.DataFrame(cols).reindex(first)[order].astype(bool)
    frame.index.name = "tg_id"
    return frame, sites.reindex(first)


# ---- tables -----------------------------------------------------------------

#: Method families the text treats as one: a TG is "only the family's" when
#: its correct set is a non-empty subset of the family.
FAMILIES: dict[str, tuple[str, ...]] = {"octant_family": ("octant_cbg_hull", "octant_cbg_spl")}

#: The pseudo-method for "some method other than the baseline".
ANY_CBG = "any_cbg"

NESTING_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "metric", "kind", "method", "method_label", "n_tgs",
    "share_of_tgs", "n_reference", "share_of_reference", "n_sites", "share_of_sites",
    "has_x_share",
)


def has_x_flags(run_ids: list[str], layout: str, *, analysis_root: Path | None = None,
                index: pd.Index | None = None, origin: pd.Series | None = None) -> pd.Series | None:
    """`tg_cell_holds_x` aligned to the mask's index, or None if not written.

    Read from `report-sp-pni-cells`'s CSV for the same run set (pooled) or the
    run's single one (per run); nothing is created looking for it.
    """
    root = analysis_root or DEFAULT_ANALYSIS_ROOT
    if layout == POOLED:
        found = [root / "_cross" / "pni-gap" / cross.cross_name(run_ids) / SP_PNI_CSV]
    else:
        found = sorted((root / run_ids[0] / "pni-gap").glob(f"*/{SP_PNI_CSV}"))
    found = [f for f in found if f.exists()]
    if len(found) != 1 or index is None or origin is None:
        return None
    flags = pd.read_csv(found[0], usecols=["run_id", "tg_id", "tg_cell_holds_x"])
    lookup = dict(zip(zip(flags.run_id, flags.tg_id), flags.tg_cell_holds_x.astype(bool)))
    run = origin.reindex(index)
    keys = [(r, str(k).split(U.RUN_KEY_SEP, 1)[-1]) for r, k in zip(run, index)]
    out = pd.Series([lookup.get(k) for k in keys], index=index, dtype=object)
    return None if out.isna().any() else out.astype(bool)


def nesting_table(mask: pd.DataFrame, sites: pd.Series, has_x: pd.Series | None = None) -> pd.DataFrame:
    """The baseline-relative and family cohorts the text quotes; see the module docstring."""
    site = sites.reindex(mask.index)
    n_sites = int(site.nunique())
    total = len(mask)
    rows: list[dict] = []

    def add(kind: str, method: str, cohort: pd.Series, reference: pd.Series | None = None) -> None:
        n_ref = int(reference.sum()) if reference is not None else total
        n = int(cohort.sum())
        rows.append({
            "kind": kind, "method": method,
            "method_label": method_label(method) if method in mask.columns else method,
            "n_tgs": n, "share_of_tgs": n / total if total else float("nan"),
            "n_reference": n_ref, "share_of_reference": n / n_ref if n_ref else float("nan"),
            "n_sites": int(site[cohort].nunique()),
            "share_of_sites": site[cohort].nunique() / n_sites if n_sites else float("nan"),
            "has_x_share": float(has_x[cohort].mean()) if has_x is not None and n else float("nan"),
        })

    base = mask[SHORTEST_PING] if SHORTEST_PING in mask.columns else None
    others = [m for m in mask.columns if m != SHORTEST_PING]
    if base is not None and others:
        candidates = [(m, mask[m]) for m in others] + [(ANY_CBG, mask[others].any(axis=1))]
        for m, col in candidates:
            add("misses_baseline", m, base & ~col, reference=base)
            add("adds_over_baseline", m, col & ~base)
    degree = mask.sum(axis=1)
    for m in mask.columns:
        add("only", m, mask[m] & (degree == 1))
    for family, members in FAMILIES.items():
        present = [m for m in members if m in mask.columns]
        if present:
            inside = mask[present].sum(axis=1)
            add("only", family, (inside > 0) & (inside == degree))
    rate = mask.astype(float).groupby(site.to_numpy()).mean()
    for m in mask.columns:
        r = rate[m]
        n = int(((r == 0) | (r == 1)).sum())
        rows.append({
            "kind": "all_or_nothing_sites", "method": m, "method_label": method_label(m),
            "n_tgs": float("nan"), "share_of_tgs": float("nan"),
            "n_reference": n_sites, "share_of_reference": n / n_sites if n_sites else float("nan"),
            "n_sites": n, "share_of_sites": n / n_sites if n_sites else float("nan"),
            "has_x_share": float("nan"),
        })
    return pd.DataFrame(rows)


def set_table(mask: pd.DataFrame, sites: pd.Series) -> pd.DataFrame:
    """Per method: TGs correct, the share (= accuracy), and TGs only it gets right."""
    return U.set_table(mask, sites).rename(columns={
        "n_champion": "n_correct", "n_sole": "n_only", "n_sites_champion": "n_sites_correct",
    })


# ---- artifacts --------------------------------------------------------------


def _manifest(layout, *, names, mask, sets, run_ids, origin, metric, source) -> str:
    degree = mask.sum(axis=1)
    order = list(mask.columns)
    per_run = {r: int(c) for r, c in origin.value_counts().sort_index().items()}
    body = {
        "figure": names["png"],
        "tables": {k: names[k] for k in ("intersections", "sets", "sites", "membership")}
        | {"nesting": nesting_name(names)},
        "layout": layout,
        "runs": list(run_ids),
        "dataset": cross.dataset_slug(run_ids),
        "methods": order,
        "methods_source": source,
        "method_terms": method_term_table(order),
        "baseline": SHORTEST_PING,
        "metric": metric,
        "rule": (
            "correct = cell_label == 'correct' and status.solved_mask (a FALLBACK row is "
            "not an answer)"
            + (f", and 0 <= {F.C.GRID_OFFSET} <= {F.BOUND_PIXELS}" if metric == "bounded" else "")
        ),
        "readings": {
            "intersections": (
                "exact combinations -- these methods are correct and no other; '(none)' "
                "is every method wrong. They partition the TGs and sum to 100%."
            ),
            "sets": "each method's accuracy over every TG; they overlap.",
            "sites": (
                f"per method, the sites where it is correct on at least one, more than "
                f"{U.MAJORITY_SHARE:.0%}, and all of the site's TGs."
            ),
            "lumped": f"the figure draws combinations below {MIN_SHARE:.0%} of TGs as one column",
        },
        "n_tgs": int(len(mask)),
        "n_tgs_per_run": per_run,
        "share_none": round(float((degree == 0).mean()), 4) if len(mask) else None,
        "share_all": round(float((degree == len(order)).mean()), 4) if len(mask) else None,
        "n_correct_per_method": {r["method"]: int(r["n_correct"]) for _, r in sets.iterrows()},
        "source_rung": {"nside": int(SOURCE_NSIDE)},
    }
    if layout == POOLED:
        body["pooling"] = {
            "rule": "micro-pool: every run's mask stacked (<run_id>::<tg_id>); a dataset "
                    "weighs by its TG count.",
            "coverage": "strict -- a method absent from any run is refused.",
        }
    return json.dumps(body, indent=2) + "\n"


def nesting_name(names: dict[str, str]) -> str:
    """`correct_upset.pooled.cell.png` -> `correct_upset.pooled.cell.nesting.csv`."""
    return names["png"].removesuffix(".png") + ".nesting.csv"


def _write(mask, sites, origin, out_dir: Path, layout: str, *, run_ids, metric, source,
           analysis_root=None):
    names = artifact_names(layout, metric)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_col = "+".join(sorted(run_ids))
    dataset = cross.dataset_slug(run_ids)

    def stamp(table: pd.DataFrame) -> pd.DataFrame:
        table.insert(0, "run_id", run_col)
        table.insert(1, "dataset", dataset)
        table.insert(2, "metric", metric)
        return table

    inter = stamp(U.intersection_table(mask, sites))
    sets = stamp(set_table(mask, sites))
    by_site = stamp(U.site_table(mask, sites))
    written = {k: out_dir / names[k] for k in KINDS}
    inter[list(INTERSECTION_COLUMNS)].to_csv(written["intersections"], index=False)
    sets[list(SET_COLUMNS)].to_csv(written["sets"], index=False)
    by_site[list(SITE_COLUMNS)].to_csv(written["sites"], index=False)
    has_x = has_x_flags(run_ids, layout, analysis_root=analysis_root, index=mask.index, origin=origin)
    nesting = stamp(nesting_table(mask, sites, has_x))
    nesting[list(NESTING_COLUMNS)].to_csv(out_dir / nesting_name(names), index=False)

    audit = mask.copy()
    audit.columns = [f"{m}_correct" for m in mask.columns]
    audit.insert(0, "run_id", origin.reindex(mask.index).to_numpy())
    audit.insert(1, "tg_id", [str(k).split(U.RUN_KEY_SEP, 1)[-1] for k in mask.index])
    audit["site_key"] = sites.reindex(mask.index).to_numpy()
    audit["n_correct"] = mask.sum(axis=1)
    audit["members"] = U.combination(mask)
    audit.to_csv(written["membership"], index=False)

    U.plot_upset(
        mask, written["png"], title=None, subtitle=None, figsize=FIGSIZE,
        set_label=METRICS[metric][1], min_share=MIN_SHARE, wspace=WSPACE,
        pct_decimals=PCT_DECIMALS,
    )
    written["manifest"].write_text(_manifest(
        layout, names=names, mask=mask, sets=sets, run_ids=run_ids, origin=origin,
        metric=metric, source=source,
    ))
    return written


def build_for_runs(
    runs: list[RunPaths],
    *,
    layouts: tuple[str, ...] = (PER_RUN,),
    metric: str = DEFAULT_METRIC,
    methods: list[str] | None = None,
    analysis_root: Path | None = None,
    source: str | None = None,
) -> list[dict[str, Path]]:
    """Render the requested layouts; one artifact set per figure, layout-major."""
    ordered = tuple(dict.fromkeys(layouts)) or (PER_RUN,)
    unknown = [x for x in ordered if x not in LAYOUTS]
    if unknown:
        raise ValueError(f"unknown layout {unknown}; pick from {list(LAYOUTS)}")
    metric = validate_metric(metric)
    G.validate_nside(SOURCE_NSIDE)
    src = methods_source(methods, source)
    out: list[dict[str, Path]] = []
    for layout in ordered:
        if layout == PER_RUN:
            for run in runs:
                mask, sites = correct_matrix(run, metric, methods=methods,
                                             analysis_root=analysis_root)
                origin = pd.Series(run.run_id, index=mask.index, name="run_id")
                out.append(_write(
                    mask, sites, origin, run.analysis_dir(CLASSIFY_KIND, root=analysis_root),
                    PER_RUN, run_ids=[run.run_id], metric=metric, source=src,
                    analysis_root=analysis_root,
                ))
            continue
        by_run = {
            run.run_id: correct_matrix(run, metric, methods=methods, analysis_root=analysis_root)
            for run in runs
        }
        mask, sites, origin = U.stack_runs(by_run)
        run_ids = [r.run_id for r in runs]
        out.append(_write(
            mask.astype(bool), sites, origin,
            cross.cross_dir(run_ids, analysis_root=analysis_root), POOLED,
            run_ids=run_ids, metric=metric, source=src, analysis_root=analysis_root,
        ))
    return out


__all__ = [
    "DEFAULT_METRIC", "METRICS", "MIN_SHARE", "artifact_names", "build_for_runs",
    "correct_matrix", "set_table", "validate_metric",
]
