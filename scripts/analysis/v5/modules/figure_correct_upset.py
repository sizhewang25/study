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
    method_order,
    method_term_table,
    methods_source,
)
from scripts.analysis.v5.modules.paths import CLASSIFY_KIND, MissingArtifactError, RunPaths
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
        "tables": {k: names[k] for k in ("intersections", "sets", "sites", "membership")},
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


def _write(mask, sites, origin, out_dir: Path, layout: str, *, run_ids, metric, source):
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
            run_ids=run_ids, metric=metric, source=src,
        ))
    return out


__all__ = [
    "DEFAULT_METRIC", "METRICS", "MIN_SHARE", "artifact_names", "build_for_runs",
    "correct_matrix", "set_table", "validate_metric",
]
