"""What a method costs per TG: runtime and peak memory, as paired boxes.

The accuracy figures say how close a method gets; this says what it pays per
geolocated TG to get there. One x slot per method, two boxes in it:

* **left box, solid, left y axis** -- runtime per TG in ms, the three stages
  **summed** (`cost.COST_SPECS["runtime"]`);
* **right box, hatched, right y axis** -- peak memory per TG in MB, the three
  stages **max-reduced** (`--memory`: `memory_heap` by default, which sees
  Shapely/GEOS; `memory_alloc` is tracemalloc). The two channels are never
  combined -- see `cost`.

Both are reduced per TG *before* any percentile (`cost.per_target_cost`).

## Slots are ranked by cost, not by `TERM_ORDER`

Left to right, cheapest first: by p50 runtime, ties broken by p50 memory,
both ascending (`cost_order`). The question the figure answers is "what does
each method pay", so the reader should find the ranking by reading along x
rather than by comparing box heights across a twin axis. Hue stays pinned to
identity (`methods.LABEL_HUES`), so a method is still findable by colour when
the data reorders the slots.

## Overlays: a variant drawn on its host's slot

`--overlay HOST=VARIANT` draws `VARIANT`'s two boxes on `HOST`'s slot as
unfilled dashed frames in ink (the memory frame keeps the memory hatch),
slightly wider than the host's boxes so the
host stays readable underneath. It is for a variant that differs from its host
in one stage (`octant_cbg_hull_geo`: OCT-H with a geometric-centroid CTR), so
the slot shows what that stage costs without spending a slot on a method the
paper does not rank. A variant takes no slot and does not enter `cost_order`.
Its rows are in the CSV with `overlay_on` naming the host.

## Whiskers are p5 and p95, not 1.5 IQR

The boxes are drawn from precomputed stats (`Axes.bxp`), so every mark is a
named percentile: whiskers p5/p95, hinges p25/p75, the line the median. No
fliers are drawn. The tails beyond p5/p95 (min/max) are in the CSV. A Tukey
whisker would move with the IQR and say nothing a reader can quote.

## Two y axes, on purpose

Runtime and memory have no common unit, and the question is how a method's
two costs sit side by side. The figure pays for the second axis with three
cues: box texture (solid = runtime, hatched = memory), each box sitting on
its own side of the slot, and the key naming the axis. Both axes are log:
across methods the per-TG cost spans three to four decades on these meshes.
The hue is the method's (`methods.LABEL_HUES`) and carries identity only. It
does not say which axis a box is read on.

## Rows

`--rows all` (default): every evaluated TG, FALLBACK included. A method pays
for the TGs it gave up on. `--rows solved` is `status.solved_mask`. Runtime
and memory share one frame and one mask (`cost.load_cost_frame`), so they are
always over the same TGs.

## What is not here

* **S-P.** Not a combo, so no stage was timed. The manifest names it as
  excluded rather than leaving a gap unexplained.
* **The LTD fit.** Once per `(combo, fold)`, in `run.json`. It amortises
  over the fold and is not a per-TG cost.

## Extrapolation to operator scale

`<stem>.extrapolation.csv` turns the per-TG runtime into a batch budget, per
method and per scope (each run, plus `all` when pooled). The **mean** pipeline
runtime sets it, not the median: every TG is processed once and independently,
so the total is mean x N. `core_hours` is that total for `--extrapolate-tgs`
TGs (default one million), `wall_hours`/`wall_days` the same on `--cores`
parallel workers (default 32). It also carries `mean_over_p50` (how far the
tail lifts the budget above the median), MTL's share of the mean runtime, and
two memory budgets for `--cores` concurrent workers: the largest per-TG peak
(the channel drawn) times the workers, and the largest per-process RSS
(`run.json` `run_peak_rss_bytes` over the method's folds) times the workers --
the second is what a machine must actually hold, interpreter and inputs
included. An extrapolation, not a measurement: it assumes contention-free
scaling, and the timings were taken with folds running in parallel.

## Pooling

Every run's per-TG rows are concatenated, then percentiled. Coverage is strict,
using the error CDF's guards: a method absent from any run is refused, and so
are shared TG ids and repeated dataset labels. A micro-average: a dataset weighs by its
TG count.

Command: `plot-cost-box`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.v5.modules import cost as C  # noqa: E402
from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules import figure_error_cdf as E  # noqa: E402
from scripts.analysis.v5.modules.methods import (  # noqa: E402
    method_colors,
    method_label,
    method_order,
    method_sort_key,
    method_term_table,
    methods_source,
)
from scripts.analysis.v5.modules.paths import COST_KIND, RunPaths  # noqa: E402
from scripts.analysis.v5.modules.status import SHORTEST_PING, solved_mask  # noqa: E402

PER_RUN = E.PER_RUN
POOLED = E.POOLED
LAYOUTS = E.LAYOUTS

RUNTIME = "runtime"
DEFAULT_MEMORY = "memory_heap"
DEFAULT_ROWS = "all"

STEM = "cost_box"

#: `memory_heap` -> `heap` in filenames.
_MEMORY_TOKEN = {"memory_heap": "heap", "memory_alloc": "alloc"}

#: Offset of each box from its slot centre, and the box width (x units).
BOX_OFFSET = 0.2
BOX_WIDTH = 0.32

#: The memory box's hatch. Diagonal, so it does not read as a gridline.
MEMORY_HATCH = "//////"

#: The key's proxies are drawn in neutral ink: they name the axis, not a method.
_KEY_INK = E._INK_2

#: An overlay frame's width relative to `BOX_WIDTH`. Wider, so its vertical
#: edges clear the host box's; at 1.2 the two frames of one slot still do not
#: touch (0.2 offset - 0.192 half-width).
OVERLAY_WIDTH_FACTOR = 1.2
OVERLAY_DASH = (0, (2.4, 1.4))

#: Key text for an overlay variant. Its `methods.METHOD_LABELS` term
#: (OCT-H-GEO) is not one the paper defines, so the key spells out the change.
OVERLAY_LABELS: dict[str, str] = {
    "octant_cbg_hull_geo": "OCT-H with GEO CTR",
    "octant_cbg_spl_geo": "OCT-S with GEO CTR",
}

#: The axis labels. "per TG" is the title's job (the paper's caption); on a
#: short panel the longer labels do not fit the axis height. The memory channel
#: (heap / alloc) is in the filename and the manifest.
RUNTIME_AXIS_LABEL = "Runtime (ms)"
MEMORY_AXIS_LABEL = "Peak memory (MB)"

#: Short and wide: one text-width row, like the has-X / no-X RTT boxes.
FIGSIZE: tuple[float, float] = (5.45, 1.5)

CSV_COLUMNS: tuple[str, ...] = (
    "run_id", "dataset", "rows", "method", "method_label", "overlay_on",
    "channel", "unit", "reduce", "stage",
    "n_rows", "n_solved", "n_null", "n",
    *C.STAT_QUANTILES, "mean", "min", "max",
)


def artifact_names(layout: str, memory: str, rows: str) -> dict[str, str]:
    """`{png, csv, manifest}` for one layout / memory channel / row policy.

    The channel and the policy are in the name, so a second figure never
    overwrites the first: `cost_box.heap.png`, `cost_box.pooled.alloc.solved.png`.
    """
    parts = [STEM]
    if layout == POOLED:
        parts.append("pooled")
    parts.append(_MEMORY_TOKEN[memory])
    if rows != DEFAULT_ROWS:
        parts.append(rows)
    stem = ".".join(parts)
    return {"png": f"{stem}.png", "csv": f"{stem}.csv", "manifest": f"{stem}.manifest.json"}


def validate(memory: str, rows: str) -> None:
    if memory not in C.MEMORY_SPECS:
        raise ValueError(f"unknown memory channel {memory!r}; pick from {list(C.MEMORY_SPECS)}")
    if rows not in C.COST_ROWS:
        raise ValueError(f"unknown rows policy {rows!r}; pick from {list(C.COST_ROWS)}")


def parse_overlays(specs: list[str] | None) -> dict[str, str]:
    """`["HOST=VARIANT", ...]` -> `{host: variant}`. One variant per host."""
    out: dict[str, str] = {}
    for spec in specs or ():
        host, sep, variant = spec.partition("=")
        host, variant = host.strip(), variant.strip()
        if not sep or not host or not variant:
            raise ValueError(f"overlay {spec!r} must read HOST=VARIANT")
        if host == variant:
            raise ValueError(f"overlay {spec!r} draws a method on itself")
        if host in out:
            raise ValueError(f"host {host!r} has two overlays: {out[host]!r}, {variant!r}")
        out[host] = variant
    if set(out) & set(out.values()):
        raise ValueError(f"an overlay variant cannot also host one: {out}")
    return out


# ---- loading ----------------------------------------------------------------


def costed_methods(run: RunPaths, methods: list[str] | None) -> list[str]:
    """The combos to draw, in `TERM_ORDER`. S-P is never one.

    Named methods the run does not hold are refused rather than skipped, so a
    typo cannot quietly drop a box.
    """
    have = set(run.combo_ids)
    if methods is None:
        return method_order(have)
    wanted = [m for m in dict.fromkeys(methods) if m != SHORTEST_PING]
    missing = sorted(set(wanted) - have)
    if missing:
        raise ValueError(f"{run.run_id} holds no combo {missing}; it has {sorted(have)}")
    if not wanted:
        raise ValueError(f"{SHORTEST_PING} has no cost; name at least one CBG combo")
    return method_order(wanted)


def drawn_methods(
    run: RunPaths, methods: list[str] | None, overlays: dict[str, str] | None = None
) -> list[str]:
    """Slot methods, then overlay variants: everything `load_run` reads.

    A variant is never a slot, even when `methods` is None (every combo on
    disk) or names it. Its host must be a slot -- an overlay with nothing
    under it would be a box drawn on no method.
    """
    overlays = overlays or {}
    slots = [m for m in costed_methods(run, methods) if m not in overlays.values()]
    if missing := sorted(set(overlays) - set(slots)):
        raise ValueError(f"overlay host {missing} is not a drawn method; slots are {slots}")
    if overlays:
        costed_methods(run, list(overlays.values()))  # refuses a variant not on disk
    return [*slots, *overlays.values()]


def load_run(
    run: RunPaths, *, memory: str, rows: str, methods: list[str] | None = None,
    overlays: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """`{method: per-TG frame}` holding both channels' raw columns, one row mask.

    Each frame also gets `run_id`, so a pooled frame keeps each row's origin.
    Overlay variants are loaded like any method; only the drawing tells them
    apart.
    """
    specs = (C.COST_SPECS[RUNTIME], C.COST_SPECS[memory])
    out: dict[str, pd.DataFrame] = {}
    for method in drawn_methods(run, methods, overlays):
        df = C.load_cost_frame(run, method, specs, rows="all")
        df["solved"] = solved_mask(df).to_numpy()
        if rows == "solved":
            df = df[df["solved"]].reset_index(drop=True)
        df.insert(0, "run_id", run.run_id)
        out[method] = df
    return out


def stack_runs(by_run: dict[str, dict[str, pd.DataFrame]]) -> dict[str, pd.DataFrame]:
    """Concatenate every run's frames per method, behind the strict guards."""
    methods = cross.guard_common_methods(
        {rid: set(frames) for rid, frames in by_run.items()}, remedy=E.REMEDY_COMMON
    )
    any_method = methods[0] if methods else None
    if any_method is not None:
        cross.guard_disjoint_tgs(
            {rid: set(frames[any_method]["tg_id"]) for rid, frames in by_run.items()},
            remedy=E.REMEDY_DISJOINT,
        )
    cross.guard_distinct_labels(cross.labels_for(list(by_run)), remedy=E.REMEDY_DISJOINT)
    return {
        m: pd.concat([frames[m] for frames in by_run.values()], ignore_index=True)
        for m in method_order(methods)
    }


# ---- stats ------------------------------------------------------------------


def stats_table(
    frames: dict[str, pd.DataFrame], *, memory: str, overlays: dict[str, str] | None = None
) -> pd.DataFrame:
    """One row per `(method, channel, stage)`, the pipeline row last per channel.

    Refuses a channel that is present but entirely NULL (`cost.require_measured`)
    -- a box for something never measured would read as a measurement.
    `overlay_on` is the host a variant is drawn on, empty for a slot method.
    """
    host_of = {variant: host for host, variant in (overlays or {}).items()}
    records = []
    for method, df in frames.items():
        for key in (RUNTIME, memory):
            spec = C.COST_SPECS[key]
            table = C.stage_cost_table(df, spec)
            C.require_measured(
                table[C.PIPELINE], spec, run_id="+".join(sorted(set(df["run_id"]))),
                method=method,
            )
            for stage in (*C.STAGES, C.PIPELINE):
                records.append({
                    "method": method,
                    "method_label": method_label(method),
                    "overlay_on": host_of.get(method, ""),
                    "channel": key,
                    "unit": spec.unit,
                    "reduce": spec.reduce,
                    "stage": stage,
                    "n_rows": len(df),
                    "n_solved": int(df["solved"].sum()),
                    **table[stage],
                })
    return pd.DataFrame.from_records(records)


def _box(block: dict[str, float]) -> dict:
    """An `Axes.bxp` stats dict: whiskers p5/p95, hinges p25/p75, no fliers."""
    return {
        "whislo": block["p5"], "q1": block["p25"], "med": block["p50"],
        "q3": block["p75"], "whishi": block["p95"], "fliers": [],
    }


def _slot_methods(pipe: pd.DataFrame) -> list[str]:
    """The pipeline rows' methods that take a slot (not overlay variants)."""
    return list(pipe.loc[pipe["overlay_on"].fillna("") == "", "method"].unique())


def cost_order(table: pd.DataFrame, *, memory: str) -> list[str]:
    """Slot methods cheapest first: p50 runtime, then p50 memory, ascending.

    `TERM_ORDER` breaks an exact tie on both, so the order is deterministic.
    Overlay variants are not ranked; they sit on their host's slot.
    """
    pipe = table[table["stage"] == C.PIPELINE]
    p50 = pipe.set_index(["method", "channel"])["p50"]
    return sorted(
        _slot_methods(pipe),
        key=lambda m: (p50[(m, RUNTIME)], p50[(m, memory)], method_sort_key(m)),
    )


# ---- drawing ----------------------------------------------------------------


def _style_log_axis(ax, label: str, *, side: str) -> None:
    from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter

    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10))  # 10^k
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(label, fontsize=E._LABEL_PT, color=E._INK_2)
    ax.tick_params(axis="y", which="both", colors=E._MUTED, labelsize=E._TICK_PT)
    ax.spines[side].set_color(E._AXIS)


def _overlay_box(ax, stats: dict, x: float, *, hatch: str | None = None) -> None:
    """A variant's box on its host's position: unfilled, dashed, in ink, on top.

    `hatch` carries the channel's texture onto the frame (the memory box's
    diagonal), so an overlay box reads as runtime or memory like any other.
    """
    line = {"color": E._INK, "linewidth": 0.9, "linestyle": OVERLAY_DASH}
    ax.bxp(
        [stats], positions=[x], widths=BOX_WIDTH * OVERLAY_WIDTH_FACTOR,
        showfliers=False, patch_artist=True, manage_ticks=False, zorder=4,
        boxprops={"facecolor": "none", "edgecolor": E._INK, "linewidth": 0.9,
                  "linestyle": OVERLAY_DASH, **({"hatch": hatch} if hatch else {})},
        medianprops=line, whiskerprops=line,
        capprops={"color": E._INK, "linewidth": 0.9},
    )


def plot_boxes(
    table: pd.DataFrame, out_path: Path, *, memory: str,
    figsize: tuple[float, float] = FIGSIZE, dpi: int = 300,
) -> Path:
    """One slot per method: runtime (left axis, solid), memory (right, hatched).

    Slots run cheapest first (`cost_order`). A row whose `overlay_on` names a
    host is drawn on that host's slot as a dashed, unfilled frame. No title:
    the dataset and TG count are in the manifest, and the paper captions it.
    The key sits on one row above the plot, clear of every box.
    """
    from matplotlib.patches import Patch

    pipe = table[table["stage"] == C.PIPELINE]
    methods = cost_order(table, memory=memory)
    colors = method_colors(methods)
    block = {
        (r.method, r.channel): r._asdict() for r in pipe.itertuples(index=False)
    }
    overlay_of = {
        r.overlay_on: r.method
        for r in pipe.itertuples(index=False) if isinstance(r.overlay_on, str) and r.overlay_on
    }
    xs = np.arange(len(methods), dtype=float)

    fig, ax_rt = plt.subplots(figsize=figsize)
    ax_mem = ax_rt.twinx()
    fig.patch.set_facecolor(E._SURFACE)
    ax_rt.set_facecolor(E._SURFACE)

    for i, method in enumerate(methods):
        hue = colors[method]
        ax_rt.bxp(
            [_box(block[(method, RUNTIME)])], positions=[xs[i] - BOX_OFFSET],
            widths=BOX_WIDTH, showfliers=False, patch_artist=True, manage_ticks=False,
            boxprops={"facecolor": hue, "edgecolor": hue, "linewidth": 0.9},
            medianprops={"color": E._SURFACE, "linewidth": 1.4},
            whiskerprops={"color": hue, "linewidth": 0.9},
            capprops={"color": hue, "linewidth": 0.9},
        )
        ax_mem.bxp(
            [_box(block[(method, memory)])], positions=[xs[i] + BOX_OFFSET],
            widths=BOX_WIDTH, showfliers=False, patch_artist=True, manage_ticks=False,
            boxprops={
                "facecolor": E._SURFACE, "edgecolor": hue, "linewidth": 0.9,
                "hatch": MEMORY_HATCH,
            },
            medianprops={"color": E._INK, "linewidth": 1.4},
            whiskerprops={"color": hue, "linewidth": 0.9, "linestyle": (0, (2, 1.2))},
            capprops={"color": hue, "linewidth": 0.9},
        )
        if (variant := overlay_of.get(method)) is not None:
            for ax, channel, dx, hatch in ((ax_rt, RUNTIME, -BOX_OFFSET, None),
                                           (ax_mem, memory, BOX_OFFSET, MEMORY_HATCH)):
                _overlay_box(ax, _box(block[(variant, channel)]), xs[i] + dx, hatch=hatch)

    _style_log_axis(ax_rt, RUNTIME_AXIS_LABEL, side="left")
    _style_log_axis(ax_mem, MEMORY_AXIS_LABEL, side="right")
    ax_rt.set_xlim(-0.6, len(methods) - 0.4)
    ax_rt.set_xticks(xs)
    ax_rt.set_xticklabels([method_label(m) for m in methods], fontsize=E._TICK_PT + 0.5,
                          color=E._INK)
    ax_rt.tick_params(axis="x", length=0)
    # Horizontal grid from the runtime axis only: one set of lines, one axis.
    ax_rt.grid(True, axis="y", which="major", color=E._GRID, linewidth=0.5, zorder=0)
    ax_rt.set_axisbelow(True)
    for spine_ax in (ax_rt, ax_mem):
        spine_ax.spines["top"].set_visible(False)
        spine_ax.spines["bottom"].set_color(E._AXIS)
    ax_rt.spines["right"].set_visible(False)
    ax_mem.spines["left"].set_visible(False)
    # Slot separators, so a reader pairs the two boxes of one method.
    for x in xs[:-1] + 0.5:
        ax_rt.axvline(x, color=E._GRID, linewidth=0.6, zorder=0)

    handles = [
        Patch(facecolor=_KEY_INK, edgecolor=_KEY_INK, label="Runtime (left axis)"),
        Patch(facecolor=E._SURFACE, edgecolor=_KEY_INK, hatch=MEMORY_HATCH,
              label="Peak memory (right axis)"),
    ]
    handles += [
        Patch(facecolor="none", edgecolor=E._INK, linestyle=OVERLAY_DASH, linewidth=0.9,
              label=OVERLAY_LABELS.get(variant, method_label(variant)))
        for host, variant in overlay_of.items() if host in methods
    ]
    # One row above the plot: on a short panel an inside key covers OCT-H's top.
    legend = ax_mem.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
        fontsize=E._LEGEND_PT, frameon=False, handlelength=1.4, columnspacing=1.6,
        borderaxespad=0.2,
    )
    for text in legend.get_texts():
        text.set_color(E._INK_2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor=E._SURFACE)
    plt.close(fig)
    return out_path


# ---- artifacts --------------------------------------------------------------


def _manifest(
    layout: str, *, names: dict[str, str], table: pd.DataFrame, run_ids: list[str],
    memory: str, rows: str, per_run: dict[str, int], source: str = "all",
) -> str:
    pipe = table[table["stage"] == C.PIPELINE]
    slots = cost_order(table, memory=memory)
    overlays = {
        r.overlay_on: r.method
        for r in pipe.itertuples(index=False) if isinstance(r.overlay_on, str) and r.overlay_on
    }
    methods = [*slots, *overlays.values()]
    body = {
        "figure": "cost_box",
        "layout": layout,
        "run_ids": sorted(run_ids),
        "dataset": cross.dataset_slug(run_ids),
        "artifacts": names,
        "methods": slots,
        "overlays": {
            host: {
                "variant": variant,
                "label": OVERLAY_LABELS.get(variant, method_label(variant)),
                "mark": (
                    "dashed unfilled frame in ink on the host's slot, both axes; the "
                    "memory frame carries the memory hatch"
                ),
            }
            for host, variant in overlays.items()
        },
        "methods_source": source,
        "method_terms": method_term_table(methods),
        "excluded": {
            SHORTEST_PING: "not a combo: no LTD/MTL/CTR stage is timed or measured",
        },
        "axes": {
            "x": "method, cheapest first: p50 runtime, then p50 memory, ascending",
            "y_left": {
                "channel": RUNTIME, "unit": "ms", "scale": "log",
                "columns": list(C.COST_SPECS[RUNTIME].stage_cols),
                "reduce": "sum across stages, per TG, before any percentile",
                "mark": "solid box, left of each slot",
            },
            "y_right": {
                "channel": memory, "unit": "MB", "scale": "log",
                "columns": list(C.COST_SPECS[memory].stage_cols),
                "reduce": "max across stages (pipeline high-water mark), per TG",
                "mark": "hatched box, right of each slot",
            },
        },
        "box": {
            "whiskers": "p5 / p95", "hinges": "p25 / p75", "line": "p50",
            "fliers": "none drawn; min/max in the CSV",
        },
        "rows": {
            "policy": rows,
            "rule": (
                "every evaluated TG, FALLBACK included -- a method pays for a TG it "
                "gave up on" if rows == "all" else "status.solved_mask rows only"
            ),
            "shared_denominator": "runtime and memory come from one frame, one mask",
            "null_stage": "a NULL stage never ran and costs 0 (e.g. CTR on FALLBACK)",
        },
        "not_counted": (
            "the per-(combo, fold) LTD fit (run.json fit_ms / fit_*_peak_bytes): "
            "amortised over the fold, not a per-TG cost"
        ),
        "n_tgs_by_method": {m: int(pipe[pipe["method"] == m]["n_rows"].iloc[0]) for m in methods},
        "p50_by_method": {
            m: {
                ch: round(float(pipe[(pipe["method"] == m) & (pipe["channel"] == ch)]["p50"].iloc[0]), 4)
                for ch in (RUNTIME, memory)
            }
            for m in methods
        },
    }
    if layout == POOLED:
        body["pooling"] = {
            "rule": "micro-pool: every run's per-TG rows concatenated, then percentiled",
            "n_tgs_by_run": per_run,
            "coverage": "strict -- a method absent from any run, shared TG ids and "
                        "repeated dataset labels are refused",
        }
    return json.dumps(body, indent=2) + "\n"


#: Defaults for the extrapolation: one million TGs on one 32-core machine.
EXTRAPOLATE_TGS = 1_000_000
CORES = 32

_MS_PER_HOUR = 3_600_000.0


def peak_rss_mb(runs: list[RunPaths], methods: list[str]) -> dict[str, dict[str, float]]:
    """`{run_id: {method: max run_peak_rss_bytes over folds, in MB}}`; NaN when unrecorded."""
    out: dict[str, dict[str, float]] = {}
    for run in runs:
        out[run.run_id] = {}
        for method in methods:
            peaks = []
            for fold in run.fold_ids:
                path = run.combo_dir(method, fold) / "run.json"
                if path.exists():
                    v = json.loads(path.read_text()).get("run_peak_rss_bytes")
                    if v is not None:
                        peaks.append(float(v))
            out[run.run_id][method] = max(peaks) * C.COST_SPECS[DEFAULT_MEMORY].scale if peaks else float("nan")
    return out


def extrapolation_table(
    frames: dict[str, pd.DataFrame], *, memory: str, n_tgs: int = EXTRAPOLATE_TGS,
    cores: int = CORES, rss: dict[str, dict[str, float]] | None = None,
    overlays: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Per `(scope, method)`: the batch budget for `n_tgs` TGs on `cores` workers."""
    host_of = {variant: host for host, variant in (overlays or {}).items()}
    runtime, mem = C.COST_SPECS[RUNTIME], C.COST_SPECS[memory]
    rows = []
    for method, df in frames.items():
        run_ids = sorted(set(df["run_id"]))
        scopes = [(r, df[df["run_id"] == r]) for r in run_ids]
        if len(run_ids) > 1:
            scopes.append(("all", df))
        for scope, g in scopes:
            total = C.per_target_cost(g, runtime)
            mtl = C.per_stage_cost(g, runtime)["mtl"].to_numpy(float)
            mean, p50 = float(np.nanmean(total)), float(np.nanmedian(total))
            heap_max = float(np.nanmax(C.per_target_cost(g, mem)))
            rss_runs = [(rss or {}).get(r, {}).get(method, float("nan")) for r in sorted(set(g["run_id"]))]
            rss_max = float(np.nanmax(rss_runs)) if np.isfinite(rss_runs).any() else float("nan")
            core_h = mean * n_tgs / _MS_PER_HOUR
            rows.append({
                "scope": scope, "method": method, "method_label": method_label(method),
                "overlay_on": host_of.get(method, ""), "n_rows": int(len(g)),
                "runtime_mean_ms": mean, "runtime_p50_ms": p50,
                "mean_over_p50": mean / p50 if p50 > 0 else float("nan"),
                "mtl_share_of_mean": float(np.nanmean(mtl)) / mean if mean > 0 else float("nan"),
                "extrapolate_tgs": int(n_tgs), "cores": int(cores),
                "core_hours": core_h, "wall_hours": core_h / cores, "wall_days": core_h / cores / 24,
                "wall_minutes": core_h / cores * 60,
                "memory_channel": memory, "peak_per_tg_max_mb": heap_max,
                "peak_per_tg_x_cores_mb": heap_max * cores,
                "process_rss_max_mb": rss_max, "process_rss_x_cores_mb": rss_max * cores,
            })
    return pd.DataFrame(rows)


def extrapolation_name(names: dict[str, str]) -> str:
    """`cost_box.pooled.heap.csv` -> `cost_box.pooled.heap.extrapolation.csv`."""
    return names["csv"].removesuffix(".csv") + ".extrapolation.csv"


def _write(
    frames: dict[str, pd.DataFrame], out_dir: Path, layout: str, *,
    run_ids: list[str], memory: str, rows: str, source: str = "all",
    overlays: dict[str, str] | None = None,
    extrapolate_tgs: int = EXTRAPOLATE_TGS, cores: int = CORES,
    rss: dict[str, dict[str, float]] | None = None,
) -> dict[str, Path]:
    names = artifact_names(layout, memory, rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = stats_table(frames, memory=memory, overlays=overlays)
    table.insert(0, "run_id", "+".join(sorted(run_ids)))
    table.insert(1, "dataset", cross.dataset_slug(run_ids))
    table.insert(2, "rows", rows)
    written = {k: out_dir / v for k, v in names.items()}
    table[list(CSV_COLUMNS)].to_csv(written["csv"], index=False)
    extrapolation_table(
        frames, memory=memory, n_tgs=extrapolate_tgs, cores=cores, rss=rss, overlays=overlays,
    ).to_csv(out_dir / extrapolation_name(names), index=False)
    plot_boxes(table, written["png"], memory=memory)
    any_frame = next(iter(frames.values()))
    per_run = {str(k): int(v) for k, v in any_frame["run_id"].value_counts().sort_index().items()}
    written["manifest"].write_text(
        _manifest(layout, names=names, table=table, run_ids=run_ids,
                  memory=memory, rows=rows, per_run=per_run, source=source)
    )
    return written


def build_for_run(
    run: RunPaths, *, memory: str = DEFAULT_MEMORY, rows: str = DEFAULT_ROWS,
    methods: list[str] | None = None, analysis_root: Path | None = None,
    source: str | None = None, overlays: dict[str, str] | None = None,
    extrapolate_tgs: int = EXTRAPOLATE_TGS, cores: int = CORES,
) -> dict[str, Path]:
    """One run's figure, CSV and manifest, into `<run>/cost/`."""
    validate(memory, rows)
    frames = load_run(run, memory=memory, rows=rows, methods=methods, overlays=overlays)
    return _write(
        frames, run.analysis_dir(COST_KIND, root=analysis_root), PER_RUN,
        run_ids=[run.run_id], memory=memory, rows=rows,
        source=methods_source(methods, source), overlays=overlays,
        extrapolate_tgs=extrapolate_tgs, cores=cores, rss=peak_rss_mb([run], list(frames)),
    )


def build_for_runs(
    runs: list[RunPaths], *, layouts: tuple[str, ...] = (PER_RUN,),
    memory: str = DEFAULT_MEMORY, rows: str = DEFAULT_ROWS,
    methods: list[str] | None = None, analysis_root: Path | None = None,
    source: str | None = None, overlays: dict[str, str] | None = None,
    extrapolate_tgs: int = EXTRAPOLATE_TGS, cores: int = CORES,
) -> list[dict[str, Path]]:
    """Render the requested layouts; one artifact set per figure, layout-major.

    `extrapolate_tgs`/`cores` size the extrapolation table (see the module docstring).
    """
    ordered = tuple(dict.fromkeys(layouts)) or (PER_RUN,)
    unknown = [x for x in ordered if x not in LAYOUTS]
    if unknown:
        raise ValueError(f"unknown layout {unknown}; pick from {list(LAYOUTS)}")
    validate(memory, rows)
    run_ids = [r.run_id for r in runs]
    out: list[dict[str, Path]] = []
    for layout in ordered:
        if layout == PER_RUN:
            out.extend(
                build_for_run(run, memory=memory, rows=rows, methods=methods,
                              analysis_root=analysis_root, source=source, overlays=overlays,
                              extrapolate_tgs=extrapolate_tgs, cores=cores)
                for run in runs
            )
            continue
        by_run = {
            run.run_id: load_run(
                run, memory=memory, rows=rows, methods=methods, overlays=overlays
            )
            for run in runs
        }
        frames = stack_runs(by_run)
        out.append(
            _write(
                frames, cross.cross_dir(run_ids, analysis_root=analysis_root, kind=COST_KIND),
                POOLED, run_ids=run_ids, memory=memory, rows=rows,
                source=methods_source(methods, source), overlays=overlays,
                extrapolate_tgs=extrapolate_tgs, cores=cores,
                rss=peak_rss_mb(runs, list(frames)),
            )
        )
    return out
