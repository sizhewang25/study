"""Accuracy against runtime per regime, with the Pareto frontier -- `plot-pareto`.

One panel per (accuracy, regime) -- unbounded or bounded accuracy, seen or
unseen sites -- each its own file, every panel pooling its runs' TGs. Each CBG
method is one dot:

* **y -- accuracy**. Unbounded (`cell`): the prediction falls in the TG's
  cell. Bounded: in the TG's cell *and* within `BOUND_PIXELS` pixels of the
  TG's pixel (`classify`'s `cell_label` and `pred_dist_to_tg_grid`). Every TG
  counts, unanswered ones included, as in every accuracy of the paper. A
  FALLBACK row is unanswered (`status.solved_mask`), although `classify`
  labels its prediction: VAN writes a fallback location on every TG it gives
  up on, which lands in the right cell nearly always and would lift VAN's
  accuracy by ~20 points if it counted. The
  gap between the two rows is the unboundedness dividend. The dot is the
  accuracy over every TG of the regime's datasets pooled, i.e. the TG-weighted
  average of the per-dataset accuracies -- the number the pooled outcome bars
  and Table 3 print. The bar spans the lowest to the highest **per-dataset**
  accuracy (one run = one content network): a spread across networks, not a
  confidence interval -- with three datasets there is no usable one. The
  unweighted mean of the datasets is in the CSV (`acc_dataset_mean`).
* **x -- runtime per TG**, the three stages summed (`cost.per_target_cost`),
  over the same TGs. The dot is the median and the bar spans p25-p75: the
  runtime is a distribution over TGs spanning decades, not an estimate with
  an error.

Each panel uses its own regime's runtime: OCT-H's tail differs ~4x between the
regimes, so a pooled runtime would be right for neither.

## The frontier

A method is on the frontier when no method is at least as fast (p50) and at
least as accurate, one of them strictly. S-P has no timed stage and costs a
sort of the RTTs, so it is drawn as a horizontal line at its accuracy rather
than a dot on a log axis, and it takes part in the frontier as the zero-cost
point: a CBG method at or below S-P's accuracy is dominated by it. Its
dataset range is a shaded band (bottom edge its worst network, top its best),
the line's equivalent of a dot's vertical bar. Most of its width is how much
easier AS-A is for every method, so a dot inside the band can still beat S-P
in every network (`pareto.pairs.csv`). The
frontier members are joined by a dotted line, fastest first.

The frontier is on the pooled accuracies. Whether two methods really differ is
`pareto.pairs.csv`: per pair, the accuracy difference in every dataset (mean,
lowest, highest) and in how many datasets the first one wins. A lead in every
dataset is the claim the figure can support; a lead in two of three is not.
`pareto.datasets.csv` holds each dataset's own accuracy and median runtime.

Memory is not drawn: every method's per-TG peak is under 25 MB, and OCT-H's
24 MB is its Monte Carlo medoid's fixed buffer (`plot-cost-box`).

Command: `plot-pareto`. Needs `classify` on every run. Writes
`pareto.<metric>.<regime>.png` (one panel each, cell/bounded x seen/unseen,
on shared axis limits so they tile as a 2x2 grid) and
`pareto.{csv,datasets.csv,pairs.csv,manifest.json}` into `_cross/pareto/<n>-runs-<hash>/`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter  # noqa: E402

from scripts.analysis.v5.modules import classify as C  # noqa: E402
from scripts.analysis.v5.modules import cost as K  # noqa: E402
from scripts.analysis.v5.modules import cross  # noqa: E402
from scripts.analysis.v5.modules import figure_error_cdf as E  # noqa: E402
from scripts.analysis.v5.modules import grid as G  # noqa: E402
from scripts.analysis.v5.modules import sites as SITES  # noqa: E402
from scripts.analysis.v5.modules.methods import (  # noqa: E402
    LABEL_HUES,
    method_colors,
    method_label,
    method_order,
    method_term_table,
)
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths  # noqa: E402
from scripts.analysis.v5.modules.status import SHORTEST_PING, solved_mask  # noqa: E402

KIND = "pareto"
SOURCE_NSIDE = G.NSIDE_LADDER[0]

#: A correct prediction must also sit within this many pixels of the TG's.
BOUND_PIXELS = C.MAX_RING

SEEN = "seen"
UNSEEN = "unseen"
REGIMES = (SEEN, UNSEEN)
REGIME_TITLES = {SEEN: "Seen sites", UNSEEN: "Unseen sites"}

#: `(metric, per-TG column, axis label)`, top row first.
METRICS = (
    ("cell", "correct", "Unbounded accuracy (%)"),
    ("bounded", "bounded", "Bounded accuracy (%)"),
)

#: One PNG per (metric, regime) panel: `pareto.<metric>.<regime>.png`.
PANEL_PNG = "pareto.{metric}.{regime}.png"

NAMES = {
    "csv": "pareto.csv",
    "datasets": "pareto.datasets.csv",
    "pairs": "pareto.pairs.csv",
    "manifest": "pareto.manifest.json",
}

#: One panel per file, short and wide: two of them side by side fill the text
#: width (2 x 2.7 in), so the four print as a 2x2 grid.
PANEL_FIGSIZE = (2.7, 1.45)
_LABEL_PT = 6.5
_TICK_PT = 6.0
_TEXT_PT = 6.0

_TGS_COLUMNS = ("tg_id", "tg_lat", "tg_lon", "status", "cell_label", C.GRID_OFFSET)


# ---- loading ----------------------------------------------------------------


def _classified(run: RunPaths, method: str, root: Path | None) -> pd.DataFrame:
    path = run.classify_dir(SOURCE_NSIDE, root=root) / C.TGS_PARQUET.format(method=method)
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; run `classify --run-id {run.run_id}` first")
    df = pd.read_parquet(path, columns=list(_TGS_COLUMNS))
    if df["tg_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate tg_id")
    return df


def cell_correct(df: pd.DataFrame) -> np.ndarray:
    """Right cell, and answered: a FALLBACK prediction is not an answer."""
    return (df["cell_label"] == "correct").to_numpy() & solved_mask(df).to_numpy()


def bounded_correct(df: pd.DataFrame) -> np.ndarray:
    """`cell_correct` and within `BOUND_PIXELS` of the TG's pixel."""
    offset = df[C.GRID_OFFSET].to_numpy()
    return cell_correct(df) & (offset >= 0) & (offset <= BOUND_PIXELS)


def load_regime(
    runs: list[RunPaths], methods: list[str], *, root: Path | None = None
) -> pd.DataFrame:
    """One row per (method, TG) over the regime's runs: site, correctness, runtime.

    Runtime is NaN for S-P (no timed stage). A CBG method whose timed TGs are
    not exactly its classified TGs is refused: accuracy and cost must share
    one denominator.
    """
    cross.guard_disjoint_tgs(
        {r.run_id: set(_classified(r, methods[0], root)["tg_id"]) for r in runs},
        remedy="pool runs over different TGs (one regime's meshes).",
    )
    rows = []
    for run in runs:
        for method in methods:
            df = _classified(run, method, root)
            frame = pd.DataFrame({
                "run_id": run.run_id,
                "method": method,
                "tg_id": df["tg_id"].to_numpy(),
                "site_key": SITES.site_key(df, run_id=run.run_id).to_numpy(),
                "correct": cell_correct(df),
                "bounded": bounded_correct(df),
                "runtime_ms": np.nan,
            })
            if method != SHORTEST_PING:
                spec = K.COST_SPECS["runtime"]
                cost = K.load_cost_frame(run, method, (spec,), rows="all")
                ms = pd.Series(K.per_target_cost(cost, spec), index=cost["tg_id"].to_numpy())
                if set(ms.index) != set(frame["tg_id"]):
                    raise ValueError(
                        f"{run.run_id}/{method}: the timed TGs are not the classified TGs; "
                        f"re-run `classify` on this benchmark tree."
                    )
                frame["runtime_ms"] = ms.loc[frame["tg_id"]].to_numpy()
            rows.append(frame)
    return pd.concat(rows, ignore_index=True)


# ---- numbers ----------------------------------------------------------------


def per_dataset(long: pd.DataFrame, regime: str) -> pd.DataFrame:
    """One row per (metric, method, run): that dataset's accuracy and runtime median."""
    rows = []
    for (method, run_id), g in long.groupby(["method", "run_id"], sort=False):
        rt = g["runtime_ms"].to_numpy(float)
        for metric, col, _ in METRICS:
            rows.append({
                "regime": regime, "metric": metric, "method": method,
                "method_label": method_label(method), "run_id": run_id,
                "dataset": cross.short_dataset(run_id),
                "n_tgs": len(g), "n_sites": int(g["site_key"].nunique()),
                "acc": float(g[col].mean()),
                "runtime_p50_ms": float(np.median(rt)) if np.isfinite(rt).any() else np.nan,
            })
    return pd.DataFrame(rows)


def summarize(long: pd.DataFrame, regime: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """`(per-(metric, method) table, per-dataset rows, pairwise differences)` for one regime.

    The dot is the TG-pooled accuracy; the bar the datasets' min-max. Runtime stays a
    distribution over every TG of the regime (p25/p50/p75).
    """
    ds = per_dataset(long, regime)
    rows = []
    for (metric, method), g in ds.groupby(["metric", "method"], sort=False):
        rt = long.loc[long["method"] == method, "runtime_ms"].to_numpy(float)
        timed = np.isfinite(rt).any()
        tgs = long[long["method"] == method]
        col = dict((m, c) for m, c, _ in METRICS)[metric]
        rows.append({
            "regime": regime, "metric": metric, "method": method,
            "method_label": method_label(method),
            "n_datasets": len(g), "n_tgs": int(g["n_tgs"].sum()),
            "n_sites": int(g["n_sites"].sum()),
            "acc": float(tgs[col].mean()),
            "acc_min": float(g["acc"].min()),
            "acc_max": float(g["acc"].max()),
            "acc_dataset_mean": float(g["acc"].mean()),
            **{f"runtime_{k}_ms": (float(np.percentile(rt, q)) if timed else np.nan)
               for k, q in (("p25", 25), ("p50", 50), ("p75", 75))},
            "runtime_mean_ms": float(rt.mean()) if timed else np.nan,
        })
    table = pd.DataFrame(rows)
    table["on_frontier"] = False
    for metric, _, _ in METRICS:
        sel = table["metric"] == metric
        table.loc[sel, "on_frontier"] = table.loc[sel, "method"].isin(frontier(table[sel]))

    pairs = []
    for metric, _, _ in METRICS:
        wide = ds[ds["metric"] == metric].pivot(index="run_id", columns="method", values="acc")
        ms = list(dict.fromkeys(ds["method"]))
        for i, a in enumerate(ms):
            for b in ms[i + 1:]:
                d = (wide[a] - wide[b]).to_numpy(float)
                pairs.append({
                    "regime": regime, "metric": metric, "a": a, "b": b,
                    "d_acc_mean": float(d.mean()), "d_acc_min": float(d.min()),
                    "d_acc_max": float(d.max()),
                    "n_datasets": len(d), "n_a_wins": int((d > 0).sum()),
                    "n_b_wins": int((d < 0).sum()),
                })
    pairs = pd.DataFrame(pairs)
    pairs["consistent"] = (pairs["n_a_wins"] == pairs["n_datasets"]) | (
        pairs["n_b_wins"] == pairs["n_datasets"])
    return table, ds, pairs


def frontier(table: pd.DataFrame) -> list[str]:
    """CBG methods no other is at least as fast and accurate as, fastest first.

    S-P, when present, is the zero-cost point: a method must beat its accuracy.
    """
    timed = table[np.isfinite(table["runtime_p50_ms"])].sort_values(
        ["runtime_p50_ms", "acc"], ascending=[True, False])
    sp = table.loc[table["method"] == SHORTEST_PING, "acc"]
    best = float(sp.iloc[0]) if len(sp) else -np.inf
    out = []
    for r in timed.itertuples(index=False):
        if r.acc > best:
            out.append(r.method)
            best = r.acc
    return out


# ---- drawing ----------------------------------------------------------------


def _style(ax) -> None:
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_major_formatter(LogFormatterSciNotation(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="both", which="both", colors=E._MUTED, labelsize=_TICK_PT,
                   length=2, pad=1.5)
    ax.grid(True, which="major", color=E._GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(E._AXIS)


#: Label nudges in points, `(dx, dy)` from the top of the method's bar, for
#: methods whose dots sit close: OCT-H and OCT-S share a runtime decade.
_LABEL_NUDGE = {"OCT-S": (-11, 0), "OCT-H": (11, 0)}

_Y_LABELS = {metric: label for metric, _, label in METRICS}

#: S-P's band: its lowest (bottom edge) to highest (top edge) network.
SP_BAND_ALPHA = 0.18


def limits(table: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    """`(xlim, ylim)` shared by every panel, so the four files line up as a grid."""
    timed = table[np.isfinite(table["runtime_p50_ms"])]
    xs = np.concatenate([timed["runtime_p25_ms"], timed["runtime_p75_ms"]])
    ys = 100 * np.concatenate([table["acc_min"], table["acc_max"]])
    return ((xs.min() / 1.6, xs.max() * 1.6),
            (max(0.0, ys.min() - 6), min(100.0, ys.max() + 12)))


def plot_panel(t: pd.DataFrame, out_png: Path, *, metric: str,
               xlim: tuple[float, float], ylim: tuple[float, float]) -> Path:
    """One (metric, regime) panel: dots with bars, frontier dotted, S-P a line."""
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    colors = method_colors(t["method"].unique())
    sp = t[t["method"] == SHORTEST_PING]
    if len(sp):
        r = sp.iloc[0]
        # S-P's worst-to-best network, as the other methods' vertical bars. Dark
        # enough to read apart from the major gridlines it overlaps.
        ax.axhspan(100 * r.acc_min, 100 * r.acc_max,
                   color=LABEL_HUES["S-P"], alpha=SP_BAND_ALPHA, lw=0, zorder=1)
        ax.axhline(100 * r.acc, color=LABEL_HUES["S-P"], lw=0.8, ls=(0, (4, 2)), zorder=2)
        ax.text(1.0, 100 * r.acc, "S-P ", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=_TEXT_PT, color=LABEL_HUES["S-P"])
    front = t[t["on_frontier"]].sort_values("runtime_p50_ms")
    ax.plot(front["runtime_p50_ms"], 100 * front["acc"], color=E._INK_2,
            lw=0.9, ls=":", zorder=3)
    for r in t[np.isfinite(t["runtime_p50_ms"])].itertuples(index=False):
        hue = colors[r.method]
        x, y = r.runtime_p50_ms, 100 * r.acc
        ax.errorbar(
            x, y,
            xerr=[[x - r.runtime_p25_ms], [r.runtime_p75_ms - x]],
            yerr=[[y - 100 * r.acc_min], [100 * r.acc_max - y]],
            fmt="o", ms=3.4, color=hue, mfc=hue if r.on_frontier else E._SURFACE,
            mec=hue, mew=0.9, elinewidth=0.8, capsize=1.6, capthick=0.8, zorder=4,
        )
        dx, dy = _LABEL_NUDGE.get(r.method_label, (0, 0))
        ax.annotate(r.method_label, (x, 100 * r.acc_max), xytext=(dx, 1.5 + dy),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=_TEXT_PT, color=hue, zorder=5)
    _style(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Runtime per TG (ms)", fontsize=_LABEL_PT, color=E._INK_2, labelpad=1)
    ax.set_ylabel(_Y_LABELS[metric], fontsize=_LABEL_PT, color=E._INK_2, labelpad=1)
    fig.tight_layout(pad=0.2)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor=E._SURFACE)
    plt.close(fig)
    return out_png


def plot(table: pd.DataFrame, out_dir: Path, *, regimes: list[str]) -> dict[str, Path]:
    """Every (metric, regime) panel as its own PNG, on shared limits."""
    xlim, ylim = limits(table)
    out = {}
    for metric, _, _ in METRICS:
        for regime in regimes:
            t = table[(table["regime"] == regime) & (table["metric"] == metric)]
            name = PANEL_PNG.format(metric=metric, regime=regime)
            out[name] = plot_panel(t, out_dir / name, metric=metric, xlim=xlim, ylim=ylim)
    return out


# ---- the whole step ---------------------------------------------------------


def _manifest(groups: dict[str, list[str]], methods: list[str], table: pd.DataFrame,
              pairs: pd.DataFrame) -> str:
    return json.dumps({
        "figures": [PANEL_PNG.format(metric=m, regime=g)
                    for m, _, _ in METRICS for g in REGIMES if g in groups],
        "artifacts": NAMES,
        "regimes": {g: {"title": REGIME_TITLES[g], "run_ids": rids} for g, rids in groups.items()},
        "methods": methods,
        "method_terms": method_term_table(methods),
        "y": {
            "rows": {"cell": "unbounded: the prediction falls in the TG's cell",
                     "bounded": "right cell and pixel distance <= BOUND_PIXELS"},
            "bound_pixels": BOUND_PIXELS,
            "denominator": "every TG, unanswered included",
            "dot": "accuracy over every TG pooled (TG-weighted average of the datasets), "
                   "as in the pooled outcome bars",
            "bar": "lowest to highest dataset: a spread across networks, not a CI",
        },
        "x": {
            "metric": "runtime per TG, LTD+MTL+CTR summed per TG, every TG",
            "dot": "p50", "bar": "p25-p75", "scale": "log",
            "regime_cost": "each panel uses its own regime's runtime",
        },
        "shortest_ping": "no timed stage: a dashed horizontal line at its pooled accuracy, a band from "
                         "its worst to its best network, and the zero-cost "
                         "frontier point; a CBG method must beat its accuracy",
        "frontier": {f"{g}/{m}": list(table[(table.regime == g) & (table.metric == m)
                                             & table.on_frontier]
                                       .sort_values("runtime_p50_ms")["method"])
                     for g in groups for m, _, _ in METRICS},
        "n_consistent_pairs": {f"{g}/{m}": int(pairs[(pairs.regime == g) & (pairs.metric == m)]
                                               ["consistent"].sum())
                               for g in groups for m, _, _ in METRICS},
        "pairs_rule": "consistent = one method wins in every dataset of the regime",
    }, indent=2) + "\n"


def build(
    groups: dict[str, list[RunPaths]], methods: list[str], *,
    analysis_root: Path | None = None,
) -> dict[str, Path]:
    """Every regime's table and pairs, one figure; into `_cross/pareto/<hash>/`."""
    unknown = sorted(set(groups) - set(REGIMES))
    if unknown or not groups:
        raise ValueError(f"regimes must be among {list(REGIMES)}, got {sorted(groups)}")
    methods = method_order(methods)
    tables, ds_tables, pair_tables = [], [], []
    for regime in [r for r in REGIMES if r in groups]:
        long = load_regime(groups[regime], methods, root=analysis_root)
        table, ds, pairs = summarize(long, regime)
        tables.append(table)
        ds_tables.append(ds)
        pair_tables.append(pairs)
    table = pd.concat(tables, ignore_index=True)
    pairs = pd.concat(pair_tables, ignore_index=True)
    run_ids = [r.run_id for rs in groups.values() for r in rs]
    out_dir = cross.cross_dir(run_ids, analysis_root=analysis_root, kind=KIND)
    written = {k: out_dir / v for k, v in NAMES.items()}
    table.to_csv(written["csv"], index=False)
    pd.concat(ds_tables, ignore_index=True).to_csv(written["datasets"], index=False)
    pairs.to_csv(written["pairs"], index=False)
    ids = {g: [r.run_id for r in rs] for g, rs in groups.items()}
    written["manifest"].write_text(_manifest(ids, methods, table, pairs))
    pngs = plot(table, out_dir, regimes=[r for r in REGIMES if r in groups])
    return {**written, **pngs}
