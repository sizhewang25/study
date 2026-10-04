"""The dataset as a bipartite graph: VP nodes, TG nodes, measured edges.

Ported from v3's `bipartite.py`, onto v5's answer space. The answer space read
only the TG side; this reads the **VPs** and the **measured (VP, TG) edges**,
and computes paper §7.3's metric list against the HEALPix grid the accuracy
numbers are scored on.

**No RTT enters and no method runs.** RTT is used for exactly one thing -- the
canonical CSV's own row filter (`rtt_ms > 0`, i.e. "was this pair measured at
all") -- and is then dropped. That is what makes these numbers the fixed
reference the RTT-dependent results are read against, rather than another
result competing with them.

**Every VP-to-TG distance is reported as a pair.** The *latent* one over all
VP x TG pairs describes where the infrastructure sits; the *observed* one over
measured edges describes what the dataset can actually deliver. An edge set is
an artifact of the campaign rather than a property of the deployment, so
reporting one half without the other is what lets sampling bias be misread as
an algorithmic result. The two are **nested** in `meta.json`
(`length_km.{observed,latent}`, `nearest_vp_km.{observed,latent}`) rather than
sitting as siblings, so neither can be read with the other out of view.

The pairing applies to distances and **not to degree**: a TG's latent degree is
the VP count for every TG, so it carries nothing, and `meta.json` says so
rather than emitting a constant column.

`measured_nearest_vp_ratio` (§7.3's measurement efficiency) is what the pair
exists to support -- nearest-measured-VP over nearest-VP per TG, `>= 1` by
construction. It separates "the VP set is badly placed" (a large latent
nearest-VP distance) from "the VP set is fine but the campaign allocated probes
badly" (a ratio above 1), and only the second is fixable by reallocating
measurement.

**Angular geometry stays observed-only**, matching §7.3's scoping of it to
measured neighbours: the arrangement term describes the constraints a method
actually receives.

## What changed from v3

* **Grid, not H3 cell.** v3 quantized on H3 res 4; v5's grid is HEALPix at the
  answer space's nside, so `dispersion.effective_count` is an occupied-grid
  count and the node frames carry `vp_grid_id` / `tg_grid_id`.
* **The TG side is `answer-space/tgs.csv`**, with its `site_id` and
  `tg_seed_id`. A seed is a complete-linkage group of sites, not a grid, so
  v3's "occupied target cells == seed count" identity no longer holds; the
  target block reports `n_sites`, `n_grids` and `n_seeds` separately.
* **A traffic-weighted run keeps the mesh's TG universe**, because its answer
  space does (see `answer_space`). Its edges are the weighted CSV's, so mesh
  TGs that lost every flow sit at degree 0 and are counted in
  `n_with_no_edge` -- the honest reading of what the filter removed.
* **`tg_*` names**, renamed from the canonical CSV's `target_*` at load time.
* v3's statistics helpers that only other v3 commands used (`ols`, `spearman`,
  `quantile_bins`, ...) are not ported.

**Scale.** The latent half is `|VP| x |TG|`. The per-TG nearest-VP distance is
**exact at any TG count** -- `nearest_across_km` chunks the cross matrix and
reduces with a per-chunk `min`. Only the latent *distribution*, which needs the
values rather than their minima, falls back to a deterministic TG subsample,
and says so in `meta.json` when it does.

Command: `build-bipartite-graph`. Writes to
`outputs/analysis/v5/<run_id>/bipartite-graph/healpix-<nside>/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.answer_space import (
    BENCHMARK_TG_COLUMNS,
    META_JSON as SPACE_META_JSON,
    AnswerSpace,
    _describe,
    load_answer_space,
    require_mesh_universe,
)
from scripts.analysis.v5.modules.edges import resolve_source_csv
from scripts.analysis.v5.modules.geodesy import elementwise_km, pairwise_km
from scripts.analysis.v5.modules.paths import MissingArtifactError, RunPaths

VP_NODES_CSV = "vp_nodes.csv"
TG_NODES_CSV = "tg_nodes.csv"
EDGE_SEGMENTS_CSV = "edge_segments.csv"
EDGE_LENGTH_CDF_CSV = "edge_length_cdf.csv"
PAIRWISE_CDF_CSV = "pairwise_distance_cdf.csv"
META_JSON = "meta.json"

#: Quantiles written to the two CDF artifacts. 101 points is enough to draw the
#: curve behind the diameter and p95 scalars and small enough to read, and it
#: fixes a grid so two datasets' CDFs can be differenced row by row.
CDF_QUANTILES = np.round(np.linspace(0.0, 1.0, 101), 3)

#: Above this node count a full pairwise matrix is subsampled rather than built.
#: The runs here are 53-134 VPs and 78-458 TGs, so this never fires; it exists
#: so an O(n^2) matrix fails into a documented approximation rather than into
#: the OOM killer.
_MAX_PAIRWISE_NODES = 5_000

#: Seed for that subsample, so a capped run is still reproducible.
_PAIRWISE_SEED = 20260903

#: TGs per chunk of the VP x TG cross matrix. Only `|VP| x chunk` floats are
#: ever resident, so the latent nearest-VP distance stays exact at scales where
#: the full cross matrix is not materializable.
_CROSS_CHUNK_TGS = 4096

#: Latent VP-TG pair distances retained for the *distribution*. Past this many
#: pairs the TG side is subsampled (deterministically) and said so.
_MAX_CROSS_PAIRS = 5_000_000

#: Relative slack around a ratio of exactly 1.0, sized from the residue.
#:
#: The observed minimum comes off per-edge `elementwise_km` and the latent one
#: off `pairwise_km`'s matrix product, so the same pair reaches the same metre
#: by two code paths that differ in the last bits. Measured on as03 (v3), twenty
#: TGs came out 1.24e-9 relative apart, and for every one of them the
#: latent-nearest VP *did* carry an edge -- so a 1e-9 threshold reported 438 of
#: 458 TGs as having missed their nearest VP when the true answer is all 458.
#: 1e-6 is ~800x that residue and ~1000x tighter than the smallest excess a
#: genuine miss could produce, since a real alternative VP is kilometres away.
_EFFICIENCY_TOL = 1e-6

#: Decimal places coordinates are rounded to before flow segments are
#: collapsed. 5 dp is ~1 m, finer than any input (the canonical CSVs carry
#: <= 6 dp) and far below the ~51 km grid, so this cannot merge two places the
#: rest of the pipeline keeps apart.
_SEGMENT_COORD_DP = 5

#: Percentiles every distribution block carries. `answer_space._describe`'s set
#: plus p90, because §7.3 specifies degree, edge length and max angular gap as
#: median/IQR/p90, all three being heavily skewed.
_PERCENTILES = (5, 25, 50, 75, 90, 95)

#: Decimals for a ratio. `_describe`'s 3 is right for kilometres and destroys a
#: ratio that lives just above 1: at 3 dp, as03's twenty TGs that missed their
#: nearest VP all round to exactly 1.000.
_RATIO_DIGITS = 6

_RATIO_NOTE = (
    "One value per TG: nearest-measured-VP km over nearest-VP km. 1.0 means the "
    "campaign measured the geometrically nearest VP; larger means it did not. "
    "Separates 'the VP set is badly placed' (a large latent nearest-VP distance) "
    "from 'the VP set is fine but the campaign allocated probes badly' (a ratio "
    "above 1), and only the second is fixable by reallocating measurement. "
    "Undefined where a VP sits exactly on the TG but carries no edge (the ratio "
    "is infinite); those TGs are excluded, so `n` here is below the TG count by "
    "exactly that many."
)

DISPERSION_NOTE = (
    "Extent says how far apart the set reaches; this says whether it is spread "
    "or stacked inside that reach, at this file's own nside. effective_count is "
    "the number of distinct HEALPix grids the set occupies and occupancy_ratio "
    "is effective_count / count -- 1.0 means every node is its own place, low "
    "means many share one. Build another rung with --nside rather than reading "
    "coarser counts out of this file."
)

_LATENT_OBSERVED_NOTE = (
    "observed is over measured edges (what the dataset can deliver); latent is "
    "over all VP x TG pairs (where the infrastructure sits). An edge set is an "
    "artifact of the campaign, so reporting one without the other is what lets "
    "sampling bias be misread as an algorithmic result (paper §7.3)."
)


# ---- distribution blocks ----------------------------------------------------


def describe_p90(values, *, digits: int = 3) -> dict:
    """`answer_space._describe`'s block shape, plus p90, at a chosen precision.

    Computed here rather than by patching `_describe`, because the ratio blocks
    need more than its hard-coded 3 decimals. Agreement on the five percentiles
    `_describe` emits is pinned by a test, so this stays a widening, not a fork.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0}
    return {
        "n": int(v.size),
        "min": round(float(v.min()), digits),
        "max": round(float(v.max()), digits),
        "mean": round(float(v.mean()), digits),
        "percentiles": {
            f"p{p}": round(float(np.percentile(v, p)), digits) for p in _PERCENTILES
        },
    }


def cdf_column(values, *, digits: int = 3) -> np.ndarray:
    """`CDF_QUANTILES` of `values`, or all-NaN when there is nothing to rank."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.full(CDF_QUANTILES.size, np.nan)
    return np.round(np.quantile(v, CDF_QUANTILES), digits)


# ---- angular geometry -------------------------------------------------------
# §7.3's "arrangement term". Two VPs on opposite sides of a TG constrain it
# better than five clustered in one metro.


def bearings_deg(tlat: float, tlon: float, vlats, vlons) -> np.ndarray:
    """Initial great-circle bearing (deg, `[0, 360)`) from a TG to each VP."""
    tlat_r, tlon_r = np.radians(float(tlat)), np.radians(float(tlon))
    vlat_r = np.radians(np.asarray(vlats, dtype=float))
    vlon_r = np.radians(np.asarray(vlons, dtype=float))
    dlon = vlon_r - tlon_r
    x = np.sin(dlon) * np.cos(vlat_r)
    y = np.cos(tlat_r) * np.sin(vlat_r) - np.sin(tlat_r) * np.cos(vlat_r) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def angular_features(bearings) -> tuple[float, float]:
    """`(max_gap_deg, circular_variance)` of a set of bearings.

    `max_gap_deg` is the largest wedge containing no VP, wrap-around included:
    VPs at 10/40/75 deg leave a 295 deg gap and barely constrain the TG, while
    20/140/260 leaves 120 deg and brackets it. A single VP leaves the whole
    turn, hence 360. `circular_variance` is `1 - |mean unit vector|` -- 0 when
    every VP lies in one direction, approaching 1 when well surrounded.

    Matches `scripts/analysis/partvp/extract_features.py`'s private
    `_angular_features`; a test pins the two against each other.
    """
    b = np.sort(np.asarray(bearings, dtype=float))
    n = b.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return 360.0, 0.0
    max_gap = float(max(np.diff(b).max(), 360.0 - (b[-1] - b[0])))
    ang = np.radians(b)
    r = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
    return max_gap, 1.0 - r


# ---- node-set geometry ------------------------------------------------------


def _pairwise_distances(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, dict]:
    """Upper-triangle pairwise great-circle km, plus a note if it was subsampled."""
    n = lat.size
    note: dict = {}
    if n > _MAX_PAIRWISE_NODES:
        rng = np.random.default_rng(_PAIRWISE_SEED)
        keep = np.sort(rng.choice(n, size=_MAX_PAIRWISE_NODES, replace=False))
        lat, lon = lat[keep], lon[keep]
        note = {
            "pairwise_subsampled_from": int(n),
            "pairwise_subsample_size": _MAX_PAIRWISE_NODES,
            "pairwise_subsample_seed": _PAIRWISE_SEED,
        }
    if lat.size < 2:
        return np.empty(0, dtype=float), note
    d = pairwise_km(lat, lon)
    return d[np.triu_indices_from(d, k=1)], note


def nearest_across_km(
    a_lat: np.ndarray,
    a_lon: np.ndarray,
    b_lat: np.ndarray,
    b_lon: np.ndarray,
    *,
    chunk: int = _CROSS_CHUNK_TGS,
) -> tuple[np.ndarray, np.ndarray]:
    """For each node in A, `(km, index)` of the nearest node in B -- over **all** of B.

    The latent half of the pair: it ignores the edge set entirely. Chunked over
    A so the resident footprint is `|B| x chunk` rather than `|A| x |B|`.

    Empty B gives all-NaN and index -1 rather than raising: a run with no VPs is
    degenerate but the caller reports it rather than crashing on it.
    """
    n = int(a_lat.size)
    if b_lat.size == 0 or n == 0:
        return np.full(n, np.nan), np.full(n, -1, dtype=int)
    step = max(int(chunk), 1)
    best = np.empty(n, dtype=float)
    idx = np.empty(n, dtype=int)
    for start in range(0, n, step):
        stop = min(start + step, n)
        d = pairwise_km(a_lat[start:stop], a_lon[start:stop], b_lat, b_lon)
        j = d.argmin(axis=1)
        best[start:stop] = d[np.arange(stop - start), j]
        idx[start:stop] = j
    return best, idx


def _latent_pair_distances(
    tg_lat: np.ndarray, tg_lon: np.ndarray, vp_lat: np.ndarray, vp_lon: np.ndarray
) -> tuple[np.ndarray, dict]:
    """Every VP-to-TG great-circle distance, measured or not, plus a note.

    Subsampled on the TG side past `_MAX_CROSS_PAIRS`; the note records that so
    a percentile is never read as exact when it is not.
    """
    n_t, n_v = int(tg_lat.size), int(vp_lat.size)
    note: dict = {}
    if n_t == 0 or n_v == 0:
        return np.empty(0, dtype=float), note
    if n_t * n_v > _MAX_CROSS_PAIRS:
        keep_n = max(_MAX_CROSS_PAIRS // n_v, 1)
        rng = np.random.default_rng(_PAIRWISE_SEED)
        keep = np.sort(rng.choice(n_t, size=keep_n, replace=False))
        tg_lat, tg_lon = tg_lat[keep], tg_lon[keep]
        note = {
            "latent_subsampled_from_tgs": n_t,
            "latent_subsample_tgs": int(keep_n),
            "latent_subsample_seed": _PAIRWISE_SEED,
        }
    out = [
        pairwise_km(
            tg_lat[s : s + _CROSS_CHUNK_TGS], tg_lon[s : s + _CROSS_CHUNK_TGS], vp_lat, vp_lon
        ).ravel()
        for s in range(0, tg_lat.size, _CROSS_CHUNK_TGS)
    ]
    return np.concatenate(out), note


def measurement_efficiency(
    nearest_measured_km: np.ndarray, nearest_latent_km: np.ndarray
) -> tuple[np.ndarray, int]:
    """Per-TG `measured / latent`, and the count where it is undefined.

    Emitted as `measured_nearest_vp_ratio`. Three cases, because the degenerate
    ones carry different meanings:

    * `latent > 0` -- the ordinary ratio, `>= 1` by construction. Values within
      `_EFFICIENCY_TOL` of 1.0 are **snapped to exactly 1.0**, so `ratio == 1.0`
      is usable downstream as "the campaign got the closest VP" with no
      tolerance repeated at each call site.
    * both zero -- a VP sits exactly on the TG *and* was measured: 1.0, not 0/0.
    * `latent == 0` with a positive or missing measured value -- a VP sits on
      the TG and carries no edge. The ratio is infinite, which no percentile can
      hold and JSON cannot encode, so it is left NaN and **counted**. Coercing
      it to 1.0 would report the worst allocation failure as the best.
    """
    measured = np.asarray(nearest_measured_km, dtype=float)
    latent = np.asarray(nearest_latent_km, dtype=float)
    ratio = np.full(measured.shape, np.nan)
    both_zero = (latent == 0.0) & (measured == 0.0)
    ratio[both_zero] = 1.0
    usable = latent > 0.0
    ratio[usable] = measured[usable] / latent[usable]
    snap = np.isfinite(ratio) & (np.abs(ratio - 1.0) <= _EFFICIENCY_TOL)
    ratio[snap] = 1.0
    undefined = int(((latent == 0.0) & ~both_zero).sum())
    return ratio, undefined


def _node_block(
    grid_ids: np.ndarray, *, asns: pd.Series | None, pairwise: np.ndarray, pairwise_note: dict
) -> dict:
    """The §7.3 node-set block for one side: extent, then dispersion.

    `pairwise` is passed in rather than computed here because the CDF artifact
    needs the same vector, and building an O(n^2) matrix twice is not free.
    `grid_ids` is the side's own quantization, so the count reported here is
    the one the caller wrote into its node frame.
    """
    pw = pairwise
    n = int(np.asarray(grid_ids).size)
    n_grids = int(np.unique(grid_ids).size)
    block: dict = {
        "count": n,
        "asn_count": None if asns is None else int(asns.dropna().nunique()),
        # A diameter is a maximum, so one near-antipodal node sets it
        # single-handedly; §7.3 requires p95 printed beside it for that reason.
        "geographic_diameter_km": round(float(pw.max()), 3) if pw.size else None,
        "pairwise_p95_km": round(float(np.percentile(pw, 95)), 3) if pw.size else None,
        "pairwise_km": _describe(pw),
        "dispersion": {
            "effective_count": n_grids,
            "occupancy_ratio": round(n_grids / n, 4) if n else None,
            "note": DISPERSION_NOTE,
        },
    }
    block.update(pairwise_note)
    return block


# ---- inputs -----------------------------------------------------------------


def load_vps(run: RunPaths) -> pd.DataFrame:
    """`vps.csv` -- the run's vantage points, shared across every fold."""
    path = run.setup_dir / "vps.csv"
    if not path.exists():
        raise MissingArtifactError(f"{path} missing; cannot draw the VP side")
    vps = pd.read_csv(path)
    missing = {"vp_id", "vp_lat", "vp_lon"} - set(vps.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return vps


def load_edges(csv_path: Path) -> tuple[pd.DataFrame, int]:
    """One row per measured `(vp_id, tg_id)` pair, RTT dropped.

    Returns `(edges, n_obs)` -- the deduplicated edge set and the observation
    count it came from, since a dataset can measure one pair repeatedly and
    density is over pairs.

    Read through `load_canonical_csv`, which owns the schema and drops NaN rows
    and `rtt_ms <= 0` exactly as the benchmark's source does -- so the graph
    described here is the graph the benchmark ran on. `target_*` columns are
    renamed to `tg_*` here.

    Raises if one id carries more than one coordinate, which would make every
    distance below ambiguous.
    """
    from scripts.libs.canonical.schema import load_canonical_csv

    df = load_canonical_csv(Path(csv_path))
    n_obs = int(len(df))
    df = df.rename(columns={**BENCHMARK_TG_COLUMNS, "target_asn": "tg_asn"})

    cols = ["vp_id", "vp_lat", "vp_lon", "tg_id", "tg_lat", "tg_lon"]
    if "tg_asn" in df.columns:
        cols.append("tg_asn")
    edges = df[cols].drop_duplicates(["vp_id", "tg_id"]).reset_index(drop=True)

    for id_col, lat_col, lon_col in (
        ("vp_id", "vp_lat", "vp_lon"),
        ("tg_id", "tg_lat", "tg_lon"),
    ):
        n_coords = edges.groupby(id_col)[[lat_col, lon_col]].nunique().max(axis=1)
        bad = n_coords[n_coords > 1]
        if not bad.empty:
            raise ValueError(
                f"{csv_path}: {len(bad)} {id_col}(s) carry more than one coordinate "
                f"(e.g. {bad.index[:3].tolist()}); every distance here would be ambiguous"
            )
    return edges, n_obs


def load_rung(run: RunPaths, nside: int, *, analysis_root: Path | None = None) -> AnswerSpace:
    """The answer space at one rung, read from disk with the weighted-run guard."""
    d = run.answer_space_dir(nside, root=analysis_root)
    if not (d / SPACE_META_JSON).exists():
        raise MissingArtifactError(
            f"{d / SPACE_META_JSON} missing; run `build-answer-space --run-id {run.run_id}` first"
        )
    space = load_answer_space(d)
    require_mesh_universe(run, space)
    return space


# ---- the graph --------------------------------------------------------------


@dataclass(frozen=True)
class BipartiteGraph:
    """Per-node geometry, the two CDFs, and the §7.3 metric block."""

    vp_nodes: pd.DataFrame
    tg_nodes: pd.DataFrame
    edge_segments: pd.DataFrame
    edge_length_cdf: pd.DataFrame
    pairwise_distance_cdf: pd.DataFrame
    meta: dict

    @property
    def n_vps(self) -> int:
        return len(self.vp_nodes)

    @property
    def n_tgs(self) -> int:
        return len(self.tg_nodes)

    def write(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.vp_nodes.to_csv(out_dir / VP_NODES_CSV, index=False)
        self.tg_nodes.to_csv(out_dir / TG_NODES_CSV, index=False)
        self.edge_segments.to_csv(out_dir / EDGE_SEGMENTS_CSV, index=False)
        self.edge_length_cdf.to_csv(out_dir / EDGE_LENGTH_CDF_CSV, index=False)
        self.pairwise_distance_cdf.to_csv(out_dir / PAIRWISE_CDF_CSV, index=False)
        (out_dir / META_JSON).write_text(json.dumps(self.meta, indent=2) + "\n")
        return out_dir


def _connected_components(edges: pd.DataFrame, vp_ids, tg_ids) -> dict:
    """Components of the bipartite graph, over **all** nodes.

    Traffic can concentrate regionally, so the edge set may fall apart into
    pieces no single calibration spans. Isolated nodes count as their own
    component: a VP that measured nothing is disconnected from the experiment,
    and hiding it in the largest component's share would overstate how
    joined-up the campaign was.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n_v, n_t = len(vp_ids), len(tg_ids)
    if n_v + n_t == 0:
        return {"n_components": 0}
    # int64 explicitly: pandas hands back int8 codes for <= 127 categories, and
    # `n_v + t` then overflows silently -- 53 + 77 wrapped to -126 on as7018.
    v = np.asarray(pd.Categorical(edges["vp_id"], categories=list(vp_ids)).codes, dtype=np.int64)
    t = np.asarray(pd.Categorical(edges["tg_id"], categories=list(tg_ids)).codes, dtype=np.int64)
    if v.min(initial=0) < 0 or t.min(initial=0) < 0:
        raise ValueError("edge endpoint outside the node sets; the join upstream failed")
    adj = coo_matrix((np.ones(len(edges)), (v, n_v + t)), shape=(n_v + n_t, n_v + n_t))
    n_comp, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels, minlength=n_comp)
    return {
        "n_components": int(n_comp),
        "n_isolated_nodes": int((sizes == 1).sum()),
        "largest_component_nodes": int(sizes.max()),
        "largest_component_node_share": round(float(sizes.max() / sizes.sum()), 6),
        "component_size_nodes": _describe(sizes.astype(float)),
    }


def build_bipartite(
    space: AnswerSpace,
    vps: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    n_obs: int | None = None,
    source_label: str | None = None,
    source_csv: str | None = None,
) -> BipartiteGraph:
    """Compute §7.3's geometry over one run's bipartite graph.

    The node sets are pinned to the run's own inputs rather than inferred from
    the edge list: VPs are `vps.csv`'s roster and TGs are the answer space's
    `tgs`. Edges naming anything outside those are dropped and counted, which is
    what makes `edge_density` read against `|VP roster| x |answer-space TGs|`
    instead of against whatever the CSV happened to contain.

    `edges` needs `vp_id, vp_lat, vp_lon, tg_id, tg_lat, tg_lon` (`tg_asn`
    optional), one row per pair -- what `load_edges` returns.
    """
    nside = space.nside
    vps = vps.drop_duplicates("vp_id").reset_index(drop=True)
    tgs = space.tgs.reset_index(drop=True)

    n_edges_raw = len(edges)
    known_vp = edges["vp_id"].isin(set(vps["vp_id"]))
    known_tg = edges["tg_id"].isin(set(tgs["tg_id"]))
    dropped_vp = int((~known_vp).sum())
    dropped_tg = int((known_vp & ~known_tg).sum())
    edges = edges.loc[known_vp & known_tg].reset_index(drop=True)
    if edges.empty:
        raise ValueError(
            "no edge survives the join against vps.csv and the answer space; "
            "the CSV and the run do not describe the same graph"
        )

    # Kept **unrounded**. Rounding here made the ratio divide a rounded
    # numerator by an unrounded denominator, which put 160 of as01's 399 TGs a
    # few 1e-4 *below* 1.0 -- a value the ratio cannot take. Rounding happens at
    # the artifact boundary instead, where it is presentation.
    edges = edges.copy()
    edges["length_km"] = elementwise_km(
        edges["vp_lat"].to_numpy(dtype=float),
        edges["vp_lon"].to_numpy(dtype=float),
        edges["tg_lat"].to_numpy(dtype=float),
        edges["tg_lon"].to_numpy(dtype=float),
    )

    # --- VP nodes ---------------------------------------------------------
    vp_lat = vps["vp_lat"].to_numpy(dtype=float)
    vp_lon = vps["vp_lon"].to_numpy(dtype=float)
    vp_grid = G.ang2pix(vp_lat, vp_lon, nside)
    vp_pw, vp_pw_note = _pairwise_distances(vp_lat, vp_lon)
    vp_block = _node_block(
        vp_grid,
        asns=vps["vp_asn"] if "vp_asn" in vps.columns else None,
        pairwise=vp_pw,
        pairwise_note=vp_pw_note,
    )
    by_vp = edges.groupby("vp_id")
    vp_nodes = vps.copy()
    vp_nodes["vp_grid_id"] = vp_grid
    # Named for the side it points at: there is no VP-to-VP edge, so a bare
    # "degree" invites reading this as one.
    vp_nodes["degree_to_tg"] = vp_nodes["vp_id"].map(by_vp.size()).fillna(0).astype(int)
    vp_nodes["nearest_measured_tg_km"] = vp_nodes["vp_id"].map(by_vp["length_km"].min()).round(3)
    vp_block["degree_to_tg"] = {
        **describe_p90(vp_nodes["degree_to_tg"].to_numpy(dtype=float)),
        "note": (
            "TGs this VP measured -- measurement effort spent. Observed by "
            "definition: the latent value is the TG count for every VP, so it "
            "carries nothing and is not emitted."
        ),
    }
    vp_block["n_with_no_edge"] = int((vp_nodes["degree_to_tg"] == 0).sum())

    # --- TG nodes ---------------------------------------------------------
    tg_lat = tgs["tg_lat"].to_numpy(dtype=float)
    tg_lon = tgs["tg_lon"].to_numpy(dtype=float)
    tg_asns = (
        edges.drop_duplicates("tg_id").set_index("tg_id")["tg_asn"]
        if "tg_asn" in edges.columns
        else None
    )
    tg_pw, tg_pw_note = _pairwise_distances(tg_lat, tg_lon)
    tg_block = _node_block(
        tgs["tg_grid_id"].to_numpy(dtype=np.int64),
        asns=tg_asns,
        pairwise=tg_pw,
        pairwise_note=tg_pw_note,
    )
    by_tg = edges.groupby("tg_id")
    nearest_idx = by_tg["length_km"].idxmin()

    tg_nodes = tgs.loc[:, ["tg_id", "tg_lat", "tg_lon", "site_id", "tg_grid_id", "tg_seed_id"]].copy()
    tg_nodes["degree_to_vp"] = tg_nodes["tg_id"].map(by_tg.size()).fillna(0).astype(int)
    measured_km = tg_nodes["tg_id"].map(by_tg["length_km"].min())
    tg_nodes["nearest_measured_vp_km"] = measured_km.round(3)
    tg_nodes["nearest_measured_vp_id"] = tg_nodes["tg_id"].map(
        edges.loc[nearest_idx].set_index("tg_id")["vp_id"]
    )

    # --- the latent half --------------------------------------------------
    tg_latent_km, tg_latent_idx = nearest_across_km(tg_lat, tg_lon, vp_lat, vp_lon)
    tg_nodes["nearest_vp_km"] = np.round(tg_latent_km, 3)
    tg_nodes["nearest_vp_id"] = np.where(
        tg_latent_idx >= 0, vps["vp_id"].to_numpy()[tg_latent_idx], None
    )
    eff, _ = measurement_efficiency(measured_km.to_numpy(dtype=float), tg_latent_km)
    tg_nodes["measured_nearest_vp_ratio"] = np.round(eff, _RATIO_DIGITS)
    # Defined on **distance**, not on id equality: co-located VPs are the normal
    # case (as01's VP nearest-neighbour p50 is 0.0 km), so any two nearest-VP
    # implementations break ties differently -- ids agreed with eval_source's
    # BallTree on only 259 of as01's 399 TGs while the distances agreed to
    # 4e-4 km. Exact, because `measurement_efficiency` already snapped.
    tg_nodes["nearest_vp_is_measured"] = eff <= 1.0

    latent_pairs, latent_note = _latent_pair_distances(tg_lat, tg_lon, vp_lat, vp_lon)

    gaps, circ = _angular_per_tg(edges, tg_nodes["tg_id"])
    tg_nodes["max_angular_gap_deg"] = np.round(gaps, 3)
    tg_nodes["circular_variance"] = np.round(circ, 6)

    tg_block["degree_to_vp"] = {
        **describe_p90(tg_nodes["degree_to_vp"].to_numpy(dtype=float)),
        "note": (
            "VPs that measured this TG -- constraints available. Observed by "
            "definition: the latent value is the VP count for every TG, so it "
            "carries nothing and is not emitted."
        ),
    }
    tg_block["n_with_no_edge"] = int((tg_nodes["degree_to_vp"] == 0).sum())
    tg_block["n_sites"] = int(space.meta["n_sites"])
    tg_block["n_grids"] = int(space.meta["n_grids"])
    tg_block["n_seeds"] = int(space.n_seeds)

    # --- meta -------------------------------------------------------------
    denom = len(vps) * len(tgs)
    segments = edge_segments(edges)
    meta = {
        "source": source_label,
        "grid": G.describe(nside),
        "scope": {
            "rtt": "not used beyond the canonical CSV's own rtt_ms > 0 row filter",
            "distances": "observed (measured edges) and latent (all VP x TG pairs), nested",
        },
        "inputs": {
            "source_csv": source_csv,
            "tgs_source": space.meta.get("targets_provenance", {}).get(
                "targets_source", "the run's own evaluated TGs"
            ),
            "n_observations": n_obs,
            "n_edges_before_join": int(n_edges_raw),
            "n_edges_dropped_vp_not_in_roster": dropped_vp,
            "n_edges_dropped_tg_not_in_answer_space": dropped_tg,
        },
        "nodes": {"vps": vp_block, "tgs": tg_block},
        "edges": {
            "n_edges": int(len(edges)),
            # Measurement completeness: the share of possible (VP, TG) pairs the
            # campaign measured. A count ratio over all pairs, unlike the
            # distance ratio at the minimum below; a campaign can score well
            # on either while failing the other.
            "edge_density": round(len(edges) / denom, 6) if denom else None,
            "connected_components": _connected_components(edges, vps["vp_id"], tgs["tg_id"]),
            "length_km": {
                "observed": describe_p90(edges["length_km"].to_numpy(dtype=float)),
                "latent": describe_p90(latent_pairs),
                "n_latent_pairs": int(denom),
                "note": _LATENT_OBSERVED_NOTE,
                **latent_note,
            },
            # The smallest disk does most of the constraining.
            "nearest_vp_km": {
                "observed": describe_p90(tg_nodes["nearest_measured_vp_km"].to_numpy(dtype=float)),
                "latent": describe_p90(tg_latent_km),
                "note": _LATENT_OBSERVED_NOTE,
            },
            "measured_nearest_vp_ratio_per_tg": {
                **describe_p90(eff, digits=_RATIO_DIGITS),
                "note": _RATIO_NOTE,
            },
            "n_distinct_geometry_edge": int(len(segments)),
            "n_distinct_geometry_edge_note": (
                "distinct (VP coord, TG coord) geometries, NOT a second edge count: "
                "all n_edges edges are distinct edges, but many share a line because "
                "several TG replicas sit at one site. This is what the flow map "
                "draws -- see edge_segments()."
            ),
        },
        "angular": {
            "note": "over measured neighbours only; the arrangement term of §7.3",
            "max_angular_gap_deg": describe_p90(gaps),
            "circular_variance": describe_p90(circ),
        },
    }

    edge_cdf = pd.DataFrame(
        {
            "quantile": CDF_QUANTILES,
            "observed_edge_km": cdf_column(edges["length_km"]),
            "latent_pair_km": cdf_column(latent_pairs),
            "measured_nearest_vp_ratio": cdf_column(eff, digits=_RATIO_DIGITS),
        }
    )
    pairwise_cdf = pd.DataFrame(
        {
            "quantile": CDF_QUANTILES,
            "vp_pairwise_km": cdf_column(vp_pw),
            "tg_pairwise_km": cdf_column(tg_pw),
        }
    )
    return BipartiteGraph(
        vp_nodes=vp_nodes,
        tg_nodes=tg_nodes,
        edge_segments=segments,
        edge_length_cdf=edge_cdf,
        pairwise_distance_cdf=pairwise_cdf,
        meta=meta,
    )


def edge_segments(edges: pd.DataFrame) -> pd.DataFrame:
    """Geometrically distinct flow lines, with the edge count each stands for.

    as01 has 53,262 edges but only 20 distinct TG coordinates, so almost every
    edge is an exact overplot of another: one line per row would blend alpha
    into a density reading that reports replicas at one site as heavy traffic.
    Collapsing to distinct `(VP coord, TG coord)` pairs and carrying `n_edges`
    as multiplicity keeps the ink honest.
    """
    cols = ["vp_lat", "vp_lon", "tg_lat", "tg_lon"]
    rounded = edges[cols].round(_SEGMENT_COORD_DP)
    return (
        edges.assign(**{c: rounded[c] for c in cols})
        .groupby(cols, as_index=False)
        .agg(n_edges=("length_km", "size"), length_km=("length_km", "first"))
        .sort_values(cols, ignore_index=True)
    )


def _angular_per_tg(edges: pd.DataFrame, tg_order: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """`(max_angular_gap_deg, circular_variance)` per TG, in `tg_order`.

    A TG with no measured edge gets NaN on both, not 360/0: 360 means "one VP,
    so the whole turn is empty", and conflating it with "no VP at all" would
    put an unmeasured TG in the same bucket as the worst-arranged measured one.
    """
    gaps: dict[str, float] = {}
    circ: dict[str, float] = {}
    for tid, sub in edges.groupby("tg_id"):
        b = bearings_deg(
            sub["tg_lat"].iloc[0], sub["tg_lon"].iloc[0],
            sub["vp_lat"].to_numpy(), sub["vp_lon"].to_numpy(),
        )
        gaps[tid], circ[tid] = angular_features(b)
    return (
        tg_order.map(gaps).to_numpy(dtype=float),
        tg_order.map(circ).to_numpy(dtype=float),
    )


def build_for_run(
    run: RunPaths,
    *,
    nside: int = G.DEFAULT_NSIDE,
    analysis_root: Path | None = None,
    source_csv: Path | None = None,
) -> tuple[BipartiteGraph, Path]:
    """`build_bipartite` for one run at one rung, written. Returns `(graph, dir)`."""
    space = load_rung(run, nside, analysis_root=analysis_root)
    csv_path = resolve_source_csv(run, source_csv)
    edges, n_obs = load_edges(csv_path)
    graph = build_bipartite(
        space,
        load_vps(run),
        edges,
        n_obs=n_obs,
        source_label=f"{run.run_id}/{run.source}/{run.setup}",
        source_csv=str(csv_path),
    )
    out_dir = graph.write(run.bipartite_dir(space.nside, root=analysis_root))
    return graph, out_dir


def load_bipartite(path: Path) -> BipartiteGraph:
    """Read back what `write` produced, with grid ids restored to int64."""
    path = Path(path)
    vp_path, tg_path = path / VP_NODES_CSV, path / TG_NODES_CSV
    if not vp_path.exists() or not tg_path.exists():
        raise MissingArtifactError(
            f"{path} has no {VP_NODES_CSV}/{TG_NODES_CSV}; run `build-bipartite-graph` first"
        )
    vp_nodes = pd.read_csv(vp_path)
    tg_nodes = pd.read_csv(tg_path)
    vp_nodes["vp_grid_id"] = vp_nodes["vp_grid_id"].astype("int64")
    for col in ("site_id", "tg_grid_id", "tg_seed_id"):
        tg_nodes[col] = tg_nodes[col].astype("int64")
    return BipartiteGraph(
        vp_nodes=vp_nodes,
        tg_nodes=tg_nodes,
        edge_segments=pd.read_csv(path / EDGE_SEGMENTS_CSV),
        edge_length_cdf=pd.read_csv(path / EDGE_LENGTH_CDF_CSV),
        pairwise_distance_cdf=pd.read_csv(path / PAIRWISE_CDF_CSV),
        meta=json.loads((path / META_JSON).read_text()),
    )
