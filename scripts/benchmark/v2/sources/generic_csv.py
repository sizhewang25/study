"""GenericCSVSource — adapt any CSV with the canonical schema into the v2 benchmark.

Canonical schema (one row per `(vp, target, rtt)` observation):

  required:
    vp_id, vp_lat, vp_lon            str, float, float    — entity acting as VP
    target_id, target_lat, target_lon str, float, float   — entity being geolocated
    rtt_ms                           float (>0)           — strictly positive RTT
  optional:
    vp_asn, vp_country, vp_continent, vp_region, vp_city
    target_asn, target_country, target_continent, target_region, target_city
    weight                      float (>=0)          — traffic weight of the
                                     (vp, target) pair, in whatever unit the
                                     dataset uses (e.g. TB). Column absent →
                                     1.0 everywhere (target weight degenerates
                                     to obs count); column present but cell
                                     NaN → 0.0 (unknown traffic = weightless).

`vp_*` columns **always** supply the VP-role data; `target_*` columns **always**
supply the target-role data. The `setup` flag is descriptive metadata only —
it does not affect column routing. Users canonicalize their CSV per the
desired role assignment before pointing this source at it. For RIPE Atlas /
Vultr-style data (anchors-as-VPs by convention), anchor data goes into the
`vp_*` columns and probe data goes into the `target_*` columns; for
IMC-2023-style data (probes-as-VPs) the mapping is reversed.

Slicing (`--slice`):
  all        — every row, no fit/eval split (smoke-test mode; leaks for stateful LTDs).
  head<k>    — keep the k targets that sort first by target_id (deterministic
               smoke slice; same no-stratification semantics as `all`).
  fold_N     — K-fold partition. Eval = the targets in fold N; fit = the
               targets in the other K-1 folds. The partition is chosen by
               `fold_by`:
                 * `distgeo` (default) — `DistGeoStratification` over single
                   targets. Deterministic in (k, seed, asn_bucket_top_n).
                   When `target_asn` is absent / missing, those targets land
                   in the `asn_none` bucket and still round-robin into the K
                   folds. Targets sharing a coordinate (IP replicas of one
                   site) are spread across folds, so a test target's site
                   is usually in its own fit set.
                 * `site` — every target sharing a coordinate (a *site*,
                   lat/lon rounded to `SITE_DECIMALS`) lands in one fold.
                   Sites are sorted by (lat, lon) and dealt round-robin, so
                   `k == n_sites` is leave-one-site-out and `k < n_sites` is
                   grouped K-fold. `k > n_sites` is refused: it would leave
                   folds empty. `seed` and `asn_bucket_top_n` are unused.

Source kwargs (defaults match the prior VultrCSVSource):
  csv_path           : Path | str   — required; canonical-schema CSV path.
  k                  : int = 5      — fold count for `fold_N`.
  seed               : int = 42     — DistGeo RNG seed.
  asn_bucket_top_n   : int = 20     — DistGeo bucket cap.
  min_obs            : int = None   — drop targets with fewer VP observations.
  fold_by            : str = "distgeo" — `distgeo` or `site`; see `fold_N`.

This source is weight-AWARE but never weight-FILTERING: it reads and validates
an optional `weight` column and carries it into `obs_weights` (so the required
`weight` field of `EVAL_OBSERVATIONS_SCHEMA` holds real traffic), but it never
restricts the eval set by it. For traffic-weighted evaluation use
`TrafficWeightedCSVSource` (`traffic_weighted_csv`), which subclasses this one
and adds the eval-side flow mask.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from scripts.benchmark.v2.sources.base import (
    DataSource,
    EvalTarget,
    TgConfig,
    VpConfig,
)
from scripts.framework.v2 import FitSample
from scripts.framework.v2.types import Coord, Latency, VpId
# The identity converter that keeps `_OPTIONAL_STR` cells out of pandas'
# NA-sentinel coercion. Defined in the canonical-CSV library rather than here:
# the precheck's reader needs the same converter, and importing it *from* this
# module is what used to pull `scripts.framework.v2` into the analysis layer.
from scripts.libs.canonical.schema import raw_str as _raw_str
from scripts.processing.ripe_atlas.stratification import (
    AnchorInfo,
    DistGeoStratification,
    normalize_asn,
)

logger = logging.getLogger(__name__)


_REQUIRED = (
    "vp_id", "vp_lat", "vp_lon",
    "target_id", "target_lat", "target_lon",
    "rtt_ms",
)

_OPTIONAL = (
    "vp_asn", "vp_country", "vp_continent", "vp_region", "vp_city",
    "target_asn", "target_country", "target_continent", "target_region", "target_city",
)

# Optional free-text columns. Read with an identity converter so pandas does
# NOT apply its default NA-sentinel set to them — otherwise common literal
# codes like the continent "NA" (North America) or country "NA" (Namibia)
# would silently parse as NaN. Converters for columns absent from the CSV are
# ignored by pandas, so passing the full tuple is safe.
_OPTIONAL_STR = (
    "vp_country", "vp_continent", "vp_region", "vp_city",
    "target_country", "target_continent", "target_region", "target_city",
)

_FOLD_SLICE_RE = re.compile(r"^fold_(\d+)$")

#: `fold_by` values: who `fold_N` partitions.
FOLD_BY = ("distgeo", "site")

#: Decimals a target coordinate is rounded to before it names a site. Matches
#: v5's `sites.SITE_DECIMALS`; kept here rather than imported, so this source
#: stays on the libs layer. Replicas carry byte-identical coordinates, so any
#: rounding from 2 to 6 decimals gives the same sites on today's datasets.
SITE_DECIMALS = 6


class GenericCSVSource(DataSource):
    """CSV-backed source with a fixed canonical schema + K-fold stratification (DistGeo or by site).

    See module docstring for the column contract and slice grammar.
    """

    name = "generic_csv"

    def __init__(
        self,
        slice: str,
        setup: str = DataSource.ANCHORS_TO_PROBES,
        csv_path: Optional[Path] = None,
        *,
        k: int = 5,
        seed: int = 42,
        asn_bucket_top_n: int = 20,
        min_obs: Optional[int] = None,
        fold_by: str = "distgeo",
    ) -> None:
        if fold_by not in FOLD_BY:
            raise ValueError(f"unknown fold_by {fold_by!r}; expected one of {FOLD_BY}")
        if setup not in DataSource.ALLOWED_SETUPS:
            raise ValueError(
                f"unknown setup {setup!r}; expected one of {DataSource.ALLOWED_SETUPS}"
            )
        if csv_path is None:
            raise ValueError(
                f"{self.name!r} requires `csv_path` (path to a canonical-schema CSV)"
            )
        fold_match = _FOLD_SLICE_RE.match(slice)
        if fold_match is not None:
            fold_index = int(fold_match.group(1))
            if fold_index >= k:
                raise ValueError(
                    f"slice fold index {fold_index} >= k={k} "
                    f"(available: fold_0..fold_{k - 1})"
                )
        elif slice == "all" or slice.startswith("head"):
            # `head<k>` is fully validated inside _apply_slice when the row
            # parser runs; constructor only confirms the prefix is legal.
            fold_index = None
        else:
            raise ValueError(
                f"unknown slice {slice!r}; expected 'all', 'head<k>', or 'fold_N'"
            )

        self._slice = slice
        self._setup = setup
        self._csv_path = Path(csv_path)
        self._fold_index = fold_index
        self._k = k
        self._seed = seed
        self._asn_bucket_top_n = asn_bucket_top_n
        self._min_obs = min_obs
        self._fold_by = fold_by
        # Set by `_normalize_weight`: whether the CSV carried a real `weight`
        # column, as opposed to the synthesized 1.0 default. Subclasses that
        # filter on traffic need this to refuse a weightless mesh.
        self._weight_column_present: bool = False

        # Lazily populated by `_ensure_loaded`.
        self._df: Optional[pd.DataFrame] = None
        self._eval_targets: Optional[set[str]] = None
        self._fit_targets: Optional[set[str]] = None

    # ---- DataSource API ------------------------------------------------------

    def slice_id(self) -> str:
        return self._slice

    def setup_id(self) -> str:
        return self._setup

    def iter_vp_configs(self) -> Iterator[VpConfig]:
        df = self._ensure_loaded()
        cols = df.columns
        for _, row in df.drop_duplicates("vp_id").iterrows():
            yield VpConfig(
                vp_id=str(row["vp_id"]),
                lat=float(row["vp_lat"]),
                lon=float(row["vp_lon"]),
                asn=normalize_asn(row.get("vp_asn")) if "vp_asn" in cols else None,
                country=_opt_col(row, "vp_country", cols),
                continent=_opt_col(row, "vp_continent", cols),
                region=_opt_col(row, "vp_region", cols),
                city=_opt_col(row, "vp_city", cols),
            )

    def iter_tg_configs(self) -> Iterator[TgConfig]:
        # Static catalog of every target the source knows about (eval ∪ fit).
        # The eval/fit split is enforced inside iter_eval_targets /
        # iter_fit_samples. Matches the convention at
        # ripe_atlas_asn_corpora.py:137-158.
        df = self._ensure_loaded()
        cols = df.columns
        for _, row in df.drop_duplicates("target_id").iterrows():
            yield TgConfig(
                tg_id=str(row["target_id"]),
                lat=float(row["target_lat"]),
                lon=float(row["target_lon"]),
                asn=normalize_asn(row.get("target_asn")) if "target_asn" in cols else None,
                country=_opt_col(row, "target_country", cols),
                continent=_opt_col(row, "target_continent", cols),
                region=_opt_col(row, "target_region", cols),
                city=_opt_col(row, "target_city", cols),
            )

    def iter_fit_samples(self) -> Iterator[FitSample]:
        df = self._ensure_loaded()
        for row in df.itertuples(index=False):
            tg_id = str(row.target_id)
            if self._fit_targets is not None and tg_id not in self._fit_targets:
                continue
            yield FitSample(
                vp_id=VpId(str(row.vp_id)),
                vp_coord=Coord(lat=float(row.vp_lat), lon=float(row.vp_lon)),
                probe_coord=Coord(lat=float(row.target_lat), lon=float(row.target_lon)),
                latency=Latency(float(row.rtt_ms)),
            )

    def iter_eval_targets(self) -> Iterator[EvalTarget]:
        df = self._ensure_loaded()
        for tg_id, group in df.groupby("target_id", sort=True):
            tg_id_str = str(tg_id)
            if self._eval_targets is not None and tg_id_str not in self._eval_targets:
                continue
            first = group.iloc[0]
            true_coord = Coord(
                lat=float(first["target_lat"]),
                lon=float(first["target_lon"]),
            )
            obs: list[tuple[VpId, Coord, Latency]] = []
            obs_weights: list[float] = []
            for r in group.itertuples(index=False):
                if not self._keep_obs(r):
                    continue
                obs.append((
                    VpId(str(r.vp_id)),
                    Coord(lat=float(r.vp_lat), lon=float(r.vp_lon)),
                    Latency(float(r.rtt_ms)),
                ))
                obs_weights.append(float(r.weight))
            yield EvalTarget(
                target_id=tg_id_str, true_coord=true_coord,
                obs=obs, obs_weights=obs_weights,
            )

    # ---- extension hooks -----------------------------------------------------

    def _keep_obs(self, row) -> bool:
        """Per-observation eval filter. Always True here — this source evaluates
        every observation of an eval target. `TrafficWeightedCSVSource` overrides
        it to keep only the flows in the traffic-weighted subset."""
        return True

    # ---- internals -----------------------------------------------------------

    def _ensure_loaded(self) -> pd.DataFrame:
        if self._df is None:
            self._load_csv()
            if self._fold_index is not None:
                self._apply_stratification()
            if self._min_obs is not None:
                self._apply_min_obs_filter()
        assert self._df is not None
        return self._df

    def _load_csv(self) -> None:
        # Lowercase all headers so CSVs with uppercase columns (e.g. VP_ID,
        # TARGET_LAT) load against the canonical lowercase schema. Converters
        # cover both cases so the NA-sentinel opt-out applies either way.
        converters = {c: _raw_str for col in _OPTIONAL_STR for c in (col, col.upper())}
        df = pd.read_csv(self._csv_path, converters=converters)
        df.columns = df.columns.str.lower()
        missing = [c for c in _REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(
                f"CSV {self._csv_path} missing required columns: {missing}. "
                f"Required: {list(_REQUIRED)}; optional: {list(_OPTIONAL)}."
            )
        df = df.dropna(subset=list(_REQUIRED))
        df = df[df["rtt_ms"] > 0].copy()
        df = self._normalize_weight(df)
        df = self._apply_slice(df, self._slice)
        self._df = df.reset_index(drop=True)

    def _normalize_weight(self, df: pd.DataFrame) -> pd.DataFrame:
        """Guarantee a numeric non-negative `weight` column.

        Two distinct defaults, deliberately:
          * column absent  → 1.0 everywhere ("this dataset has no weight
            notion"; summed target weight degenerates to obs count, and
            pair-weight-min thresholds <= 1.0 stay no-ops).
          * column present but cell NaN/non-numeric → 0.0 ("traffic unknown
            = weightless"): the flow can never outrank genuinely heavy flows
            in the kept-traffic ranking nor survive a weighted-eval
            threshold. Filling 1.0 here would masquerade as 1 unit of real
            traffic (e.g. 1 TB) among measured values.
        """
        if "weight" not in df.columns:
            self._weight_column_present = False
            df["weight"] = 1.0
            return df
        self._weight_column_present = True
        df["weight"] = pd.to_numeric(df["weight"], errors="coerce")
        n_missing = int(df["weight"].isna().sum())
        if n_missing:
            logger.info(
                "weight: %d missing/non-numeric cells defaulted to 0.0 "
                "(unknown traffic = weightless)",
                n_missing,
            )
            df["weight"] = df["weight"].fillna(0.0)
        if (df["weight"] < 0).any():
            raise ValueError(
                f"CSV {self._csv_path}: weight must be >= 0 "
                f"(found negative values)"
            )
        return df

    @staticmethod
    def _apply_slice(df: pd.DataFrame, slice_name: str) -> pd.DataFrame:
        """Row-level slice. `fold_N` keeps every row — the eval/fit partition
        lives in cached id sets, not in row drops."""
        if slice_name == "all":
            return df
        if _FOLD_SLICE_RE.match(slice_name):
            return df
        if slice_name.startswith("head"):
            try:
                k = int(slice_name.removeprefix("head"))
            except ValueError as e:
                raise ValueError(f"invalid head-k slice: {slice_name!r}") from e
            if k < 1:
                raise ValueError(f"head-k must be >=1, got {k}")
            keep = sorted(df["target_id"].astype(str).unique())[:k]
            if not keep:
                raise ValueError(
                    f"slice {slice_name!r} selected no targets (empty CSV?)"
                )
            return df[df["target_id"].astype(str).isin(keep)].copy()
        raise ValueError(
            f"unknown slice {slice_name!r}; expected 'all', 'head<k>', or 'fold_N'"
        )

    def _apply_stratification(self) -> None:
        """Partition unique target_ids into K folds (`fold_by`) and cache the
        eval / fit target-id sets."""
        assert self._df is not None and self._fold_index is not None
        unique = self._df.drop_duplicates("target_id")
        if self._fold_by == "site":
            fold_by_id = self._site_fold_assignments(unique)
        else:
            fold_by_id = self._distgeo_fold_assignments(unique)

        eval_targets: set[str] = set()
        fit_targets: set[str] = set()
        for tg_id, fold in fold_by_id.items():
            (eval_targets if fold == self._fold_index else fit_targets).add(tg_id)
        if self._fold_by == "site" and not (eval_targets and fit_targets):
            # Unreachable while 2 <= k <= n_sites; a guard, not a policy.
            raise ValueError(
                f"fold_by='site' fold_{self._fold_index}: empty "
                f"{'eval' if not eval_targets else 'fit'} set"
            )
        self._eval_targets = eval_targets
        self._fit_targets = fit_targets
        logger.info(
            "stratified %d targets (fold_by=%s) into K=%d folds: eval=fold_%d "
            "(%d targets), fit=union of %d other folds (%d targets)",
            len(fold_by_id), self._fold_by, self._k, self._fold_index,
            len(eval_targets), self._k - 1, len(fit_targets),
        )

    def _site_fold_assignments(self, unique: pd.DataFrame) -> dict[str, int]:
        """`target_id -> fold`, one site per fold slot, sites dealt round-robin.

        A site is a target coordinate rounded to `SITE_DECIMALS`. Sites are
        ranked by (lat, lon), so the assignment depends on the coordinates
        alone -- not on row order, target ids, or a seed.
        """
        if self._k < 2:
            raise ValueError(f"fold_by='site' needs k >= 2; got k={self._k}")
        lat = unique["target_lat"].astype(float).round(SITE_DECIMALS)
        lon = unique["target_lon"].astype(float).round(SITE_DECIMALS)
        sites = sorted(set(zip(lat, lon)))
        if self._k > len(sites):
            raise ValueError(
                f"fold_by='site' with k={self._k} but the CSV holds only "
                f"{len(sites)} sites; k > n_sites would leave folds empty. "
                f"Use k={len(sites)} for leave-one-site-out."
            )
        fold_of_site = {s: r % self._k for r, s in enumerate(sites)}
        assignments = {
            str(tg): fold_of_site[(la, lo)]
            for tg, la, lo in zip(unique["target_id"], lat, lon)
        }
        if self._fold_index is not None:
            held = [s for s, f in fold_of_site.items() if f == self._fold_index]
            logger.info(
                "fold_by=site: %d sites over K=%d folds; fold_%d holds %s",
                len(sites), self._k, self._fold_index,
                ", ".join(f"({la:g},{lo:g})" for la, lo in held),
            )
        return assignments

    def _distgeo_fold_assignments(self, unique: pd.DataFrame) -> dict[str, int]:
        """`target_id -> fold` via `DistGeoStratification` over single targets."""
        targets: list[AnchorInfo] = []
        for _, row in unique.iterrows():
            asn = row.get("target_asn")
            country = row.get("target_country")
            targets.append(AnchorInfo(
                ip=str(row["target_id"]),
                lat=float(row["target_lat"]),
                lon=float(row["target_lon"]),
                country=str(country) if pd.notna(country) else None,
                asn=normalize_asn(asn),
            ))

        algo = DistGeoStratification(
            k=self._k,
            fold_index=0,  # full assignment is fold-index-independent
            seed=self._seed,
            asn_bucket_top_n=self._asn_bucket_top_n,
        )
        return algo.compute_fold_assignments(targets)

    def _apply_min_obs_filter(self) -> None:
        assert self._df is not None and self._min_obs is not None
        counts = self._df.groupby("target_id")["target_id"].transform("count")
        before = self._df["target_id"].nunique()
        self._df = self._df[counts >= self._min_obs].reset_index(drop=True)
        after = self._df["target_id"].nunique()
        surviving = set(self._df["target_id"].astype(str))
        if self._eval_targets is not None:
            self._eval_targets &= surviving
        if self._fit_targets is not None:
            self._fit_targets &= surviving
        logger.info("min_obs=%d: %d → %d targets", self._min_obs, before, after)


def _opt_col(row: "pd.Series", col: str, cols: "pd.Index") -> Optional[str]:
    """Stringify an optional column value, or None when the column is absent
    from the CSV, the cell is NaN, or the cell is empty/whitespace (empty
    cells arrive as "" under the `_raw_str` converter, not NaN)."""
    if col not in cols:
        return None
    value = row.get(col)
    if not pd.notna(value):
        return None
    text = str(value).strip()
    return text or None
