"""Artifact access for slug getters: one cached loader per file, under the group's pools."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import pandas as pd

from scripts.analysis.v5.modules.labels import load_group
from scripts.analysis.v5.modules.paths import DEFAULT_ANALYSIS_ROOT, DEFAULT_OUTPUTS_ROOT
from scripts.analysis.v5.paper.core import Context

SP, SOI, VAN, OCTH, OCTS, SPO = (
    "shortest_ping", "million_scale_cbg", "vanilla_cbg", "octant_cbg_hull", "octant_cbg_spl", "spotter_cbg")
OCTH_GEO = "octant_cbg_hull_geo"
TERM = {SP: "S-P", SOI: "SOI", VAN: "VAN", OCTH: "OCT-H", OCTS: "OCT-S", SPO: "SPO"}
COMBO = {v: k for k, v in TERM.items()}
#: Template keys name methods by these (OCT-H -> oct-h).
METHODS = ("s-p", "soi", "van", "oct-h", "oct-s", "spo")
KEY_TO_TERM = {k: t for k, t in zip(METHODS, ("S-P", "SOI", "VAN", "OCT-H", "OCT-S", "SPO"))}
CBG = ("soi", "van", "oct-h", "oct-s", "spo")
LIGHT = ("soi", "van", "spo")


def combo(mkey: str) -> str:
    return COMBO[KEY_TO_TERM[mkey]]


class Ctx:
    """The group's artifacts. Every loader is cached; a missing file raises
    `MissingArtifactError` (from `Context`) or `FileNotFoundError`, which
    `generate` records as the slug's `missing` reason."""

    def __init__(self, group_id: str, analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
                 outputs_root: Path = DEFAULT_OUTPUTS_ROOT) -> None:
        self.group = load_group(group_id)
        self.group_id = group_id
        self.analysis_root = Path(analysis_root)
        self.outputs_root = Path(outputs_root)
        self.base = Context(group=self.group, analysis_root=self.analysis_root)
        self._cache: dict = {}

    def runs(self, role: str) -> list[str]:
        return self.base.runs(role)

    def _get(self, key, load):
        if key not in self._cache:
            self._cache[key] = load()
        return self._cache[key]

    def path(self, kind: str, name: str, pool: str = "seen") -> Path:
        d = self.base.seen_dir(kind) if pool == "seen" else self.base.all_dir(kind)
        p = d / name
        if not p.is_file():
            raise FileNotFoundError(f"{p.relative_to(self.analysis_root)} missing")
        return p

    def csv(self, kind: str, name: str, pool: str = "seen") -> pd.DataFrame:
        return self._get(("csv", kind, name, pool), lambda: pd.read_csv(self.path(kind, name, pool)))

    def json(self, kind: str, name: str, pool: str = "seen") -> dict:
        return self._get(("json", kind, name, pool), lambda: json.loads(self.path(kind, name, pool).read_text()))

    def memo(self, key, load):
        """Cache any computed frame under `key`."""
        return self._get(("memo", key), load)

    # ---- the frames most slugs read ------------------------------------------------

    @cached_property
    def bars(self) -> pd.DataFrame:
        """Unbounded outcome bars, pooled seen sites, by combo id (shares are fractions)."""
        return self.csv("classify", "outcome_bars.unbounded.pooled.healpix-128.csv").set_index("method")

    @cached_property
    def ecdf(self) -> pd.DataFrame:
        return self.csv("classify", "error_cdf.pooled.norm.cut.csv").set_index("method")

    def acc(self, mkey: str) -> float:
        """Pooled seen-site cell accuracy, percent."""
        return float(self.bars.loc[combo(mkey), "share_n_cell_correct"] * 100)

    def unanswered(self, mkey: str) -> float:
        return float(self.bars.loc[combo(mkey), "share_n_cell_unanswered"] * 100)

    def err(self, mkey: str, p: int) -> float:
        """Pooled normalized error distance percentile (x1e-3), answered TGs."""
        return float(self.ecdf.loc[combo(mkey), f"pred_dist_to_tg_norm_e3_p{p}"])

    @cached_property
    def pareto(self) -> pd.DataFrame:
        return self.csv("pareto", "pareto.csv", pool="all").set_index(["regime", "metric", "method"])

    def pa(self, regime: str, mkey: str, col: str) -> float:
        return float(self.pareto.loc[(regime, "cell", combo(mkey)), col])
