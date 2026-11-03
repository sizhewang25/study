"""TG coverage: how many distinct VPs measured each TG, and the scoring floor.

A mesh measures almost every TG from almost every VP. On the pro meshes
1,267 of 1,269 TGs were measured by at least 130 of the 134 VPs; the other
two by 26 and 80. A TG missing most of the fleet is a measurement gap, not a
property of the network: the one no-X TG that S-P "classifies correctly" is
the 26-VP one, and it is right only because the VPs that win for every other
replica of its site never measured it.

So a run may declare a floor, `analysis.common.min_vps_per_tg`
(`labels.declared_min_vps_per_tg`), and v5 then **scores** only the TGs at or
above it. Scoring only:

* the benchmark's outputs and the measurements are untouched -- the dropped
  TGs still ran, and still served as calibration TGs in the other folds;
* the answer space is untouched -- built from every evaluated TG, so a site
  and its cell do not depend on the floor (every dropped TG's site keeps its
  other replicas anyway);
* every v5 frame that starts from `classify` or from the `pni_gap` clusters
  drops the same TGs, and each writer records which (`record`).

The count is distinct VPs per TG in the run's edge CSV
(`edges.load_min_rtt`), the same table every PNI and VP-distance step reads.
On a mesh it equals the scored observations' count.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from scripts.analysis.v5.modules import edges
from scripts.analysis.v5.modules.paths import RunPaths

#: The manifest key every writer records the floor and its drops under.
RECORD_KEY = "min_vps_per_tg"


def vp_counts(run: RunPaths, *, source_csv: Path | None = None) -> pd.Series:
    """Distinct VPs per TG, indexed by `tg_id`, from the run's edge CSV."""
    rtt = edges.load_min_rtt(run, source_csv=source_csv)
    return rtt.groupby("tg_id")["vp_id"].nunique().rename("n_vps")


def low_coverage(
    run: RunPaths, min_vps: int | None, *, source_csv: Path | None = None,
    counts: pd.Series | None = None,
) -> pd.DataFrame:
    """`(tg_id, n_vps)` for every TG measured by fewer than `min_vps` VPs.

    Empty when `min_vps` is None. `counts` skips the CSV read when the caller
    already holds `vp_counts`.
    """
    if min_vps is None:
        return pd.DataFrame({"tg_id": pd.Series(dtype=str), "n_vps": pd.Series(dtype=int)})
    n = counts if counts is not None else vp_counts(run, source_csv=source_csv)
    low = n[n < int(min_vps)]
    return (
        pd.DataFrame({"tg_id": low.index.astype(str), "n_vps": low.to_numpy().astype(int)})
        .sort_values(["n_vps", "tg_id"])
        .reset_index(drop=True)
    )


def drop(frame: pd.DataFrame, low: pd.DataFrame, *, col: str = "tg_id") -> pd.DataFrame:
    """`frame` without the TGs in `low`, index reset."""
    if low.empty:
        return frame
    return frame[~frame[col].astype(str).isin(set(low["tg_id"]))].reset_index(drop=True)


def record(min_vps: int | None, low: pd.DataFrame) -> dict:
    """The manifest entry: the floor, and each TG it dropped with its count."""
    return {
        "floor": None if min_vps is None else int(min_vps),
        "n_tgs_dropped": int(len(low)),
        "dropped": [
            {"tg_id": str(t), "n_vps": int(n)} for t, n in zip(low["tg_id"], low["n_vps"])
        ],
        "note": (
            "scoring only: TGs measured by fewer distinct VPs than `floor` are left "
            "out of every v5 frame; the benchmark outputs and the answer space keep them"
        ),
    }


def from_record(record: dict) -> pd.DataFrame:
    """The dropped `(tg_id, n_vps)` a writer recorded under `RECORD_KEY`.

    For readers that pair a recorded artifact (the `pni_gap` clusters) with
    the raw edge CSV: they drop exactly what the artifact dropped, so their
    TG-set checks still catch any other disagreement.
    """
    rows = ((record or {}).get(RECORD_KEY) or {}).get("dropped") or []
    return pd.DataFrame(
        {"tg_id": pd.Series([str(r["tg_id"]) for r in rows], dtype=str),
         "n_vps": pd.Series([int(r["n_vps"]) for r in rows], dtype=int)}
    )


def declared_low_coverage(run: RunPaths) -> pd.DataFrame:
    """`low_coverage` under the run's declared floor (`labels.declared_min_vps_per_tg`).

    For readers that start from the benchmark tree rather than from a v5
    artifact (`cost.load_cost_frame`). Empty, without reading the edge CSV,
    when the run declares no floor.
    """
    from scripts.analysis.v5.modules.labels import declared_min_vps_per_tg

    floor = declared_min_vps_per_tg(run.run_id)
    if floor is None:
        return low_coverage(run, None)
    return _low_cached(run.run_id, str(edges.resolve_source_csv(run)), int(floor), run).copy()


@lru_cache(maxsize=None)
def _low_cached(run_id: str, csv: str, floor: int, run: RunPaths) -> pd.DataFrame:
    return low_coverage(run, floor, source_csv=Path(csv))
