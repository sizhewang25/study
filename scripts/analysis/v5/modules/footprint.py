"""The footprint span D: the value to declare as `dist_norm_km.max`.

The paper never prints a distance in km. It reports error distance min-max
normalized with **fixed** bounds -- 0, a prediction on its TG, and D, the
largest great-circle distance between any two of the VPs and sites in the
datasets -- so a normalized value means the same fraction of the footprint in
every figure, and the ratio between two methods' errors survives.

The figures do not compute D. They read the bounds each run's config declares
(`analysis.common.dist_norm_km: {min, max}`, `labels.declared_dist_norm_km`),
so D is one constant for every figure and every regime. This module computes
the value to declare, from the canonical CSVs' VP and target coordinates,
never from predictions:

    python -m scripts.analysis.v5.modules.footprint pro-as01-mesh pro-as02-mesh pro-as03-mesh

Pass every dataset the paper reports; D over a subset is a different
constant. A LOSO run reads the same CSV as its mesh run, so mixing the two
changes nothing. On the three meshes D is 4,387.257 km (130 distinct
coordinates); the value is confidential in the paper.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.analysis.v5.modules import edges, geodesy
from scripts.analysis.v5.modules.paths import RunPaths

#: Canonical-CSV coordinate columns, benchmark names.
_VP = ("vp_lat", "vp_lon")
_TG = ("target_lat", "target_lon")


def coordinates(run: RunPaths, *, source_csv: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """`(vp, site)` distinct `(lat, lon)` arrays from the run's canonical CSV."""
    from scripts.libs.canonical.schema import load_canonical_csv

    df = load_canonical_csv(edges.resolve_source_csv(run, source_csv))
    vp = df[list(_VP)].drop_duplicates().to_numpy(dtype=float)
    site = df[list(_TG)].drop_duplicates().to_numpy(dtype=float)
    return vp, site


def footprint_span(runs: list[RunPaths]) -> dict:
    """D over every VP and site of `runs`, pooled, plus what it was taken over.

    `span_km` is the largest pairwise great-circle distance among the distinct
    coordinates; a VP and a site at one place count once.
    """
    if not runs:
        raise ValueError("footprint_span needs at least one run")
    vps, sites = [], []
    for run in runs:
        vp, site = coordinates(run)
        vps.append(vp)
        sites.append(site)
    vp = np.unique(np.vstack(vps), axis=0)
    site = np.unique(np.vstack(sites), axis=0)
    points = np.unique(np.vstack([vp, site]), axis=0)
    span = float(geodesy.pairwise_km(points[:, 0], points[:, 1]).max()) if len(points) > 1 else 0.0
    if not span > 0:
        raise ValueError(
            f"footprint span over {[r.run_id for r in runs]} is {span} km; "
            "normalizing by it is undefined"
        )
    return {
        "span_km": round(span, 3),
        "runs": sorted(r.run_id for r in runs),
        "n_vp_coords": int(len(vp)),
        "n_site_coords": int(len(site)),
        "n_coords": int(len(points)),
        "definition": (
            "largest great-circle distance between any two distinct VP or site "
            "(TG) coordinates in the runs' canonical CSVs, pooled over the runs"
        ),
    }


if __name__ == "__main__":
    import json
    import sys

    from scripts.analysis.v5.modules.paths import resolve_run

    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m scripts.analysis.v5.modules.footprint RUN_ID [RUN_ID ...]")
    print(json.dumps(footprint_span([resolve_run(r) for r in sys.argv[1:]]), indent=2))
