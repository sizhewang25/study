"""Seeds: sites grouped by complete linkage, one seed per group.

A **seed** is the spherical centroid of a group of sites whose pairwise
distances are all at most `grid_km`. The seeds generate the cell partition:
a cell is the Voronoi cell of one seed, unbounded.

## Why group sites at all

Two sites closer than one grid are one place at that granularity. Without
grouping, EWR and JFK (~33 km apart) would split the New York area into two
slivers of cells, and a prediction landing between them would be `wrong` on
what is at most a naming difference. Grouping at `grid_km` makes the cell
partition as coarse as the grid partition it sits beside.

## Why complete linkage

It caps the group **diameter** -- the largest pairwise distance -- which is the
quantity "these sites are within one grid of each other" is about. Single
linkage would chain sites 40 km apart into a group of any length; a cap on the
distance to the centroid is a different quantity (a group's diameter can reach
twice its radius), and an earlier answer space in this repo was caught by
exactly that difference.

## Logical, not geospatial

A group depends only on the distances between sites and on `grid_km`. It does
not depend on where the HEALPix grid boundaries fall: two sites in one grid can
sit in two groups, and two sites in adjacent grids can share one. That
independence is the point of having a second partition.

## Peripheral seeds

A seed is **peripheral** when its cell is unbounded within the footprint's
hemisphere: in a Voronoi diagram, that holds exactly for the seeds on the
convex hull of all seeds. The cells are built from great-circle distances, so
the hull is the spherical one, computed in a gnomonic projection centred on
the seeds' mean direction (great circles map to straight lines, so the planar
hull there is the spherical hull). Seeds on a hull edge between two hull
vertices count as peripheral too: their cells are unbounded strips. On the
globe every cell is finite, but the peripheral ones are the cells that split
the rest of the planet between them -- on the pro meshes, exactly the largest
cells. A flat lat/lon projection would bend the hull and miss a nearly
collinear seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.analysis.v5.modules.geodesy import pairwise_km, spherical_centroid


def group_sites(site_lat, site_lon, diameter_km: float) -> np.ndarray:
    """Seed id per site: complete linkage, every group's diameter <= `diameter_km`.

    Ids are dense, `0..K-1`, ordered by each group's smallest input position,
    so they follow the site ids (which are sorted keys) and not scipy's
    internal labelling. The caller passes sites in `site_id` order.
    """
    lat = np.asarray(site_lat, dtype=float).ravel()
    lon = np.asarray(site_lon, dtype=float).ravel()
    n = lat.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    d = pairwise_km(lat, lon)
    np.fill_diagonal(d, 0.0)
    # Symmetrise: the chord form is symmetric up to the last ulp, and
    # squareform refuses anything that is not exactly so.
    d = (d + d.T) / 2.0
    labels = fcluster(
        linkage(squareform(d, checks=False), method="complete"),
        t=float(diameter_km),
        criterion="distance",
    )
    first_seen: dict[int, int] = {}
    for label in labels:
        first_seen.setdefault(int(label), len(first_seen))
    return np.array([first_seen[int(x)] for x in labels], dtype=np.int64)


#: Tolerance (gnomonic plane units, ~radians near the centre) for a seed lying
#: on a hull edge.
_HULL_EDGE_TOL = 1e-9


def peripheral_seeds(seed_lat, seed_lon) -> np.ndarray:
    """Per seed, whether its cell is unbounded: on the spherical convex hull.

    Three or fewer seeds, or seeds on one great circle, are all peripheral.
    Refuses seeds that do not fit in
    an open hemisphere, where "the footprint's outside" is undefined.
    """
    from scipy.spatial import ConvexHull

    from scripts.analysis.v5.modules.geodesy import unit_vectors

    lat = np.asarray(seed_lat, dtype=float)
    lon = np.asarray(seed_lon, dtype=float)
    n = len(lat)
    if n <= 3:
        return np.ones(n, dtype=bool)
    u = unit_vectors(lat, lon)
    centre = u.mean(axis=0)
    centre /= np.linalg.norm(centre)
    depth = u @ centre
    if not (depth > 0).all():
        raise ValueError("seeds do not fit in an open hemisphere; the hull is undefined")
    east = np.cross([0.0, 0.0, 1.0], centre)
    if np.linalg.norm(east) < 1e-12:  # centre at a pole
        east = np.array([1.0, 0.0, 0.0])
    east /= np.linalg.norm(east)
    north = np.cross(centre, east)
    xy = np.column_stack([(u @ east) / depth, (u @ north) / depth])
    # All seeds on one great circle: every cell is an unbounded strip.
    if np.linalg.matrix_rank(xy - xy.mean(axis=0), tol=_HULL_EDGE_TOL) < 2:
        return np.ones(n, dtype=bool)
    hull = ConvexHull(xy)
    on = np.zeros(n, dtype=bool)
    on[hull.vertices] = True
    # Seeds on a hull edge between two vertices: distance to some facet ~ 0.
    # `hull.equations` rows are (a, b, c) with a*x + b*y + c <= 0 inside.
    slack = xy @ hull.equations[:, :2].T + hull.equations[:, 2]
    on |= (np.abs(slack) <= _HULL_EDGE_TOL).any(axis=1)
    return on


def build_seeds(sites: pd.DataFrame, diameter_km: float) -> tuple[np.ndarray, pd.DataFrame]:
    """`(seed id per site, seeds frame)` from a `sites` frame.

    `sites` needs `site_id, site_lat, site_lon, n_tgs`, sorted by `site_id`.
    The seeds frame is `seed_id, seed_lat, seed_lon, n_sites, n_tgs,
    seed_diameter_km, nearest_seed_km, peripheral` (`peripheral_seeds`).
    """
    seed_of_site = group_sites(sites["site_lat"], sites["site_lon"], diameter_km)
    rows = []
    for seed_id in range(int(seed_of_site.max()) + 1 if len(seed_of_site) else 0):
        member = sites.loc[seed_of_site == seed_id]
        lat, lon = spherical_centroid(member["site_lat"], member["site_lon"])
        span = pairwise_km(member["site_lat"], member["site_lon"])
        rows.append(
            {
                "seed_id": seed_id,
                "seed_lat": lat,
                "seed_lon": lon,
                "n_sites": int(len(member)),
                "n_tgs": int(member["n_tgs"].sum()),
                "seed_diameter_km": round(float(span.max()), 3),
            }
        )
    seeds = pd.DataFrame(
        rows,
        columns=["seed_id", "seed_lat", "seed_lon", "n_sites", "n_tgs", "seed_diameter_km"],
    )
    if len(seeds) > 1:
        mesh = pairwise_km(seeds["seed_lat"], seeds["seed_lon"])
        np.fill_diagonal(mesh, np.inf)
        seeds["nearest_seed_km"] = np.round(mesh.min(axis=1), 3)
    else:
        seeds["nearest_seed_km"] = np.nan
    seeds["peripheral"] = peripheral_seeds(seeds["seed_lat"], seeds["seed_lon"])
    return seed_of_site, seeds
