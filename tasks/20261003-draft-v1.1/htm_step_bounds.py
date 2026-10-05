"""HTM comparison for Appendix A: over CONUS, HTM trixel-area max/min and the
largest TG-prediction distance within k steps (neighbours = trixels touching at
an edge or a corner, as for HEALPix), in units of each trixel's own sqrt(area).
HTM: octahedron, every spherical triangle split into four at each level.
Run: PATH=$PWD/.venv/bin:$PATH python tasks/20261003-draft-v1.1/htm_step_bounds.py"""
import numpy as np
from collections import defaultdict
from scripts.analysis.v5.modules.geodesy import pairwise_km

R = 6371.0088
rng = np.random.default_rng(0)

def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)

def htm(level):
    V = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
    T = np.array([[1, 0, 4], [4, 0, 3], [3, 0, 2], [2, 0, 1], [1, 5, 2], [2, 5, 3], [3, 5, 4], [4, 5, 1]])
    tri = V[T]  # (n, 3, 3)
    for _ in range(level):
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, bc, ca = unit(a + b), unit(b + c), unit(c + a)
        tri = np.concatenate([np.stack(x, 1) for x in ((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca))])
    return tri

def sph_area(tri):
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    num = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    den = 1 + np.einsum("ij,ij->i", a, b) + np.einsum("ij,ij->i", b, c) + np.einsum("ij,ij->i", c, a)
    return 2 * np.arctan2(num, den) * R * R

def latlon(v):
    return np.degrees(np.arcsin(v[..., 2])), np.degrees(np.arctan2(v[..., 1], v[..., 0]))

def boundary(t):
    P = [unit(t.mean(0))]
    for i in range(3):
        for s in np.linspace(0, 1, 6, endpoint=False):
            P.append(unit(t[i] * (1 - s) + t[(i + 1) % 3] * s))
    return np.array(P)

for level in (6, 7, 8):
    tri = htm(level)
    cen = unit(tri.mean(1)); clat, clon = latlon(cen)
    us = np.flatnonzero((clat > 22) & (clat < 52) & (clon > -128) & (clon < -64))
    tri_us = tri[us]; area = sph_area(tri_us)
    # vertex -> trixels, for the edge-or-corner neighbourhood
    key = lambda v: tuple(np.round(v, 9))
    v2t = defaultdict(set)
    for i, t in enumerate(tri_us):
        for v in t:
            v2t[key(v)].add(i)
    nbr = [set().union(*(v2t[key(v)] for v in t)) - {i} for i, t in enumerate(tri_us)]
    ulat, ulon = latlon(unit(tri_us.mean(1)))
    inner = np.flatnonzero((ulat > 27) & (ulat < 47) & (ulon > -120) & (ulon < -72))
    mx = {k: 0.0 for k in range(5)}; deg = []
    for c in rng.choice(inner, 40, replace=False):
        deg.append(len(nbr[c]))
        A = boundary(tri_us[c]); alat, alon = latlon(A); w = np.sqrt(area[c])
        seen = {c}; frontier = {c}
        for k in range(5):
            if k:
                frontier = set().union(*(nbr[f] for f in frontier)) - seen; seen |= frontier
            for q in frontier:
                blat, blon = latlon(boundary(tri_us[q]))
                mx[k] = max(mx[k], pairwise_km(alat, alon, blat, blon).max() / w)
    print(f"level {level}: w~{np.sqrt(area.mean()):.0f} km, CONUS area max/min {area.max() / area.min():.3f}, "
          f"neighbours {min(deg)}-{max(deg)}; largest error within k (w): " + " ".join(f"{mx[k]:.2f}" for k in range(5)))
