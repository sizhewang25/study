"""Distance between a TG and a prediction placed uniformly at random in their
pixels, by pixel distance k (exact) and cumulative (<= k), in pixel widths."""
import numpy as np
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.geodesy import elementwise_km

rng = np.random.default_rng(1)

def sample_in(pix, nside, n):
    """Uniform points inside one pixel by rejection from a small cap."""
    c = G.pix2ang(np.array([pix]), nside)[0]
    w_deg = np.degrees(G.grid_km(nside) / 6371.0088)
    out = []
    while sum(len(o) for o in out) < n:
        m = 4 * n
        # uniform in a lat/lon box, area-weighted by cos(lat)
        z0, z1 = np.sin(np.radians(c[0] - 2 * w_deg)), np.sin(np.radians(c[0] + 2 * w_deg))
        lat = np.degrees(np.arcsin(rng.uniform(z0, z1, m)))
        lon = c[1] + rng.uniform(-3, 3, m) * w_deg / np.cos(np.radians(c[0]))
        keep = G.ang2pix(lat, lon, nside) == pix
        out.append(np.column_stack([lat[keep], lon[keep]]))
    return np.concatenate(out)[:n]

for nside in (64, 128, 256):
    w = G.grid_km(nside)
    lat = rng.uniform(26, 48, 15); lon = rng.uniform(-122, -70, 15)
    by_k = {k: [] for k in range(5)}
    for p in np.unique(G.ang2pix(lat, lon, nside)):
        rings = G.ring_grids(int(p), nside, 4)
        a = sample_in(int(p), nside, 400)
        for k, ring in enumerate(rings):
            for q in ring:
                b = sample_in(int(q), nside, 60)
                ia = rng.integers(0, len(a), len(b))
                by_k[k].append(elementwise_km(a[ia, 0], a[ia, 1], b[:, 0], b[:, 1]) / w)
    ex = {k: np.concatenate(v) for k, v in by_k.items()}
    cum = {k: np.concatenate([ex[j] for j in range(k + 1)]) for k in range(5)}
    print(f"nside {nside}: exact median " + " ".join(f"k{k}={np.median(ex[k]):.2f}" for k in range(5))
          + " | exact max " + " ".join(f"{ex[k].max():.2f}" for k in range(5))
          + " | cum(<=k) median " + " ".join(f"{np.median(cum[k]):.2f}" for k in range(5)))
