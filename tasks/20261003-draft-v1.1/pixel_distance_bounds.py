"""Pixel-distance bounds for the paper (Appendix A, tab:pixel-distance-bounds): min/max
point of its ring-k pixels, in units of pixel width w = sqrt(area), over CONUS."""
import numpy as np, astropy.units as u
from astropy_healpix import HEALPix
from scripts.analysis.v5.modules import grid as G
from scripts.analysis.v5.modules.geodesy import pairwise_km

rng = np.random.default_rng(0)
STEP = 6  # boundary points per edge
for nside in (32, 64, 128, 256, 512):
    hp = HEALPix(nside=nside, order="nested")
    w = G.grid_km(nside)
    lat = rng.uniform(25, 49, 60); lon = rng.uniform(-124, -68, 60)
    cen = np.unique(G.ang2pix(lat, lon, nside))
    res = {k: [np.inf, 0.0] for k in range(5)}
    def pts(pix):
        lo, la = hp.boundaries_lonlat(np.atleast_1d(pix), step=STEP)
        lo = lo.to_value(u.deg).ravel(); la = la.to_value(u.deg).ravel()
        lo = ((lo + 180) % 360) - 180
        # interior samples too (centre), boundaries carry the extremes
        c = G.pix2ang(np.atleast_1d(pix), nside)
        return np.concatenate([la, c[:, 0]]), np.concatenate([lo, c[:, 1]])
    for p in cen:
        rings = G.ring_grids(int(p), nside, 4)
        a_la, a_lo = pts(p)
        for k, ring in enumerate(rings):
            for q in ring:
                b_la, b_lo = pts(q)
                d = pairwise_km(a_la, a_lo, b_la, b_lo) / w
                res[k][0] = min(res[k][0], d.min()); res[k][1] = max(res[k][1], d.max())
    print(f"nside {nside:4d} w={w:6.1f} km  " + "  ".join(f"r{k}:[{lo:.2f},{hi:.2f}]" for k, (lo, hi) in res.items()))
