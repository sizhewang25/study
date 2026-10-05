"""H3 comparison for Appendix A (why HEALPix, not H3): over CONUS, H3 cell-area
max/min and the largest TG-prediction distance within k grid steps, in units of
each cell's own sqrt(area). Measured 2026-10-04: area max/min 1.60-1.71 at
res 3-5; largest error within k = 1.3, 2.4, 3.6, 4.7, 5.9 w for k = 0..4.
Run: PATH=$PWD/.venv/bin:$PATH python tasks/20261003-draft-v1.1/h3_step_bounds.py"""
import numpy as np, h3
from scripts.analysis.v5.modules.geodesy import pairwise_km

rng = np.random.default_rng(0)

def pts(c):
    b = np.array(h3.cell_to_boundary(c)); P = [np.array(h3.cell_to_latlng(c))]
    for i in range(len(b)):
        for t in np.linspace(0, 1, 6, endpoint=False):
            P.append(b[i] * (1 - t) + b[(i + 1) % len(b)] * t)
    return np.array(P)

for res in (3, 4, 5):
    lat = rng.uniform(26, 48, 60); lon = rng.uniform(-122, -70, 60)
    cells = list({h3.latlng_to_cell(a, b, res) for a, b in zip(lat, lon)})
    areas = np.array([h3.cell_area(c, "km^2") for c in cells])
    mx = {k: 0.0 for k in range(5)}
    for c in cells:
        A = pts(c); w = np.sqrt(h3.cell_area(c, "km^2"))
        for k in range(5):
            for q in h3.grid_ring(c, k):
                B = pts(q)
                mx[k] = max(mx[k], pairwise_km(A[:, 0], A[:, 1], B[:, 0], B[:, 1]).max() / w)
    print(f"res {res}: area max/min {areas.max() / areas.min():.3f}; largest error within k (w): "
          + " ".join(f"{mx[k]:.2f}" for k in range(5)))
