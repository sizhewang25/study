"""figure_cbg_mechanism: projection, constraint highlight, drawn = used, rendering (synthetic)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Point

from scripts.analysis.v5.modules import figure_cbg_mechanism as F

D = 1000.0


def _p(vp, lat, lon, upper, lower=0.0, rtt=5.0, mu=None, sigma=None):
    return {"vp_id": vp, "vp_lat": lat, "vp_lon": lon, "rtt_ms": rtt, "upper_km": upper, "lower_km": lower,
            "mu_km": mu, "sigma_km": sigma}


def test_frame_puts_the_tg_at_the_origin_in_units_of_d():
    fr = F.Frame(40.0, -90.0, D)
    x, y = fr.xy(40.0, -90.0)
    assert abs(float(x)) < 1e-9 and abs(float(y)) < 1e-9
    # Due north by D km (1 degree of latitude ~ 111.2 km): y ~ 1, x ~ 0.
    x, y = fr.xy(40.0 + D / 111.2, -90.0)
    assert float(y) == pytest.approx(1.0, rel=0.01) and abs(float(x)) < 1e-6
    lat, lon = fr.latlon(x, y)
    assert float(lat) == pytest.approx(40.0 + D / 111.2) and float(lon) == pytest.approx(-90.0)


def test_annulus_ring_has_a_dotted_inner_line_and_a_disk_does_not():
    fr = F.Frame(40.0, -90.0, D)
    outer, inner = F.ring_lines(fr, "planar_annulus_weighted", _p("a", 40.0, -90.5, 80.0, 30.0))
    assert len(outer) == 1 and len(inner) == 1
    outer, inner = F.ring_lines(fr, "planar_circle", _p("a", 40.0, -90.5, 80.0))
    assert len(outer) == 1 and inner == []


def test_bearing_gap_is_small_when_vps_surround_the_tg():
    fr = F.Frame(0.0, 0.0, D)
    around = [_p(str(i), 0.5 * math.sin(a), 0.5 * math.cos(a), 1) for i, a in enumerate(np.linspace(0, 2 * np.pi, 4,
                                                                                                    endpoint=False))]
    one_side = [_p("a", 0.5, 0.5, 1), _p("b", 0.5, 0.6, 1)]
    assert F._bearing_gap_deg(fr, around) == pytest.approx(90.0, abs=1.0)
    assert F._bearing_gap_deg(fr, one_side) > 300


def _examples():
    import scripts.framework.v2  # noqa: F401
    from scripts.analysis.v5.modules import map_mtl as MM
    from scripts.framework.v2.registry import MTL_REGISTRY

    parts = [_p("a", 0.0, -0.5, 70.0), _p("b", 0.4, 0.3, 70.0), _p("c", -0.4, 0.3, 70.0)]
    m = MTL_REGISTRY["planar_circle"](n_pts=64, enable_circle_filter=True).multilaterate(MM.ltd_results(parts))
    c = m.intersection.centroid
    ex = {}
    for combo in F.METHODS:
        if combo == F.SPO:
            sp = [dict(q, mu_km=55.0, sigma_km=8.0) for q in parts]
            ex[combo] = F.MethodExample(combo, 0, sp, None, (0.02, 0.0), True)
        elif combo.startswith("octant"):
            ann = [dict(q, lower_km=20.0) for q in parts]
            reg = MTL_REGISTRY["planar_annulus_weighted"](n_pts=64).multilaterate(MM.ltd_results(ann)).intersection
            ex[combo] = F.MethodExample(combo, 0, ann, reg, (reg.centroid.y, reg.centroid.x), True)
        else:
            ex[combo] = F.MethodExample(combo, 0, parts, m.intersection, (c.y, c.x), True)
    return ex


def test_centroid_star_lies_in_the_disk_region():
    ex = _examples()["vanilla_cbg"]
    assert ex.region.covers(Point(ex.estimate[1], ex.estimate[0]))


def test_bind_share_sees_every_disk_forming_the_region():
    fr = F.Frame(0.0, 0.0, D)
    ex = _examples()["vanilla_cbg"]
    shares = [F._bind_share(fr, ex, p, 0.002) for p in ex.participants]
    assert all(s > 0.15 for s in shares) and sum(shares) == pytest.approx(1.0, abs=0.1)


def test_maps_carry_no_ticks_and_draw_only_participants(tmp_path):
    import matplotlib.pyplot as plt

    fr = F.Frame(0.0, 0.0, D)
    exs = _examples()
    ext = F.fit_extent(fr, exs, "a")
    for combo, ex in exs.items():
        for hl, st in ((True, False), (False, True)):
            fig, ax = F._fig_map()
            F.draw_map(ax, fr, ex, "a", ext, highlight=hl, star=st)
            assert list(ax.get_xticks()) == [] and list(ax.get_yticks()) == []
            n_vp_markers = sum(len(c.get_offsets()) for c in ax.collections
                               if getattr(c, "get_offsets", None) is not None and c.get_offsets().shape[0] == 1
                               and c.get_paths() and c.get_sizes()[0] in (16, 26))
            assert n_vp_markers == len(ex.participants)
            out = F._save(fig, tmp_path / f"{combo}.{hl}.png")
            assert out.exists() and out.stat().st_size > 0
    plt.close("all")


def test_ltd_panel_marks_the_bound_the_vp_rtt_yields(tmp_path):
    p = _p("a", 0.0, 0.0, 300.0, 120.0, rtt=10.0)
    grid = np.linspace(0.05, 40.0, 50)
    data = F.LtdData(np.array([[5.0, 200.0], [20.0, 900.0]]),
                     {"lower": grid * 12, "upper": grid * 30, "mu": np.full(50, np.nan), "sigma": np.full(50, np.nan)},
                     grid, None, 30.0)
    fig, ax = F._fig()
    F.draw_ltd(ax, "octant_cbg_hull", data, p, 250.0, D_km=D, R_ms=50.0, lim=(1.0, 1.0))
    from matplotlib.colors import to_rgba

    reds = [c for c in ax.collections if c.get_offsets().shape[0] == 2
            and np.allclose(c.get_facecolor()[0], to_rgba(F.STAR))]
    ys = sorted(reds[0].get_offsets()[:, 1])
    assert ys == pytest.approx([0.12, 0.30]) and reds[0].get_offsets()[0, 0] == pytest.approx(0.2)
    assert F._save(fig, tmp_path / "ltd.png").exists()
