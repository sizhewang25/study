"""MTL replay: geometry primitives, spec overrides, one-TG metrics, and the real-data replay gate."""

from __future__ import annotations

import json
import math

import pytest

from scripts.analysis.v5.modules import mtl_replay as R


def _p(vp, lat, lon, upper, lower=0.0, rtt=5.0):
    return {"vp_id": vp, "vp_lat": lat, "vp_lon": lon, "rtt_ms": rtt, "upper_km": upper, "lower_km": lower}


DISK = json.dumps({"n_pts": 64, "enable_circle_filter": True})


def test_override_spec_sets_and_keeps_keys():
    name, kw = R.override_spec(("planar_annulus_weighted", json.dumps({"n_pts": 64, "enable_circle_filter": True})),
                               (("enable_circle_filter", False),))
    assert name == "planar_annulus_weighted"
    assert json.loads(kw) == {"n_pts": 64, "enable_circle_filter": False}


def test_spo_1sigma_uses_octant_spec_with_filter_off():
    v = R.VARIANTS[R.SPO_1SIGMA]
    assert (v.participants_combo, v.spec_combo, v.synthetic) == ("spotter_cbg", "octant_cbg_hull", True)
    assert dict(v.overrides) == {"enable_circle_filter": False}
    assert dict(R.VARIANTS[R.SPO_1SIGMA_FILTERED].overrides) == {"enable_circle_filter": True}


def test_single_disk_area_matches_the_cap_within_two_percent():
    m = R.tg_metrics("planar_circle", DISK, [_p("a", 0.0, 0.0, 100.0)],
                     tg=(0.0, 0.0), pred=(0.0, 0.0), d_near_km=1.0, D_km=1000.0)
    cap = 2 * math.pi * R._EARTH_R_KM**2 * (1 - math.cos(100.0 / R._EARTH_R_KM))
    assert m["replay_success"] and m["replay_kind"] == "polygon"
    assert m["area_over_D2"] * 1000.0**2 == pytest.approx(cap, rel=0.02)
    assert m["contains_tg"] and m["tg_to_region_over_D"] == 0.0 and m["pred_to_region_over_D"] == 0.0


def test_area_of_a_region_past_the_pole_is_finite():
    # A planar lon/lat polygon reaching lat 95: pyproj alone returns NaN there.
    from shapely.geometry import box

    a = R.geodesic_area_km2(box(-10.0, 80.0, 10.0, 95.0))
    assert math.isfinite(a) and a == pytest.approx(R.geodesic_area_km2(box(-10.0, 80.0, 10.0, 90.0)))


def test_outside_tg_distance_and_exclusion():
    # Two overlapping disks around lon 0..0.5; the TG sits ~1 degree east, outside both.
    parts = [_p("a", 0.0, 0.0, 60.0), _p("b", 0.0, 0.5, 60.0)]
    m = R.tg_metrics("planar_circle", DISK, parts, tg=(0.0, 1.5), pred=None, d_near_km=111.0 * 1.0, D_km=1000.0)
    assert m["replay_success"] and not m["contains_tg"]
    # The intersection's east edge is disk a's (60/111 deg east of lon 0), so the TG is
    # (1.5 - 60/111) deg beyond it along the equator.
    assert m["tg_to_region_over_D"] * 1000.0 == pytest.approx((1.5 - 60 / 111) * 111.2, rel=0.03)
    assert m["excl_true_share"] == 1.0 and m["excl_impl_share"] == 1.0 and m["any_excluder"]
    assert math.isnan(m["pred_to_region_over_D"])


def test_disjoint_disks_replay_empty():
    parts = [_p("a", 0.0, 0.0, 30.0), _p("b", 0.0, 2.0, 30.0)]
    m = R.tg_metrics("planar_circle", DISK, parts, tg=(0.0, 1.0), pred=None, d_near_km=111.0, D_km=1000.0)
    assert not m["replay_success"] and m["replay_kind"] == "none"
    assert math.isnan(m["area_over_D2"]) and not m["contains_tg"]


def test_ring_polygon_annulus_excludes_its_centre():
    g = R.ring_polygon("planar_annulus_weighted", _p("a", 10.0, 10.0, 100.0, 50.0), 64)
    from shapely.geometry import Point

    assert not g.covers(Point(10.0, 10.0)) and g.covers(Point(10.0 + 75 / (111 * math.cos(math.radians(10))), 10.0))
    assert R.ring_polygon("spherical_circle", _p("a", 0, 0, 10), 64) is None


def test_site_stratified_sample_keeps_every_site():
    import pandas as pd

    s = pd.DataFrame({"tg_id": [f"t{i:02d}" for i in range(40)], "site_id": [i // 10 for i in range(40)],
                      "fold": 0})
    out = R.sample_tgs(s, limit=None, frac=0.2)
    assert sorted(out.site_id.unique()) == [0, 1, 2, 3] and len(out) == 8
    assert len(R.sample_tgs(s, limit=3, frac=1.0)) == 3


class TestRealReplayGate:
    """Replay of the recorded combos must reproduce what each run recorded (a sample)."""

    @pytest.fixture(scope="class")
    def rows(self, tmp_path_factory):
        import shutil

        import pandas as pd

        from scripts.analysis.v5.modules.paths import DEFAULT_ANALYSIS_ROOT, resolve_run

        try:
            run = resolve_run("pro-as01-mesh")
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"benchmark run not available: {exc}")
        src = DEFAULT_ANALYSIS_ROOT / "pro-as01-mesh" / "classify"
        if not src.exists():
            pytest.skip("run `classify --run-id pro-as01-mesh` first")
        root = tmp_path_factory.mktemp("v5")
        shutil.copytree(src, root / "pro-as01-mesh" / "classify")
        path = R.replay_run(run, ("vanilla_cbg", "million_scale_cbg"), D_km=4387.257, limit=12, analysis_root=root)
        return pd.read_parquet(path)

    def test_replay_matches_recorded(self, rows):
        assert rows.replay_matches.all()

    def test_soi_region_holds_its_point(self, rows):
        soi = rows[rows.variant == "million_scale_cbg"]
        assert (soi.pred_to_region_over_D <= R.ON_REGION_TOL).all()
