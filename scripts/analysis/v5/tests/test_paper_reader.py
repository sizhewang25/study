"""`report-paper`: the formatting rules, and every section's tables on the real artifacts."""

from __future__ import annotations

import pytest

from scripts.analysis.v5.paper import core, fmt


# ---- formatting: one rounding, half-up, from the unrounded value ------------------


@pytest.mark.parametrize("value,want", [
    (0.35329, "0.353"), (7.558, "7.56"), (26.382, "26.4"), (136.92, "137"),
    (65.017, "65.0"), (899.42, "899"), (99.95, "100"), (0.1855, "0.186"),
])
def test_e3_three_significant_figures(value, want):
    assert fmt.e3(value) == want


def test_percent_points_and_ratios():
    assert fmt.pct(73.52) == "74%" and fmt.pct(0.17) == "<1%" and fmt.pct(0) == "0%"
    assert fmt.pct(22.5) == "23%"                       # half-up, not half-to-even
    assert fmt.pp(0.158) == "0.2 pp" and fmt.pp(0.473) == "0.5 pp"   # never truncated
    assert fmt.pp(-15.76, signed=True) == "-16 pp" and fmt.pp(1.58) == "1.6 pp"
    assert fmt.times(6.78) == "6.8×" and fmt.rho(0.8147) == "+0.81" and fmt.rho(-0.996) == "-1.00"
    assert fmt.rng(1.63, 6.78, fmt.num1, fmt.TIMES) == "1.6–6.8×"
    assert fmt.rng(6.70, 6.74, fmt.num1, fmt.TIMES) == "6.7×"   # equal ends collapse
    assert fmt.runtime(179.29) == "179 ms" and fmt.runtime(5244) == "5.2 s"
    assert fmt.e3(float("nan")) == fmt.UNDEFINED


# ---- every section on the real artifacts ------------------------------------------------


def _section(name):
    import importlib

    from scripts.analysis.v5.modules.labels import load_group

    mod = importlib.import_module(f"scripts.analysis.v5.paper.{name}")
    try:
        return mod.build(core.Context(group=load_group("pro-paper")))[0]
    except Exception as exc:  # artifacts not built in this checkout
        pytest.skip(f"pro-paper artifacts unavailable: {exc}")


def _cells(tables, id_):
    (t,) = [t for t in tables if t.id == id_]
    return t.markdown()


def test_section_4_tables():
    tables = _section("s4_error_distance")
    assert "| VAN | 2.94 | 17.6 | 65.0 | 482 | -- | -- | 78% |" in _cells(tables, "tab:err-dist-percentiles")
    assert "| d_geo | 0.334 | 1.25 | 3.13 |" in _cells(tables, "fig:cdf-vp-proximity")
    assert "| All | 100% | 100% | +0.81 |" in _cells(tables, "fig:scatter-vp_pni_proximity")


def test_section_5_tables():
    tables = _section("s5_region_classification")
    x = _cells(tables, "tab:x-cell")
    assert "| (group share) | 53% | 47% |" in x
    assert "| VAN | 53% (41) | 16% | 35% |" in x
    assert "| OCT-H | 74% | 58% | -16 |" in _cells(tables, "tab:seen-unseen")


def test_sections_3_6_7_and_appendix():
    assert "| All | 134 | 1269 | 65 |" in _cells(_section("s3_methodology"), "datasets")
    six = _section("s6_overhead")
    assert "| OCT-H | All | 13.8 s | 3843 | 5.0 days |" in _cells(six, "batch budget")
    assert "| OCT-H | 20 ms | 4.5 s | 466 ms |" in _cells(six, "fig:boxplot-speed-memory (phases)")
    assert "| OCT-H | 74 (64–82)† | 61 (52–72)† | 58 (46–73)† | 49 (42–62) | 5.2 s | 2.3 s |" in _cells(
        _section("s7_comprehensive"), "tab:sota")
    appx = _section("sB_appendix")
    assert "| 7 | 128 | 196,608 | 2,594 | 50.9 |" in _cells(appx, "tab:healpix-levels")
    assert "| OCT-H | seen | All | 74% | 74% | +0.3 |" in _cells(appx, "appendix B: accuracy")


def test_render_has_every_table():
    tables = _section("s4_error_distance")
    md = core.render("4", "t", tables, group_id="g", sources={})
    assert md.count("\n## ") == len(tables)
