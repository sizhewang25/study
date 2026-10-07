"""`report-paper`: the formatting rules, the report, and §4 on the real artifacts."""

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


# ---- §4 on the real artifacts ---------------------------------------------------------


def test_section_4_on_the_real_artifacts():
    from scripts.analysis.v5.modules.labels import load_group
    from scripts.analysis.v5.paper import s4_error_distance as S4

    try:
        claims, tables, _ = S4.build(core.Context(group=load_group("pro-paper")))
    except Exception as exc:  # artifacts not built in this checkout
        pytest.skip(f"pro-paper artifacts unavailable: {exc}")
    by_id = {c.id: c for c in claims}
    assert "by 1.6–6.8×" in by_id["E.6"].text
    assert "6.7–6.8×" in by_id["E.8"].text
    assert "within a normalized distance of 3.13" in by_id["E.19"].text
    assert "ρ = +0.81" in by_id["E.27"].text
    assert "of 4.26" in by_id["E.30"].text
    assert "**7.56**" in tables[0].markdown
    assert "| VAN | 2.94 | 17.6 | 65.0 | 482 | -- | -- | 78% |" in tables[0].markdown
    md = core.render(S4.SECTION, S4.TITLE, claims, tables, group_id="pro-paper", sources={})
    assert md.count("| E.") == len(claims)
