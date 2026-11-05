import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import figure_x_side_regions as F


def _err():
    rows = []
    for term in ("S-P", "VAN"):
        for side, base in (("has-X", 1e-3), ("no-X", 1e-1)):
            for k in range(5):
                answered = not (term == "VAN" and side == "has-X" and k < 2)
                rows.append(dict(run_id="r", tg_id=f"{term}{side}{k}", term=term, has_x=side, answered=answered,
                                 error_over_D=base * (k + 1) if answered else np.nan))
    return pd.DataFrame(rows)


def _reg():
    rows = []
    for side, area in (("has-X", 1e-7), ("no-X", 1e-3)):
        for k in range(4):
            rows.append(dict(term="OCT-H", has_x=side, replay_success=k < 3, area_over_D2=area if k < 3 else np.nan,
                             contains_tg=(k == 0)))
    return pd.DataFrame(rows)


def test_summary_uses_answered_tgs_only():
    s = F.summarize(_err(), _reg())
    assert s["answered_pct"]["VAN"]["has-X"] == 60.0
    # VAN has-X answered errors are 3e-3, 4e-3, 5e-3: the unanswered two do not enter.
    assert s["error_over_D"]["VAN"]["has-X"]["p50"] == pytest.approx(4e-3)
    assert s["error_over_D"]["S-P"]["no-X"]["p50"] == pytest.approx(3e-1)


def test_region_share_is_over_non_empty_regions():
    s = F.summarize(_err(), _reg())
    assert s["region_nonempty_pct"]["OCT-H"]["has-X"] == 75.0
    assert s["tg_inside_pct"]["OCT-H"]["has-X"] == pytest.approx(100 / 3, abs=0.1)
    assert s["area_over_D2"]["OCT-H"]["no-X"]["p50"] == pytest.approx(1e-3)


def test_region_panels_leave_out_sp_and_spo():
    terms = {F.method_label(c) for c in F.REGION_COMBOS}
    assert terms == {"SOI", "VAN", "OCT-H", "OCT-S"}
    assert {"S-P", "SPO"} <= {F.method_label(c) for c in F.ERROR_COMBOS}


def test_render_writes_the_three_panels(tmp_path):
    s = F.summarize(_err(), _reg())
    a = F.render_boxes(s["error_over_D"], ["S-P", "VAN"], scale=F.ERROR_SCALE, floor=F.ERROR_FLOOR,
                       ylabel="e", size=F.SIZE_IN["error"], out_png=tmp_path / "a.png")
    b = F.render_boxes(s["area_over_D2"], ["OCT-H"], scale=1.0, floor=F.AREA_FLOOR, ylabel="A",
                       size=F.SIZE_IN["area"], out_png=tmp_path / "b.png", legend=False)
    c = F.render_inside(s["tg_inside_pct"], ["OCT-H"], size=F.SIZE_IN["inside"], out_png=tmp_path / "c.png")
    assert all(p.exists() and p.stat().st_size > 0 for p in (a, b, c))
