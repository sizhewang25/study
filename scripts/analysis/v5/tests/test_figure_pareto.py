"""Accuracy-vs-runtime Pareto panels: scoring, pooling and the frontier."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import figure_pareto as F
from scripts.analysis.v5.modules.status import SHORTEST_PING


def test_fallback_prediction_is_not_an_answer():
    df = pd.DataFrame({
        "status": ["SUCCESS", "FALLBACK", "SUCCESS", "SUCCESS"],
        "cell_label": ["correct", "correct", "wrong", "correct"],
        C.GRID_OFFSET: [0, 0, 0, F.BOUND_PIXELS + 1],
    })
    assert list(F.cell_correct(df)) == [True, False, False, True]
    assert list(F.bounded_correct(df)) == [True, False, False, False]


def _long(spec):
    """`{(method, run): (n_tgs, n_correct, n_bounded, runtime_ms)}` -> a `load_regime` frame."""
    rows = []
    for (method, run), (n, k_cell, k_bounded, ms) in spec.items():
        for i in range(n):
            rows.append({
                "run_id": run, "method": method, "tg_id": f"{run}-{i}",
                "site_key": f"{run}|{i // 2}", "correct": i < k_cell, "bounded": i < k_bounded,
                "runtime_ms": np.nan if method == SHORTEST_PING else ms,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def long(monkeypatch):
    monkeypatch.setattr(F.cross, "short_dataset", lambda r: r.upper())
    return _long({
        (SHORTEST_PING, "a"): (10, 5, 5, 0), (SHORTEST_PING, "b"): (30, 15, 15, 0),
        ("fast", "a"): (10, 6, 5, 10.0), ("fast", "b"): (30, 15, 15, 10.0),
        ("slow", "a"): (10, 9, 8, 1000.0), ("slow", "b"): (30, 27, 24, 1000.0),
        ("worse", "a"): (10, 4, 1, 100.0), ("worse", "b"): (30, 10, 3, 100.0),
    })


def test_dot_is_tg_pooled_and_bar_is_the_datasets_range(long):
    table, ds, _ = F.summarize(long, F.SEEN)
    fast = table[(table.method == "fast") & (table.metric == "cell")].iloc[0]
    # a: 6/10, b: 15/30 -> pooled 21/40, not the 0.55 mean of the datasets.
    assert fast.acc == pytest.approx(21 / 40)
    assert fast.acc_dataset_mean == pytest.approx(0.55)
    assert (fast.acc_min, fast.acc_max) == (pytest.approx(0.5), pytest.approx(0.6))
    assert set(ds["dataset"]) == {"A", "B"}


def test_frontier_starts_above_shortest_ping(long):
    table, _, _ = F.summarize(long, F.SEEN)
    cell = table[table.metric == "cell"]
    # fast beats S-P (21/40 > 20/40); worse is below S-P; slow is the most accurate.
    assert F.frontier(cell) == ["fast", "slow"]
    bounded = table[table.metric == "bounded"]
    # fast only ties S-P on bounded, so it is not on that frontier.
    assert F.frontier(bounded) == ["slow"]


def test_pairs_count_dataset_wins(long):
    _, _, pairs = F.summarize(long, F.SEEN)
    row = pairs[(pairs.metric == "cell") & (pairs.a == SHORTEST_PING) & (pairs.b == "fast")].iloc[0]
    assert (row.n_a_wins, row.n_b_wins, bool(row.consistent)) == (0, 1, False)
    row = pairs[(pairs.metric == "cell") & (pairs.a == SHORTEST_PING) & (pairs.b == "slow")].iloc[0]
    assert (row.n_b_wins, bool(row.consistent)) == (2, True)


def test_panels_share_limits_and_are_written(long, tmp_path):
    table, _, _ = F.summarize(long, F.SEEN)
    pngs = F.plot(table, tmp_path, regimes=[F.SEEN])
    assert sorted(pngs) == ["pareto.bounded.seen.png", "pareto.cell.seen.png"]
    assert all(p.stat().st_size > 0 for p in pngs.values())
