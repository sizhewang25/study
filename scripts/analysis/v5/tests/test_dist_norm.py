"""`dist_norm`: the one min-max rule every normalized distance axis shares."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.analysis.v5.modules import dist_norm as DN


def test_a_distance_shifts_then_scales():
    got = DN.distance([100.0, 300.0, 500.0], (100.0, 500.0))
    assert got.tolist() == pytest.approx([0.0, 500.0, 1_000.0])


def test_a_difference_only_scales():
    """The shift cancels in d_sp - d_geo."""
    a, b = np.array([300.0]), np.array([200.0])
    bounds = (100.0, 500.0)
    assert DN.difference(a - b, bounds) == pytest.approx(
        DN.distance(a, bounds) - DN.distance(b, bounds)
    )


def test_a_value_outside_the_bounds_is_refused():
    with pytest.raises(ValueError, match="outside the declared"):
        DN.distance([10.0, 2_000.0], (0.0, 1_000.0))


def test_nan_is_passed_through_not_refused():
    got = DN.distance([np.nan, 500.0], (0.0, 1_000.0))
    assert np.isnan(got[0]) and got[1] == pytest.approx(500.0)


@pytest.mark.parametrize("bounds", [(5.0, 5.0), (-1.0, 5.0), (10.0, 5.0)])
def test_bad_bounds_are_refused(bounds):
    with pytest.raises(ValueError, match="0 <= min < max"):
        DN.checked(bounds)


def test_common_bounds_agree_or_refuse():
    b = (0.0, 4387.257)
    assert DN.common_bounds(["a", "b"], {"a": b, "b": b}) == b
    assert DN.common_bounds(["a", "b"], {}) is None
    with pytest.raises(ValueError, match="same analysis.common.dist_norm_km"):
        DN.common_bounds(["a", "b"], {"a": b, "b": None})


def test_the_manifest_entry_names_the_bounds():
    assert DN.manifest_entry(None) is None
    entry = DN.manifest_entry((0, 10))
    assert (entry["min"], entry["max"], entry["scale"]) == (0.0, 10.0, DN.SCALE)


@pytest.mark.parametrize(
    "value, linthresh, want",
    [(0.01, 0.0, "$10^{-2}$"), (1.0, 0.0, "$10^{0}$"), (1000.0, 0.0, "$10^{3}$"),
     (0.0, 0.0, "0"), (20.0, 0.0, "20"), (10.0, 22.8, "10"), (100.0, 22.8, "$10^{2}$")],
)
def test_power_labels_decades_only_in_the_log_region(value, linthresh, want):
    assert DN.power_label(value, linthresh) == want
