"""Min-max normalized distances: the one rule every distance axis shares.

Absolute distances are confidential in the paper. A run whose config declares
`analysis.common.dist_norm_km: {min, max}` (`labels.declared_dist_norm_km`)
has every distance d drawn as (d - min) / (max - min), in units of 10^-3:
min 0, a prediction on its TG, and max the footprint span D (`footprint.py`).
The bounds are fixed, not taken from the data, so a value means the same
fraction of the footprint in every figure and ratios survive.

Two kinds of quantity, two transforms:

- a **distance** (prediction to TG, TG to its nearest VP) shifts by `min`
  and scales: `distance(...)`;
- a **difference of distances** (the S-P gap d_sp - d_geo) only scales, since
  the shift cancels: `difference(...)`. With min 0 the two agree.

Figures read the bounds through `common_bounds`, which refuses a pooled figure
over runs that declare different bounds or only some of which declare one.
"""

from __future__ import annotations

import numpy as np

#: Normalized values are reported in units of 10^-3 of (max - min).
SCALE = 1_000.0

#: Axis-label suffix for a normalized axis.
UNIT = r"($\times 10^{-3}$)"

#: The normalized log axis: five decades, ending at the declared max.
X_MIN = 0.01
X_MAX = SCALE

Bounds = tuple[float, float]


def power_label(value: float, linthresh: float = 0.0) -> str:
    """A normalized axis tick: `$10^{k}$` for a decade at or above
    `linthresh`, the plain number otherwise (0, or a symlog linear-region
    tick)."""
    if value > 0 and value >= linthresh:
        k = np.log10(value)
        if np.isclose(k, round(k)):
            return rf"$10^{{{int(round(k))}}}$"
    return f"{value:g}"


def checked(bounds: Bounds) -> Bounds:
    """`bounds` as floats, refused unless 0 <= min < max."""
    lo, hi = (float(b) for b in bounds)
    if not 0 <= lo < hi:
        raise ValueError(f"normalization bounds need 0 <= min < max, got {bounds}")
    return lo, hi


def distance(km, bounds: Bounds, *, what: str = "distance") -> np.ndarray:
    """(km - min) / (max - min) x `SCALE`; refuses a value outside the bounds.

    The paper states that no distance exceeds D, and normalized axes end there,
    so a value past it is either a wrong declaration or a claim the text can no
    longer make.
    """
    lo, hi = checked(bounds)
    values = np.asarray(km, dtype=float)
    finite = values[np.isfinite(values)]
    outside = (finite < lo) | (finite > hi)
    if outside.any():
        raise ValueError(
            f"{what}: {int(outside.sum())} values fall outside the declared "
            f"analysis.common.dist_norm_km [{lo:g}, {hi:g}] km "
            f"(range {finite.min():,.1f}-{finite.max():,.1f} km)"
        )
    return (values - lo) / (hi - lo) * SCALE


def difference(km, bounds: Bounds) -> np.ndarray:
    """A difference of two distances, normalized: km / (max - min) x `SCALE`."""
    lo, hi = checked(bounds)
    return np.asarray(km, dtype=float) / (hi - lo) * SCALE


def common_bounds(
    run_ids: list[str], declared: dict[str, Bounds | None] | None
) -> Bounds | None:
    """The one `(min, max)` every run in `run_ids` declares, None if none does;
    mixed raises.

    A pooled figure normalizes every run's distances by one pair of bounds, so
    runs declaring different values, or some declaring none, cannot share it.
    """
    values = {r: (declared or {}).get(r) for r in run_ids}
    distinct = set(values.values())
    if len(distinct) > 1:
        raise ValueError(
            f"pooled runs must declare the same analysis.common.dist_norm_km; got {values}"
        )
    return distinct.pop() if distinct else None


def manifest_entry(bounds: Bounds | None) -> dict | None:
    """What a manifest records about the normalization, or None for km."""
    if bounds is None:
        return None
    lo, hi = checked(bounds)
    return {
        "min": lo,
        "max": hi,
        "scale": SCALE,
        "source": "analysis.common.dist_norm_km in each run's config",
        "note": (
            "distances drawn as (d - min) / (max - min) x scale, differences of "
            "distances as d / (max - min) x scale; values outside [min, max] are "
            "refused. max is the footprint span D, confidential in the paper."
        ),
    }
