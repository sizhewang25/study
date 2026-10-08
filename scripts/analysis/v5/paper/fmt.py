"""How the paper writes a number. Every function takes the unrounded value.

Rounding is half-up (`decimal.ROUND_HALF_UP`), not Python's half-to-even, so
0.25 pp prints as 0.3 the way a reader rounds it. Each value is rounded exactly
once, here; a value rounded upstream and then again here is the double-rounding
slip (0.8147 -> 0.815 -> 0.82) this module exists to prevent.

| Kind | Rule | Example |
|---|---|---|
| `e3` | normalized distance x10^-3, 3 significant figures | 0.353, 7.56, 26.4, 137 |
| `pct` | whole percent; above 0 and below 1 shown as `<1%` | 74%, <1% |
| `pp` | 1 decimal below 10 pp, whole at 10 pp and above | 0.2 pp, 1.6 pp, 16 pp |
| `times` | 1 decimal | 6.8x |
| `runtime` | whole ms below 1 s, 1 decimal s at or above | 179 ms, 5.2 s |
| `rng` | lo-hi in one format, the unit once | 1.6-6.8x |
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

#: What `rng` and the paper put between the two ends of a range.
DASH = "–"
MINUS = "-"  # the paper's LaTeX source writes a hyphen-minus; the .tex check compares against it
TIMES = "×"
UNDEFINED = "--"


def _q(value: float, decimals: int) -> str:
    """`value` rounded half-up to `decimals` places, as a string with exactly that many."""
    exp = Decimal(1).scaleb(-decimals) if decimals > 0 else Decimal(1)
    return str(Decimal(repr(float(value))).quantize(exp, rounding=ROUND_HALF_UP))


def defined(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def dp(value: float, decimals: int, *, signed: bool = False) -> str:
    """Fixed decimals; `signed` prefixes `+` on a positive value."""
    if not defined(value):
        return UNDEFINED
    out = _q(value, decimals)
    if out.startswith("-") and float(out) == 0:
        out = out[1:]
    return f"+{out}" if signed and float(out) > 0 else out


def e3(value: float) -> str:
    """Three significant figures, trailing zeros kept (65.0, not 65)."""
    if not defined(value):
        return UNDEFINED
    if value == 0:
        return "0"
    decimals = max(0, 2 - math.floor(math.log10(abs(value))))
    out = _q(value, decimals)
    # Rounding can carry into a new digit (99.95 -> 100.0); re-round at the new magnitude.
    if len(out.replace("-", "").replace(".", "").lstrip("0")) > 3 and decimals > 0:
        out = _q(value, decimals - 1)
    return out


def pct(value: float, *, sign: bool = True) -> str:
    """`value` in percent. Whole; above 0 and below 1 is `<1`. `sign` appends `%`."""
    if not defined(value):
        return UNDEFINED
    out = "<1" if 0 < value < 1 else _q(value, 0)
    return out + ("%" if sign else "")


def frac(value: float, *, sign: bool = True) -> str:
    """A share given as a fraction (0.735), printed as `pct` (74%)."""
    return pct(100 * value, sign=sign) if defined(value) else UNDEFINED


def frac1(value: float, *, sign: bool = True) -> str:
    """A fraction printed to one decimal percent (0.0016 -> 0.2%)."""
    return pct1(100 * value, sign=sign) if defined(value) else UNDEFINED


def pct1(value: float, *, sign: bool = True) -> str:
    """One decimal percent, for a share the paper states below 1% (0.1%)."""
    if not defined(value):
        return UNDEFINED
    return _q(value, 1) + ("%" if sign else "")


def pp(value: float, *, signed: bool = False, unit: bool = True) -> str:
    """Percentage points: 1 decimal below 10, whole at 10 and above."""
    if not defined(value):
        return UNDEFINED
    out = _q(value, 1 if abs(value) < 10 else 0)
    if out.startswith("-") and float(out) == 0:
        out = out[1:]
    if signed and float(out) > 0:
        out = "+" + out
    return out + (" pp" if unit else "")


def times(value: float, *, unit: bool = True) -> str:
    if not defined(value):
        return UNDEFINED
    return _q(value, 1) + (TIMES if unit else "")


def runtime(ms: float) -> str:
    """179 ms below one second, 5.2 s at or above."""
    if not defined(ms):
        return UNDEFINED
    return f"{_q(ms, 0)} ms" if ms < 1000 else f"{_q(ms / 1000, 1)} s"


def rng(lo: float, hi: float, num, unit: str = "") -> str:
    """`lo–hi` in one number format, the unit once: `rng(1.63, 6.78, times_num, "×")`."""
    a, b = num(lo), num(hi)
    return f"{a}{unit}" if a == b else f"{a}{DASH}{b}{unit}"


def num1(value: float) -> str:
    """A bare 1-decimal number, for a range's ends."""
    return dp(value, 1)


def rho(value: float) -> str:
    """A correlation: 2 decimals, signed (+0.81, -1.00)."""
    return dp(value, 2, signed=True)
