"""Slugs, derived slugs and claims: every data-derived number in the paper template.

A **slug** is one number the paper prints, keyed `<section>.<subject>.<quantity>`
(lowercase, `[a-z0-9.-]`). Its getter reads v5 artifacts and returns the
unrounded value; its `fmt` says how the paper writes it (`FORMATS`). The
template writes `\\pv{key}` (or `\\pv[fmt]{key}` to print the same value in
another format, e.g. `bare` for a range's first end); `render` replaces it.

A **derived** slug combines other slugs' raw values. Derived slugs are
evaluated at render time, so a slug of one group (pro-tf, on the TF machine)
can be compared with one of another (pro-paper) without either machine
holding both.

A **claim** is a worded statement the template keeps as text (an order of
magnitude, a time bound, a method ranking). Its check returns
(holds, detail); `generate` records it and `inventory` lists the ones that
fail, so a wording the data no longer supports is caught without the text
being generated.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

from scripts.analysis.v5.paper import fmt as F

KEY_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*$")

MESH, TF = "pro-paper", "pro-tf"


# ---- formats ------------------------------------------------------------------------
# Each takes the unrounded value (percent for pct/pp kinds) and returns LaTeX.


def _tex(s: str) -> str:
    return s.replace("%", r"\%").replace(F.TIMES, r"$\times$").replace(F.DASH, "--")


def _sig(value: float, n: int) -> str:
    """`n` significant figures, half-up, thousands separated (1234 -> 1,200 at n=2)."""
    if value == 0:
        return "0"
    decimals = n - 1 - math.floor(math.log10(abs(value)))
    if decimals > 0:
        out = F._q(value, decimals)
    else:  # round half-up to 10^-decimals
        step = 10 ** (-decimals)
        out = str(int(math.floor(abs(value) / step + 0.5) * step) * (1 if value > 0 else -1))
    if decimals > 0 and len(out.replace("-", "").replace(".", "").lstrip("0")) > n:  # 9.96 -> 10.0
        out = F._q(value, decimals - 1)
    if "." not in out and abs(float(out)) >= 1000:
        out = f"{int(float(out)):,}"
    return out


def _s(sec: float) -> str:
    """Seconds: 1 decimal below 10 s, whole above."""
    return F._q(sec, 1) if sec < 9.95 else F._q(sec, 0)


def _ms(ms: float) -> str:
    """Milliseconds as fmt.runtime writes them below a second."""
    return F._q(ms, 1) if ms < 9.95 else F._q(ms, 0)


def _times(value: float, decimals: int) -> str:
    return F._q(value, decimals) + r"$\times$"


FORMATS: dict[str, Callable[[float], str]] = {
    "pct": lambda v: _tex(F.pct(v)),  # 12\%
    "bare": lambda v: F.pct(v, sign=False),  # 12 (a percent without the sign)
    "pp": lambda v: F._q(abs(v), 0),  # 12 (whole pp, unsigned)
    "pp1": lambda v: F.pp(abs(v), unit=False),  # 1.2, 12 (1 decimal below 10 pp, as fmt.pp)
    "pp1s": lambda v: f"${F.pp(v, signed=True, unit=False)}$",  # $+1.2$
    "negpp": lambda v: str(int(F._q(-v, 0))),  # a drop printed as a change, whole pp: 15.7 -> -16, 0 -> 0
    "e3w": lambda v: _sig(v, 2) if v < 9.95 else F._q(v, 0),  # 2 significant figures below 10, whole above: 0.93, 6.2, 137
    "e3": lambda v: F.e3(v),  # 3 significant figures
    "e3s2": lambda v: _sig(v, 2),  # 2 significant figures
    "x1": lambda v: _times(v, 1),  # 1.2$\times$
    "x0": lambda v: _times(v, 0),  # 1$\times$
    "x1bare": lambda v: F._q(v, 1),  # 1.2 (the template prints the sign)
    "xsig1": lambda v: _sig(v, 1) + r"$\times$",  # 1 significant figure, $\times$
    "frac2": lambda v: F.dp(v / 100.0, 2),  # a percent printed as a fraction: 12 -> 0.12
    "f2": lambda v: F.dp(v, 2),  # 2 decimals
    "f1": lambda v: F.dp(v, 1),  # 1 decimal
    "int": lambda v: str(int(round(v))),  # whole number (degrees, VP counts, p-numbers)
    "sig2": lambda v: _sig(v, 2),  # 2 significant figures, thousands separated
    "ms": _ms,  # milliseconds (unit in the text)
    "s": lambda v: _s(v / 1000.0),  # value in ms, printed in seconds (unit in the text)
}


# ---- the registry -------------------------------------------------------------------


@dataclass
class Slug:
    key: str
    group: str
    fmt: str
    getter: Callable
    source: str
    section: str
    note: str = ""


@dataclass
class Derived:
    key: str
    needs: list[str]
    fmt: str
    f: Callable[..., float]
    source: str
    section: str
    note: str = ""


@dataclass
class Claim:
    key: str
    group: str
    text: str
    check: Callable
    section: str
    needs: list[str] = field(default_factory=list)   # derived claims read slugs at render


SLUGS: dict[str, Slug] = {}
DERIVED: dict[str, Derived] = {}
CLAIMS: dict[str, Claim] = {}


def _register(table: dict, key: str, item) -> None:
    if not KEY_RE.match(key):
        raise ValueError(f"bad slug key {key!r}: lowercase [a-z0-9.-], starting alphanumeric")
    if key in SLUGS or key in DERIVED or key in CLAIMS:
        raise ValueError(f"slug key {key!r} registered twice")
    table[key] = item


def slug(key: str, fmt: str, source: str, *, group: str = MESH, section: str = "", note: str = ""):
    """Decorator: `getter(ctx) -> unrounded value` for the slug `key`."""
    if fmt not in FORMATS:
        raise ValueError(f"{key}: unknown fmt {fmt!r}")

    def wrap(getter):
        _register(SLUGS, key, Slug(key, group, fmt, getter, source, section, note))
        return getter
    return wrap


def put(key: str, fmt: str, source: str, getter: Callable, *, group: str = MESH, section: str = "",
        note: str = "") -> None:
    """`slug` without the decorator, for slugs registered in a loop."""
    slug(key, fmt, source, group=group, section=section, note=note)(getter)


def derive(key: str, needs: list[str], fmt: str, f: Callable[..., float], *, source: str = "",
           section: str = "", note: str = "") -> None:
    if fmt not in FORMATS:
        raise ValueError(f"{key}: unknown fmt {fmt!r}")
    _register(DERIVED, key, Derived(key, needs, fmt, f, source or f"derived from {', '.join(needs)}",
                                    section, note))


def claim(key: str, text: str, *, group: str = MESH, section: str = "", needs: list[str] | None = None):
    """Decorator: `check(ctx) -> (holds, detail)`; with `needs`, `check(*values)` at render time."""
    def wrap(check):
        _register(CLAIMS, key, Claim(key, group, text, check, section, list(needs or [])))
        return check
    return wrap


def render_value(value, fmt: str) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("undefined value")
    return FORMATS[fmt](float(value))


CLAIMS_FILE = "paper/claims.py"   # under the repo root; kept out of git with the template


def load_all() -> None:
    """Import every section module (slugs), then the claims file if present."""
    import importlib.util
    import sys

    from scripts.analysis.v5.modules.paths import REPO_ROOT
    from scripts.analysis.v5.paper_values import sections  # noqa: F401

    name = "paper_values_claims"
    path = REPO_ROOT / CLAIMS_FILE
    if name in sys.modules or not path.is_file():
        return
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
