"""One way to write a figure: PNG, PDF and a state JSON, side by side.

`save_figure(fig, out_png, **savefig_kw)` replaces a module's `fig.savefig`:

  <stem>.png         exactly what `fig.savefig(out_png, **savefig_kw)` wrote before
  <stem>.pdf         the same figure as vector, same kwargs (`dpi` only sets the
                     resolution of raster parts), fonts embedded as TrueType
                     (Type 42; ACM rejects Type 3)
  <stem>.state.json  provenance: module, command, git commit, time, savefig kwargs

The figure's numbers stay in the module's own manifest; `paper_figures.publish`
joins the two into `outputs/paper/json/<paper name>.json`.
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib

from scripts.analysis.v5.modules.paths import REPO_ROOT

#: `<stem>.state.json` beside `<stem>.png`.
STATE_SUFFIX = ".state.json"
#: Fonts as TrueType in PDF/PS, never Type 3.
PDF_RC = {"pdf.fonttype": 42, "ps.fonttype": 42}


def pdf_path(out_png: Path) -> Path:
    return Path(out_png).with_suffix(".pdf")


def state_path(out_png: Path) -> Path:
    p = Path(out_png)
    return p.with_name(p.stem + STATE_SUFFIX)


@lru_cache(maxsize=1)
def git_state() -> dict:
    """HEAD and whether the tree is dirty; nulls when git is unavailable."""
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args],
                capture_output=True, text=True, timeout=10, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {"commit": sha, "dirty": None if status is None else bool(status)}


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def save_figure(fig, out_png: Path, *, module: str | None = None, **savefig_kw) -> Path:
    """Write `out_png`, its PDF twin and its state JSON. Returns `out_png`.

    The caller still closes the figure. `module` defaults to the caller's
    module name.
    """
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, **savefig_kw)
    with matplotlib.rc_context(PDF_RC):
        fig.savefig(pdf_path(out_png), format="pdf", **savefig_kw)
    if module is None:
        module = sys._getframe(1).f_globals.get("__name__")
    state = {
        "figure": out_png.stem,
        "png": out_png.name,
        "pdf": pdf_path(out_png).name,
        "module": module,
        "command": " ".join(sys.argv),
        "git": git_state(),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "savefig": _jsonable(savefig_kw),
    }
    state_path(out_png).write_text(json.dumps(state, indent=1) + "\n")
    return out_png
