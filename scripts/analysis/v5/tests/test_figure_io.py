"""save_figure: PNG as before, a PDF twin with TrueType fonts, and a state JSON."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.analysis.v5.modules import figure_io as IO  # noqa: E402


def _fig():
    fig, ax = plt.subplots(figsize=(1.5, 1.0))
    ax.plot([0, 1], [0, 1])
    ax.set_xlabel("x label")
    return fig


def test_png_matches_plain_savefig(tmp_path):
    kw = dict(dpi=100, bbox_inches="tight", pad_inches=0.02)
    fig = _fig()
    fig.savefig(tmp_path / "plain.png", **kw)
    IO.save_figure(fig, tmp_path / "a.b.png", **kw)
    plt.close(fig)
    assert (tmp_path / "a.b.png").read_bytes() == (tmp_path / "plain.png").read_bytes()


def test_writes_pdf_and_state(tmp_path):
    fig = _fig()
    out = IO.save_figure(fig, tmp_path / "sub" / "x.pooled.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    assert out == tmp_path / "sub" / "x.pooled.png"
    pdf = tmp_path / "sub" / "x.pooled.pdf"
    assert pdf.read_bytes().startswith(b"%PDF")
    assert b"/Type3" not in pdf.read_bytes()
    state = json.loads((tmp_path / "sub" / "x.pooled.state.json").read_text())
    assert {"figure", "png", "pdf", "module", "command", "git", "created_utc", "savefig"} <= set(state)
    assert state["figure"] == "x.pooled" and state["pdf"] == "x.pooled.pdf"
    assert state["module"] == __name__
    assert state["savefig"] == {"dpi": 150, "bbox_inches": "tight"}
