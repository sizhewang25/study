"""The paper-figure registry matches the paper; publish names, copies and embeds."""

from __future__ import annotations

import json
import re

import pytest

from scripts.analysis.v5.modules import paper_figures as P
from scripts.analysis.v5.modules.paths import REPO_ROOT

PAPER = REPO_ROOT.parent / "cbg-benchmark-paper"
#: Included figures that are drawn, not built from data.
NOT_DATA = {"benchmark-framework"}


def _uncommented(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def _paper_figures() -> set[str]:
    main = _uncommented((PAPER / "main.tex").read_text())
    names = set()
    for sec in re.findall(r"\\input\{(sections/[^}]+)\}", main):
        body = _uncommented((PAPER / sec).read_text())
        for path in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{figs/([^}]+)\}", body):
            names.add(re.sub(r"\.(png|pdf)$", "", path))
    return names - NOT_DATA


@pytest.mark.skipif(not (PAPER / "main.tex").is_file(), reason="paper repo not checked out beside this one")
def test_registry_is_the_papers_data_figures():
    registered = {e.name for es in P.PAPER_FIGURES.values() for e in es}
    assert registered == _paper_figures()


def test_names_are_unique_across_groups():
    names = [e.name for es in P.PAPER_FIGURES.values() for e in es]
    assert len(names) == len(set(names))


def test_publish(tmp_path, monkeypatch):
    group = {"group_id": "g", "runs": {"seen": ["r1", "r2"], "unseen": ["r3"]},
             "analysis": {"plot-mech": {"run-id": "r1"}}}
    reg = [
        P.Entry("fig.a", "classify", P.SEEN, "fig.a.healpix-128", "fig.a.manifest.json", "plot-a"),
        P.Entry("fig.m", "mech", "run:plot-mech", "m", "m.json", "plot-mech"),
        P.Entry("fig.gone", "classify", P.ALL, "gone", "gone.json", "plot-gone"),
    ]
    monkeypatch.setitem(P.PAPER_FIGURES, "g", reg)
    monkeypatch.setattr(P, "load_group", lambda gid: group)
    root = tmp_path / "analysis"
    seen_dir = root / "_cross" / "classify" / "g.seen"
    seen_dir.mkdir(parents=True)
    (seen_dir / "fig.a.healpix-128.pdf").write_bytes(b"%PDF-a")
    (seen_dir / "fig.a.healpix-128.state.json").write_text(json.dumps({"figure": "fig.a.healpix-128"}))
    (seen_dir / "fig.a.manifest.json").write_text(json.dumps({"n": 3}))
    mech_dir = root / "r1" / "mech"
    mech_dir.mkdir(parents=True)
    (mech_dir / "m.pdf").write_bytes(b"%PDF-m")

    out, figs = tmp_path / "paper", tmp_path / "figs"
    done, problems = P.publish("g", analysis_root=root, out_root=out, copy_to=figs)
    assert (out / "pdf" / "fig.a.pdf").read_bytes() == b"%PDF-a"
    assert (figs / "fig.m.pdf").read_bytes() == b"%PDF-m"
    rec = json.loads((out / "json" / "fig.a.json").read_text())
    assert rec["paper_name"] == "fig.a" and rec["data"] == {"n": 3}
    assert rec["state"] == {"figure": "fig.a.healpix-128"}
    rec_m = json.loads((out / "json" / "fig.m.json").read_text())
    assert rec_m["data"] is None and "state_missing" in rec_m and "manifest_missing" in rec_m
    assert any("fig.gone" in p and "not built" in p for p in problems)
    assert any("fig.m" in p and "manifest" in p for p in problems)
    assert len(done) == 2

    done, problems = P.publish("g", analysis_root=root, out_root=out, check=figs)
    assert {"same    fig.a", "same    fig.m"} <= set(done)
    (figs / "fig.a.pdf").write_bytes(b"%PDF-changed")
    _, problems = P.publish("g", analysis_root=root, out_root=out, check=figs)
    assert any("fig.a: differs" in p for p in problems)
