"""paper_values: formats, the spec against the registry, slugify, render."""

from __future__ import annotations

import json

import pytest

from scripts.analysis.v5.paper_values import registry as R
from scripts.analysis.v5.paper_values import render as RD
from scripts.analysis.v5.paper_values import slugify as SG

R.load_all()


@pytest.mark.parametrize("fmt, value, out", [
    ("pct", 12.4, r"12\%"), ("pct", 0.4, r"<1\%"), ("bare", 12.6, "13"),
    ("pp", -12.3, "12"), ("pp1", 1.25, "1.3"), ("pp1s", 0.25, "$+0.3$"), ("pp1s", -0.06, "$-0.1$"),
    ("pp1s", 0.0, "$0.0$"), ("e3", 12.0, "12.0"), ("e3s2", 1.234, "1.2"), ("x1", 1.25, r"1.3$\times$"),
    ("x0", 1.4, r"1$\times$"), ("xsig1", 1234.0, r"1,000$\times$"), ("frac2", 12.34, "0.12"),
    ("sig2", 1234.0, "1,200"), ("sig2", 1.996, "2.0"), ("sig2", 9.96, "10"), ("sig2", 12.3, "12"),
    ("ms", 1.23, "1.2"), ("ms", 123.4, "123"), ("s", 1234.0, "1.2"), ("s", 12300.0, "12"), ("int", 12.4, "12"),
])
def test_formats(fmt, value, out):
    assert R.render_value(value, fmt) == out


def test_render_value_refuses_undefined():
    with pytest.raises(ValueError):
        R.render_value(float("nan"), "pct")


def test_every_spec_key_is_registered_with_a_known_format():
    if not SG.DEFAULT_SPEC.is_file():
        pytest.skip("paper/spec.py is not checked out (kept out of git)")
    for entry in SG.load_spec():
        if len(entry) == 3:
            continue
        _, _, uses = SG.unmark(entry[1])
        assert uses, entry[1][:60]
        for literal, key, fmt in uses:
            assert key in R.SLUGS or key in R.DERIVED, key
            assert fmt is None or fmt in R.FORMATS, (key, fmt)


def test_derived_and_claims_need_registered_slugs():
    for d in R.DERIVED.values():
        assert all(k in R.SLUGS for k in d.needs), d.key
    for c in R.CLAIMS.values():
        assert all(k in R.SLUGS for k in c.needs), c.key


def test_unmark():
    old, new, uses = SG.unmark(r"at [[12\%|a.b]] and p[[34|c.d|bare]].")
    assert old == r"at 12\% and p34."
    assert new == r"at \pv{a.b} and p\pv[bare]{c.d}."
    assert uses == [(r"12\%", "a.b", None), ("34", "c.d", "bare")]


def test_apply_spec_skips_comments_and_reports_absent():
    spec = [("a.tex", r"reach [[12\%|a.b]]."), ("a.tex", "absent passage")]
    texts = {"a.tex": "% old: reach 12\\%.\nNow they reach 12\\%.\n"}
    out, uses, problems = SG.apply_spec(texts, spec)
    assert out["a.tex"] == "% old: reach 12\\%.\nNow they reach \\pv{a.b}.\n"
    assert uses == [{"key": "a.b", "fmt": None, "literal": r"12\%", "file": "a.tex", "line": 2}]
    assert len(problems) == 1 and "found 0 times" in problems[0]


def _values(tmp_path, group, slugs):
    p = tmp_path / f"{group}.json"
    p.write_text(json.dumps({"group": group, "created_utc": "t", "git": {}, "slugs": slugs, "claims": {}}))
    return p


def test_render_end_to_end(tmp_path):
    t = tmp_path / "template"
    (t / "sections").mkdir(parents=True)
    (t / "figs").mkdir()
    (t / "main.tex").write_text(
        "\\documentclass{article}\n\\newcommand{\\rednote}[1]{#1}\n" + SG.MACRO
        + "\\begin{document}\n\\input{sections/a.tex}\n\\end{document}\n")
    (t / "sections" / "a.tex").write_text(
        "A at \\pv{acc.seen.s-p}, table \\pv[frac2]{acc.seen.s-p}, B \\pv{tf.acc.s-p}.\n"
        "\\includegraphics{figs/f1.pdf}\n\\includegraphics{figs/f2.pdf}\n")
    figs = tmp_path / "pdf"
    figs.mkdir()
    (figs / "f1.pdf").write_bytes(b"%PDF-1")
    v = _values(tmp_path, "pro-paper", {"acc.seen.s-p": {"value": 12.34, "latex": r"12\%", "status": "ok"}})

    out = tmp_path / "project"
    prov = RD.render(t, [v], figs, out, allow_missing=True)
    body = (out / "sections" / "a.tex").read_text()
    assert r"A at 12\%, table 0.12" in body
    assert r"\textcolor{red}{[tf.acc.s-p]}" in body
    assert r"\fbox{\texttt{missing figure f2}}" in body
    assert (out / "figs" / "f1.pdf").read_bytes() == b"%PDF-1"
    assert SG.MACRO_BEGIN not in (out / "main.tex").read_text()
    assert prov["slugs"]["missing"] == ["tf.acc.s-p"] and prov["figures"]["missing"] == ["figs/f2.pdf"]
    assert "\\pv" not in body

    with pytest.raises(ValueError):
        RD.render(t, [v], figs, t / "inside")


def test_merge_refuses_a_key_in_two_files(tmp_path):
    rec = {"value": 1.0, "latex": "1", "status": "ok"}
    a, b = _values(tmp_path, "a", {"k": rec}), _values(tmp_path / ".." / tmp_path.name, "b", {"k": rec})
    with pytest.raises(ValueError):
        RD.merge_values([a, b])


def test_template_slugs_are_registered():
    tmpl = SG.DEFAULT_TEMPLATE
    if not (tmpl / "main.tex").is_file():
        pytest.skip("no template built")
    keys = {m.group(2) for p in tmpl.rglob("*.tex") for m in SG.PV.finditer(p.read_text())}
    assert keys and all(k in R.SLUGS or k in R.DERIVED for k in keys)
