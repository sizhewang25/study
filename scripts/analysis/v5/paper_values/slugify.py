r"""Build the slugged template from the paper project.

    python -m scripts.analysis.v5.paper_values.slugify --paper ../cbg-benchmark-paper

Copies what the paper compiles (main.tex, the sections main.tex inputs, the
bibliography and class files, and the non-data figures) into
`paper/template/`, applies the spec (`paper/spec.py`: every generated number becomes
`\pv{key}`), adds the `\pv` macro to main.tex, and writes
`paper/template/slugs.orig.json`: for every slug use, the file, line and the
literal the paper printed. The data figures are not copied: render takes
them from the published PDF folder.

Re-run it after the paper changes on Overleaf. A spec passage the paper no
longer contains is reported with its file, and nothing is written: fix the
passage in paper/spec.py (or the paper) and re-run.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import typer

from scripts.analysis.v5.modules.paths import REPO_ROOT

DEFAULT_TEMPLATE = REPO_ROOT / "paper" / "template"
DEFAULT_SPEC = REPO_ROOT / "paper" / "spec.py"   # kept out of git with the template
MARK = re.compile(r"\[\[(.+?)\|([a-z0-9][a-z0-9.-]*)(?:\|([a-z0-9]+))?\]\]")
PV = re.compile(r"\\pv(?:\[([a-z0-9]+)\])?\{([a-z0-9][a-z0-9.-]*)\}")
INPUT = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHIC = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
TOP_FILES = ("main.tex", "references.bib")
OPTIONAL_FILES = (".latexmkrc", "Makefile", ".gitignore")
TOP_GLOBS = ("*.cls", "*.bst", "*.bbx", "*.cbx", "*.dbx")

MACRO_BEGIN = "% >>> paper_values"
MACRO_END = "% <<< paper_values"
MACRO = (
    f"{MACRO_BEGIN}: slug macros. render.py replaces every \\pv and drops this block.\n"
    "\\providecommand{\\pv}{}\n"
    "\\renewcommand{\\pv}[2][]{\\textcolor{magenta}{\\texttt{\\detokenize{#2}}}}\n"
    f"{MACRO_END}\n"
)


def uncomment(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in text.splitlines())


def compiled_sections(main_tex: str) -> list[str]:
    return [p if p.endswith(".tex") else p + ".tex" for p in INPUT.findall(uncomment(main_tex))
            if p.startswith("sections/")]


def referenced_figs(root: Path, files: list[str]) -> list[str]:
    """`figs/...` paths the compiled files include or input."""
    out = []
    for f in files:
        body = uncomment((root / f).read_text())
        for p in [*GRAPHIC.findall(body), *INPUT.findall(body)]:
            if p.startswith("figs/"):
                out.append(p if Path(p).suffix else p + ".tex")
    return list(dict.fromkeys(out))


def unmark(marked: str) -> tuple[str, str, list[tuple[str, str, str | None]]]:
    """(paper text, template text, [(literal, key, fmt)])."""
    uses = [(m.group(1), m.group(2), m.group(3)) for m in MARK.finditer(marked)]
    old = MARK.sub(lambda m: m.group(1), marked)
    new = MARK.sub(lambda m: f"\\pv[{m.group(3)}]{{{m.group(2)}}}" if m.group(3) else f"\\pv{{{m.group(2)}}}",
                   marked)
    return old, new, uses


def _occurrences(text: str, sub: str) -> list[int]:
    out, i = [], text.find(sub)
    while i >= 0:
        out.append(i)
        i = text.find(sub, i + 1)
    return out


def _in_comment(text: str, i: int) -> bool:
    """Whether position `i` follows an unescaped `%` on its line."""
    line_start = text.rfind("\n", 0, i) + 1
    return re.search(r"(?<!\\)%", text[line_start:i]) is not None


def load_spec(path: Path = DEFAULT_SPEC) -> list[tuple]:
    """`SPEC` from the spec file (a list of `(file, marked)` / `(file, old, new)`)."""
    ns: dict = {}
    exec(compile(Path(path).read_text(), str(path), "exec"), ns)
    return ns["SPEC"]


def apply_spec(texts: dict[str, str], spec: list[tuple]) -> tuple[dict[str, str], list[dict], list[str]]:
    """Apply `spec` to `texts` (file -> content). Returns (texts, uses, problems)."""
    texts = dict(texts)
    uses, problems = [], []
    for entry in spec:
        f = entry[0]
        if f not in texts:
            problems.append(f"{f}: not a compiled file")
            continue
        if len(entry) == 3:
            old, new, marks = entry[1], entry[2], []
        else:
            old, new, marks = unmark(entry[1])
        hits = [i for i in _occurrences(texts[f], old) if not _in_comment(texts[f], i)]
        if len(hits) != 1:
            problems.append(f"{f}: passage found {len(hits)} times outside comments: {old[:90]!r}...")
            continue
        start = hits[0]
        line = texts[f][:start].count("\n") + 1
        texts[f] = texts[f][:start] + new + texts[f][start + len(old):]
        for literal, key, fmt in marks:
            uses.append({"key": key, "fmt": fmt, "literal": literal, "file": f, "line": line})
    return texts, uses, problems


def build(paper: Path, out: Path, spec_path: Path = DEFAULT_SPEC) -> tuple[list[dict], list[str]]:
    from scripts.analysis.v5.modules.paper_figures import PAPER_FIGURES

    main = (paper / "main.tex").read_text()
    sections = compiled_sections(main)
    texts = {"main.tex": main, **{s: (paper / s).read_text() for s in sections}}
    texts, uses, problems = apply_spec(texts, load_spec(spec_path))
    if MACRO_BEGIN not in texts["main.tex"]:
        anchor = re.search(r"^\\newcommand\{\\rednote\}.*$", texts["main.tex"], re.M)
        if anchor is None:
            problems.append("main.tex: no \\newcommand{\\rednote} line to put the \\pv macro after")
        else:
            i = anchor.end() + 1
            texts["main.tex"] = texts["main.tex"][:i] + MACRO + texts["main.tex"][i:]
    if problems:
        return uses, problems

    data_figs = {e.name for es in PAPER_FIGURES.values() for e in es}
    figs = [p for p in referenced_figs(paper, sections) if Path(p).stem not in data_figs]
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f, body in texts.items():
        (out / f).parent.mkdir(parents=True, exist_ok=True)
        (out / f).write_text(body)
    for f in TOP_FILES[1:]:
        shutil.copy2(paper / f, out / f)
    for f in OPTIONAL_FILES:
        if (paper / f).is_file():
            shutil.copy2(paper / f, out / f)
    for g in TOP_GLOBS:
        for p in paper.glob(g):
            shutil.copy2(p, out / p.name)
    if (paper / "references").is_dir():
        shutil.copytree(paper / "references", out / "references")
    for p in figs:
        (out / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paper / p, out / p)
        gen = (paper / p).with_name(Path(p).stem + "-gen.py")
        if gen.is_file():
            shutil.copy2(gen, out / "figs" / gen.name)
    (out / "slugs.orig.json").write_text(json.dumps(
        {"paper": str(paper), "uses": uses}, indent=1, ensure_ascii=False) + "\n")
    return uses, problems


app = typer.Typer(add_completion=False)


@app.command()
def main(
    paper: Path = typer.Option(REPO_ROOT.parent / "cbg-benchmark-paper", help="The paper project."),
    out: Path = typer.Option(DEFAULT_TEMPLATE, help="The template to (re)write."),
    spec: Path = typer.Option(DEFAULT_SPEC, help="The spec: where each slug sits in the paper."),
) -> None:
    uses, problems = build(paper.resolve(), out, spec)
    if problems:
        for p in problems:
            typer.echo(f"problem: {p}", err=True)
        typer.echo("nothing written", err=True)
        raise typer.Exit(1)
    typer.echo(f"wrote {out}: {len(uses)} slug uses of {len({u['key'] for u in uses})} keys")


if __name__ == "__main__":
    app()
