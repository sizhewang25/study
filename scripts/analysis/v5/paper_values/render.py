r"""Render the template into a complete, self-contained paper project.

    python -m scripts.analysis.v5.paper_values.render \
        --values outputs/paper/values/pro-paper.json --values outputs/paper/values/pro-tf.json \
        --figs outputs/paper/pdf --out outputs/paper/project --build --zip

Copies `paper/template/` to `--out`, replaces every `\pv{key}` (and
`\pv[fmt]{key}`) with the value from the values files, evaluates the derived
slugs and claims, drops the `\pv` macro block, copies the data-figure PDFs
into `figs/`, checks every included file exists, and writes `PROVENANCE.json`.
`--build` compiles it (latexmk) and fails on a LaTeX error or an unresolved
reference; `--zip` packs it for Overleaf.

A slug with no value, or a figure with no PDF, is an error. `--allow-missing`
renders it as a red `[key]` (a boxed name for a figure) and lists it instead,
for a draft built before every machine has delivered its values.

Every slot is written `\new{\pv{key}}` in the template, so each generated number
prints in the paper's `\new` colour (blue) and stands out from the hand-written
text. `--no-mark` drops the wrapper and prints the plain number, for submission.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import subprocess
from pathlib import Path

import typer

from scripts.analysis.v5.modules.figure_io import git_state
from scripts.analysis.v5.modules.paths import REPO_ROOT
from scripts.analysis.v5.paper_values import registry as R
from scripts.analysis.v5.paper_values.slugify import (
    DEFAULT_TEMPLATE, GRAPHIC, INPUT, MACRO_BEGIN, MACRO_END, NEW_PV, PV, compiled_sections, uncomment,
)

PAPER_OUT = REPO_ROOT / "outputs" / "paper"


def merge_values(paths: list[Path]) -> tuple[dict, dict, list[dict]]:
    """(slugs, claims, sources) from every values file; a key in two files is an error."""
    slugs, claims, sources = {}, {}, []
    for p in paths:
        body = json.loads(Path(p).read_text())
        sources.append({"file": str(p), "group": body["group"], "created_utc": body["created_utc"],
                        "git": body["git"], "analysis_root": body.get("analysis_root")})
        for table, part in ((slugs, "slugs"), (claims, "claims")):
            for k, v in body[part].items():
                if k in table:
                    raise ValueError(f"{k} is in two values files ({body['group']} and another)")
                table[k] = {**v, "group": body["group"]}
    return slugs, claims, sources


def evaluate_derived(slugs: dict, claims: dict) -> None:
    """Add derived slugs and cross-group claims, in place, from the merged raw values."""
    R.load_all()

    def raw(keys):
        vals = [slugs.get(k, {}).get("value") for k in keys]
        missing = [k for k, v in zip(keys, vals) if v is None]
        return vals, missing

    for d in R.DERIVED.values():
        vals, missing = raw(d.needs)
        rec = {"fmt": d.fmt, "section": d.section, "source": d.source, "note": d.note, "group": "derived"}
        if missing:
            rec.update(value=None, latex=None, status="missing", error=f"needs {', '.join(missing)}")
        else:
            v = float(d.f(*vals))
            rec.update(value=v, latex=R.render_value(v, d.fmt), status="ok")
        slugs[d.key] = rec
    for k in R.CLAIMS.values():
        if not k.needs:
            continue
        vals, missing = raw(k.needs)
        rec = {"text": k.text, "section": k.section, "group": "derived"}
        if missing:
            rec.update(holds=None, detail=None, status="missing", error=f"needs {', '.join(missing)}")
        else:
            holds, detail = k.check(*vals)
            rec.update(holds=bool(holds), detail=detail, status="ok")
        claims[k.key] = rec


def substitute(text: str, slugs: dict, *, allow_missing: bool, missing: list[str], mark: bool = True) -> str:
    """Replace every slug with its value; `mark=False` also drops each slot's `\\new{}` highlight."""
    if not mark:
        text = NEW_PV.sub(lambda m: m.group(1), text)

    def one(m):
        fmt, key = m.group(1), m.group(2)
        rec = slugs.get(key)
        if rec is None or rec.get("status") != "ok":
            missing.append(key)
            if not allow_missing:
                return m.group(0)
            return r"\textcolor{red}{[" + key + "]}"
        return R.render_value(rec["value"], fmt) if fmt else rec["latex"]
    return PV.sub(one, text)


def drop_macro(main: str) -> str:
    return re.sub(rf"^{re.escape(MACRO_BEGIN)}.*?^{re.escape(MACRO_END)}\n", "", main, flags=re.S | re.M)


def render(template: Path, values: list[Path], figs: Path, out: Path, *, allow_missing: bool = False,
           mark: bool = True) -> dict:
    template, out = Path(template).resolve(), Path(out).resolve()
    if out == template or template in out.parents:
        raise ValueError("--out must be outside the template")
    slugs, claims, sources = merge_values(values)
    evaluate_derived(slugs, claims)

    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(template, out, ignore=shutil.ignore_patterns("slugs.orig.json", "outputs"))

    missing: list[str] = []
    used: set[str] = set()
    tex_files = sorted(p for p in out.rglob("*.tex"))
    for p in tex_files:
        body = p.read_text()
        used |= {m.group(2) for m in PV.finditer(body)}
        if "\\pv" in body:
            p.write_text(substitute(body, slugs, allow_missing=allow_missing, missing=missing, mark=mark))
    main = out / "main.tex"
    main.write_text(drop_macro(main.read_text()))

    # figures: the data PDFs, then check every include resolves
    copied, absent = [], []
    files = ["main.tex", *compiled_sections(main.read_text())]
    for f in files:
        body = (out / f).read_text()
        for ref in GRAPHIC.findall(uncomment(body)):
            target = out / ref
            if target.is_file():
                continue
            src = Path(figs) / Path(ref).name
            if src.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                copied.append(ref)
            else:
                absent.append(ref)
        for ref in INPUT.findall(uncomment(body)):
            r = ref if Path(ref).suffix else ref + ".tex"
            if not (out / r).is_file():
                absent.append(r)
    if absent and allow_missing:
        for f in files:
            p = out / f
            body = p.read_text()
            for ref in absent:
                body = re.sub(r"\\includegraphics(\[[^\]]*\])?\{" + re.escape(ref) + r"\}",
                              lambda m, ref=ref: r"\fbox{\texttt{missing figure " + Path(ref).stem.replace("_", r"\_")
                              + "}}", body)
            p.write_text(body)

    # figure provenance, if the published json sits beside the pdf folder
    fig_json = Path(figs).parent / "json"
    fig_state = {}
    for ref in copied:
        j = fig_json / (Path(ref).stem + ".json")
        if j.is_file():
            rec = json.loads(j.read_text())
            fig_state[Path(ref).stem] = {"group": rec.get("group"), "command": rec.get("command"),
                                         "git": (rec.get("state") or {}).get("git"),
                                         "created_utc": (rec.get("state") or {}).get("created_utc")}
    failing = {k: v for k, v in claims.items() if v.get("holds") is False}
    prov = {
        "rendered_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "renderer_git": git_state(),
        "template": str(template),
        "values": sources,
        "figures": {"copied": copied, "missing": sorted(set(absent)), "state": fig_state},
        "slugs": {"used": len(used), "missing": sorted(set(missing)), "marked": mark},
        "claims_failing": {k: {"text": v["text"], "detail": v["detail"]} for k, v in failing.items()},
        "claims_unevaluated": sorted(k for k, v in claims.items() if v.get("status") != "ok"),
    }
    (out / "PROVENANCE.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False) + "\n")
    return prov


def build(out: Path) -> tuple[bool, str]:
    """latexmk in `out`; (ok, summary). Fails on `!` errors or `??` in the PDF."""
    r = subprocess.run(["latexmk", "-g", "-quiet", "main.tex"], cwd=out, capture_output=True, text=True)
    log = next(iter(sorted(out.rglob("main.log"))), None)
    pdf = next(iter(sorted(out.rglob("main.pdf"))), None)
    if log is None or pdf is None:
        return False, f"latexmk exit {r.returncode}, no main.log/main.pdf\n{r.stdout[-2000:]}{r.stderr[-2000:]}"
    errors = [ln for ln in log.read_text(errors="replace").splitlines() if ln.startswith("!")]
    text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True).stdout
    unresolved = text.count("??")
    pages = re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout)
    summary = f"{pdf.relative_to(out)}: {pages.group(1) if pages else '?'} pages, {len(errors)} errors, {unresolved} '??'"
    return not errors and unresolved == 0, summary + ("\n" + "\n".join(errors[:10]) if errors else "")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    values: list[Path] = typer.Option(None, "--values", help="Values JSON (repeatable). Default: all in outputs/paper/values."),
    template: Path = typer.Option(DEFAULT_TEMPLATE, help="The slugged template."),
    figs: Path = typer.Option(PAPER_OUT / "pdf", help="The published figure PDFs."),
    out: Path = typer.Option(PAPER_OUT / "project", help="Where to write the paper project."),
    allow_missing: bool = typer.Option(False, help="Render missing slugs/figures as red placeholders."),
    do_build: bool = typer.Option(False, "--build", help="Compile with latexmk and check."),
    do_zip: bool = typer.Option(False, "--zip", help="Also write <out>.zip."),
    mark: bool = typer.Option(True, "--mark/--no-mark",
                              help="Keep each generated number in \\new{} (blue); --no-mark prints it plain."),
) -> None:
    values = values or sorted((PAPER_OUT / "values").glob("*.json"))
    if not values:
        raise typer.BadParameter("no values files; run paper_values.generate first")
    try:
        prov = render(template, values, figs, out, allow_missing=allow_missing, mark=mark)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"wrote {out} from {', '.join(s['group'] for s in prov['values'])}: "
               f"{len(prov['figures']['copied'])} figures copied")
    problems = []
    if prov["slugs"]["missing"]:
        problems.append(f"{len(prov['slugs']['missing'])} slugs without a value: {', '.join(prov['slugs']['missing'])}")
    if prov["figures"]["missing"]:
        problems.append(f"{len(prov['figures']['missing'])} files missing: {', '.join(prov['figures']['missing'])}")
    for k, v in prov["claims_failing"].items():
        typer.echo(f"  claim fails: {k}: {v['text']} ({v['detail']})")
    for p in problems:
        typer.echo(("warning: " if allow_missing else "error: ") + p, err=True)
    if problems and not allow_missing:
        raise typer.Exit(1)
    if do_zip:  # before the build, so the archive holds sources only
        z = shutil.make_archive(str(out), "zip", root_dir=out)
        typer.echo(f"wrote {z}")
    if do_build:
        ok, summary = build(out)
        typer.echo(summary)
        if not ok:
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
