"""Evaluate every slug and claim of one group into `outputs/paper/values/<group>.json`.

    python -m scripts.analysis.v5.paper_values.generate --group pro-paper
    python -m scripts.analysis.v5.paper_values.generate --group pro-tf      # on the TF machine

Reads only v5 artifacts (create_paper_artifacts.sh / create_tf_artifacts.sh
build them). A slug whose artifact is missing is written with
`status: "missing"` and its reason; the command then exits 1, after writing
everything it could.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import traceback
from pathlib import Path

import typer

from scripts.analysis.v5.modules.figure_io import git_state
from scripts.analysis.v5.modules.paths import DEFAULT_ANALYSIS_ROOT, DEFAULT_OUTPUTS_ROOT, REPO_ROOT
from scripts.analysis.v5.paper_values import registry as R
from scripts.analysis.v5.paper_values.common import Ctx

DEFAULT_OUT = REPO_ROOT / "outputs" / "paper" / "values"
EXPECTED = (FileNotFoundError, KeyError, IndexError)

app = typer.Typer(add_completion=False)


def _clean(v):
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def evaluate(group_id: str, *, analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
             outputs_root: Path = DEFAULT_OUTPUTS_ROOT) -> dict:
    from scripts.analysis.v5.paper.core import MissingArtifactError

    R.load_all()
    ctx = Ctx(group_id, analysis_root, outputs_root)
    slugs, claims = {}, {}
    for s in R.SLUGS.values():
        if s.group != group_id:
            continue
        rec = {"fmt": s.fmt, "section": s.section, "source": s.source, "note": s.note}
        try:
            v = _clean(s.getter(ctx))
            rec.update(value=v, latex=R.render_value(v, s.fmt), status="ok")
        except (*EXPECTED, MissingArtifactError) as exc:
            rec.update(value=None, latex=None, status="missing", error=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # a bug in a getter: keep going, but say where
            rec.update(value=None, latex=None, status="error",
                       error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc(limit=3))
        slugs[s.key] = rec
    for k in R.CLAIMS.values():
        if k.group != group_id or k.needs:
            continue
        rec = {"text": k.text, "section": k.section}
        try:
            holds, detail = k.check(ctx)
            rec.update(holds=bool(holds), detail=detail, status="ok")
        except (*EXPECTED, MissingArtifactError) as exc:
            rec.update(holds=None, detail=None, status="missing", error=f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            rec.update(holds=None, detail=None, status="error", error=f"{type(exc).__name__}: {exc}")
        claims[k.key] = rec
    return {
        "group": group_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "analysis_root": str(analysis_root),
        "slugs": slugs,
        "claims": claims,
    }


@app.command()
def main(
    group: str = typer.Option(..., "--group", help="pro-paper (mesh) or pro-tf (traffic-filtered)."),
    analysis_root: Path = typer.Option(DEFAULT_ANALYSIS_ROOT, help="Where v5 wrote the artifacts."),
    outputs_root: Path = typer.Option(DEFAULT_OUTPUTS_ROOT, help="Benchmark output root."),
    out: Path = typer.Option(None, help="Default: outputs/paper/values/<group>.json."),
) -> None:
    body = evaluate(group, analysis_root=analysis_root, outputs_root=outputs_root)
    out = out or DEFAULT_OUT / f"{group}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=1, ensure_ascii=False) + "\n")
    bad = {k: v for k, v in body["slugs"].items() if v["status"] != "ok"}
    cbad = {k: v for k, v in body["claims"].items() if v["status"] != "ok"}
    failing = [k for k, v in body["claims"].items() if v.get("holds") is False]
    typer.echo(f"wrote {out}: {len(body['slugs']) - len(bad)}/{len(body['slugs'])} slugs, "
               f"{len(body['claims']) - len(cbad)}/{len(body['claims'])} claims evaluated, {len(failing)} claims fail")
    for k, v in {**bad, **cbad}.items():
        typer.echo(f"  {v['status']}: {k}: {v['error']}", err=True)
    for k in failing:
        typer.echo(f"  claim fails: {k}: {body['claims'][k]['text']} ({body['claims'][k]['detail']})")
    if bad or cbad:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
