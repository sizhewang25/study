"""The paper's data figures: paper name -> v5 artifact, and their publication.

`PAPER_FIGURES[group]` lists, for each figure the paper (cbg-benchmark-paper)
includes from a group's artifacts, the name the paper uses and where the
artifact lives. It is the one place that mapping is kept; the paper's
`% data:` comment above each `\\includegraphics` names the command.

`publish(group)` copies every listed figure's PDF (written beside its PNG by
`figure_io.save_figure`) to `<out_root>/pdf/<name>.pdf` and writes
`<out_root>/json/<name>.json`: the figure's state (`<stem>.state.json`, its
provenance) plus the module's data manifest under `data`. `out_root` defaults
to `<repo>/outputs/paper`. pro-paper and pro-tf publish into the same folder:
the TF names carry `.tf`.
"""

from __future__ import annotations

import filecmp
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules.figure_io import pdf_path, state_path
from scripts.analysis.v5.modules.labels import group_members, load_group
from scripts.analysis.v5.modules.paths import DEFAULT_ANALYSIS_ROOT, REPO_ROOT

DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "paper"

#: `pool` values: a role of the group (`seen`), every member (`all`), or the
#: run id that a command's block in the group file names (`run:<command>`).
SEEN, ALL = "seen", "all"


@dataclass(frozen=True)
class Entry:
    name: str        # the paper's name, figs/<name>.pdf
    kind: str        # _cross/<kind>/ (pooled) or <run>/<kind>/ (`run:` pools)
    pool: str        # SEEN, ALL or "run:<command>"
    stem: str        # artifact stem: <stem>.png / .pdf / .state.json
    manifest: str    # the module's data manifest in the same directory
    command: str     # the v5 command that writes it


def _seen(name, kind, stem, manifest, command):
    return Entry(name, kind, SEEN, stem, manifest, command)


_MECH = [
    Entry(f"cbg_mech.{m}.{ph}", "cbg-mechanism", "run:plot-cbg-mechanism",
          f"cbg_mech.{m}.{ph}", "cbg_mechanism.json", "plot-cbg-mechanism")
    for m in ("VAN", "SOI", "OCT-H", "OCT-S", "SPO")
    for ph in ("ltd", "mtl", "est")
]

PAPER_FIGURES: dict[str, list[Entry]] = {
    "pro-paper": [
        # §6.1
        _seen("error_cdf.pooled.norm.cut", "classify", "error_cdf.pooled.norm.cut",
              "error_cdf.pooled.norm.cut.manifest.json", "plot-error-cdf"),
        _seen("outcome_bars.unbounded.pooled", "classify", "outcome_bars.unbounded.pooled.healpix-128",
              "outcome_bars.unbounded.pooled.healpix-128.manifest.json", "plot-outcome-bars"),
        # §6.2
        *[_seen(n, "pni-gap", n, "sp_baseline.manifest.json", "plot-sp-baseline")
          for n in ("sp_vp_scatter", "sp_x_scatter", "sp_accuracy_by_x", "sp_rtt_error_has_x")],
        # §6.3
        _seen("x_cell_rtt.all", "pni-gap", "x_cell_rtt.all", "x_cell_rtt.manifest.json", "plot-x-cell-rtt"),
        _seen("sp_rho_by_side", "pni-gap", "sp_rho_by_side", "sp_taxonomy.report.json", "report-sp-taxonomy"),
        # §6.4
        *[_seen(n, "x-side-regions", n, "x_side_regions.json", "plot-x-side-regions")
          for n in ("x_side_error", "x_side_area", "x_side_inside")],
        # §6.5
        *[_seen(f"step_heading.{m}", "step-heading", f"step_heading.{m}", "step_heading.json",
                "plot-step-heading")
          for m in ("SOI", "VAN", "OCT-H", "OCT-S", "SPO")],
        # §7
        _seen("cost_box.pooled.heap", "cost", "cost_box.pooled.heap",
              "cost_box.pooled.heap.manifest.json", "plot-cost-box"),
        *[Entry(n, "pareto", ALL, n, "pareto.manifest.json", "plot-pareto")
          for n in ("pareto.cell.seen", "pareto.cell.unseen")],
        # Appendix: CBG mechanism figures
        *_MECH,
    ],
    # Built on the TF machine (create_tf_artifacts.sh): the TF runs are only there.
    "pro-tf": [
        _seen("error_cdf.pooled.tf", "classify", "error_cdf.pooled.norm.cut",
              "error_cdf.pooled.norm.cut.manifest.json", "plot-error-cdf"),
        _seen("outcome_bars.unbounded.pooled.tf", "classify", "outcome_bars.unbounded.pooled.healpix-128",
              "outcome_bars.unbounded.pooled.healpix-128.manifest.json", "plot-outcome-bars"),
        _seen("pareto.cell.seen.tf", "pareto", "pareto.cell.seen", "pareto.manifest.json", "plot-pareto"),
    ],
}


def entries(group_id: str) -> list[Entry]:
    try:
        return PAPER_FIGURES[group_id]
    except KeyError:
        raise ValueError(f"no paper figures registered for group {group_id!r}; have {sorted(PAPER_FIGURES)}") from None


def source_dir(entry: Entry, group: dict, analysis_root: Path) -> Path:
    """Where `entry`'s artifact lives for `group`, without creating anything."""
    if entry.pool.startswith("run:"):
        command = entry.pool[len("run:"):]
        run_id = ((group.get("analysis") or {}).get(command) or {}).get("run-id")
        if not isinstance(run_id, str) or run_id.startswith("@"):
            raise ValueError(f"group {group['group_id']!r}: analysis.{command}.run-id must name one run")
        return analysis_root / run_id / entry.kind
    runs = group_members(group) if entry.pool == ALL else group_members(group, entry.pool)
    with cross.using_group(group):
        name = cross.cross_name(runs)
    return analysis_root / "_cross" / entry.kind / name


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def publish(
    group_id: str,
    *,
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT,
    out_root: Path = DEFAULT_OUT_ROOT,
    copy_to: Path | None = None,
    check: Path | None = None,
) -> tuple[list[str], list[str]]:
    """Publish the group's paper figures. Returns (done lines, problems).

    `copy_to`: also copy each PDF to `<copy_to>/<name>.pdf` (the paper's figs/).
    `check`: compare each PDF to `<check>/<name>.pdf` instead of writing anything.
    """
    group = load_group(group_id)
    analysis_root = Path(analysis_root)
    pdf_dir, json_dir = Path(out_root) / "pdf", Path(out_root) / "json"
    done: list[str] = []
    problems: list[str] = []
    if check is None:
        pdf_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
    for e in entries(group_id):
        try:
            src_dir = source_dir(e, group, analysis_root)
        except ValueError as exc:
            problems.append(f"{e.name}: {exc}")
            continue
        png = src_dir / f"{e.stem}.png"
        pdf = pdf_path(png)
        if not pdf.is_file():
            problems.append(f"{e.name}: not built ({pdf}); run `{e.command} --group {group_id}`")
            continue
        if check is not None:
            theirs = Path(check) / f"{e.name}.pdf"
            if not theirs.is_file():
                problems.append(f"{e.name}: absent from {check}")
            elif filecmp.cmp(pdf, theirs, shallow=False):
                done.append(f"same    {e.name}")
            else:
                problems.append(f"{e.name}: differs from {theirs}")
            continue
        shutil.copyfile(pdf, pdf_dir / f"{e.name}.pdf")
        state = _read_json(state_path(png))
        manifest = _read_json(src_dir / e.manifest)
        record = {
            "paper_name": e.name,
            "group": group_id,
            "command": f"python -m scripts.analysis.v5.cli {e.command} --group {group_id}",
            "source_dir": str(src_dir.relative_to(REPO_ROOT)) if src_dir.is_relative_to(REPO_ROOT) else str(src_dir),
            "source_pdf": pdf.name,
            "state": state,
            "manifest": e.manifest,
            "data": manifest,
        }
        if state is None:
            record["state_missing"] = f"no {state_path(png).name}: the figure predates figure_io; rebuild it"
        if manifest is None:
            record["manifest_missing"] = f"no {e.manifest} in {src_dir}"
            problems.append(f"{e.name}: manifest {e.manifest} missing")
        (json_dir / f"{e.name}.json").write_text(json.dumps(record, indent=1) + "\n")
        if copy_to is not None:
            Path(copy_to).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(pdf, Path(copy_to) / f"{e.name}.pdf")
        done.append(f"published {e.name}")
    return done, problems
