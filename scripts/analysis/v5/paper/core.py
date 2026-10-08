"""The data behind a paper section's figures and tables, and the report writer.

Each section module returns **tables**: one per paper figure or table (or a
dataset the section describes), holding the absolute values and percentages
the figure draws or the table prints -- percentiles, medians, accuracies,
shares -- each cell formatted once by `fmt` from its unrounded artifact value.
The unrounded rows go to the JSON beside the Markdown. No sentence of the
text is tracked, and nothing is read from the paper project: comparing the
report with the text is left to its reader.

Artifacts are located under the group's pooled names (`cross.group_name`):
`seen_dir(kind)` is `_cross/<kind>/<group>.seen/`, `all_dir(kind)` the folder
pooling every member, `run_dir(run, kind)` one run's own folder.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules.labels import dataset_label, group_members
from scripts.analysis.v5.modules.paths import DEFAULT_ANALYSIS_ROOT, MissingArtifactError

KIND = "paper"

#: The paper's method order, by term.
METHOD_ORDER = ("S-P", "SOI", "VAN", "OCT-H", "OCT-S", "SPO")


@dataclass
class Table:
    id: str               # the paper label it backs (tab:..., fig:...) or a short name
    title: str
    source: str           # artifact file(s), relative to the sources listed in the report
    columns: list[str]
    rows: list[list[str]]  # formatted cells
    raw: list[dict] = field(default_factory=list)  # unrounded rows
    note: str = ""

    def markdown(self) -> str:
        head = "| " + " | ".join(self.columns) + " |"
        sep = "|" + "---|" * len(self.columns)
        body = ["| " + " | ".join(str(c) for c in r) + " |" for r in self.rows]
        return "\n".join([head, sep, *body])


#: `(header, function of a raw row -> formatted cell)`.
Column = tuple[str, Callable[[pd.Series], str]]


def from_frame(id: str, title: str, source: str, frame: pd.DataFrame, columns: list[Column],
               *, note: str = "") -> Table:
    """A table whose rows are `frame`'s rows, one formatted cell per column spec."""
    rows = [[f(r) for _, f in columns] for _, r in frame.iterrows()]
    raw = json.loads(frame.to_json(orient="records"))
    return Table(id, title, source, [h for h, _ in columns], rows, raw, note)


def network(name: str) -> str:
    """A dataset label (`PRO-MESH AS-A`) or a run id -> `AS-A`; `all` / `pooled` -> `All`."""
    if name in ("all", "pooled", "All"):
        return "All"
    label = name if " " in name else dataset_label(name)
    return label.split()[-1]


def method_rank(labels) -> list[int]:
    """Sort key for the paper's method order; unknown methods last."""
    return [METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER) for m in labels]


@dataclass
class Context:
    group: dict
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT

    def runs(self, role: str) -> list[str]:
        return group_members(self.group, role)

    def _check(self, d: Path, what: str) -> Path:
        if not d.is_dir():
            raise MissingArtifactError(
                f"{d} missing; build it with create_paper_artifacts.sh (or the {what} "
                f"command) under --group {self.group['group_id']}"
            )
        return d

    def _dir(self, kind: str, run_ids: list[str]) -> Path:
        # Named from this context's group, not from whichever group the CLI activated.
        name = cross.group_name(run_ids, self.group) or cross.cross_name(run_ids)
        return self._check(Path(self.analysis_root) / "_cross" / kind / name, kind)

    def seen_dir(self, kind: str) -> Path:
        return self._dir(kind, self.runs("seen"))

    def all_dir(self, kind: str) -> Path:
        return self._dir(kind, group_members(self.group))

    def run_dir(self, run_id: str, *parts: str) -> Path:
        return self._check(Path(self.analysis_root).joinpath(run_id, *parts), parts[0])

    def source(self, d: Path) -> str:
        """`d` relative to the analysis root, for the report's source list."""
        return str(d.relative_to(self.analysis_root)) + "/"


def render(section: str, title: str, tables: list[Table], *, group_id: str,
           sources: dict[str, str]) -> str:
    """One section's Markdown: its sources, then every table in paper order."""
    lines = [
        f"# §{section} {title} — data",
        "",
        f"Group `{group_id}`. Each cell is formatted once from its unrounded artifact value "
        f"(`scripts/analysis/v5/paper/fmt.py`); the unrounded rows are in the JSON beside this file.",
        "",
        "Sources (under `outputs/analysis/v5/`):",
        "",
        *[f"- **{k}** = `{v}`" for k, v in sources.items()],
    ]
    for t in tables:
        lines += ["", f"## `{t.id}` — {t.title}", "", f"Source: {t.source}", ""]
        if t.note:
            lines += [t.note, ""]
        lines.append(t.markdown())
    return "\n".join(lines) + "\n"


def write(out_dir: Path, section: str, md: str, tables: list[Table]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"paper_numbers.s{section}"
    paths = {"md": out_dir / f"{stem}.md", "json": out_dir / f"{stem}.json"}
    paths["md"].write_text(md)
    paths["json"].write_text(json.dumps(
        {"section": section, "tables": [asdict(t) for t in tables]}, indent=2, default=float,
    ) + "\n")
    return paths
