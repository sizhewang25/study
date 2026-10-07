"""Claims, where their artifacts live, and the report writer.

A **claim** is one statistic a paper section quotes: an ID from the task
inventory (`tasks/20261007-paper-numbers-reader/plan.md`), the sentence with
the values filled in by `fmt`, the raw unrounded values, and the artifact it
came from. A section module returns its claims in paper order, plus any tables
it renders whole.

Artifacts are located under the group's pooled names (`cross.group_name`):
`seen_dir(kind)` is `_cross/<kind>/<group>.seen/`, `all_dir(kind)` the folder
pooling every member. So the reader reads exactly what `create_paper_artifacts.sh`
wrote under `--group`. It needs nothing else -- in particular not the paper's
sources; comparing the report with the text is left to its reader.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from scripts.analysis.v5.modules import cross
from scripts.analysis.v5.modules.labels import group_members
from scripts.analysis.v5.modules.paths import DEFAULT_ANALYSIS_ROOT, MissingArtifactError

KIND = "paper"


@dataclass
class Claim:
    id: str
    sub: str              # the subsection it belongs to
    text: str             # the sentence, values filled in
    source: str           # artifact and column, relative to the section's folders
    values: dict = field(default_factory=dict)  # raw, unrounded
    note: str = ""


@dataclass
class Table:
    id: str
    sub: str
    title: str
    markdown: str
    source: str


@dataclass
class Context:
    group: dict
    analysis_root: Path = DEFAULT_ANALYSIS_ROOT

    def runs(self, role: str) -> list[str]:
        return group_members(self.group, role)

    def _dir(self, kind: str, run_ids: list[str]) -> Path:
        # Named from this context's group, not from whichever group the CLI activated.
        name = cross.group_name(run_ids, self.group) or cross.cross_name(run_ids)
        d = Path(self.analysis_root) / "_cross" / kind / name
        if not d.is_dir():
            raise MissingArtifactError(
                f"{d} missing; build it with create_paper_artifacts.sh (or the {kind} "
                f"command) under --group {self.group['group_id']}"
            )
        return d

    def seen_dir(self, kind: str) -> Path:
        return self._dir(kind, self.runs("seen"))

    def all_dir(self, kind: str) -> Path:
        return self._dir(kind, group_members(self.group))


def render(section: str, title: str, claims: list[Claim], tables: list[Table], *,
           group_id: str, sources: dict[str, str]) -> str:
    """The Markdown report for one section: claims in order, tables beside their claims."""
    lines = [
        f"# §{section} {title} — statistics",
        "",
        f"Group `{group_id}`. Each value is formatted once from its unrounded artifact value "
        f"(`scripts/analysis/v5/paper/fmt.py`); the raw values are in the JSON beside this file.",
        "",
        "Sources (under `outputs/analysis/v5/_cross/`):",
        "",
        *[f"- **{k}** = `{v}`" for k, v in sources.items()],
    ]
    for sub in dict.fromkeys(i.sub for i in [*tables, *claims]):
        lines += ["", f"## {sub}", ""]
        for t in (t for t in tables if t.sub == sub):
            lines += [f"**{t.id}** {t.title} — source: `{t.source}`", "", t.markdown, ""]
        rows = [c for c in claims if c.sub == sub]
        if rows:
            lines += ["| ID | Statistic | Source | Note |", "|---|---|---|---|"]
            lines += [f"| {c.id} | {c.text} | `{c.source}` | {c.note} |" for c in rows]
    return "\n".join(lines) + "\n"


def write(out_dir: Path, section: str, md: str, claims: list[Claim], tables: list[Table]) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"paper_numbers.s{section}"
    paths = {"md": out_dir / f"{stem}.md", "json": out_dir / f"{stem}.json"}
    paths["md"].write_text(md)
    paths["json"].write_text(json.dumps(
        {"section": section, "claims": [asdict(c) for c in claims],
         "tables": [asdict(t) for t in tables]},
        indent=2, default=float,
    ) + "\n")
    return paths
