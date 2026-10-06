"""Group files: a pooled set of runs declared once.

`analysis.common` merges into every member run's lookups and must agree with
the run's own config; `analysis.<command>` is that command's CLI flags, loaded
by `--group` as defaults that a flag on the line still overrides.
"""

from __future__ import annotations

import pytest
import typer
import yaml
from typer.testing import CliRunner

from scripts.analysis.v5 import cli
from scripts.analysis.v5.modules import labels as L

RUN_CONFIG = {
    "analysis": {
        "common": {"dataset_label": "AS-A", "dist_norm_km": {"min": 0.0, "max": 100.0}},
        "plot-error-cdf": {"combo_ids": ["shortest_ping", "octant_cbg_hull"]},
    }
}


def _group(tmp_path, body: dict, name: str = "g") -> str:
    d = tmp_path / "groups"
    d.mkdir(exist_ok=True)
    (d / f"{name}.yaml").write_text(yaml.safe_dump({"group_id": name, **body}))
    return str(d)


@pytest.fixture
def configs(tmp_path, monkeypatch):
    """`run-a` has RUN_CONFIG; `run-b` has no config at all."""
    cfg = tmp_path / "run-a.yaml"
    cfg.write_text(yaml.safe_dump(RUN_CONFIG))
    monkeypatch.setattr(L, "config_path", lambda run_id, root=None: cfg if run_id == "run-a" else None)
    return tmp_path


def _use(monkeypatch, groups_dir: str) -> None:
    monkeypatch.setattr(L, "GROUPS_DIR", groups_dir)


# ---- analysis.common merges; per-command blocks do not -----------------------------


def test_group_common_reaches_a_run_without_its_own(configs, monkeypatch):
    _use(monkeypatch, _group(configs, {
        "runs": {"seen": ["run-a", "run-b"]},
        "analysis": {"common": {"rtt_norm_ms": {"min": 0.0, "max": 92.395}}},
    }))
    assert L.declared_rtt_norm_ms("run-b") == (0.0, 92.395)
    assert L.declared_rtt_norm_ms("run-a") == (0.0, 92.395)
    assert L.declared_dist_norm_km("run-a") == (0.0, 100.0)   # the run's own, no group value
    assert L.declared_dist_norm_km("run-b") is None


def test_agreeing_copies_are_fine_and_conflicts_raise(configs, monkeypatch):
    _use(monkeypatch, _group(configs, {
        "runs": ["run-a"], "analysis": {"common": {"dist_norm_km": {"min": 0.0, "max": 100.0}}},
    }, "same"))
    assert L.declared_dist_norm_km("run-a") == (0.0, 100.0)
    _use(monkeypatch, _group(configs, {
        "runs": ["run-a"], "analysis": {"common": {"dist_norm_km": {"min": 0.0, "max": 99.0}}},
    }, "other"))
    with pytest.raises(L.ConflictingDeclarationError, match="dist_norm_km"):
        L.declared_dist_norm_km("run-a")


def test_group_command_blocks_are_flags_not_config(configs, monkeypatch):
    _use(monkeypatch, _group(configs, {
        "runs": ["run-a", "run-b"],
        "analysis": {"plot-error-cdf": {"method": ["vanilla_cbg"]}},
    }))
    assert L.declared_combo_ids("run-a", "plot-error-cdf") == ["shortest_ping", "octant_cbg_hull"]
    assert L.declared_combo_ids("run-b", "plot-error-cdf") is None


def test_group_file_checks(tmp_path):
    d = _group(tmp_path, {"runs": ["a", "a"]})
    with pytest.raises(ValueError, match="twice"):
        L._groups(d)
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "x.yaml").write_text(yaml.safe_dump({"group_id": "y", "runs": []}))
    with pytest.raises(ValueError, match="file stem"):
        L._groups(str(bad))


def test_roles():
    g = {"group_id": "g", "runs": {"seen": ["a", "b"], "unseen": ["c", "d"]}}
    assert L.group_members(g) == ["a", "b", "c", "d"]
    assert L.group_members(g, "unseen") == ["c", "d"]
    with pytest.raises(ValueError, match="no role"):
        L.group_members(g, "ripe")


# ---- analysis.<command> -> the command's flags ------------------------------------------


def test_default_map_expands_roles_and_pairs(tmp_path):
    d = _group(tmp_path, {
        "runs": {"seen": ["m1", "m2"], "unseen": ["l1", "l2"]},
        "analysis": {
            "common": {"dist_norm_km": {"min": 0.0, "max": 1.0}},
            "plot-error-cdf": {"layout": "pooled", "unanswered": "cut", "run-id": "@seen"},
            "report-loso-delta": {"pair": "@seen:@unseen"},
            "plot-vp-distance-cdf": {"run-id": ["@seen", "extra"], "share-at": [3.4]},
        },
    })
    dm = cli.group_default_map("g", d)
    assert "common" not in dm
    assert dm["plot-error-cdf"] == {"layout": ["pooled"], "unanswered": "cut", "run_id": ["m1", "m2"]}
    assert dm["report-loso-delta"] == {"pair": ["m1:l1", "m2:l2"]}
    assert dm["plot-vp-distance-cdf"]["run_id"] == ["m1", "m2", "extra"]


def test_default_map_refuses_what_is_not_a_flag(tmp_path):
    d = _group(tmp_path, {"runs": ["a"], "analysis": {"plot-error-cdf": {"combo_ids": ["x"]}}}, "g1")
    with pytest.raises(typer.BadParameter, match="not a flag of plot-error-cdf"):
        cli.group_default_map("g1", d)
    d = _group(tmp_path, {"runs": ["a"], "analysis": {"plot-nothing": {}}}, "g2")
    with pytest.raises(typer.BadParameter, match="not a v5 command"):
        cli.group_default_map("g2", d)
    d = _group(tmp_path, {"runs": {"s": ["a"], "u": ["b", "c"]},
                          "analysis": {"report-loso-delta": {"pair": "@s:@u"}}}, "g3")
    with pytest.raises(typer.BadParameter, match="different sizes"):
        cli.group_default_map("g3", d)


def test_split_group_anywhere():
    assert cli._split_group(["plot-error-cdf", "--group", "g", "--layout", "pooled"]) == (
        "g", ["plot-error-cdf", "--layout", "pooled"])
    assert cli._split_group(["--group=g", "classify"]) == ("g", ["classify"])
    assert cli._split_group(["classify"]) == (None, ["classify"])


def test_a_flag_on_the_line_wins(tmp_path):
    """Click's default_map: the group supplies --run-id, the line overrides it."""
    command = cli.app
    defaults = {"report-bounds": {"run_id": ["from-group"],
                                  "outputs_root": str(tmp_path), "analysis_root": str(tmp_path)}}
    def said(args):
        r = CliRunner().invoke(command, args, default_map=defaults)
        assert r.exit_code != 0  # no such run either way
        return f"{r.output} {r.exception!r}"

    assert "from-group" in said(["report-bounds"])
    out = said(["report-bounds", "--run-id", "from-line"])
    assert "from-line" in out and "from-group" not in out


def test_the_real_group_file_resolves():
    """configs/groups/pro-paper.yaml: every block is a real command's flags."""
    dm = cli.group_default_map("pro-paper")
    assert dm["plot-pareto"]["seen"] == ["pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh"]
    assert dm["report-variant-delta"]["pair"][0] == "pro-as01-mesh:pro-as01-loso"
