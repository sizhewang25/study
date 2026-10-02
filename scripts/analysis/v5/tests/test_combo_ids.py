"""Per-figure method selection: `analysis.<command>.combo_ids` in the run config.

Three layers. `labels.declared_combo_ids` reads the key and refuses a malformed
one -- unlike `dataset_label`, a broken value must not fall back to "draw
everything", because that is exactly the figure the author meant to change.
`cli._methods_for` decides precedence (`--method`, then the config, then all)
and refuses runs drawn together that disagree. The end-to-end tests run the
real command and read the manifest back, since the filtered figure keeps the
unfiltered one's filename and the manifest is all that tells them apart.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import typer
import yaml
from typer.testing import CliRunner

from scripts.analysis.v5 import cli
from scripts.analysis.v5.modules import classify as C
from scripts.analysis.v5.modules import cross, labels
from scripts.analysis.v5.modules import figure_error_cdf as E
from scripts.analysis.v5.modules.paths import CLASSIFY_KIND, resolve_run
from scripts.analysis.v5.modules.status import SHORTEST_PING

COMBOS = ("million_scale_cbg", "octant_cbg_hull", "octant_cbg_spl", "vanilla_cbg")
NO_OCT_S = [SHORTEST_PING, "million_scale_cbg", "octant_cbg_hull", "vanilla_cbg"]


def _run(tmp_path, run_id, analysis=None, *, combos=COMBOS, first_id=0):
    """A resolvable run: folds holding `combos`, and a config with `analysis`."""
    root = tmp_path / "outputs"
    setup = root / run_id / "generic_csv" / "anchors_to_probes"
    for combo in combos:
        (setup / "fold_0" / combo).mkdir(parents=True, exist_ok=True)
        (setup / "fold_0" / combo / "targets.parquet").touch()
    cfg = tmp_path / "configs" / f"{run_id}.yaml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(yaml.safe_dump({"run_id": run_id, "analysis": analysis or {}}))
    (setup / "target_space.json").write_text(json.dumps({"config": str(cfg)}))
    return root


def _scored(tmp_path, root, run_id, *, combos=COMBOS, first_id=0):
    """`classify/healpix-128/` for the run: four solved TGs per method."""
    run = resolve_run(run_id, root)
    out = run.classify_dir(128, root=tmp_path / "analysis")
    frames = {}
    for m in (SHORTEST_PING, *combos):
        frames[m] = pd.DataFrame({
            "tg_id": [f"tg-{first_id + i}" for i in range(4)],
            "status": ["SUCCESS"] * 4,
            "pred_lat": [40.0] * 4,
            "pred_lon": [-100.0] * 4,
            "tg_lat": [40.0 + first_id + i for i in range(4)],
            "tg_lon": [-100.0] * 4,
            C.GRID_OFFSET: [0] * 4,
            "cell_label": ["correct"] * 4,
            "pred_dist_to_tg_km": [10.0, 20.0, 30.0, 40.0],
            "pred_dist_to_seed_km": [10.0, 20.0, 30.0, 40.0],
        })
        frames[m].to_parquet(out / C.TGS_PARQUET.format(method=m), index=False)
    C.summarize(frames, 128).to_csv(out / C.ACCURACY_CSV, index=False)
    return run


def _block(cmd, ids):
    return {cmd: {"combo_ids": ids}}


@pytest.fixture(autouse=True)
def _clear_label_cache():
    labels.dataset_label.cache_clear()
    yield
    labels.dataset_label.cache_clear()


class TestTheConfigReader:
    def test_no_block_means_everything(self, tmp_path):
        root = _run(tmp_path, "r1")
        assert labels.declared_combo_ids("r1", "plot-error-cdf", root) is None

    def test_an_empty_placeholder_means_everything(self, tmp_path):
        root = _run(tmp_path, "r1", {"plot-error-cdf": {}})
        assert labels.declared_combo_ids("r1", "plot-error-cdf", root) is None

    def test_a_null_block_means_everything(self, tmp_path):
        root = _run(tmp_path, "r1", {"plot-error-cdf": None})
        assert labels.declared_combo_ids("r1", "plot-error-cdf", root) is None

    def test_a_list_is_returned_in_order(self, tmp_path):
        root = _run(tmp_path, "r1", _block("plot-error-cdf", NO_OCT_S))
        assert labels.declared_combo_ids("r1", "plot-error-cdf", root) == NO_OCT_S

    def test_another_commands_block_is_not_read(self, tmp_path):
        root = _run(tmp_path, "r1", _block("plot-outcome-bars", NO_OCT_S))
        assert labels.declared_combo_ids("r1", "plot-error-cdf", root) is None

    @pytest.mark.parametrize("bad", [[], "vanilla_cbg", [1, 2], ["vanilla_cbg", "vanilla_cbg"]])
    def test_a_malformed_list_raises_rather_than_draw_everything(self, tmp_path, bad):
        root = _run(tmp_path, "r1", _block("plot-error-cdf", bad))
        with pytest.raises(ValueError, match="combo_ids"):
            labels.declared_combo_ids("r1", "plot-error-cdf", root)

    def test_the_scalar_reader_still_refuses_lists(self, tmp_path):
        """`declared` was split, not loosened: a list label still falls back."""
        root = _run(tmp_path, "r1", {"common": {"dataset_label": ["a", "b"]}})
        assert cross.short_dataset("r1", outputs_root=root) == "r1"


class TestPrecedence:
    def test_nothing_declared_is_all(self, tmp_path):
        root = _run(tmp_path, "r1")
        runs = [resolve_run("r1", root)]
        assert cli._methods_for("plot-error-cdf", runs, None, root) == (None, "all")

    def test_the_config_is_used(self, tmp_path):
        root = _run(tmp_path, "r1", _block("plot-error-cdf", NO_OCT_S))
        runs = [resolve_run("r1", root)]
        assert cli._methods_for("plot-error-cdf", runs, None, root) == (NO_OCT_S, "config")

    def test_the_cli_beats_the_config(self, tmp_path):
        root = _run(tmp_path, "r1", _block("plot-error-cdf", NO_OCT_S))
        runs = [resolve_run("r1", root)]
        got = cli._methods_for("plot-error-cdf", runs, ["octant_cbg_spl"], root)
        assert got == (["octant_cbg_spl"], "cli")

    def test_a_combo_the_run_does_not_hold_is_refused(self, tmp_path):
        """`octant_cbg` is the as7018 spelling; the pro runs say `octant_cbg_spl`."""
        root = _run(tmp_path, "r1", _block("plot-outcome-bars", ["octant_cbg"]))
        runs = [resolve_run("r1", root)]
        with pytest.raises(typer.BadParameter, match="does not hold"):
            cli._methods_for("plot-outcome-bars", runs, None, root)


class TestRunsDrawnTogether:
    def _two(self, tmp_path, a, b):
        _run(tmp_path, "r1", a)
        root = _run(tmp_path, "r2", b)
        return root, [resolve_run("r1", root), resolve_run("r2", root)]

    def test_agreeing_runs_share_the_list(self, tmp_path):
        blk = _block("plot-error-cdf", NO_OCT_S)
        root, runs = self._two(tmp_path, blk, blk)
        assert cli._methods_for("plot-error-cdf", runs, None, root) == (NO_OCT_S, "config")

    def test_different_lists_are_refused(self, tmp_path):
        root, runs = self._two(tmp_path, _block("plot-error-cdf", NO_OCT_S),
                               _block("plot-error-cdf", [SHORTEST_PING]))
        with pytest.raises(typer.BadParameter, match="differs"):
            cli._methods_for("plot-error-cdf", runs, None, root)

    def test_a_list_in_one_config_only_is_refused(self, tmp_path):
        root, runs = self._two(tmp_path, _block("plot-error-cdf", NO_OCT_S), None)
        with pytest.raises(typer.BadParameter, match="differs"):
            cli._methods_for("plot-error-cdf", runs, None, root)

    def test_per_run_layouts_take_each_runs_own_list(self, tmp_path):
        root, runs = self._two(tmp_path, _block("plot-error-cdf", NO_OCT_S), None)
        calls = cli._layout_calls("plot-error-cdf", runs, None, (E.PER_RUN,), E.PER_RUN, root)
        got = {tuple(r.run_id for r in g): (m, s) for g, _, m, s in calls}
        assert got == {("r1",): (NO_OCT_S, "config"), ("r2",): (None, "all")}

    def test_pooled_beside_per_run_still_needs_agreement(self, tmp_path):
        root, runs = self._two(tmp_path, _block("plot-error-cdf", NO_OCT_S), None)
        with pytest.raises(typer.BadParameter, match="differs"):
            cli._layout_calls(
                "plot-error-cdf", runs, None, (E.PER_RUN, E.POOLED), E.PER_RUN, root)


class TestCommandsWithoutAMethodAxis:
    def test_an_empty_placeholder_is_fine(self, tmp_path):
        root = _run(tmp_path, "r1", {"plot-answer-space": {}})
        cli._refuse_combo_ids("plot-answer-space", ["r1"], root)

    def test_a_list_there_is_refused_not_ignored(self, tmp_path):
        root = _run(tmp_path, "r1", _block("plot-answer-space", NO_OCT_S))
        with pytest.raises(typer.BadParameter, match="no effect"):
            cli._refuse_combo_ids("plot-answer-space", ["r1"], root)

    def test_the_pni_clustering_method_is_not_a_combo_command(self):
        assert "plot-pni-gap" not in cli.COMBO_COMMANDS

    def test_every_combo_command_is_registered(self):
        names = set(typer.main.get_command(cli.app).commands)
        assert cli.COMBO_COMMANDS <= names


class TestEndToEnd:
    """The real command, read back through its manifest."""

    def _invoke(self, tmp_path, root, *args):
        result = CliRunner().invoke(cli.app, [
            *args, "--outputs-root", str(root), "--analysis-root", str(tmp_path / "analysis"),
        ])
        assert result.exit_code == 0, result.output
        return result

    def _manifest(self, tmp_path, run, layout=E.PER_RUN):
        name = E.NAMES[layout][2]
        return json.loads(
            (run.analysis_dir(CLASSIFY_KIND, root=tmp_path / "analysis") / name).read_text())

    def test_the_config_list_drops_oct_s_from_the_error_cdf(self, tmp_path):
        root = _run(tmp_path, "r1", _block("plot-error-cdf", NO_OCT_S))
        run = _scored(tmp_path, root, "r1")
        self._invoke(tmp_path, root, "plot-error-cdf", "--run-id", "r1")
        body = self._manifest(tmp_path, run)
        assert "octant_cbg_spl" not in body["methods"]
        assert sorted(body["methods"]) == sorted(NO_OCT_S)
        assert body["methods_source"] == "config"

    def test_without_a_list_everything_is_drawn(self, tmp_path):
        root = _run(tmp_path, "r1", {"plot-error-cdf": {}})
        run = _scored(tmp_path, root, "r1")
        self._invoke(tmp_path, root, "plot-error-cdf", "--run-id", "r1")
        body = self._manifest(tmp_path, run)
        assert sorted(body["methods"]) == sorted([SHORTEST_PING, *COMBOS])
        assert body["methods_source"] == "all"

    def test_the_cli_flag_overrides_and_says_so(self, tmp_path):
        root = _run(tmp_path, "r1", _block("plot-error-cdf", NO_OCT_S))
        run = _scored(tmp_path, root, "r1")
        self._invoke(tmp_path, root, "plot-error-cdf", "--run-id", "r1",
                     "-m", "octant_cbg_spl", "-m", SHORTEST_PING)
        body = self._manifest(tmp_path, run)
        assert sorted(body["methods"]) == sorted(["octant_cbg_spl", SHORTEST_PING])
        assert body["methods_source"] == "cli"

    def test_s_p_is_not_implicit(self, tmp_path):
        root = _run(tmp_path, "r1", _block("plot-error-cdf", ["vanilla_cbg", "octant_cbg_hull"]))
        run = _scored(tmp_path, root, "r1")
        self._invoke(tmp_path, root, "plot-error-cdf", "--run-id", "r1")
        assert SHORTEST_PING not in self._manifest(tmp_path, run)["methods"]

    def test_the_pooled_champion_upset_honours_its_own_block(self, tmp_path):
        blk = {**_block("plot-champion-upset", NO_OCT_S), "plot-error-cdf": {}}
        _run(tmp_path, "r1", blk)
        root = _run(tmp_path, "r2", blk)
        _scored(tmp_path, root, "r1")
        _scored(tmp_path, root, "r2", first_id=100)
        self._invoke(tmp_path, root, "plot-champion-upset", "--layout", "pooled",
                     "--run-id", "r1", "--run-id", "r2")
        manifests = list((tmp_path / "analysis").rglob("champion_upset*.pooled*manifest.json"))
        assert len(manifests) == 1
        body = json.loads(manifests[0].read_text())
        assert sorted(body["methods"]) == sorted(NO_OCT_S)
        assert body["methods_source"] == "config"
