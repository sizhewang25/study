# Paper artifacts wired into v5 (2026-10-05)

Follows [the paper ↔ v5 sweep](2026-10-05-paper-v5-support-sweep.md). That
sweep found every paper figure already came from v5, but about 20 numbers in
the text existed only as hand calculations, and the build script missed 7 of
the paper's outputs. This change fixes the code side only. **The paper is
unchanged**, so its text errors (sweep §A, §B, §E) are still open.

The rule now enforced: **a number the paper quotes is a file, not a
calculation.**

## The script

`scripts/analysis/v5/create_paper_artifacts.sh` builds exactly what
`cbg-benchmark-paper` cites, section by section in the paper's order:

1. prerequisites
2. §3 methodology
3. §4 error distance
4. §5.1–5.4 region classification
5. §6 overhead
6. §7 comprehensive evaluation
7. §8 discussion
8. appendix

Each command is commented with the figure, table or sentence it backs. The
runs are fixed:
- the meshes `pro-as0{1,2,3}-mesh`;
- their LOSO twins `pro-as0{1,2,3}-loso`;
- the RIPE mesh, `ripe-asmix-mesh`.

```bash
./scripts/analysis/v5/create_paper_artifacts.sh                      # into outputs/analysis/v5
./scripts/analysis/v5/create_paper_artifacts.sh --analysis-root DIR  # a sandbox
./scripts/analysis/v5/create_paper_artifacts.sh --check-figs ../cbg-benchmark-paper/figs
./scripts/analysis/v5/create_paper_artifacts.sh --copy-figs  ../cbg-benchmark-paper/figs
```

The script's `FIGS` table is the only place where paper figure names are
mapped to artifact paths. `--check-figs` compares the files byte for byte, and
`--copy-figs` copies them in under the paper's names.

**Verified:**
- **Empty analysis root:** all 39 commands succeed, and all 17 v5 figures are
  byte-identical to `cbg-benchmark-paper/figs`.
- **Tables:** the rebuilt tables match the existing artifacts row for row, so
  no paper number moved.
- **Real tree:** the script was then run on `outputs/analysis/v5`, with the
  same result.
- **Tests:** v5 has 1,085 tests and all pass, 15 of them new
  (`tests/test_paper_readings.py`).

The healpix partition figure is not built: it is Górski et al.'s, and its
table is plain arithmetic.

## Where each hand calculation now lives

| Paper | Reading | File | Command |
|---|---|---|---|
| §3 | VP fleet and that every network shares it; TGs measured by fewer than all VPs; replicas per site; sites sharing a seed (12%); K-fold and LOSO fold layout; D against the largest error; filter-removal estimate (2.4 / 6.4 / 0.25 / 0.43%) | `_cross/dataset/<hash>/dataset.csv` | **new** `report-dataset` |
| §4 | the p99 column; pairwise ratios per percentile (6.8× at p25, 1.8× at p99); p50/p5 spread (SPO 3.4×, OCT-H 41×) | `error_cdf.pooled.norm.cut{,.ratios}.csv` | `plot-error-cdf` |
| §4 | shares of TGs at 3.4, 22.8 and 228×10⁻³; d_sp/d_geo median ratio (8.4×) | `vp_distance_cdf.norm.shares.csv`, manifest `median_ratio` | `plot-vp-distance-cdf --share-at` |
| §5.1 | per-network gaps between methods, unrounded pp (SPO−OCT-H in AS-A = 6.3) | `outcome_bars[.pooled].healpix-128.gaps.csv` | `plot-outcome-bars` |
| §5.1 | sites on the wrong side of the RTT split, with cluster and RTT range | `x_cell_rtt.exceptions.csv` | `plot-x-cell-rtt` |
| §5.1 | S-P on each side of has-X/no-X: correct share, wrong answers in an interconnect cell, misses per run and cluster, near-tied VPs within 1 ms | `sides` in `sp_pni_cells.report.json`; `n_near_tie` per TG | `report-sp-pni-cells` |
| §5.2 | misses within S-P's correct set (SOI 0.16%, OCT-H 2.4% at 3% of sites); CBG-adds share (41.9%); family-only cohorts with their has-X share (Octant family 98.7% no-X); all-or-nothing sites | `correct_upset[.pooled].cell.nesting.csv` | `plot-correct-upset` |
| §5.3 | bounded accuracy; each method's correct TGs by pixel distance (SPO beyond 2 px 59%) | `accuracy_bounded`, `of_correct_*` in the outcome-bars CSV | `plot-outcome-bars` |
| §5.4 | own-pixel share per regime and network (OCT-H in AS-A 57→17%); bounded accuracy per regime; share of the gain over S-P kept under LOSO (OCT-H on no-X: 0.43) | `own_pixel_*`, `acc_bounded_*`, `gain_*` in `loso_delta*.csv` | `report-loso-delta` |
| §6 | batch budget for 1M IPs on 32 cores (mean runtime), per network and pooled; mean/p50; MTL share; worker memory as heap × cores and RSS × cores | `cost_box.pooled.heap.extrapolation.csv` | `plot-cost-box --extrapolate-tgs --cores` |
| §7 | lead over S-P, best−worst spread, frontier steps; pairwise pp and runtime ratios (OCT-H vs SPO 29×) | `pareto.csv`, `pareto.pairs.csv` | `plot-pareto` |
| App. B | geometric-centroid ablation: Δ accuracy, sites that flip, displacement, difference between drops, EST runtime ratio | `_cross/variant-delta/<hash>/variant_delta*.csv` | **new** `report-variant-delta` |

The new readings reproduce every hand-calculated value the sweep recomputed.

## Two new commands

**`report-dataset --run-id ... [--pair BASE:LOSO] [--replicas-per-site 20]`**
(`modules/dataset_summary.py`). It writes one row per run and one pooled row.
No pre-filter CSV survives, because the final CSVs are reconstructions. The
filter-removal share is therefore an estimate that assumes each site started
with `--replicas-per-site` replicas, and its columns are named `est_*`.

**`report-variant-delta --pair BASE:LOSO [--variant ORIGINAL=VARIANT]`**
(`modules/variant_delta.py`). It compares a variant that differs in one phase
against its original, under both seen and unseen sites. The default compares
OCT-H and OCT-S with their `_geo` twins.

- **Scoring:** both arms are scored in memory with the v5 scorer, because no
  config's `combo_ids` includes the variants.
- **Check:** the module confirms its scoring reproduces `classify`'s parquet
  for the original.
- **Refusals:** it refuses arms whose TGs, folds or `run.json` differ beyond
  the swapped phase.
- **Statistics:** none are bootstrapped, as in `report-loso-delta`; site
  counts sit beside every share instead.

It replaces `tasks/20261004-geo-centroid-variants/compare_geo.py`. That script
no longer runs: it imports a `site_bootstrap` that `loso_delta` dropped in
94d619c.

## Other changes

- **Configs:** built from scratch, `classify` scores every combo on disk,
  including the `_geo` variants. The outcome bars then drew eight methods,
  which was caught by `--check-figs`. `analysis.plot-outcome-bars.combo_ids`
  now names the paper's six methods in all six pro configs. The script passes
  the same six to `report-loso-delta`, which has no config block.
- **Stale strings:**
  - the stability caveat now quotes the live `spread_max` (SPO 15.8, OCT-H 8.9)
    instead of a hard-coded "6.5";
  - the `figure_pareto` docstring no longer says peak heap stays under 25 MB;
  - `plot-ripe-vs-databases --help` now documents `cut`.
- **README:** a new section, "The paper's artifacts", with the table above.

## Still open

- **Paper text vs the new artifacts**, from sweep §A. The code reproduces the
  real values; the paper still states the slipped ones:
  - d_geo median 3.13, not 3.4;
  - Δ_VP median 25.9, not 22.8;
  - ρ +0.81, not +0.82;
  - C3 interconnect distance 4.26, not 4.6;
  - SPO−OCT-H gap in AS-A 6.3 pp, not 7;
  - largest SOI−S-P gap 2.5 pp, not "at most 2";
  - bounded leads over S-P 0.16 / 0.47 pp, not 0.1 / 0.4.

  There is also one appendix mismatch. The paper says accuracy "changes at no
  more than 5% of the sites in every comparison". That holds pooled (≤4.6%),
  but per network the swap changes accuracy at up to 9.1% of sites
  (`variant_delta.csv`, `sites_flip_pct`).
- **Description errors and confidentiality**, sweep §B and §E: Octant's
  face-selection wording, the filter description, the "one week" window, the
  testbed spec, D being recoverable from the §4 thresholds, and the absolute
  site counts in §5.
- **Command line in manifests**, sweep §D 2: not done. The script is the
  record of the invocations.
- **Side effect:** each per-run `classify/` now also holds the `_geo` parquets.
  A `create_analysis_artifacts.sh` figure whose config block is empty, such as
  `plot-outcome-map: {}`, will draw them on its next run.

## Follow-up: group files (same day)

A pooled command's flags, and the values that belong to the set of runs, are
now declared once in `configs/groups/pro-paper.yaml`.

- **`analysis.<command>`** is that command's CLI flags, under their command-line
  names. `--group pro-paper`, before or after the command, loads them as
  defaults. A flag on the line still wins, and a key that is not a flag is
  refused. `@seen` expands to the seen runs, and `@seen:@unseen` zips them
  into pairs.
- **`analysis.common`** holds the bounds, D = 4,387.257 km and R_max =
  92.395 ms. It merges into every member run's lookups; a run config that
  also declares a key must agree exactly, or the lookup raises.
- **New command, `report-bounds`.** It computes both bounds from the data
  (D over the VP and site coordinates, R_max the largest per-pair minimum
  RTT, which is AS-B's) and fails if any run's resolved declaration
  disagrees. On the three meshes both match.
- **`create_paper_artifacts.sh` takes its run lists from the group's roles.**
  Every pooled step is `<command> --group pro-paper`, and `report-bounds`
  runs first.
  - From an empty root: 39 commands, all 17 figures byte-identical.
  - Tests: 1,095 pass, 10 of them new (`tests/test_groups.py`).
- **Not done.** The bounds are still also copied in the seven run configs.
  They agree, so nothing breaks. Deleting them there would make the group
  file the only source.

### Pooled directories named after the group

When a command runs with `--group`, a pool whose runs are exactly a set of the
group's roles lands in `_cross/<kind>/<id>[.<role>]/` instead of
`<n>-runs-<hash>/`.

| Pool | Directory |
|---|---|
| the meshes | `pro-paper.seen` |
| meshes + LOSO (pareto, loso-delta, dataset, variant-delta) | `pro-paper` |
| any other run set | hashed, as before |

- **Membership guard.** These names are not content-addressed, so a directory
  whose `runs.json` lists other runs (the membership was edited) is refused.
- **Paths in the script.** `create_paper_artifacts.sh` builds its `FIGS` paths
  from the same rule.
- **Checked.** From an empty root, no hash directory is created, all 17
  figures are identical, and 1,097 tests pass.
- **Old directories.** The hash-named directories already in
  `outputs/analysis/v5/_cross` stay until deleted. A rerun of the script writes
  the group-named ones beside them.
