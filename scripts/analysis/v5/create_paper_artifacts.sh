#!/usr/bin/env bash
#
# Build every v5 artifact the paper (cbg-benchmark-paper) cites, in the order
# the paper cites it.
#
#   ./create_paper_artifacts.sh                         # build into outputs/analysis/v5
#   ./create_paper_artifacts.sh --group ID              # another group file (default pro-paper)
#   ./create_paper_artifacts.sh --analysis-root DIR     # build somewhere else (a sandbox)
#   ./create_paper_artifacts.sh --check-figs DIR        # ...then compare the PDFs to the paper's figs/
#   ./create_paper_artifacts.sh --copy-figs DIR         # ...then copy the PDFs in, paper names
#
# The last step, publish-paper-figs, writes every paper figure under its paper
# name to outputs/paper/pdf/<name>.pdf, with its provenance and data manifest in
# outputs/paper/json/<name>.json (registry: modules/paper_figures.py).
#
# `create_analysis_artifacts.sh` builds the exploratory set for a group of
# runs; this script builds the paper's set and nothing else. Each section below
# is a paper section, and each command names the figure, table or sentence it
# backs, so "where does this number come from" is answered by reading down
# this file. A number the paper quotes that is not a file here is a bug in this
# script or in a module, not a calculation to redo by hand.
#
# Every pooled command takes its flags from the group file
# configs/groups/$GROUP.yaml (`--group`): which runs, which methods, which
# layout. This script only orders the calls and says what each backs; the
# flags live in one place. The runs are the group's `seen` (the three operator
# meshes) and `unseen` (their LOSO twins), plus the RIPE Atlas mesh of the
# discussion, which is a different dataset and not a group member.
#
# Prerequisite: the benchmark runs themselves (this script only reads them).
# Runtimes are quotable only from runs benchmarked after 2026-11-07, when the
# benchmark stopped timing under tracemalloc (3-7x slower); the `preflight`
# section's `check-timing` fails on older runs. Snakemake skips cells already
# on disk, so re-time an existing run with
#   snakemake -s scripts/benchmark/v2/Snakefile --configfile configs/<run>.yaml \
#       --forcerun run_combo -j <cores>
# Predictions are unchanged by the re-run; only the timing and heap columns are.
#
# Deliberately NOT `set -e`, as in create_analysis_artifacts.sh: failures are
# collected and summarised, and the script exits non-zero if any occurred.

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$PATH"     # the CLI is invoked as bare `python`

V5="python -m scripts.analysis.v5.cli"
GROUP=pro-paper
RIPE=ripe-asmix-mesh

# ---- arguments ----------------------------------------------------------------
ROOT_ARGS=()
CHECK_DIR=""
COPY_DIR=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --group)         GROUP=$2; shift 2 ;;
    --analysis-root) ROOT_ARGS=(--analysis-root "$2"); shift 2 ;;
    --check-figs)    CHECK_DIR=$2; shift 2 ;;
    --copy-figs)     COPY_DIR=$2; shift 2 ;;
    *)
      echo "create_paper_artifacts.sh: unknown argument: $1" >&2
      echo "usage: $0 [--group ID] [--analysis-root DIR] [--check-figs DIR | --copy-figs DIR]" >&2
      exit 2 ;;
  esac
done
ROOT=${ROOT_ARGS[1]:-outputs/analysis/v5}

# The group's runs, by role, read from the group file.
members() {
  python -c 'import sys
from scripts.analysis.v5.modules.labels import group_members, load_group
print("\n".join(group_members(load_group(sys.argv[1]), sys.argv[2])))' "$GROUP" "$1"
}
mapfile -t MESH < <(members seen) || exit 1
mapfile -t LOSO < <(members unseen) || exit 1
if [ "${#MESH[@]}" -eq 0 ] || [ "${#LOSO[@]}" -eq 0 ]; then
  echo "group $GROUP has no seen/unseen runs" >&2
  exit 1
fi

FAILED=()
N_OK=0
R=_setup

# run <label> <command...> -- execute, keep going on failure, record which.
run() {
  local label=$1; shift
  local rc=0
  "$@" ${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"} || rc=$?
  if [ "$rc" -eq 0 ]; then
    N_OK=$((N_OK + 1))
  else
    FAILED+=("$R :: $label")
  fi
  return "$rc"
}

# grun <command> -- one pooled command, flags from the group file.
grun() { run "$1" $V5 "$1" --group "$GROUP"; }

section() { R=$1; printf '\n==================== %s ====================\n' "$1"; }

# ---- prerequisites: every run's answer space and labels --------------------------
# Not a paper section: what every section below reads. Per run, so without
# --group (its blocks are the pooled calls). The PNI clusters and S-P cell
# report are built per run as well as pooled: report-loso-delta and
# report-variant-delta take each mesh's has-X flags from its per-run one.
# ---- preflight: runtimes must be timed without tracemalloc ---------------------------
# The cost boxes (§6) and the Pareto x axis (§7) quote runtimes. Runs benchmarked
# before 2026-11-07 timed every stage under tracemalloc, inflated 3-7x; this
# fails (and is listed in the summary) until those runs are re-run. Everything
# else still builds: accuracy does not depend on it.
section "preflight"
run check-timing $V5 check-timing $(printf -- '--run-id %s ' "${MESH[@]}" "${LOSO[@]}")

section "prerequisites"
for r in "${MESH[@]}" "${LOSO[@]}" "$RIPE"; do
  if run "build-answer-space $r" $V5 build-answer-space --run-id "$r"; then
    run "classify $r" $V5 classify --run-id "$r"
  fi
done
for r in "${MESH[@]}"; do
  if run "plot-pni-gap $r" $V5 plot-pni-gap --run-id "$r"; then
    run "report-sp-pni-cells $r" $V5 report-sp-pni-cells --run-id "$r"
  fi
done

# ---- §3 Methodology ---------------------------------------------------------------
section "§3 methodology"
# D and the largest RTT, computed from the data and checked against the
# group's analysis.common (every normalized axis below divides by them).
grun report-bounds
# The VP fleet and that it is shared, replicas per site, sites sharing a seed,
# the K-fold and LOSO fold layouts, D against the largest error, and the
# filter-removal estimate (2% / 6% / <1%).
grun report-dataset
# "At least one VP colocated with each interconnect": interconnect_coverage in
# sp_interconnect.report.json. Needs the pooled clusters, which §4's scatter
# and every pooled PNI step in §5 read too, so they are built here, first.
grun plot-pni-gap
grun plot-sp-interconnect

# ---- §4 Evaluation on error distance ------------------------------------------------
section "§4 error distance"
# Fig. error CDF and Table err-dist-percentiles (p5..p99); the ratios between
# methods at each percentile ("6.8x at p25", "1.8x at p99", "p50 = 3.4x p5")
# in error_cdf.pooled.norm.cut.ratios.csv.
grun plot-error-cdf
# Fig. VP-distance CDF; the shares the text reads off it (thresholds in the
# group file) in vp_distance_cdf.norm.shares.csv, the median ratio in the manifest.
grun plot-vp-distance-cdf
# No longer figures in the paper, but they still back numbers: the Delta_VP vs
# D_X-TG scatter (built above by plot-pni-gap; rho and clusters in
# pni_gap.manifest.json) and the RTT boxes per cluster.
grun plot-pni-cluster-rtt
# §4.2 (Understand the Baseline: Shortest Ping), all four PNGs from this one
# command, written beside the pooled clusters in _cross/pni-gap/<group>.seen/:
#   Fig. 5(a) sp_vp_scatter       S-P error vs. distance to the nearest VP
#   Fig. 5(b) sp_x_scatter        S-P error vs. distance to the nearest X (rho)
#   Fig. 6    sp_accuracy_by_x    S-P region classification, has-X vs. no-X
#   Fig. 7    sp_rtt_error_has_x  has-X TGs' min RTT vs. S-P error
# Shares, medians, the has-X misses (inflated RTT / near-tie) and the
# conjecture test are in sp_baseline.manifest.json.
grun plot-sp-baseline

# ---- §5 Evaluation on region classification -------------------------------------------
section "§5.1 unbounded accuracy and the interconnect"
# Fig. outcome bars, unbounded (§5.1) and bounded (§5.3), in one call;
# per-network gaps between methods ("SPO 6 pp ahead of OCT-H in AS-A") in the
# compare layout's .gaps.csv; own-pixel shares, the share of each method's
# correct TGs beyond 2 pixels and bounded accuracy in the pooled CSV.
grun plot-outcome-bars
# Fig. has-X / no-X min-RTT boxes, per network and pooled (x_cell_rtt.all.png, §5);
# the exception sites in x_cell_rtt.exceptions.csv.
grun plot-x-cell-rtt
# S-P on each side of the split, where its wrong answers land, and why it
# misses (sides in sp_pni_cells.report.json).
grun report-sp-pni-cells
# §4.2: a cause for every S-P prediction (RTT ignores distance / near-tie on
# has-X; X_in != X and the expected miss on no-X) and the nearest-VP RTT ratio
# per side, in sp_taxonomy.report.json. report-paper's sp-mechanism and
# sp-taxonomy blocks read it with the three S-P artifacts above.
grun report-sp-taxonomy
# §4.3: each method's region replayed (SOI, VAN, OCT-H, OCT-S; cached per TG), then
# Fig. x-side-regions: error distance by side, region area / D^2 and TG inside the
# region (numbers in x_side_regions.json).
grun report-mtl-replay
grun plot-x-side-regions
# §6.5: step and heading of each CBG method's no-X predictions off the ingress
# point (step_heading.<method>.png, numbers in step_heading.json).
grun plot-step-heading
# Appendix: each CBG method's three phases on one TG and VP (cbg_mech.<method>.<phase>.png),
# after checking that the all-VP rebuild reproduces the recorded predictions.
grun plot-cbg-mechanism
# Table x-cell (has-X / no-X accuracy, seen sites) and, in §5.4, Table
# seen-unseen: both from loso_delta.{csv,by_has_x.csv}.
grun report-loso-delta

section "§5.2 shared and single-method successes"
# Fig. correct UpSet and Table cell-correct-sites; nesting against S-P, the
# CBG-adds share, family-only cohorts with their has-X share, and the
# all-or-nothing site shares in correct_upset.pooled.cell.nesting.csv.
# Needs the pooled S-P cell report (above) for has_x_share.
grun plot-correct-upset

section "§5.3 unbounded cells and bounded accuracy"
# The three SPO-vs-OCT-H panels. The bounded outcome bars are built in §5.1.
grun plot-exclusive-error
grun plot-stability
grun plot-peripherality

section "§5.4 seen vs unseen sites"
# Table seen-unseen, built in §5.1 (report-loso-delta); own-pixel shares per
# network and the no-X gain kept (gain_kept) are in the same files.
echo "(report-loso-delta, built in §5.1)"

# ---- §6 Evaluation on overhead ---------------------------------------------------------
section "§6 overhead"
# Fig. cost boxes; the 1M-IP batch budget on 32 cores (core-hours, wall clock,
# per network), mean/p50, MTL share and the worker memory budgets in
# cost_box.pooled.heap.extrapolation.csv.
grun plot-cost-box

# ---- §7 Comprehensive evaluation --------------------------------------------------------
section "§7 comprehensive"
# The four Pareto panels and Table sota; leads over S-P, best-worst spreads,
# frontier steps (pareto.csv) and pairwise differences / runtime ratios
# (pareto.pairs.csv).
grun plot-pareto

# ---- §8 Discussion ------------------------------------------------------------------------
section "§8 discussion"
# Fig. RIPE Atlas methods vs commercial databases. Not a group member.
run plot-ripe-vs-databases $V5 plot-ripe-vs-databases --run-id "$RIPE" --unanswered cut

# ---- Appendices ---------------------------------------------------------------------------
section "appendix"
# A, HEALPix: the partition figure is Gorski et al.'s and the table is
# arithmetic -- nothing to build.
# B, CBG implementation: the geometric-centroid ablation (accuracy change,
# flipped sites, displacement, drop difference, EST runtime ratio).
grun report-variant-delta

# ---- the data behind every section, as the paper writes it -------------------------
section "data report"
# One table per paper figure / table, every value formatted once from its
# unrounded artifact (paper_numbers.s<N>.md and paper_numbers.md under
# _cross/paper/<group>/). Reads only what the sections above built.
grun report-paper
# The text's numbers report-paper does not print (ratios, §6.4-6.5 replay and
# step/heading shares, §7 budget words, §8 ratios, App. D), by paper section:
# _cross/paper/<group>/paper_stats.json. A one-off for the current draft.
run paper-stats python -m scripts.analysis.v5.paper_stats --group "$GROUP"

# ---- the paper's figures --------------------------------------------------------------------
# paper name -> artifact lives in modules/paper_figures.py (PAPER_FIGURES), the one
# place the mapping is kept. Writes outputs/paper/{pdf,json}/<name>.*; with
# --copy-figs also copies the PDFs into DIR, with --check-figs only compares.
section "figures"
PUB_ARGS=()
[ -n "$COPY_DIR" ] && PUB_ARGS+=(--copy-to "$COPY_DIR")
[ -n "$CHECK_DIR" ] && PUB_ARGS+=(--check "$CHECK_DIR")
run publish-paper-figs $V5 publish-paper-figs --group "$GROUP" ${PUB_ARGS[@]+"${PUB_ARGS[@]}"}

printf '\n==================== summary ====================\n'
printf 'commands ok: %d   failed: %d\n' "$N_OK" "${#FAILED[@]}"
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\nfailed:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
