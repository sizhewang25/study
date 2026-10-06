#!/usr/bin/env bash
#
# Build every v5 artifact the paper (cbg-benchmark-paper) cites, in the order
# the paper cites it.
#
#   ./create_paper_artifacts.sh                         # build into outputs/analysis/v5
#   ./create_paper_artifacts.sh --group ID              # another group file (default pro-paper)
#   ./create_paper_artifacts.sh --analysis-root DIR     # build somewhere else (a sandbox)
#   ./create_paper_artifacts.sh --check-figs DIR        # ...then compare to the paper's figs/
#   ./create_paper_artifacts.sh --copy-figs DIR         # ...then copy them in, paper names
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
# Fig. Delta_VP vs D_X-TG scatter (rho, clusters: pni_gap.manifest.json) --
# built above -- and Fig. RTT boxes per cluster.
grun plot-pni-cluster-rtt

# ---- §5 Evaluation on region classification -------------------------------------------
section "§5.1 unbounded accuracy and the interconnect"
# Fig. outcome bars, unbounded (§5.1) and bounded (§5.3), in one call;
# per-network gaps between methods ("SPO 6 pp ahead of OCT-H in AS-A") in the
# compare layout's .gaps.csv; own-pixel shares, the share of each method's
# correct TGs beyond 2 pixels and bounded accuracy in the pooled CSV.
grun plot-outcome-bars
# Fig. has-X / no-X min-RTT boxes; the exception sites in x_cell_rtt.exceptions.csv.
grun plot-x-cell-rtt
# S-P on each side of the split, where its wrong answers land, and why it
# misses (sides in sp_pni_cells.report.json).
grun report-sp-pni-cells
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

# ---- the paper's figures --------------------------------------------------------------------
# paper name -> artifact under the analysis root. The one place the mapping lives.
X3=$(python -c 'import sys;from scripts.analysis.v5.modules.cross import cross_name;print(cross_name(sys.argv[1:]))' "${MESH[@]}")
X6=$(python -c 'import sys;from scripts.analysis.v5.modules.cross import cross_name;print(cross_name(sys.argv[1:]))' "${MESH[@]}" "${LOSO[@]}")
FIGS=(
  "error_cdf.pooled.norm.cut|_cross/classify/$X3/error_cdf.pooled.norm.cut.png"
  "vp_distance_cdf|_cross/vp-distance-cdf/$X3/vp_distance_cdf.norm.png"
  "vp_pni_proximity_scatter|_cross/pni-gap/$X3/pni_gap_scatter.norm.png"
  "vp_pni_proximity_rtt|_cross/pni-gap/$X3/pni_cluster_rtt.png"
  "outcome_bars.unbounded.pooled|_cross/classify/$X3/outcome_bars.unbounded.pooled.healpix-128.png"
  "x_cell_rtt|_cross/pni-gap/$X3/x_cell_rtt.png"
  "correct_upset.pooled.cell|_cross/classify/$X3/correct_upset.pooled.cell.png"
  "spo_vs_octh_exclusive_pixel_cdf|_cross/exclusive-error/$X3/exclusive_error_cdf.spo_vs_octh.png"
  "spo_vs_octh_replica_spread|_cross/stability/$X3/paired_std_grid_offset.spo_vs_octh.png"
  "spo_vs_octh_peripherality|_cross/peripherality/$X3/peripherality.SPO-vs-OCT-H.png"
  "outcome_bars.bounded.pooled|_cross/classify/$X3/outcome_bars.pooled.healpix-128.png"
  "cost_box.pooled.heap|_cross/cost/$X3/cost_box.pooled.heap.png"
  "pareto.cell.seen|_cross/pareto/$X6/pareto.cell.seen.png"
  "pareto.bounded.seen|_cross/pareto/$X6/pareto.bounded.seen.png"
  "pareto.cell.unseen|_cross/pareto/$X6/pareto.cell.unseen.png"
  "pareto.bounded.unseen|_cross/pareto/$X6/pareto.bounded.unseen.png"
  "ripe_vs_databases.cut|$RIPE/ripe-vs-databases/ripe_vs_databases.cut.png"
)

if [ -n "$CHECK_DIR" ] || [ -n "$COPY_DIR" ]; then
  section "figures"
  for entry in "${FIGS[@]}"; do
    name=${entry%%|*}
    src="$ROOT/${entry#*|}"
    if [ ! -f "$src" ]; then
      FAILED+=("figures :: $name (not built: $src)")
      continue
    fi
    if [ -n "$COPY_DIR" ]; then
      cp "$src" "$COPY_DIR/$name.png" && echo "copied  $name"
    else
      if [ ! -f "$CHECK_DIR/$name.png" ]; then
        FAILED+=("figures :: $name (absent from $CHECK_DIR)")
      elif cmp -s "$src" "$CHECK_DIR/$name.png"; then
        echo "same    $name"
      else
        FAILED+=("figures :: $name differs from $CHECK_DIR/$name.png")
      fi
    fi
  done
fi

printf '\n==================== summary ====================\n'
printf 'commands ok: %d   failed: %d\n' "$N_OK" "${#FAILED[@]}"
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\nfailed:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
