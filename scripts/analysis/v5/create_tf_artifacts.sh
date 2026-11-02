#!/usr/bin/env bash
#
# Build the traffic-filtered (TF) figures: error CDF, outcome bars (unbounded
# and bounded) and Pareto frontier (unbounded and bounded), seen sites only.
#
#   ./create_tf_artifacts.sh                         # build into outputs/analysis/v5
#   ./create_tf_artifacts.sh --group ID              # another group file (default pro-tf)
#   ./create_tf_artifacts.sh --analysis-root DIR     # build somewhere else (a sandbox)
#
# Run on the TF machine: the pro-as0X-weighted runs and their source CSVs are
# there, and the committed configs carry placeholder paths. Flags live in
# configs/groups/$GROUP.yaml; this script only orders the calls.
#
# None of the three figures opens the edge CSV, so a TF run filtering on the
# fly needs no materialized weighted CSV here. build-answer-space does need
# the run's pre-filter mesh (`mesh_csv_path`): a TF run's cells are its mesh's.
#
# Deliberately NOT `set -e`, as in create_paper_artifacts.sh: failures are
# collected and summarised, and the script exits non-zero if any occurred.

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$PATH"     # the CLI is invoked as bare `python`

V5="python -m scripts.analysis.v5.cli"
GROUP=pro-tf

ROOT_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --group)         GROUP=$2; shift 2 ;;
    --analysis-root) ROOT_ARGS=(--analysis-root "$2"); shift 2 ;;
    *)
      echo "create_tf_artifacts.sh: unknown argument: $1" >&2
      echo "usage: $0 [--group ID] [--analysis-root DIR]" >&2
      exit 2 ;;
  esac
done

mapfile -t SEEN < <(python -c 'import sys
from scripts.analysis.v5.modules.labels import group_members, load_group
print("\n".join(group_members(load_group(sys.argv[1]), "seen")))' "$GROUP") || exit 1
if [ "${#SEEN[@]}" -eq 0 ]; then
  echo "group $GROUP has no seen runs" >&2
  exit 1
fi

FAILED=()
N_OK=0

run() {
  local label=$1; shift
  local rc=0
  "$@" ${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"} || rc=$?
  if [ "$rc" -eq 0 ]; then N_OK=$((N_OK + 1)); else FAILED+=("$label"); fi
  return "$rc"
}
grun() { run "$1" $V5 "$1" --group "$GROUP"; }
section() { printf '\n==================== %s ====================\n' "$1"; }

# Per run, without --group: every figure below reads each run's labels.
section "prerequisites"
for r in "${SEEN[@]}"; do
  if run "build-answer-space $r" $V5 build-answer-space --run-id "$r"; then
    run "classify $r" $V5 classify --run-id "$r"
  fi
done

section "figures"
grun plot-error-cdf       # error_cdf.pooled.norm.cut
grun plot-outcome-bars    # outcome_bars.unbounded.pooled, outcome_bars.pooled (bounded)
grun plot-pareto          # pareto.cell.seen (unbounded), pareto.bounded.seen

printf '\n==================== summary ====================\n'
printf 'commands ok: %d   failed: %d\n' "$N_OK" "${#FAILED[@]}"
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf '\nfailed:\n'
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
