#!/usr/bin/env bash
#
# Production-VM entry point: build a group's paper artifacts, its numbers and
# figures, and pack them into one tarball to bring back.
#
#   ./create_paper_bundle.sh --group pro-paper                 # mesh machine
#   ./create_paper_bundle.sh --group pro-tf                    # TF machine
#   ./create_paper_bundle.sh --group G --skip-artifacts        # artifacts already built
#   ./create_paper_bundle.sh --group G --analysis-root DIR     # artifacts elsewhere
#
# Steps:
#   1. create_paper_artifacts.sh (a group with an `unseen` role) or
#      create_tf_artifacts.sh (otherwise); both end with publish-paper-figs,
#      which writes outputs/paper/{pdf,json}/<paper name>.*. With
#      --skip-artifacts, only publish-paper-figs runs.
#   2. paper_values.generate -> outputs/paper/values/<group>.json: every slug
#      the template prints (§1-§8 and Apps. E-G; registry in
#      paper_values/{mesh,tf,derived}.py) and every claim of paper/claims.py.
#      paper/ is not in git: without paper/claims.py on this machine the claims
#      are not checked against its data (a warning, and MANIFEST says so).
#   3. tar outputs/paper/{pdf,json}/<this group's figures> and values/<group>.json,
#      plus MANIFEST.json (commit, failures), into
#      outputs/paper/bundle.<group>.<utc>.tar.gz.
#
# Bring the tarball back and unpack it where the template lives:
#   tar -xzf bundle.<group>.<utc>.tar.gz -C outputs/paper
# Bundles of different groups never share a file name, so both unpack into
# one folder. Then render (see paper/README.md):
#   python -m scripts.analysis.v5.paper_values.render --build --zip
# Every generated number prints inside \new{} (blue) so it stands out from the
# hand-written text; add --no-mark for the plain submission copy.
#
# Not `set -e`: failures are collected, written to MANIFEST.json, and the
# script exits non-zero after packing whatever was built.

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
cd "$REPO" || exit 1
export PATH="$REPO/.venv/bin:$PATH"

GROUP=""
SKIP=0
ROOT_ARGS=()
while [ "$#" -gt 0 ]; do
  case "$1" in
    --group)          GROUP=$2; shift 2 ;;
    --skip-artifacts) SKIP=1; shift ;;
    --analysis-root)  ROOT_ARGS=(--analysis-root "$2"); shift 2 ;;
    *)
      echo "usage: $0 --group ID [--skip-artifacts] [--analysis-root DIR]" >&2
      exit 2 ;;
  esac
done
[ -n "$GROUP" ] || { echo "--group is required" >&2; exit 2; }

FAILED=()
step() { printf '\n==================== %s ====================\n' "$1"; }

HAS_UNSEEN=$(python -c 'import sys
from scripts.analysis.v5.modules.labels import load_group
print(1 if "unseen" in (load_group(sys.argv[1]).get("runs") or {}) else 0)' "$GROUP") || exit 1

step "artifacts ($GROUP)"
if [ "$SKIP" -eq 1 ]; then
  python -m scripts.analysis.v5.cli publish-paper-figs --group "$GROUP" ${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"} \
    || FAILED+=("publish-paper-figs")
elif [ "$HAS_UNSEEN" -eq 1 ]; then
  scripts/analysis/v5/create_paper_artifacts.sh --group "$GROUP" ${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"} \
    || FAILED+=("create_paper_artifacts.sh (see its summary above)")
else
  scripts/analysis/v5/create_tf_artifacts.sh --group "$GROUP" ${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"} \
    || FAILED+=("create_tf_artifacts.sh (see its summary above)")
fi

step "numbers ($GROUP)"
CLAIMS_FILE=paper/claims.py
if [ ! -f "$CLAIMS_FILE" ]; then
  echo "warning: $CLAIMS_FILE is absent (paper/ is not in git): claims are not checked on this machine" >&2
fi
python -m scripts.analysis.v5.paper_values.generate --group "$GROUP" ${ROOT_ARGS[@]+"${ROOT_ARGS[@]}"} \
  || FAILED+=("paper_values.generate (missing slugs listed above)")

step "bundle"
OUT=outputs/paper
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BUNDLE="$OUT/bundle.$GROUP.$STAMP.tar.gz"
mapfile -t NAMES < <(python -c 'import sys
from scripts.analysis.v5.modules.paper_figures import entries
print("\n".join(e.name for e in entries(sys.argv[1])))' "$GROUP")
FILES=()
for n in "${NAMES[@]}"; do
  for f in "pdf/$n.pdf" "json/$n.json"; do
    if [ -f "$OUT/$f" ]; then FILES+=("$f"); else FAILED+=("bundle: $f not built"); fi
  done
done
if [ -f "$OUT/values/$GROUP.json" ]; then FILES+=("values/$GROUP.json"); else FAILED+=("bundle: values/$GROUP.json"); fi

python - "$OUT/MANIFEST.$GROUP.json" "$GROUP" "$STAMP" "$CLAIMS_FILE" "${FAILED[@]}" <<'EOF'
import json, sys
from pathlib import Path
from scripts.analysis.v5.modules.figure_io import git_state
path, group, stamp, claims, *failed = sys.argv[1:]
json.dump({"group": group, "created_utc": stamp, "git": git_state(), "failures": failed,
           "claims_checked": Path(claims).is_file()},
          open(path, "w"), indent=1)
EOF
FILES+=("MANIFEST.$GROUP.json")
tar -czf "$BUNDLE" -C "$OUT" "${FILES[@]}" && echo "wrote $BUNDLE (${#FILES[@]} files)"

printf '\n==================== summary ====================\n'
if [ "${#FAILED[@]}" -gt 0 ]; then
  printf 'failed:\n'; printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
echo "ok"
