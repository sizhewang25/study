#!/usr/bin/env bash
# Run only the new *_geo combos (and re-summarize) for the six pro-as configs.
# --rerun-triggers mtime input: ignore the stale params record on the mesh
# octant_cbg_spl jobs, but still pick up summarize's new inputs.
set -euo pipefail
export PATH=$PWD/.venv/bin:$PATH
for c in pro-as01-mesh pro-as02-mesh pro-as03-mesh pro-as01-loso pro-as02-loso pro-as03-loso; do
  echo "== $c: start $(date +%T)"
  CBG_SKIP_INSPECT=1 ./cli.sh --configfile configs/$c.yaml --rerun-triggers mtime input
  echo "== $c: DONE $(date +%T)"
done
echo "ALL DONE"
