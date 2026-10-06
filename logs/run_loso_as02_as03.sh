#!/usr/bin/env bash
set -euo pipefail
export PATH=$PWD/.venv/bin:$PATH
for r in 02 03; do
  c=configs/pro-as$r-loso.yaml
  echo "== pro-as$r-loso: inspection (target_space, eval_source)"
  python -m snakemake target_space eval_source -s scripts/benchmark/v2/inspect_dataset.smk --cores 1 --configfile $c
  echo "== pro-as$r-loso: benchmark"
  CBG_SKIP_INSPECT=1 ./cli.sh --configfile $c
  echo "== pro-as$r-loso: DONE"
done
echo "ALL DONE"
