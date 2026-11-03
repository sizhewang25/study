"""Cut a seeded share of TGs in a traffic-weighted CSV down to 1 or 2 flows.

Test-fixture generator, like `assign_random_weights`, and its companion. That
fixture's uniform weights leave every TG with most of its VPs (a 0.95 cut keeps
~78% of flows evenly), so nothing downstream ever sees the sparse targets real,
heavy-tailed traffic produces. This one makes them on purpose: it takes a
precomputed traffic-weighted CSV (`filter_weighted_flows`' output) and, for a
seeded random share of its TGs, keeps only that TG's `k` heaviest flows.

* `--single-frac` of the TGs keep 1 flow, `--double-frac` keep 2, the rest are
  untouched. The two sets are disjoint and drawn per TG, not per site: replicas
  of one site carry different traffic, so they can land in different sets.
* "Heaviest" is by the CSV's `weight`, ties broken by the seed. With
  `assign_random_weights`' uniform weights that is a VP chosen at random with
  respect to geometry -- so a single-VP TG's lone VP is usually *not* its
  nearest one, which is what makes Shortest Ping miss there.
* The output is a subset of the input's flows, hence of the mesh's, so it
  passes `TrafficWeightedCSVSource`'s precomputed-mode subset check. The fit
  side never reads it: only what is scored gets sparse.

Writes `<stem>.sparse.traffic-weighted.csv` beside the input (the input's
`.traffic-weighted` tag moved after `.sparse`), and a `.summary.json` with the
counts per set.

CLI::

    .venv/bin/python -m scripts.processing.source.sparsify_weighted_flows \\
        --input datasets/final/<stem>.traffic-weighted.csv --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TARGET = "target_id"
_WEIGHT = "weight"
_IN_SUFFIX = ".traffic-weighted"
_OUT_SUFFIX = ".sparse.traffic-weighted.csv"


def default_output(input_path: Path) -> Path:
    stem = input_path.stem
    stem = stem[: -len(_IN_SUFFIX)] if stem.endswith(_IN_SUFFIX) else stem
    return input_path.with_name(stem + _OUT_SUFFIX)


def sparsify(
    df: pd.DataFrame,
    *,
    seed: int,
    single_frac: float = 0.15,
    double_frac: float = 0.10,
) -> tuple[pd.DataFrame, dict]:
    """`(subset, summary)`: the chosen TGs cut to their 1 or 2 heaviest flows."""
    for name, v in (("single_frac", single_frac), ("double_frac", double_frac)):
        if not 0 <= v <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    if single_frac + double_frac > 1:
        raise ValueError(f"single_frac + double_frac must be <= 1, got {single_frac + double_frac}")
    if _WEIGHT not in df.columns:
        raise ValueError(f"input has no {_WEIGHT!r} column; 'heaviest flow' is undefined")

    rng = np.random.default_rng(seed)
    tgs = np.array(sorted(df[_TARGET].astype(str).unique()))
    order = rng.permutation(len(tgs))
    n1, n2 = round(single_frac * len(tgs)), round(double_frac * len(tgs))
    keep_k = pd.Series(0, index=tgs)  # 0 = untouched
    keep_k.iloc[order[:n1]] = 1
    keep_k.iloc[order[n1 : n1 + n2]] = 2

    out = df.copy()
    out["_tie"] = rng.random(len(out))
    out["_rank"] = (
        out.sort_values([_WEIGHT, "_tie"], ascending=[False, True])
        .groupby(_TARGET, sort=False)
        .cumcount()
        .reindex(out.index)
    )
    k = out[_TARGET].astype(str).map(keep_k).to_numpy()
    kept = out[(k == 0) | (out["_rank"].to_numpy() < k)].drop(columns=["_tie", "_rank"])

    per_tg = kept.groupby(_TARGET).size()
    summary = {
        "seed": seed,
        "single_frac": single_frac,
        "double_frac": double_frac,
        "n_tgs": int(len(tgs)),
        "n_tgs_1_flow": int((per_tg == 1).sum()),
        "n_tgs_2_flows": int((per_tg == 2).sum()),
        "n_flows_in": int(len(df)),
        "n_flows_out": int(len(kept)),
        "rule": "chosen TGs keep their k highest-weight flows; ties broken by the seed",
    }
    return kept, summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", type=Path, required=True, help="A traffic-weighted CSV.")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--single-frac", type=float, default=0.15)
    p.add_argument("--double-frac", type=float, default=0.10)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out_path = args.output or default_output(args.input)
    kept, summary = sparsify(
        pd.read_csv(args.input), seed=args.seed,
        single_frac=args.single_frac, double_frac=args.double_frac,
    )
    summary["input"] = str(args.input)
    kept.to_csv(out_path, index=False)
    out_path.with_name(out_path.stem + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("wrote %s: %s", out_path, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
