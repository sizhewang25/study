# Octant with a geometric centroid: the CTR step is not where OCT-H loses

**Date:** 2026-10-04
**Task:** [tasks/20261004-geo-centroid-variants/](../tasks/20261004-geo-centroid-variants/)
(full tables in `report.md`; backing script `compare_geo.py`)
**Prior notes:** [2026-10-04-allfail-cohort-label-identifiability.md](2026-10-04-allfail-cohort-label-identifiability.md)
(the LOSO result), [2026-10-04-paper-eval-flow-plan.md](2026-10-04-paper-eval-flow-plan.md)
(§6.5 lists these runs as an optional ablation)

## Question

OCT-H loses 15.8 pp of cell accuracy from seen sites (K-fold) to unseen sites
(LOSO). Calibration (LTD), the feasible region (MTL) and the point picked inside
it (CTR) could each carry part of that loss. Swapping only the CTR step,
`monte_carlo_medoid` -> `geometric_centroid`, isolates the last one. Does it
change accuracy, error distance, or the drop?

## Setup

- New combos `octant_cbg_hull_geo` (OCT-H-GEO) and `octant_cbg_spl_geo`
  (OCT-S-GEO): same LTD and MTL kwargs as OCT-H / OCT-S, `ctr:
  geometric_centroid`. Added to `configs/pro-as0{1,2,3}-{mesh,loso}.yaml` and
  run on all folds (mesh 5; LOSO 20/22/23). No existing combo was rerun.
- Scored in memory with the v5 scorer against each run's existing nside-128
  answer space; re-scoring the originals reproduces the stored v5 outputs
  exactly. Pooled over AS-A/B/C, paired site-clustered bootstrap (2,000 reps).
- Invariants: identical TGs per fold, identical `run.json` apart from the CTR,
  identical `status` per TG (all SUCCESS, 0% unanswered in both arms).

## Result

| Regime | Pair | Cell accuracy | Δ cell, pp [CI] | Δ p50, km [CI] |
|---|---|---|---|---|
| Seen sites | OCT-H | 73.5 -> 73.8% | +0.32 [-0.16, +1.10] | +1.1 [-2.2, +9.1] |
| Seen sites | OCT-S | 69.1 -> 69.1% | 0.00 | +0.1 [-9.9, +3.0] |
| Unseen sites | OCT-H | 57.8 -> 57.7% | -0.08 [-0.24, 0.00] | -0.3 [-11.6, +1.2] |
| Unseen sites | OCT-S | 60.2 -> 59.0% | -1.18 [-4.23, +0.70] | -0.2 [-5.6, +7.6] |

- **Not separable anywhere.** Every CI spans 0, ring-bounded accuracy (<=1,
  <=2 rings) moves by at most 0.7 pp, and every change is on no-X TGs (has-X
  Δ is 0.00 in all four comparisons).
- **The LOSO drop is unchanged.** OCT-H drops 16.15 pp with GEO vs 15.76 pp
  with the medoid; diff-in-diff -0.39 pp [-1.18, +0.16]. The loss sits in
  LTD/MTL, not CTR.
- **Answers barely move.** Median displacement between the two arms is
  0.4-0.9 km (p90 6-12 km); 1-6 TGs flip per comparison, at <= 3 sites.
- **The one larger move is a cell-edge artefact.** LOSO OCT-S -1.18 pp is
  one no-X AS-B site: both arms place its TGs ~220 km off, and a ~30 km shift
  crosses a nearest-seed boundary. Its <=2-ring accuracy is unchanged.
- **Cost is the only real difference.** CTR falls from ~470 ms to ~0.4 ms per
  TG (~1,000x). Totals are partly confounded by run timing (identical MTL
  differs 4,512 vs 3,989 ms p50), so only the CTR column measures the swap.

This does not contradict the earlier "MTL+CTR is regime-dependent" finding,
which compared different CTR families; medoid and geometric centroid land on
nearly the same point inside an Octant region.

## Implication for the paper

- §6.5 does not need an ablation: one sentence stating that the centroid step
  explains none of the LOSO drop, with a pointer to the appendix.
- Appendix: that sentence plus the CTR cost line.
- The `_geo` combos stay out of every figure's `combo_ids`.
