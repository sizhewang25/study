# Task: Geometric-centroid variants of OCT-H / OCT-S vs the originals

## Background
`octant_cbg_hull_geo` and `octant_cbg_spl_geo` are OCT-H and OCT-S with one
change: the CTR step uses `geometric_centroid` instead of `monte_carlo_medoid`.
Calibration (LTD) and the feasible region (MTL) are the same, so any difference
is the centroid step alone. Both ran on the three pro-as meshes (seen sites,
5 folds) and their LOSO twins (unseen sites, 20/22/23 folds) on 2026-10-04.

The paper plan (`notes/2026-10-04-paper-eval-flow-plan.md`, §6.5) lists them as
an optional ablation: do they separate what calibration buys from what the
centroid step buys, especially under LOSO where OCT-H loses ~16 pp?

## Goals
- Measure the paired difference GEO − original in error distance and in cell
  (serving-region) classification accuracy, per network and pooled, under both
  regimes.
- Say whether the difference is separable (site-clustered CI) and whether it
  changes the K-fold → LOSO drop.

## Requirements
- Read-only on benchmark outputs. Do not add the `_geo` combos to any config's
  `combo_ids`, and do not regenerate or write any figure (user, 2026-10-04).
- Write nothing under `outputs/analysis/v5/` — scratch work goes to the session
  scratchpad or `tasks/20261004-geo-centroid-variants/`.
- Same denominators and conventions as v5 `classify` / `report-loso-delta`.

## Success Criteria
- [x] `report.md` with the paired tables and a verdict for each of OCT-H, OCT-S
- [x] Invariants checked (same TGs, same status/unanswered per pair)
- [x] Recommendation: §6.5 ablation, appendix, or drop

## Related Files
- [loso_delta.py](../../scripts/analysis/v5/modules/loso_delta.py)
- [classify.py](../../scripts/analysis/v5/modules/classify.py)
- [answer_space.py](../../scripts/analysis/v5/modules/answer_space.py)
- [methods.py](../../scripts/analysis/v5/modules/methods.py)
- [paper eval-flow plan](../../notes/2026-10-04-paper-eval-flow-plan.md)
