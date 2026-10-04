# Geometric-centroid variants vs the originals — Report

## Question
OCT-H-GEO and OCT-S-GEO are OCT-H and OCT-S with one change: the CTR step uses
`geometric_centroid` where the originals use `monte_carlo_medoid`. Does the swap
change cell accuracy or error distance, and does it change the K-fold → LOSO
drop? If so, the centroid step accounts for part of what OCT-H loses under LOSO.

## Method
- **Data.** `outputs/benchmark/v2/pro-as0{1,2,3}-{mesh,loso}`, all folds (mesh 5;
  LOSO 20/22/23), 1,269 TGs at 65 sites per regime. AS-A/B/C = as01/02/03.
- **Scoring.** Both arms were scored in memory with the v5 scorer
  (`classify.load_method_frame` + `score_method`) against each run's existing
  nside-128 answer space (grid ≈ 51 km). Nothing was written under `outputs/`.
- **Conventions** (v5 / `report-loso-delta`). The denominator is every evaluated
  TG. Unanswered rows (`solved_mask` false) count as wrong and are left out of
  the error percentiles. Cell accuracy is the unbounded nearest-seed
  `cell_label == correct`. Ring-bounded accuracy is cell-correct with grid
  offset ≤ k.
- **Bootstrap.** Paired and site-clustered: 2,000 reps, seed 0, 95% percentile
  CI. `loso_delta.site_bootstrap` is used with the original as "base" and GEO as
  the other arm. The ring-bounded CIs and the K-fold→LOSO diff-in-diff use the
  same resampling: a site is drawn with all its TGs, in both arms (and both
  regimes).
- **Sanity.** Re-scoring OCT-H and OCT-S in memory reproduces the stored v5
  `<method>_tgs.parquet` exactly in all 6 runs: 0 mismatches in `cell_label`,
  `status`, grid offset, seeds, `pred_lat/lon` or `pred_dist_to_tg_km`. The
  pooled originals match `6-runs-197fee` (OCT-H 73.5 → 57.8%, OCT-S 69.1 → 60.2%).
- **Invariants. All pass.**
  - Every fold has both arms (5/5/5 and 20/22/23).
  - The tg_id sets are identical per run and per fold.
  - Apart from `ctr`/`ctr_kwargs`, each fold's `run.json` matches between the
    arms (LTD, MTL, kwargs, seed).
  - `status` is identical for every TG: all 7,614 rows per arm are SUCCESS, so
    the unanswered share is 0% in both arms.

## Results

**Cell and ring-bounded accuracy, pooled over AS-A/B/C (GEO − original, pp, site CI)**

| Regime | Pair | Cell orig → GEO | Δ cell [CI] | ≤2 rings orig → GEO | Δ [CI] | ≤1 ring orig → GEO | Δ [CI] |
|---|---|---|---|---|---|---|---|
| K-fold | OCT-H | 73.52 → 73.84 | +0.32 [−0.16, +1.10] | 61.47 → 61.78 | +0.32 [0.00, +0.96] | 59.10 → 58.94 | −0.16 [−1.64, +0.94] |
| K-fold | OCT-S | 69.11 → 69.11 | 0.00 [0.00, 0.00] | 56.97 → 57.05 | +0.08 [0.00, +0.24] | 55.24 → 55.63 | +0.39 [0.00, +0.87] |
| LOSO | OCT-H | 57.76 → 57.68 | −0.08 [−0.24, 0.00] | 49.09 → 49.17 | +0.08 [0.00, +0.24] | 42.32 → 42.32 | 0.00 [0.00, 0.00] |
| LOSO | OCT-S | 60.20 → 59.02 | −1.18 [−4.23, +0.70] | 49.65 → 49.88 | +0.24 [0.00, +0.72] | 37.59 → 38.30 | +0.71 [−0.16, +2.21] |

Per network, Δ cell (pp) is 0.00 in 7 of the 12 network × regime × pair cells.
The rest:
- K-fold OCT-H: AS-B +1.21 [0, +3.45] and AS-C −0.22.
- LOSO OCT-H: AS-C −0.22.
- LOSO OCT-S: AS-B −4.37 [−13.64, 0] and AS-C +0.66.

Every per-network CI includes 0. By has-X/no-X: Δ cell is 0.00 on has-X in all
four arms. Every change is on no-X TGs. The largest is LOSO OCT-S at −2.50
[−8.99, +1.45].

**Error distance on solved rows, pooled (km)**

| Regime | Pair | p25 orig / GEO | p50 orig / GEO | Δp50 [site CI] | p75 orig / GEO | p90 orig / GEO |
|---|---|---|---|---|---|---|
| K-fold | OCT-H | 3.74 / 3.72 | 33.16 / 34.27 | +1.12 [−2.18, +9.05] | 293.5 / 293.6 | 500.9 / 500.8 |
| K-fold | OCT-S | 3.78 / 3.66 | 36.89 / 36.98 | +0.09 [−9.94, +3.00] | 361.5 / 358.1 | 641.4 / 638.7 |
| LOSO | OCT-H | 41.17 / 41.19 | 118.22 / 117.95 | −0.27 [−11.63, +1.23] | 407.7 / 403.4 | 692.4 / 695.0 |
| LOSO | OCT-S | 54.53 / 53.64 | 129.37 / 129.20 | −0.17 [−5.64, +7.63] | 433.0 / 433.0 | 796.0 / 796.7 |

Per network, |Δp50| ≤ 5 km everywhere, and every CI includes 0.

**K-fold → LOSO drop, original vs GEO (paired on the same TGs; diff-in-diff = GEO drop − original drop)**

| Pair | Δacc orig [CI] | Δacc GEO [CI] | Diff-in-diff, acc (pp) [CI] | Δp50 orig (km) [CI] | Δp50 GEO (km) [CI] | Diff-in-diff, p50 (km) [CI] |
|---|---|---|---|---|---|---|
| OCT-H | −15.76 [−24.32, −7.79] | −16.15 [−24.78, −7.92] | −0.39 [−1.18, +0.16] | +85.1 [49.8, 170.4] | +83.7 [48.9, 164.6] | −1.4 [−15.2, +2.1] |
| OCT-S | −8.90 [−16.54, −2.54] | −10.09 [−18.10, −3.15] | −1.18 [−4.23, +0.70] | +92.5 [27.8, 168.2] | +92.2 [31.7, 170.8] | −0.3 [−5.9, +12.4] |

**Transitions and prediction displacement, pooled (displacement = km between the two arms' predictions for the same TG)**

| Regime | Pair | correct→wrong | wrong→correct | Sites touched (c→w / w→c / any) | Displacement p50 / p90 (km) | Share moved < 1 km |
|---|---|---|---|---|---|---|
| K-fold | OCT-H | 1 (0.08%) | 5 (0.39%) | 1 / 2 / 3 | 0.41 / 11.6 | 59.8% |
| K-fold | OCT-S | 1 (0.08%) | 1 (0.08%) | 1 / 1 / 1 | 0.40 / 6.1 | 60.7% |
| LOSO | OCT-H | 1 (0.08%) | 0 | 1 / 0 / 1 | 0.94 / 10.0 | 51.3% |
| LOSO | OCT-S | 18 (1.42%) | 3 (0.24%) | 1 / 1 / 2 | 0.91 / 6.7 | 52.9% |

All 18 correct→wrong rows of LOSO OCT-S come from one no-X AS-B site, 18 of its
20 TGs. Both arms place those TGs about 220 km away, and GEO moves them about
30 km, which carries them across a nearest-seed boundary. OCT-S was credited
"correct" there only by the unbounded rule. The ≤2-ring accuracy on that
network is unchanged (Δ 0.00). Displacement is larger on no-X TGs: p50
2–3.5 km, p90 ≈ 20 km. On has-X TGs it is 0.05–0.4 km.

**Cost (per TG, ms; all 1,269 TGs per regime)**

| Regime | Method | ctr p50 | ctr p95 | total p50 | total p95 |
|---|---|---|---|---|---|
| K-fold | OCT-H | 466.4 | 1845.8 | 5244 | 60642 |
| K-fold | OCT-H-GEO | 0.35 | 0.46 | 4000 | 53753 |
| K-fold | OCT-S | 475.6 | 1787.5 | 3899 | 60800 |
| K-fold | OCT-S-GEO | 0.36 | 0.46 | 2874 | 55891 |
| LOSO | OCT-H | 480.4 | 1307.8 | 2313 | 39565 |
| LOSO | OCT-H-GEO | 0.40 | 0.51 | 1561 | 35123 |
| LOSO | OCT-S | 521.7 | 1413.9 | 2412 | 38513 |
| LOSO | OCT-S-GEO | 0.41 | 0.54 | 1537 | 32589 |

Total = ltd_ms + mtl_ms + ctr_ms. Part of the gap in the total comes from the
arms running at different times: their LTD and MTL are identical, yet mesh
OCT-H MTL p50 is 4512 vs 3989 ms. Only the CTR column measures the swap.

## Verdict
- **OCT-H-GEO vs OCT-H.** No separable difference in either regime. Δ cell is
  +0.32 pp K-fold and −0.08 pp LOSO; Δp50 is +1.1 km and −0.3 km. Every site
  CI includes 0. Half the predictions move < 1 km, and 1–6 TGs flip, at ≤ 3
  sites. The LOSO drop is unchanged: −16.15 vs −15.76 pp, diff-in-diff −0.39
  [−1.18, +0.16]. The CTR step explains none of OCT-H's LOSO loss, so that
  loss sits in LTD/MTL.
- **OCT-S-GEO vs OCT-S.** No separable difference. K-fold is identical on cell
  accuracy, with Δp50 +0.1 km. In LOSO, Δ cell −1.18 pp [−4.23, +0.70] is one
  boundary-straddling no-X site in AS-B whose predictions were about 220 km
  off in both arms. Ring-bounded accuracy moves the other way (+0.24 /
  +0.71 pp, n.s.). The LOSO drop diff-in-diff is −1.18 [−4.23, +0.70], driven
  by that same site, so the LOSO story does not change.
- The only real difference is cost: CTR falls from about 470 ms to about
  0.4 ms per TG (≈ 1,000×), roughly 10–30% of total per-TG inference. MTL
  dominates the total.

## Recommendation
**Appendix, one sentence plus the cost table, not a §6.5 ablation.** Swapping
the medoid for the geometric centroid moves the median prediction < 1 km and
leaves accuracy and the LOSO drop unchanged within the site CI. That settles
the §6.5 question ("calibration vs centroid": it is not the centroid) and needs
no figure. GEO's ≈ 1,000× cheaper CTR is worth one line.

Backing script: `tasks/20261004-geo-centroid-variants/compare_geo.py`. Its
tables are in `tasks/20261004-geo-centroid-variants/tables/` (`pooled.csv`,
`per_run.csv`, `by_has_x.csv`, `kfold_to_loso_drop.csv`, `timing.csv`,
`invariants.json`).
