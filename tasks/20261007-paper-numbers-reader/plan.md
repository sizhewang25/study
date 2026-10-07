# Paper Numbers Reader — Plan

## Background

Checking the paper's numbers by hand against v5 artifacts is slow and error-prone. The
2026-10-05 sweep (`notes/2026-10-05-paper-v5-support-sweep.md`) found rounding slips
(truncation 0.16 → 0.1, double rounding 0.8147 → 0.815 → 0.82). It also found numbers
read off a table instead of a file. Since then every quoted number has an artifact
(`notes/2026-10-05-paper-artifacts-wiring.md`). What is missing is a reader that prints
those numbers **the way the paper writes them**, section by section, so a section can be
checked in one pass.

## Context

- **Paper:** `cbg-benchmark-paper/sections/*.tex`, input order from `main.tex`.
- **Sections with statistics:**
  - Intro (key findings)
  - §3 Methodology
  - §4 Error distance
  - §5 Region classification
  - §6 Overhead
  - §7 Comprehensive evaluation
  - Appendix A (HEALPix)
  - Appendix B (CBG implementation)
- **Sections with no statistics today:** Motivation, Discussion, Related Work (not input),
Appendix C (traffic-filtered).
- **Artifacts:** built by `scripts/analysis/v5/create_paper_artifacts.sh` under
`--group pro-paper`, so pooled outputs sit in group-named folders under
`outputs/analysis/v5/_cross/<kind>/`.

  | Key              | Folder or file                                                                                                                                       |
  | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
  | **E**            | `classify/pro-paper.seen/error_cdf.pooled.norm.cut{.csv,.ratios.csv}`                                                                                |
  | **V**            | `vp-distance-cdf/pro-paper.seen/vp_distance_cdf.norm{.csv,.shares.csv,.manifest.json}`                                                               |
  | **P**            | `pni-gap/pro-paper.seen/`: `pni_gap.manifest.json`, `pni_cluster_rtt.csv`, `x_cell_rtt{.csv,.exceptions.csv}`, `sp_pni_cells.report.json`            |
  | **OB**           | `classify/pro-paper.seen/outcome_bars{.unbounded,}.pooled.healpix-128.csv`; per network: `outcome_bars{.unbounded,}.healpix-128.csv` and `.gaps.csv` |
  | **U**            | `classify/pro-paper.seen/correct_upset.pooled.cell.{intersections,sites,nesting}.csv`                                                                |
  | **XE / ST / PE** | `exclusive-error/`, `stability/`, `peripherality/` under `pro-paper.seen/` (manifests)                                                               |
  | **AS**           | `<run>/answer-space/healpix-128/meta.json`                                                                                                           |
  | **L**            | `loso-delta/pro-paper/loso_delta{,.by_has_x}.csv`                                                                                                    |
  | **C**            | `cost/pro-paper.seen/cost_box.pooled.heap{.csv,.extrapolation.csv}`                                                                                  |
  | **PA**           | `pareto/pro-paper/pareto{,.pairs,.datasets}.csv`                                                                                                     |
  | **D**            | `dataset/pro-paper/dataset.csv`                                                                                                                      |
  | **VD**           | `variant-delta/pro-paper/variant_delta{,.by_has_x,.drop,.timing}.csv`                                                                                |
  | **cfg**          | a config parameter or design constant, not a measurement                                                                                             |




## Goals

1. **This step:** an agreed list of every statistic the paper quotes, in section order
  (below), each with its source and format. No code until the user confirms each section.
2. **Later:** `report-paper --group pro-paper [--section N]`, which prints each statistic
  in the paper's wording and format from the artifacts only. It writes `.md` + `.json`.
   It does not read the paper project; the user compares the report with the text.



## Approach

**Formatting rules** (proposed, to confirm). They are applied to unrounded values only.


| Kind       | Rule                                             | Example                |
| ---------- | ------------------------------------------------ | ---------------------- |
| `e3`       | normalized distance ×10⁻³, 3 significant figures | 0.353, 7.56, 26.4, 137 |
| `%`        | whole percent; below 1 shown as `<1%`            | 74%, <1%               |
| `pp`       | 1 decimal below 10 pp, whole at 10 pp and above  | 0.2 pp, 1.6 pp, 16 pp  |
| `×`        | 1 decimal                                        | 6.8×                   |
| `ms` / `s` | whole ms below 1 s, 1 decimal s at or above      | 179 ms, 5.2 s          |
| range      | lo–hi in the same format                         | 1.6–6.8×, 42–64        |


Each section gets one reader module of claims in paper order; a claim is ID + sentence
template + artifact reader. We go section by section: confirm the list → implement →
user reviews the report → next section.

## Statistics inventory (to confirm, section by section)

"Paper" is the value the text prints today, as of cbg-benchmark-paper `258afd7`.
⚠ marks a value the sweep found differs from the artifact; the reader settles each one.

### Intro: key findings (recaps of later sections)


| ID  | Statistic                                                          | Paper                    | Fmt     | Source                                                                |
| --- | ------------------------------------------------------------------ | ------------------------ | ------- | --------------------------------------------------------------------- |
| I.1 | no-X accuracy of Octant and Spotter (range over OCT-H, OCT-S, SPO) | 41–50%                   | % range | L `by_has_x` (no-X, seen)                                             |
| I.2 | no-X accuracy of S-P                                               | <1%                      | %       | L `by_has_x`                                                          |
| I.3 | 1M IPs on 32 cores: OCT-H / SPO / SOI wall clock                   | ~5 days / <2 h / ~20 min | time    | C `extrapolation` (scope all)                                         |
| I.4 | memory constrains none at 32 GB                                    | 32 GB                    | cfg     | testbed (not in data); C `process_rss_x_cores_mb` gives the real need |




### §3 Methodology


| ID   | Statistic                                                  | Paper        | Fmt                 | Source                                                |
| ---- | ---------------------------------------------------------- | ------------ | ------------------- | ----------------------------------------------------- |
| M.1  | VPs: "over a hundred", same set for all networks           | >100; same   | count (qualitative) | D `n_vps`, manifest `vp_sets_identical`               |
| M.2  | TGs "over a thousand" at "several tens of sites"           | >1,000; tens | qualitative         | D `n_tgs`, `n_sites` (all)                            |
| M.3  | TGs per site: 20 (10 IPv4 + 10 IPv6)                       | 20           | cfg                 | D `replicas_max` (cap check)                          |
| M.4  | traffic share kept per network                             | 95%          | cfg                 | not in data                                           |
| M.5  | probing: 5 ICMP / hour; one-week window                    | —            | cfg                 | not in data                                           |
| M.6  | "full mesh"                                                | full         | check               | D `partial_mesh_pct` (19.5% of TGs lack some VPs) ⚠   |
| M.7  | filter removes 2% of TGs; AS-B 6%; AS-A and AS-C <1%       | 2 / 6 / <1   | %                   | D `est_removed_pct` (an estimate)                     |
| M.8  | sites sharing a seed                                       | 12%          | %                   | D `sites_merged_pct` (all)                            |
| M.9  | seen sites: 5 folds; replicas of a site in different folds | 5            | count               | D `kfold_n_folds`, `kfold_sites_all_folds_pct`        |
| M.10 | unseen sites: folds = sites                                | =            | check               | D `loso_folds_equal_sites`, `loso_sites_per_fold_max` |
| M.11 | pixels with 7 neighbours (footnote)                        | 24           | cfg                 | grid constant (`grid.neighbours`)                     |
| M.12 | no error exceeds D                                         | ≤1           | check               | D `max_error_over_d`                                  |
| M.13 | SOI speed                                                  | 2/3 c        | cfg                 | benchmark config `speed_ratio`                        |




### §4 Evaluation on error distance

**§4.1 Achievable error distances: Fig.** `cdf-err-dist-pooled-mesh`**, Table** `err-dist-percentiles`


| ID   | Statistic                                                         | Paper                   | Fmt                   | Source                                 |
| ---- | ----------------------------------------------------------------- | ----------------------- | --------------------- | -------------------------------------- |
| E.1  | Table: p5, p25, p50, p75, p95, p99 × 6 methods                    | (table)                 | e3; `--` if undefined | E `.csv`                               |
| E.2  | Table: answered share × 6 methods                                 | 100% / VAN 78%          | %                     | E `n_solved / n_tgs`                   |
| E.3  | Table: bold = best per column                                     | OCT-S p5, OCT-H rest    | check                 | E `.csv`                               |
| E.4  | caption and text: VAN unanswered; curve ends at                   | 22% / 78%               | %                     | E                                      |
| E.5  | ranking by median, with values                                    | OCT-H 7.56 … VAN 65.0   | e3 list               | E `p50`                                |
| E.6  | OCT-H ahead of S-P at every percentile, range                     | 1.6–6.8×                | × range               | E `.ratios` (ref S-P, `ratio_inv`)     |
| E.7  | OCT-S ahead of S-P, range                                         | 1.3–6.7×                | × range               | E `.ratios`                            |
| E.8  | p25: "the two OCT methods" ahead of the next method               | 6.8×                    | ×                     | E `.ratios` (OCT-H 6.78, OCT-S 6.70) ⚠ |
| E.9  | margin at p50 / p75 / p95 (text: "p75 and ." missing p95)         | 2.9 / 1.6 / 1.6×        | ×                     | E `.ratios` (OCT-H vs best non-OCT)    |
| E.10 | OCT-S within x% of OCT-H through p50                              | 11%                     | %                     | E `.ratios` (OCT-S/OCT-H)              |
| E.11 | OCT-S / OCT-H from p75 on                                         | 1.2–1.3×                | × range               | E `.ratios`                            |
| E.12 | SOI = S-P through p25; ahead at p50 / p75; tied p95; ahead at p99 | =; 1.2–1.3×; tied; 1.8× | ×                     | E `.ratios` (ref S-P)                  |
| E.13 | VAN behind S-P at every defined percentile                        | 2.5–8.3×                | × range               | E `.ratios`                            |
| E.14 | SPO behind S-P through p50 and at p95                             | (qualitative)           | check                 | E `.ratios`                            |
| E.15 | SPO p5 vs the best methods' p5                                    | ~2 orders               | × (91×)               | E `.ratios` (SPO / OCT-S, p5)          |
| E.16 | SPO p50/p5; OCT-H p50/p5                                          | 3.4× / 41×              | ×                     | E `.ratios` (`p50/p5` rows)            |
| E.17 | "SPO rarely lands … very far"                                     | qualitative             | check                 | E: SPO p95 432 is the worst p95 ⚠      |
| E.18 | S-P median / OCT-H median                                         | 3.5×                    | ×                     | E `.ratios`                            |


**§4.2 What limits Shortest Ping: Figs.** `cdf-vp-proximity`**,** `scatter-vp_pni_proximity`**,** `boxplot-vp_pni_proximity_rtt`


| ID   | Statistic                                        | Paper       | Fmt      | Source                                            |
| ---- | ------------------------------------------------ | ----------- | -------- | ------------------------------------------------- |
| E.19 | half of TGs have a VP within (d_geo median)      | 3.4         | e3       | V `.csv` d_geo `p50` = 3.13 ⚠                     |
| E.20 | S-P median error                                 | 26.4        | e3       | V d_sp `p50` (= E S-P p50)                        |
| E.21 | S-P median / d_geo median ("nearly eight times") | ~8×         | ×        | V manifest `median_ratio.d_sp_over_d_geo` = 8.4 ⚠ |
| E.22 | zero divergence (text and caption)               | 19%         | %        | V gap `zero_share_pct`                            |
| E.23 | Δ_VP exceeds x for half of TGs                   | 22.8        | e3       | V gap `p50` = 25.9 ⚠ (or V `.shares` at 22.8)     |
| E.24 | Δ_VP exceeds 228 for x%                          | 7%          | %        | V `.shares` gap `share_gt_pct` at 228             |
| E.25 | Δ_VP approaches the max distance                 | qualitative | e3       | V gap `max_norm_e3` (894)                         |
| E.26 | Ward clustering, 3 clusters                      | 3           | count    | P manifest `clustering`                           |
| E.27 | Spearman ρ, all TGs                              | +0.82       | 2 dp     | P manifest `spearman.rho_tgs` (0.8147 → +0.81) ⚠  |
| E.28 | C2 ρ                                             | +0.77       | 2 dp     | P manifest `clusters`                             |
| E.29 | C3 share of TGs; C3 ρ                            | 2% / −1.00  | % / 2 dp | P manifest `clusters`                             |
| E.30 | C3 interconnect within x                         | 4.6         | e3       | P manifest `clusters` d_pni max = 4.26 ⚠          |
| E.31 | C3 median normalized min RTT                     | 0.64        | 2 dp     | P `pni_cluster_rtt.csv` `p50_norm`                |
| E.32 | C1 ρ                                             | +0.20       | 2 dp     | P manifest `clusters`                             |
| E.33 | 1 ms ↔ 100 km under speed of Internet            | 100 km      | cfg      | constant (2/3 c)                                  |




### §5 Evaluation on region classification

**§5.1 Overall classification accuracy: Fig.** `stackbar-cls-acc-overall`


| ID  | Statistic                                                     | Paper              | Fmt      | Source                                     |
| --- | ------------------------------------------------------------- | ------------------ | -------- | ------------------------------------------ |
| R.1 | pooled accuracy × 6 methods                                   | 74/69/64/50/49/35  | %        | OB unbounded pooled `share_n_cell_correct` |
| R.2 | OCT-H per network and its rank (AS-B 1st, AS-C 1st, AS-A 2nd) | 75 / 64 / 82       | % + rank | OB per-network CSV                         |
| R.3 | SPO per network (AS-A 1st), and its lead over OCT-H in AS-A   | 89 / 51 / 53; 7 pp | % / pp   | OB per network; `.gaps` (6.3 pp) ⚠         |
| R.4 | SOI vs S-P: pooled gap; largest per-network gap               | 1 pp; ≤2 pp        | pp       | OB `.gaps` (AS-A 2.5 pp) ⚠                 |
| R.5 | VAN accuracy and unanswered share                             | 35% / 22%          | %        | OB `share_n_cell_unanswered`               |
| R.6 | ranking = median-error ranking except SPO (5th → 3rd)         | check              | rank     | OB + E `p50`                               |


**§5.2 The interconnect explains accuracy: Table** `x-cell`**, Fig.** `boxplot-rtt-of-x-cell-by-asn`


| ID   | Statistic                                                                   | Paper                    | Fmt        | Source                                                             |
| ---- | --------------------------------------------------------------------------- | ------------------------ | ---------- | ------------------------------------------------------------------ |
| R.7  | has-X / no-X share, pooled                                                  | 53 / 47%                 | %          | P `x_cell_rtt.csv` `tgs_pct` (all)                                 |
| R.8  | X in a has-X cell is always the nearest interconnect                        | always                   | check      | P `x_cell_rtt.manifest` `n_tgs_x_vs_any_interconnect_disagree = 0` |
| R.9  | no-X = C2; has-X = C1 + C3                                                  | check                    | crosstab   | P `sp_pni_cells_tgs.csv` (cluster × `tg_cell_holds_x`)             |
| R.10 | median normalized min RTT: has-X range; no-X range (per network)            | 0.015–0.018; 0.095–0.148 | 3 dp range | P `x_cell_rtt.csv` `p50_norm`                                      |
| R.11 | every no-X above the split; has-X at or below it                            | 0.032; 92%               | 3 dp / %   | P `x_cell_rtt.csv` `le_split_pct`, manifest `split_norm`           |
| R.12 | exception sites: AS-A C3 ×2 at 0.64–0.69; AS-C ×1 at 0.10; AS-B ×2 at 0.033 | (text)                   | 2–3 dp     | P `x_cell_rtt.exceptions.csv` (absolute site counts ⚠ convention)  |
| R.13 | Table: has-X / no-X / Acc × 6 methods × {All, A, B, C}                      | (table)                  | %, `<1`    | L `by_has_x` (seen) + OB                                           |
| R.14 | Table: group shares (All 53/47, A 75/25, B 42/58, C 43/57)                  | (header)                 | %          | P `x_cell_rtt.csv` `tgs_pct`                                       |
| R.15 | Table: VAN unanswered within has-X (All 41, A 36, B 44, C 47)               | (parentheses)            | %          | L `by_has_x` `unanswered_base`                                     |
| R.16 | Table: bold = best per column                                               | check                    | —          | from R.13                                                          |
| R.17 | Fig. caption: tick labels = share of sites                                  | check                    | %          | P `x_cell_rtt.csv` `sites_pct`                                     |


**§5.2.x Per-method characterization**


| ID   | Statistic                                                                        | Paper                | Fmt      | Source                                                     |
| ---- | -------------------------------------------------------------------------------- | -------------------- | -------- | ---------------------------------------------------------- |
| R.18 | S-P prediction outside any interconnect cell                                     | 0.1%                 | % (1 dp) | P `sp_pni_cells.report` `pred_in_interconnect_cell_pct`    |
| R.19 | S-P has-X / no-X accuracy                                                        | 93% / <1%            | %        | P report `sides`                                           |
| R.20 | every wrong S-P no-X prediction lands in an interconnect cell                    | 100%                 | check    | P `sides.no-X.wrong_in_interconnect_cell_pct`              |
| R.21 | all S-P has-X misses in AS-A; causes: inflated RTT (C3) or near-ties within 1 ms | AS-A only            | check    | P `sides.has-X.misses_by_run_cluster`, `n_near_tie_median` |
| R.22 | SOI = S-P on has-X; SOI no-X                                                     | identical; 2%        | %        | L `by_has_x`                                               |
| R.23 | VAN no-X vs SOI vs S-P                                                           | 16 / 2 / <1%         | %        | L `by_has_x`                                               |
| R.24 | VAN no-answers only on has-X; share of has-X declined                            | all; 41%             | %        | L `by_has_x` `unanswered_base`                             |
| R.25 | OCT has-X vs S-P; OCT-H / OCT-S no-X                                             | 95 vs 93%; 50 / 41%  | %        | L `by_has_x`                                               |
| R.26 | OCT-H has-X vs S-P: gains in AS-A, loses in AS-C                                 | check                | %        | L `by_has_x` per pair                                      |
| R.27 | SPO no-X (= OCT-S); SPO has-X vs OCT vs S-P                                      | 41%; 84 vs 95 vs 93% | %        | L `by_has_x`                                               |
| R.28 | SPO has-X gap vs S-P only in AS-B, AS-C; SPO has-X there                         | 77 / 70%             | %        | L `by_has_x` per pair                                      |
| R.29 | AS-A: SPO has-X / no-X vs OCT-H                                                  | 98 / 60 vs 93 / 51%  | %        | L `by_has_x` per pair                                      |


**§5.3 Method uniqueness: Fig.** `upset-cell-correct`**, Table** `cell-correct-sites`


| ID   | Statistic                                                                      | Paper                | Fmt      | Source                                    |
| ---- | ------------------------------------------------------------------------------ | -------------------- | -------- | ----------------------------------------- |
| R.30 | Table: share of sites with ≥1 / >50% / all correct × 6 methods; bold           | (table)              | %        | U `.sites`                                |
| R.31 | correct for every method except VAN; for all six                               | 23 / 19%             | %        | U `.intersections`                        |
| R.32 | wrong for every method: TGs, sites                                             | 9% at 17%            | %        | U `.intersections` `(none)`               |
| R.33 | SOI misses <1% of S-P-correct; OCT-H misses 2%, at 3% of sites                 | <1 / 2 / 3%          | %        | U `.nesting` `misses_baseline`            |
| R.34 | S-P-correct with no CBG method correct                                         | 0.1%                 | % (1 dp) | U `.nesting` `misses_baseline any_cbg`    |
| R.35 | CBG correct where S-P is wrong (rednote "42% (how?)")                          | 42%                  | %        | U `.nesting` `adds_over_baseline any_cbg` |
| R.36 | per site: OCT-H ≥1 / all; SPO all vs OCT-S all; SPO ≥1                         | 88/62; 58 vs 57; 65% | %        | U `.sites`                                |
| R.37 | all-or-nothing sites: S-P, SOI, SPO range; VAN, OCT-H, OCT-S                   | 94–97% / 74%         | % range  | U `.nesting` `all_or_nothing_sites`       |
| R.38 | only correct method: S-P, SOI each                                             | <1%                  | %        | U `.nesting` `only`                       |
| R.39 | Octant family only 12%: both 7%, OCT-H alone 4% (12% of sites), OCT-S alone 1% | 12/7/4/12/1          | %        | U `.nesting` + `.intersections`           |
| R.40 | SPO only 8% (11% of sites); VAN only 4% (6% of sites)                          | 8/11; 4/6            | %        | U `.nesting` `only`                       |
| R.41 | Octant-only cohort is almost all no-X                                          | (98.7%)              | %        | U `.nesting` `has_x_share`                |
| R.42 | (rednote) SPO-only share in peripheral cells and >2 px                         | pending              | %        | not yet in an artifact                    |


**§5.4 Unbounded cells and bounded accuracy: Figs.** `stackbar-bounded-pooled`**,** `spo-exclusive-pixel`**,** `spo-replica-spread`**,** `spo-peripherality`


| ID   | Statistic                                                                                         | Paper            | Fmt    | Source                                                |
| ---- | ------------------------------------------------------------------------------------------------- | ---------------- | ------ | ----------------------------------------------------- |
| R.43 | own pixel: OCT-H, OCT-S, S-P, SOI (~twice)                                                        | 42/40/22/20%     | %      | OB pooled `share_n_ring0_cell_correct`                |
| R.44 | S-P, SOI correct almost only within 1 pixel                                                       | check            | %      | OB `of_correct_le_ring1`                              |
| R.45 | SPO own pixel; SPO correct beyond 2 px (of correct; of all)                                       | 0; 59% = 38%     | %      | OB `of_correct_beyond`, `share_n_beyond_cell_correct` |
| R.46 | beyond 2 px of correct: OCT-H, OCT-S, VAN                                                         | 16/18/25%        | %      | OB `of_correct_beyond`                                |
| R.47 | SPO-not-OCT-H cohort share                                                                        | 11%              | %      | XE manifest `only_a` share                            |
| R.48 | its median pixel distance; share at 3–10 px; OCT-H max                                            | 4; ~half; 8      | px / % | XE manifest                                           |
| R.49 | replica spread p50/p75: SPO; OCT-H                                                                | 0.3/0.5; 0.5/1.4 | 1 dp   | ST manifest                                           |
| R.50 | SPO-win sites more peripheral than OCT-H-win sites                                                | check            | 2 dp   | PE manifest (p50 0.87 vs 0.68)                        |
| R.51 | SPO exclusive wins >10 px                                                                         | 22%              | %      | XE manifest `share_beyond_10`                         |
| R.52 | peripheral cell share: AS-A, AS-B, AS-C                                                           | 39/29/32%        | %      | AS `peripheral_seed_share` per run                    |
| R.53 | bounded threshold                                                                                 | 2 px             | cfg    | `BOUND_PIXELS`                                        |
| R.54 | bounded: OCT-H 74→61, OCT-S 69→57, S-P 49, SOI loss <1 pp, SPO 64→26 (23 pp below S-P), VAN 35→27 | (text)           | % / pp | OB `accuracy_bounded` (= PA bounded seen)             |
| R.55 | SPO bounded behind S-P in every network                                                           | check            | %      | PA `.datasets`                                        |


**§5.5 Seen vs unseen sites: Table** `seen-unseen`


| ID   | Statistic                                                                                          | Paper             | Fmt          | Source                             |
| ---- | -------------------------------------------------------------------------------------------------- | ----------------- | ------------ | ---------------------------------- |
| R.56 | Table: acc seen / unseen / Δ pp, sites ↓ / ↑ %, no-X seen / unseen, median seen / unseen × 6; bold | (table)           | % / pp / e3  | L `.csv` + `by_has_x` (pooled)     |
| R.57 | OCT-H 74→58 (−16 pp), ↓35% ↑5%; OCT-S 69→60 (−9), ↓26% ↑3%                                         | (text)            | % / pp       | L                                  |
| R.58 | OCT-H correct→wrong; wrong→correct                                                                 | 17% / 1%          | %            | L `n_correct_to_wrong / n_tgs`     |
| R.59 | no-X: OCT-H 50→21, OCT-S 41→24; has-X both >90%                                                    | (text)            | %            | L `by_has_x`                       |
| R.60 | median: OCT-H 7.56→26.9, OCT-S 8.41→29.5; S-P 26.4; SOI 21.9 lowest unseen                         | (text)            | e3           | L `p50_*_norm_e3`                  |
| R.61 | OCT-H own pixel AS-A 57→17; AS-B, AS-C ≤5%                                                         | (text)            | %            | L `own_pixel_*` per pair           |
| R.62 | OCT-H no-X unseen vs S-P, SOI; gain kept (~two fifths)                                             | 21 vs <1, 2; 0.43 | % / fraction | L `by_has_x` `gain_kept`           |
| R.63 | OCT-S coverage parameter                                                                           | 90%               | cfg          | benchmark config `target_coverage` |
| R.64 | VAN unanswered 22→41, has-X 41→77; acc 35→16; ↓31% ↑3%                                             | (text)            | %            | L + `by_has_x`                     |
| R.65 | SPO Δ −2 pp; ↓3% ↑0; median unchanged                                                              | (text)            | pp / %       | L                                  |




### §6 Evaluation on overhead: Fig. `boxplot-speed-memory`


| ID   | Statistic                                            | Paper                  | Fmt      | Source                                                        |
| ---- | ---------------------------------------------------- | ---------------------- | -------- | ------------------------------------------------------------- |
| O.1  | runtime spans >2 orders of magnitude                 | >100×                  | ×        | C (OCT-H p50 / SOI p50 = 154×)                                |
| O.2  | median runtime: SOI, VAN, SPO                        | 34/101/179 ms          | ms       | C pipeline `p50`                                              |
| O.3  | none of SOI, VAN, SPO exceeds 0.6 s                  | <0.6 s                 | s        | C pipeline `max`                                              |
| O.4  | OCT-H median; p95                                    | 5.2 s / 61 s           | s        | C pipeline                                                    |
| O.5  | OCT-H MTL share of runtime ("nearly all")            | (86% median; 95% mean) | %        | C extrapolation `mtl_share_of_mean`                           |
| O.6  | peak heap <0.3 MB for SOI, VAN, SPO                  | <0.3 MB                | MB       | C memory_heap `max`                                           |
| O.7  | OCT-H heap from EST                                  | 24 MB                  | MB       | C ctr heap `p50`                                              |
| O.8  | geo-centroid heap median; runtime 5.2 → 4.0 s        | 3.2 MB; 4.0 s          | MB / s   | C OCT-H-GEO rows                                              |
| O.9  | OCT-H mean; mean/median                              | 13.8 s; 2.6×           | s / ×    | C extrapolation `runtime_mean_ms`, `mean_over_p50`            |
| O.10 | core-hours for 1M IPs: SOI, VAN, SPO, OCT-H          | 11/34/61/~3,800        | h        | C extrapolation `core_hours`                                  |
| O.11 | 32 cores: SOI, VAN, SPO within 2 h; OCT-H ~5 days    | (text)                 | h / days | C extrapolation `wall_*` (all; note AS-C SPO 2.6 h ⚠)         |
| O.12 | p95 under 0.4 s: SOI, VAN, SPO; OCT-H up to a minute | <0.4 s; ~1 min         | s        | C pipeline `p95`                                              |
| O.13 | 32 workers × OCT-H largest peak heap                 | ~1.2 GB                | GB       | C extrapolation `peak_per_tg_x_cores_mb` (RSS-based ~9 GiB ⚠) |
| O.14 | OCT-S cost similar to OCT-H                          | check                  | s        | PA seen OCT-S runtime                                         |




### §7 State of the art under operator criteria: Fig. `pareto` (4 panels), Table `sota`


| ID   | Statistic                                                                                                                                    | Paper        | Fmt             | Source                                                     |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------- | ------------ | --------------- | ---------------------------------------------------------- |
| S.1  | Table: pooled accuracy (worst–best network) × 6 methods × {seen, unseen} × {unbounded, bounded}                                              | (table)      | % (lo–hi)       | PA `.csv` `acc`, `acc_min`, `acc_max`                      |
| S.2  | Table: † = beats S-P in all three networks                                                                                                   | flags        | check           | PA `.pairs` `consistent`                                   |
| S.3  | Table: bold = best pooled per column                                                                                                         | check        | —               | PA                                                         |
| S.4  | Table: median runtime seen / unseen × 5 CBG methods                                                                                          | (table)      | ms / s          | PA `runtime_p50_ms`                                        |
| S.5  | spread worst–best ≥18 pp for all but VAN; max 41 pp (SPO unseen unbounded, 48–89)                                                            | 18 / 41      | pp              | PA `spread_pp`                                             |
| S.6  | SU frontier SOI→SPO→OCT-S→OCT-H; SPO 64% at 179 ms; OCT-H 74% at 5.2 s, 29× slower, +10 pp                                                   | (text)       | % / ms / × / pp | PA `on_frontier`, `.pairs` OCT-H vs SPO                    |
| S.7  | SB frontier SOI→OCT-S→OCT-H; SOI over S-P by                                                                                                 | 0.1 pp       | pp              | PA `lead_vs_sp_pp` (0.16 → 0.2) ⚠                          |
| S.8  | UU frontier SOI, SPO; SPO ~10× faster than OCT; lead over OCT-S                                                                              | ~10×; 1.6 pp | × / pp          | PA `.pairs` `runtime_p50_ratio` (13.6×), `d_acc_pooled_pp` |
| S.9  | UB: SOI and OCT-S over S-P by; neither ahead in every network                                                                                | 0.1 / 0.4 pp | pp              | PA `lead_vs_sp_pp` (0.16 / 0.47) ⚠, `.pairs`               |
| S.10 | across criteria: S-P stays 49%; SPO 23–24 pp below S-P bounded; OCT within 0.4 pp hardest; VAN behind everywhere; SOI within 1 pp everywhere | (text)       | % / pp          | PA (0.47 → "0.5") ⚠                                        |




### Appendix A: HEALPix granularities, Table `healpix-levels`


| ID  | Statistic                                                  | Paper   | Fmt            | Source                                        |
| --- | ---------------------------------------------------------- | ------- | -------------- | --------------------------------------------- |
| A.1 | N_side, N_pix, pixel area (km²), width (km) for levels 3–8 | (table) | integer / 1 dp | arithmetic, R = 6,371 km (ℓ=3 area 664,146 ⚠) |
| A.2 | caption: N_side 1, 2, 4, 8 → N_pix 12, 48, 192, 768        | (text)  | integer        | arithmetic                                    |




### Appendix B: CBG implementation, the geometric-centroid ablation


| ID  | Statistic                                                        | Paper                      | Fmt               | Source                                                     |
| --- | ---------------------------------------------------------------- | -------------------------- | ----------------- | ---------------------------------------------------------- |
| B.1 | Δ accuracy: OCT-H seen / unseen; OCT-S seen / unseen             | +0.3 / −0.1; 0.0 / −1.2 pp | pp (1 dp, signed) | VD `.csv` pooled `d_acc_pp`                                |
| B.2 | OCT-S −1.2 pp from a single site                                 | 1 site                     | count             | VD `n_sites_flip` (unseen, OCT-S)                          |
| B.3 | accuracy changes at no more than x% of sites, "every comparison" | 5%                         | %                 | VD `sites_flip_pct` (pooled ≤4.6; per network up to 9.1 ⚠) |
| B.4 | every change falls on no-X TGs                                   | check                      | —                 | VD `.by_has_x` (has-X Δ = 0)                               |
| B.5 | median prediction moves less than                                | 0.25 ×10⁻³                 | e3                | VD `disp_p50_norm_e3` (max 0.21)                           |
| B.6 | OCT-H drop difference; drop differs at x% of sites               | −0.4 pp; 6%                | pp / %            | VD `.drop` pooled `did_acc_pp`, `sites_drop_differs_pct`   |
| B.7 | EST ~1,000× cheaper; (rednote) 470 ms vs 0.4 ms                  | ~1,000×; 470 / 0.4 ms      | × / ms            | VD `.timing` `ctr_p50_ratio`, `ctr_p50_ms_*`               |




## Caveats

- **A ✓ only means the string was found.** The ⚠ check against the `.tex` is a heuristic;
the same digits can appear elsewhere in the section.
- **The reader does no analysis.** A statistic with no artifact column must first get one
in its module. Pending: R.42, the SPO-only peripheral / >2 px share.
- `cfg` **rows** report a config or design constant, or note "not in data" (M.4, M.5, I.4).
- **Absolute counts.** The reader may print counts to check a share, but the paper must not
print absolute TG or site counts. R.12 currently does, which is a convention issue.
- **The paper is still being edited**, so this inventory is a snapshot of `258afd7`.
Re-read each section before implementing its reader.

