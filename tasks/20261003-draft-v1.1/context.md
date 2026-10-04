# Draft v1.1 — Context

Discussion of 2026-10-03 on restructuring the paper's two evaluation sections:
`cbg-benchmark-paper/sections/eval-on-error-distance.tex` (§4) and
`cbg-benchmark-paper/sections/eval-on-region-classification.tex` (§5), plus where
the notion of *site* is introduced (`sections/methodology.tex`).

## Starting point (state of the paper before v1.1)

- §4 has the pooled error-distance table and CDF, seven percentile-by-percentile
  observations, then "Capability Ceiling and Floor of each Method". Only the S-P
  subsection is written: d_sp = d_geo + Δ_VP, Δ_VP ≈ D_X-TG, Ward clusters C1/C2/C3,
  and an RTT boxplot per cluster. SOI/VAN/OCT/SPO subsections are empty. VAN has
  only an italic conclusion. A champion UpSet block is commented out.
- §5 has overall accuracy (§5.1) and the has-X / no-X split with the per-network
  RTT figure (`figs/x_cell_rtt.png`) and Table `tab:x-cell` (within-group integer
  rates). It then has one subsection per method (S-P → SOI → VAN → OCT → SPO), an
  UpSet of per-target correctness (figure is a placeholder) with a per-site table,
  and stubs for the unbounded answer space and the combined evaluation.
- The user's motivation for §5: error distance has no direction.
  1. A large error can still be correct, because a large cell absorbs it.
  2. Equal errors can classify differently, because direction matters: crossing a
     cell edge fails, moving away from it survives.

  Hence the Voronoi serving-region partition, then HEALPix-bounded distance, then
  a combined evaluation under operator requirements.

## Problem the user raised

The characterization is duplicated. §4 and §5 both use one subsection per method
and retell each method's story under a second metric.

Concrete duplications found:
- VAN's "double-edged sword" conclusion appears almost verbatim in §4 and §5.
- §5 S-P restates §4's inference (S-P picks the VP at the interconnect).
- Two RTT boxplots show the same partition. no-X = C2 exactly; has-X = C1 ∪ C3.
- §5's per-method paragraphs read Table `tab:x-cell` row by row.

## Decisions (agreed with the user)

1. **X is the first-order expectation.**
   - In §5 the expectation is: has-X → correct, no-X → wrong.
   - Whatever the expectation explains gets no further text.
   - Other metrics are used only for deviations and counterintuitive results:
     - has-X misses
     - no-X hits
     - per-network volatility
     - within-site variation
   - The other metrics are cell size, distance to the boundary, peripheral vs
     central, and error distance.
   - They are likely confounded with X, since interconnects sit in dense metros.
     Use them to explain what remains within each group, not as parallel factors.
2. **Characterize the best tier, not every method.** The goal is to tell an
   operator which method fits their requirements.
   - Lower tier (S-P, SOI, VAN) gets about one sentence each, explained by X or by
     its §4 mechanism.
   - No per-method subsections in §5.
   - Each method's characterization lives once, in a capability table in the final
     combined-evaluation subsection. It replaces the scattered italic conclusions.
3. **§4 gets the same skeleton as §5: overall → X expectation → best-tier
   deviations.**
   - In §4, X is continuous: a method answering at the interconnect errs by about
     D_X-TG.
   - In §5, X is binary (has-X / no-X).
   - §4 builds the expectation; §5 cites it in a paragraph.
4. **Introduce *site* in Methodology, before either evaluation section.**

## Proposed §4

- **4.1 Achievable error distances.** Keep the table and the CDF. Cut the seven
  observations to three:
  1. The OCT tier is best at every percentile and holds its rank.
  2. S-P and SOI match OCT at p5 but are about 7× worse at p25. The near-exact low
     end is VP proximity, shared by all four.
  3. The other curves cross. SPO is never near-exact (71 km at p5). VAN refuses on
     22% of TGs.

  Define the tier with a paired site bootstrap. Memory
  `finding_error_distance_medians_not_separable` says only OCT-H vs SOI separates at
  p50 and p99 rests on 1–2 sites. Drop noise claims such as "OCT-H beats OCT-S at
  p25 by a tiny margin".
- **4.2 The interconnect sets the expected error.** Retitle the current S-P
  subsection; it describes the dataset's latency structure seen through S-P.
  - d_sp = d_geo + Δ_VP, Δ_VP ≈ D_X-TG, clusters C1–C3, one RTT figure.
  - End on the expectation: answering at the interconnect costs about D_X-TG.
  - The expectation is per site, since D_X-TG is the same for every TG at a site.
- **4.3 Where methods depart from the expectation.** Replaces the four empty
  per-method subsections with one paired comparison per cluster, e.g. each method's
  error minus S-P's error, per TG.
  - Best tier beats the expectation on C2 (far from X). This is what pays off as
    OCT-H's 50% and OCT-S's 41% of no-X TGs in §5.
  - Best tier falls short next to a VP, where S-P is near-exact. Memory
    `finding_cbg_squanders_vp_adjacent`: OCT-H p90 is about 250 km on VP-adjacent
    TGs.
  - Lower tier, mechanism only:
    - SOI tracks S-P.
    - VAN's tight envelope fails to intersect, mostly on C1 (~1.5 ms RTTs). This
      also explains the CDF sentinel.
    - SPO snaps to a region, so it is never near-exact.
  - SPO's snapping explains both its weak error distance (§4) and its
    classification strength (§5). Introduce it in §4 so §5 can point back.
- Delete the commented-out champion UpSet block. 4.3's deviation statistics
  replace it.

## Proposed §5

1. **Overall accuracy.** Unchanged, plus the tier from the same site bootstrap
   (paired accuracy differences). OCT-H 74, OCT-S 69, SPO 64 may not be separable.
   That favours a "tier" framing over a ranking: within the tier, choose by the
   network's geometry.
2. **Accuracy vs error distance.** Measure the two direction effects:
   - absorption: correct despite a large error, per method
   - direction: wrong despite a small error, per method

   This explains SPO's move from fifth by median error to third by accuracy.
3. **X as the expectation.**
   - Reading of `tab:x-cell`: the has-X column is how often a method meets the
     expectation; the no-X column is how often it beats it.
   - S-P and SOI are fully explained by X (93% has-X, ≤2% no-X).
   - Their remaining has-X misses are all in AS-1, explained in §4: C3's inflated
     RTTs, and C1 VPs whose RTTs differ by under 1 ms.
   - VAN: refusals on has-X (41%), pointing to §4.
   - One sentence ties to the §4 RTT figure: no-X is C2, has-X is C1 ∪ C3.
4. **Deviations in the best tier (OCT-H, OCT-S, SPO, against S-P).**
   - no-X hits: OCT-H 50%, OCT-S 41%, SPO 41%. Split them into *located* vs
     *absorbed* by error distance and cell size. This also tests the
     directionless-metric argument.
   - has-X misses: SPO in AS-2/3 (77% and 70%); OCT at 93% in AS-1 and AS-3.
   - Within the tier only two comparisons matter:
     - OCT-H vs OCT-S: same has-X (95%), OCT-S loses 9 pp on no-X. This is the
       hull-vs-spline bound; the mechanism goes in §4.
     - OCT-H vs SPO: SPO wins AS-1 on both groups and loses has-X TGs in AS-2/3.
       This is the existing case study and leads into §5.5.
   - Within-site variation: all-or-nothing holds at only 74% of sites for VAN and
     OCT, against 94–97% for S-P/SOI/SPO. The best tier is sensitive to per-TG RTT
     noise.
   - The UpSet can shrink into this subsection. Its useful results are deviation
     statistics: 9% of TGs (at 17% of sites) wrong for every method, and SPO the
     only correct method on 8%.
5. **Unbounded answer space.** The SPO-vs-OCT-H case (Seattle, peripheral cells)
   leads into the HEALPix-bounded bars.
6. **Combined evaluation.** One capability table: method × {error ceiling/floor,
   has-X, no-X, cell-geometry effects}, plus 1–2 sentences per method.

## Site: where and how to introduce it

The site is one target location within one network. The same location in two
networks is two sites, because the interconnects can differ. TGs at a site are
replicas sharing every VP distance. The code's site key is (run_id, lat, lon).

Its three roles all come before any result:
1. **Unit of geometry.** Answer-space seeds are sites, so a cell is a site's cell
   and has-X is a site property.
2. **Unit of independence.** Co-located TGs are not independent, so uncertainty is
   resampled by site. This is needed by §4.1's tier.
3. **Unit of train/test separation.** Memory
   `finding_operator_datasets_20_regions` says every site spans all 5 folds, so
   calibrated methods are tuned on replicas co-located with the test TG. Verify it
   still holds, then discuss it.

| Place | What it says about sites |
|---|---|
| §3.1.2 Dataset Preprocessing | definition |
| §3.3.1 Model Training and Testing | fold / leakage |
| §3.3.2 Answer Space Construction | seeds = sites |
| §3.3.3 Evaluation Metrics | per-TG results, site-resampled uncertainty |
| §4.2 | the X expectation is per site |
| §5 | has-X is a site property; the all-or-nothing result is the evidence |

Move the definition paragraph out of §5.3, where it appears today after §5.2 and
the RTT caption have already used "site".

Qualifiers on the claim that site geometry decides:
- **X nearby is not sufficient.** The C3 sites have X within 20 km but RTTs of
  59–64 ms from every VP. The claim is that a site is easy when X is nearby *and*
  the RTTs show it, so has-X is a latency property of the site.
- **Within-site variation exists for VAN and OCT** (see §5 item 4).

(The user wrote "a GDOP nearby"; confirmed to mean "an interconnect nearby", not
VP angular geometry.)

## Constraints

- No absolute TG or site counts anywhere in the paper; percentages only (memory
  `feedback_paper_no_absolute_counts`).
- Leave each method's definition sentences as the user wrote them.
- Method abbreviations: OCT-H, OCT-S, SOI, S-P, SPO, VAN.
- Figures are deferred ("we will deal with figs later").

## Issues found to fix

- **§4 table, Answered column** (`eval-on-error-distance.tex` lines 24–29) prints
  "1,269 (100%)" and "994 (78.3%)". Keep percentages only.
- **§4 commented-out block** (lines 101, 112, 134) says "1,269 targets at 65 sites"
  and "10 to 20 targets per site". arXiv publishes .tex source with comments;
  delete the block.
- **x_cell_rtt** caption ("about 20 TGs per site") together with §5.3 ("4–5% of the
  targets in their network") lets a reader derive network size (~440 TGs). Keep
  only the share. The figure's tick labels print TG/site counts and must become
  percentages (`scripts/analysis/v5/modules/figure_x_cell_rtt.py`).
- **One RTT figure.** Suggestion: redraw x_cell_rtt with C1/C2/C3 colours per
  network, keep it in §4.2, and have §5 cite it.
- **Typos.**
  - §4: vantange, drived, unqiue, propriotary, acutal, memasuring, superioty.
  - §5: incapabale, diffrent.
- **Placeholders still in red:** methodology ref, VAN example figure, the `\S~\ref{}`
  for the SPO case study, and the §5 UpSet figure. The UpSet figure needs a
  correctness criterion in `figure_champion_upset.py`.

## Analyses needed (scratch; do not touch the paper)

1. **Site bootstrap of tiers, both metrics:** paired differences of error
   percentiles and of accuracy for OCT-H vs OCT-S, OCT-H vs SPO, OCT-S vs SPO. Run
   this first, since a different tier changes what 4.3 and §5.4 characterize.
2. **Paired error vs S-P per cluster** for the tier, plus the VP-adjacent subset.
   Feeds 4.3.
3. **The tier's no-X hits split into located vs absorbed** (error distance, cell
   size). Feeds §5.4.
4. **SPO's p5 floor:** confirm region snapping is the cause.
5. **Cross-tab of the geometry factors against has-X:** cell size, boundary
   distance, peripheral vs central. Shows which factors carry signal beyond X.

## Related memories

`feedback_paper_characterize_deviations_best_tier`, `feedback_paper_no_absolute_counts`,
`finding_sp_cell_accuracy_is_pni_cell`, `finding_sp_locates_interconnect`,
`finding_top4_not_separable`, `finding_error_distance_medians_not_separable`,
`finding_cbg_squanders_vp_adjacent`, `finding_exclusive_region_verified`,
`finding_operator_datasets_20_regions`, `project_v4_site_key`.
