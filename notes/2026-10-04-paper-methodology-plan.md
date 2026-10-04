# Paper methodology section: what to change

**Date:** 2026-10-04
**Paper:** `cbg-benchmark-paper/sections/methodology.tex`
**Prior note:** [2026-10-04-paper-eval-flow-plan.md](2026-10-04-paper-eval-flow-plan.md)
(the agreed evaluation flow; its "Methodology" decisions are expanded here)

## Question

The evaluation flow now rests on two regimes (K-fold, LOSO), a bounded and
an unbounded cell accuracy, site-clustered CIs and an inference-only cost
scope. The methodology section defines none of them yet, and three of its
subsections are empty. What should it say, and in what order should it be
written?

## State of the section

- 4.1 Dataset collection: written; a few gaps (see Housekeeping).
- 4.2 Three-phase CBG implementation: empty.
- 4.3.1 Model training and testing: K-fold only. Says "Sites without any
  ground truth are left to future work", which the LOSO arm now answers, and
  points to `\S~\ref{sec:eval-on-tf}`, a section whose file was deleted.
- 4.3.2 Answer space construction: empty. The definition currently lives in
  §6.2 behind a red "As defined in §methodology" note.
- 4.3.3 Evaluation metrics: three placeholder sentences.
- Benchmark runtime environment / Testbed setup: empty headings.

## Suggestions, by priority

### 4.3.1 Training and testing: two regimes (blocking)

- **K-fold (label propagation, seen sites).** Keep the current greedy
  geographic-spread paragraph as is.
- **LOSO (unseen sites).** Each test fold holds every replica of one site;
  the model is trained on all other sites. Write "k equals the number of
  sites in each network" with no numbers: site counts are confidential.
- One sentence: S-P and SOI fit nothing, so their predictions are identical
  under both regimes, a built-in check that both arms used the same inputs
  (`report-loso-delta` enforces it).
- Delete "Sites without any ground truth are left to future work" and the
  traffic-filtered sentence pointing to `sec:eval-on-tf`. §5's intro has the
  same dangling reference.

### 4.3.2 Answer-space construction: move the definition here

Move the inline Voronoi definition out of §6.2:

- **site**: a unique labelled location;
- **seed**: sites within one grid width grouped by complete linkage
  (diameter cap, not centroid radius), placed at their spherical centroid.
  EWR and JFK become one place;
- **cell** (serving region): a seed's nearest-site Voronoi cell, unbounded;
- **grid**: HEALPix nside 128, equal-area, 50.9 km wide (sqrt of the pixel
  area).

Both partitions are built at the same width (`answer_space.py`,
`seeds.py`), which makes this one clean paragraph. §6.2 then only defines
has-X / no-X.

### 4.3.3 Metrics: replace the placeholders

1. **Error distance.** `motivation.tex` promises min-max normalized
   distances; the §5 table caption says km. Pick one; it also decides how
   the ring span below is stated.
2. **Cell accuracy (unbounded).** Each prediction is correct, wrong or
   unanswered. The denominator is every evaluated TG: a method that declines
   to answer does not earn a smaller denominator.
3. **Ring bound.** Grid offset = HEALPix steps from the TG's grid to the
   prediction's. Bounded accuracy = cell correct **and** offset <= 2.
   - State the span. An earlier measurement gave a max error of ~193 km
     inside ring 2, against a ~51 km grid. **Re-measure on the current
     classify outputs before quoting it.**
   - Name it "ring-bounded serving-region accuracy", never "Voronoi"; the
     bound exists to defuse the Seattle/Arctic objection.
   - The one-ring result goes to the appendix as a sensitivity check, so the
     choice of two does not look tuned.
   - This paragraph sets up §6.4 (the mechanism on SPO) and §8 (bounded
     accuracy for every method).
4. **Uncertainty.** Bootstrap CIs resample sites, not TGs, because
   replicas at a site are correlated (effective n ~66-99). §8 uses this to
   report tied sets, so define it once here.
5. **Overhead scope.** Per-target inference time and peak heap, summed over
   the LTD, MTL and CTR phases. Fit cost is excluded as a one-off offline
   step amortized over all inferences.

### 4.2 Three-phase CBG implementation

- A short paragraph plus a table mapping the six methods to their LTD / MTL
  / CTR choices. `framework.tex` already lists the variants in comments.
- Details and parameter tuning stay in the appendix.
- §7's per-phase cost and any `_geo` ablation in §6.5 (a CTR swap:
  geometric centroid instead of Monte Carlo medoid) need this vocabulary.
- Open: table or prose (user's call).

### Housekeeping

- Merge "Benchmark runtime environment" and "Testbed setup" into one short
  paragraph: hardware; heap measured with `tracemalloc`; runs used all cores
  in parallel, which explains the caveat on OCT-H's inference-time tail.
- 4.1 "Traffic-filtered datasets": cut, or keep one line as future work,
  since its evaluation section is gone.
- 4.1 gaps: empty `\cite{}`s, the red "what granularity?", "several weeks".
- 4.1 "the 20 IPs (10 IPv4 and 10 IPv6)" per site: confirm this per-site
  number may be published. It is not a total, but an earlier pass flagged
  "10 to 20 targets per site" as a count to remove.

## Writing order

1. 4.3.1 and 4.3.3: decisions settled; §6-§8 depend on them.
2. 4.3.2.
3. 4.2, after the table-vs-prose call.
4. Housekeeping.

## Open

- Normalized vs km error distance (blocks the wording of 4.3.3 items 1 and 3).
- Current max error per ring at nside 128 (not yet re-measured).
- Whether the per-site replica count may be published.
