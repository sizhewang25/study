# Paper evaluation flow after the LOSO result

**Date:** 2026-10-04
**Paper:** `cbg-benchmark-paper/main.tex`, sections `eval-on-region-classification.tex`,
`eval-on-overhead.tex`, `comprehensive-eval.tex`, `discussion.tex`, `methodology.tex`
**Prior note:** [2026-10-04-allfail-cohort-label-identifiability.md](2026-10-04-allfail-cohort-label-identifiability.md)
(the memorization finding and the measured LOSO numbers this plan builds on)

## Question

The LOSO arm showed that part of OCT's cell accuracy is memorized labels from
same-site replicas, and the classification section already flagged that
unbounded cells reward SPO's far-but-directionally-right answers. How should
the evaluation sections be restructured around these two findings?

## Agreed outline

```
§5 Error distance (K-fold)
§6 Region classification (K-fold)
   6.1 Overall accuracy
   6.2 Per-method: has-X / no-X
   6.3 Method uniqueness (UpSet + site table)
       -> ends on the single-method cohorts: SPO-only 8%, OCT-H-only 4%, VAN-only 4%
   6.4 SPO: unbounded answer space          (subsection, not subsubsection)
   6.5 Seen vs unseen sites (OCT as lead case)
§7 Overhead (inference only)
§8 SOTA under operator criteria
   2x2: {unbounded, 2-ring bounded} x {K-fold, LOSO}; Pareto per regime
§9 Discussion (operator guidance by regime)
```

The two special topics each expose one flaw of plain cell accuracy; §8 then
applies both corrections (bounded cells, LOSO) to every method. That is what
makes the final Pareto figure earned rather than asserted.

## Decisions

### §6.3–6.5

- 6.4 and 6.5 are promoted to subsections. Uniqueness is the lens that
  surfaces them, not their topic. 6.3 closes by naming the single-method
  cohorts; VAN-only (4%) gets at least one sentence so the bridge is complete.
- 6.4 shows the *mechanism* on SPO under K-fold only: its error-distance
  distribution inside the cells it gets right, and the Seattle case. The full
  bounded table for all methods is left to §8, so the numbers are not repeated.
- 6.5 is framed as seen vs unseen sites, not "OCT memorizes". Pooled measured
  LOSO (`_cross/loso-delta/6-runs-197fee`):
  - OCT-H 73.5 -> 57.8% cell (-15.8 pp, site CI -24.3..-7.8), p50 33 -> 118 km;
    OCT-S 69.1 -> 60.2%.
  - The OCT verdict is "both": on no-X TGs OCT-H drops 49.9 -> 21.5% while
    S-P/SOI sit at 0-2%, so ~40% of OCT-H's no-X gain is real geometry.
  - VAN 57.0 -> 55.4% but unanswered 22 -> 41%; SPO 63.7 -> 61.8% (no
    memorization, consistent with min sum z^2 on pooled mu/sigma).
  - S-P/SOI identical by construction; SOI has the best LOSO p50 (96 km).
- Optional: the `octant_cbg_{hull,spl}_geo` runs (geometric_centroid instead
  of monte_carlo_medoid) can be an ablation in 6.5 if they separate calibration
  from the centroid step; otherwise appendix.

### §7 Overhead

- **Inference cost only.** Fit cost is ignored: calibration is a one-off
  offline step amortized over millions of inferences. Methodology states this
  scope in one sentence. It also removes the LOSO fold-count question (20-23
  folds vs 5 only changes fit cost).
- One pooled boxplot per method at p5 / p25 / p50 / p75 / p95 of per-target
  prediction time and heap, pooling both splits. State once that each TG
  appears twice (once per split).
- Caveat: OCT-H's prediction tail differs ~4x between splits on AS-1
  (p90 41 s K-fold vs 9 s LOSO; p50 2.2 s vs 1.6 s), from different fitted
  annuli and/or CPU contention (both ran `--cores all` at different times).
  Heap is identical. The ranking (OCT-H >> SPO > VAN > SOI, orders of
  magnitude) is unaffected, so do not quote the pooled OCT-H tail as a
  single-regime number.

### §8 SOTA

- 2x2 accuracy: unbounded vs bounded-within-2-rings, for K-fold and LOSO.
  Paired bars per method, unbounded light and bounded overlaid, one panel per
  split. The gap is the unboundedness dividend.
- Pareto figures on bounded accuracy x {inference time, inference heap}, one
  per regime, each using **its own split's** inference cost.
- **Ties.** Clustered by site the effective n is ~66-99; under LOSO
  SPO / OCT-S / OCT-H / VAN (61.8 / 60.2 / 57.8 / 55.4% unbounded) likely
  overlap. Every Pareto point carries a site-clustered bootstrap CI, and the
  SOTA per regime may be a tied set; when accuracy ties, cost decides.
  Needs a between-method site bootstrap per regime (`report-loso-delta` only
  bootstraps each method's K-fold -> LOSO delta).
- Keep LOSO error distance visible: a p50 column in the §8 table (SOI best
  under LOSO), since §5 is K-fold only.

### Methodology

- §4.3.1 currently says "Sites without any ground truth are left to future
  work" — contradicts the LOSO arm. Define both regimes: geo-stratified K-fold
  (label propagation) and LOSO (unseen sites, k = number of sites, one site's
  replicas per test fold).
- §4.3.3 defines the ring rule, with the km span of two rings at HEALPix
  nside 128. One-ring result goes to the appendix as a sensitivity check so
  the choice does not look tuned.
- State the inference-only cost scope (above).
- Fix the dangling `\S~\ref{sec:eval-on-tf}` (no traffic-filtered section is
  included): cut the sentence or restore the section.

### §9 Discussion

- Replace the limitation stub "OCT's memorization could get confused. SPO can
  be a good fit." (now covered by 6.5 / §8) with operator guidance by regime:
  ground truth already at the site -> choose from the K-fold frontier; new
  PoPs or stale-label sites -> choose from the LOSO frontier.
- Limitations: Hillsboro + San Jose flat ~63 ms sites (1.8% of TGs) are
  unlocatable from VP RTTs and label-suspect.

## Open

- Bounded (2-ring) LOSO numbers are not yet read; whether SOI joins the LOSO
  frontier is a hypothesis until they are.
- Between-method site bootstrap per regime is not implemented.
- Clean inference tails would need a serial single-core re-timing from the
  saved fit checkpoints; not planned unless the tail matters.
