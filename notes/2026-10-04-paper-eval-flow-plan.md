# Paper evaluation flow after the LOSO result

**Date:** 2026-10-04 (revised after the methodology first draft, `cbg-benchmark-paper` e62b0ca)
**Paper:** `cbg-benchmark-paper/main.tex`, sections `eval-on-error-distance.tex`,
`eval-on-region-classification.tex`, `eval-on-overhead.tex`, `comprehensive-eval.tex`,
`discussion.tex`, `methodology.tex`
**Prior notes:** [2026-10-04-allfail-cohort-label-identifiability.md](2026-10-04-allfail-cohort-label-identifiability.md)
(the memorization finding and the measured LOSO numbers),
[2026-10-04-octant-geo-centroid-variants.md](2026-10-04-octant-geo-centroid-variants.md)
(the CTR swap explains none of the LOSO drop)

## Question

The LOSO arm showed that part of OCT's cell accuracy is memorized labels from
same-site replicas, and the classification section already flagged that
unbounded cells reward SPO's far-but-directionally-right answers. How should
the evaluation sections be structured around these two findings, now that the
methodology defines every metric and both regimes?

## Spine: defaults first, then relax each one

The methodology (§4.3.3) evaluates a prediction on two operator questions plus
cost, and (§4.3.1) uses the seen-site regime unless stated otherwise. The
evaluation follows that: the default metric under the default regime first,
then one subsection per default that the results put in doubt.

```
§5 Error distance (seen sites)                 how far
§6 Region classification (seen sites)          which serving region
   6.1 Overall accuracy
   6.2 Per-method: has-X / no-X
   6.3 Method uniqueness (UpSet + site table)
       -> ends on the single-method cohorts: SPO-only 8%, OCT-H-only 4%, VAN-only 4%
   6.4 Unbounded cells -> bounded accuracy      relaxes "cells are unbounded"  (SPO lead case)
   6.5 Seen vs unseen sites                     relaxes "seen sites"           (OCT lead case)
§7 Inference overhead                          what it costs
§8 SOTA under operator criteria
   2x2: {cell, bounded} x {seen, unseen}; Pareto against cost per regime
§9 Discussion (operator guidance by regime)
```

Each special topic tests one default that the methodology set; §8 then applies
both alternatives to all six methods. The methodology already promises this
order ("introduce the unseen-site regime in §6.5, compare under both in §8"),
so the evaluation only has to deliver it. This is what makes the final Pareto
figure earned rather than asserted.

## Two threads, set up early and paid off later

| | Set up | Built up | Payoff |
|---|---|---|---|
| SPO | 6.1: methods rank as by median error, except SPO | 6.2: SPO matches OCT-S on no-X; 6.3: SPO-only 8% | 6.4: part of it is right-direction-only answers in periphery cells |
| OCT | §5: best at every percentile; 6.1: leads | 6.2: 50% no-X where S-P/SOI get ~0; 6.3: OCT-H-only 4% | 6.5: part of it is same-site replicas in training |

Wiring:
- 6.1's last sentence points to 6.4, not to 6.2.
- 6.3 ends by naming the single-method cohorts as the two open questions, plus
  one sentence on VAN-only (4%). VAN's unanswered share doubles under unseen
  sites (22 -> 41%), so 6.5 picks it back up.

## Decisions per section

### §5 Error distance

- Intro shrinks to one line: seen sites, pooled over AS-A/B/C, error distance
  normalized as in §4.3.3. Drop the six-method list and the mesh vs
  traffic-filtered recap (methodology covers both).
- Convert to d/D x 10^-3 (fixed bounds [0, D]); cut each CDF line after the
  method's last answered TG; no 10,000 km sentinel, no dagger marks, no
  "VAN from p79 is the sentinel" sentence. Done later, with the figure rerun.

### §6 Region classification

- Intro shrinks to one line (seen sites, cell accuracy as in §4.3.3).
- 6.2: replace the inline Voronoi definition (and its red "As defined in
  §methodology") with a callback to §4.3.2. Keep the beat "§5 used Delta_VP as a
  proxy for distance to X; the answer space defines it directly". has-X / no-X
  stays defined here, as a result-level concept.
- 6.4 and 6.5 are subsections, not subsubsections. Uniqueness is the lens that
  surfaces them, not their topic.
- **6.4 Unbounded cells.** Mechanism on SPO under seen sites only.
  - First place the bounded-accuracy threshold is set: **pixel distance <= 2**,
    framed as one operator's choice, not a tuned value. §8 reuses it.
  - Show SPO's correct-cell predictions by **pixel distance** from the TG's
    pixel, not km. Seattle case reads "right cell, many pixels away".
  - The full bounded table for all methods is left to §8, so numbers are not
    repeated.
- **6.5 Seen vs unseen sites**, not "OCT memorizes". Pooled measured LOSO
  (`_cross/loso-delta/6-runs-197fee`):
  - OCT-H 73.5 -> 57.8% cell (-15.8 pp, site CI -24.3..-7.8), p50 33 -> 118 km;
    OCT-S 69.1 -> 60.2%.
  - The OCT verdict is "both": on no-X TGs OCT-H drops 49.9 -> 21.5% while
    S-P/SOI sit at 0-2%, so ~40% of OCT-H's no-X gain is real geometry.
  - VAN 57.0 -> 55.4% but unanswered 22 -> 41%; SPO 63.7 -> 61.8% (no
    memorization, consistent with min sum z^2 on pooled mu/sigma).
  - S-P/SOI identical by construction; SOI has the best LOSO p50.
  - CTR ablation (`_geo` combos) explains none of the drop: one sentence plus
    an appendix pointer. `_geo` combos never enter a figure.

### §7 Inference overhead

- Inference only; fitting excluded (already stated in §4.3.3). Time summed over
  LTD/MTL/CTR, peak heap = max over phases (`memory_heap`). S-P left out.
- One pooled boxplot per method at p5 / p25 / p50 / p75 / p95, pooling both
  regimes. State in §7 that each TG appears twice (once per regime).
- Caveat lives in §7 with its own cause (Testbed no longer mentions parallel
  runs): OCT-H's time tail differs ~4x between regimes on AS-A (p90 41 s seen vs
  9 s unseen; p50 2.2 vs 1.6 s), from different fitted annuli and/or CPU
  contention of parallel folds and methods. Heap is identical. The ranking
  (OCT-H >> SPO > VAN > SOI, orders of magnitude) is unaffected, so do not
  quote the pooled OCT-H tail as a single-regime number.

### §8 SOTA

- 2x2 accuracy: cell vs bounded (pixel distance <= 2), for seen and unseen
  sites. Paired bars per method, cell light and bounded overlaid, one panel per
  regime. The gap is the unboundedness dividend.
- Pareto figures on bounded accuracy x {inference time, inference heap}, one
  per regime, each using **its own regime's** inference cost.
- **Ties.** Clustered by site the effective n is ~66-99; under unseen sites
  SPO / OCT-S / OCT-H / VAN (61.8 / 60.2 / 57.8 / 55.4% cell) likely overlap.
  Every Pareto point carries a site-clustered bootstrap CI and the SOTA per
  regime may be a tied set; when accuracy ties, cost decides.
  - The methodology skips uncertainty, so **§8 defines the CI where it is first
    used**: one sentence, resampling sites because ~20 replicas per site are
    not independent. Without it, §8 must not say "tied".
  - Needs a between-method site bootstrap per regime (`report-loso-delta` only
    bootstraps each method's seen -> unseen delta).
- Keep unseen-site error distance visible: a normalized p50 column in the §8
  table (SOI best under unseen sites), since §5 is seen sites only.

### §9 Discussion

- Replace the limitation stub "OCT's memorization could get confused. SPO can
  be a good fit." (now covered by 6.5 / §8) with operator guidance by regime:
  labels already at the site -> choose from the seen-site frontier; new PoPs or
  stale-label sites -> choose from the unseen-site frontier.
- Limitations: Hillsboro + San Jose flat ~63 ms sites (1.8% of TGs) are
  unlocatable from VP RTTs and label-suspect.

### Intro

- RQs have no seen/unseen question. Add one clause to RQ1 ("... for sites with
  and without location labels") so 6.5 and §8 answer something and §9's
  guidance has a question.

### Methodology (done in e62b0ca; small tweaks left)

- Done: both regimes defined (seen = geo-stratified greedy 5-fold, unseen =
  LOSO); answer space (seeds, cells, pixels); metrics (normalized error
  distance, pixel distance, region classification accuracy with bounded
  variant, inference overhead); testbed. The dangling
  `\S~\ref{sec:eval-on-tf}` is gone.
- Superseded, do **not** do: "km span of two rings at nside 128" (never state
  km, widths or nside) and "one-ring result in the appendix as a sensitivity
  check" (dropped: the threshold is the operator's choice). The threshold is
  "a threshold" in methodology, "2" first in 6.4. No tau.
- Tweak: methodology intro ("an unbounded and a distance-bounded form besides
  error distance") -> follow §4.3.3 order and names: error distance, region
  classification accuracy with a bounded variant, inference overhead.
- Tweak: Testbed footnote -> "S-P only sorts VPs by RTT, so we leave it out of
  the overhead analysis."
- Open: 4.2 Three-phase CBG Implementation (empty); Testbed CPU TODO.

## Confidentiality leaks (fix)

The HEALPix granularity (nside) and the seed threshold must never appear in the
paper or in its source on Overleaf.

- **LEAK, visible in the PDF:** `sections/eval-on-region-classification.tex:116`,
  the UpSet placeholder `\fbox` prints "HEALPix nside 128" in red. Already
  synced to Overleaf (e62b0ca). Fix: drop the nside from the placeholder text.
- **LEAK, source only:** figure filenames carry `healpix-128`. Not in the PDF,
  but visible to anyone with the Overleaf source.
  - Referenced: `figs/outcome_bars.unbounded.pooled.healpix-128.png`
    (`eval-on-region-classification.tex:22`); commented
    `outcome_bars.unbounded.healpix-128.png` (:29) and
    `champion_upset.pooled.cell.healpix-128.png` (:117).
  - Committed in `figs/` but unreferenced: `outcome_bars.healpix-128.png`,
    `outcome_bars.unbounded.healpix-128.png`.
  - Commented caption at `eval-on-region-classification.tex:30` says
    "HEALPix nside 128".
  - Fix: rename on the figure rerun (export names without the nside), update
    the `\includegraphics` paths, delete the stale files and the commented
    caption. Checked 2026-10-04: the pooled outcome-bars image itself shows no
    nside.
- Check on every figure rerun: no nside in captions, axis labels, legends or
  filenames; no km on pixel-distance axes; AS-A/B/C labels.

## Other knock-on cleanup (later)

- Motivation §3.4: shrink the partition/Starlink paragraph and line 39 to the
  "why"; definitions live in §4.3.2-4.3.3.
- §8 stub: "geostratified and LOSO" -> seen / unseen sites.
- §5 duplicate X sentences (X is defined in §4.1).
- Unused `uber:h3` bib entry.

## Open

- Bounded (pixel distance <= 2) unseen-site numbers are not yet read; whether
  SOI joins the unseen-site frontier is a hypothesis until they are.
- Between-method site bootstrap per regime is not implemented.
- Clean inference tails would need a serial single-core re-timing from the
  saved fit checkpoints; not planned unless the tail matters.
