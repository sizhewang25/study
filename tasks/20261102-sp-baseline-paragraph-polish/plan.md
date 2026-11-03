# §4.2 S-P Baseline Paragraph Polish — Plan

## Background

On 2026-11-02 the review of `cbg-benchmark-paper/sections/eval-on-error-distance.tex` §4.2
(lines 40–54) found several problems:

- **Wrong or misread numbers:**
  - "none" of the no-X TGs, but the data says <1%;
  - "88% in other cells", which is really the share in an *adjacent* cell;
  - nearest VP 3.5, which should be 3.6;
  - RTT ratio 6.3–8.2×, which should be 5.4–9.6×.
- **An overstated "clear separation"** in min RTT.
- **A loose has-X-miss explanation:** the near-tie label was a residual, and the third
  inflated marker in Fig. 5(b) went unmentioned.
- **Typos** throughout.

The S-P taxonomy prototype (`tasks/20261102-sp-taxonomy-module`) then split the misses into
tested causes:

- has-X misses = 3.7 pp inflated + 3.0 pp near-tie, all AS-A;
- X_in ≠ X only in no-X TGs (≈17%).

This task rewrites §4.2 around those findings in plain prose (humanizer pass).

## Context

- **Paper:** `cbg-benchmark-paper/sections/eval-on-error-distance.tex`, §4.2 = lines 40–54;
  Table `tab:x-cell`; Figs 5–7 and `fig:boxplot-rtt-of-x-cell-by-asn`.
- **Draft:** `draft-v1.tex` in this folder (humanizer pass, 2026-11-02). The paper file is
  not edited yet.
- **Number sources:**
  - `_cross/pni-gap/pro-paper.seen/`: `sp_baseline.manifest.json`, `sp_pni_cells.report.json`,
    `x_cell_rtt.csv`, `sp_interconnect.report.json`.
  - The taxonomy prototype, for anything marked `\rednote{taxonomy}`.
- **Paper rules (memory):**
  - percentages only, no TG or site counts;
  - method abbreviations (S-P, OCT-H, …);
  - "site", never "seed";
  - networks are AS-A/B/C;
  - characterize deviations from X only.

## Goals

1. Replace lines 40–54 with the polished text once the user approves `draft-v1.tex`.
2. Every number traces to a v5 artifact: `\rednote{taxonomy}` numbers switch to
   `report-sp-taxonomy` output when that module lands.
3. Fix the typos in §4.1 (Quantatitive ×3, closelytracks, literatures) in the same pass.
4. After the min-VPs-per-TG filter lands, refresh every §4 number and drop "almost none
   (<1%)" for "none" if the measurement-gap TG is the only no-X hit.

## Approach

- Keep the user's argument order: mechanism, then has-X/no-X, then deviations, then
  takeaways. Add one short paragraph on X_in ≠ X for no-X TGs.
- Recast the inflated cases as a shared delay that erases the distance signal, not as
  X_in ≠ X. The prototype showed that K ≠ X there is an artifact of a 6-cell tie.
- Run the humanizer checks on each revision: no not-X-but-Y, no one-line closers, no
  connector dashes, no forced triads.

## Caveats

- **The taxonomy numbers are provisional** until the module reproduces them.
- **Check figure references.** `fig:network-topology` and the two `\citeneeded` keys still
  need real targets.
- **Wording to confirm with the user:** "at least one VP at every interconnect" in the
  takeaways rests on §3's `interconnect_coverage` (every interconnect has a VP within 50 km).
- **Routing-policy claim:** "a deliberate routing policy we confirmed internally" is the
  user's claim. The data cannot name the ingress the policy uses (best-fit ρ only
  0.21–0.55).
