# S-P Interconnect Story — Lessons

## 2026-10-01

- **Interconnect lists must include settlement-free peering, not only private
  business interconnects.** The PNI-only AS03 list produced a spurious cluster
  (sites 340–875 km from any listed PNI with Δ_VP = 0) and 10 physics
  violations (RTT shorter than any path through the nearest PNI). Adding two
  operator-confirmed peering locations removed both. A floor violation is the
  diagnostic: RTT < (d(VP, X) + d(X, TG)) / 100 means traffic cannot cross at X.
- **"By construction" can be the finding.** Routing through the interconnect
  beside the S-P VP fits every RTT because S-P *is* the VP at the crossing.
  Back it with a non-circular baseline (random-VP share at an interconnect)
  and an independent test (nearest interconnect, chosen without the S-P result).
- **Don't call a near-tie "inflated".** C3's lowest RTT is ~1.5× its own floor.
  What fails is that RTT stops varying with VP distance (per-target ρ ≈ 0, 15
  VPs within 1 ms). State the condition as measured, not as a judgement.
- **Gap vs d_sp.** For the interconnect argument use Δ_VP (it removes the
  coverage objection by construction); for RTT physics use d_sp (RTT ≥ d_sp/100;
  Δ_VP's correlation with RTT is inherited from d_sp, partial ρ ≈ −0.10).
- **Ward over k-means on few points.** k-means misplaced a boundary point into
  a 5-point cluster (only negative silhouette of 80); Ward kept it with its own
  site. Number clusters by site count, and note that numbering changes when
  inputs do.
- **Unconfirmed mechanisms stay red.** Routing policy as the reason traffic
  crosses a non-nearest interconnect is a conjecture until the operator
  confirms it.
- **Any edit to an interconnect list invalidates its clusters, even a cosmetic
  one.** Anonymising `pni_asn` changed every list's sha256, and
  `checked_source_csv` refused the old clusters until `plot-pni-gap` reran. That
  is intended (the output directory is keyed on the file stem, not content);
  the driver reruns the chain in order.
