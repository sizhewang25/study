# S-P Interconnect Story — Report

**Status**: In Progress
**Created**: 2026-10-01
**Last Updated**: 2026-10-01

## Summary

Pooled over pro-as01/02/03 (1,269 TGs, 65 sites), with the interconnect lists
completed by settlement-free peering locations, the S-P VP sits at an
interconnect for 92.3% of targets (random VP: 31.5%), and every lowest RTT is
consistent with traffic crossing at the interconnect beside the S-P VP. The
condition is that RTT still varies with VP distance; it holds for 95.6% of
targets and fails for the AS01 West-Coast pair (C3). Every number below is now
reproduced by v5 — `plot-sp-interconnect` writes them all to
`_cross/pni-gap/3-runs-379a99/sp_interconnect.report.json` — and the headline
ones are pinned by `test_figure_sp_interconnect.TestTheRealPaperNumbers`.

**2026-10-01 — v5 implementation.** New `figure_sp_interconnect.py` + CLI
`plot-sp-interconnect` (Fig. B, per-TG CSV, report), wired into the driver
per run and pooled (driver: 41 ok, 0 failed, 0 skipped). Fig. C relabelled
"interconnect". Shared staleness check moved to `pni_gap.checked_source_csv`.
Figs. A/B/C exported to `cbg-benchmark-paper/figs/` (byte-identical to the
driver's). Every red number in the S-P subsection turned black against the
report; the 25/100 km sensitivity added; still red: the routing-policy
conjecture and the three caption TODOs. 930 v5 tests pass.

**2026-10-01 — paragraph reframed.** The S-P subsection of
`cbg-benchmark-paper/sections/eval-on-error-distance.tex` now follows
definitions → distribution → "S-P locates the interconnect" (with the
latency condition) → ceiling (C1) → floor (C2, C3) → non-nearest interconnect
→ conclusion. Every scratchpad-only number, and Fig. B (an `\fbox`
placeholder, label `fig:rtt-via-interconnect`), is red; v5-backed numbers
(CDF zero share, cluster sizes, per-cluster lowest-RTT medians) are black. The
cluster RTT boxplot is commented out. The routing-policy conjecture is red.
Builds with latexmk.

## Findings

### Decomposition and distribution [v5: plot-vp-distance-cdf]
- d_sp = d_geo + Δ_VP. Δ_VP = 0 for 19.1% of TGs; 50.3% > 100 km; 6.8% > 1,000 km; max 3,922 km.
- d_geo: about half of the 65 sites within 15 km (median 14.2 km per site, 13.8 per TG); p90 80 km; max 374.5 km.

### S-P locates the interconnect
- Every interconnect (14 / 9 / 11) has a VP within 50 km (all 34).
- S-P VP within 50 km of an interconnect: 92.3% of TGs (AS01 90%, AS02 95%, AS03 92%) vs 31.5% for a random VP measuring the same TG (2.9×). Distinct S-P VPs at an interconnect: 14/19, 12/14, 19/20.
- Sensitivity: 25 km 59.9% vs 17.9% (3.4×); 100 km 99.8% vs 36.7% (2.7×).
- Farthest interconnect from its nearest VP: 48.6 km (AS01), 35.6 km (AS02, AS03).
- RTT of the S-P VP vs path length (floor = d / 100 km·ms⁻¹):

  | x | ρ | ρ (x>100 km) | RTT/floor | spread | below floor |
  |---|---|---|---|---|---|
  | direct d(S-P VP, TG) | 0.88 | 0.84 | 1.98 | 0.16 | 0 |
  | via TG's nearest interconnect | 0.92 | 0.90 | 1.90 | 0.15 | 4 sites / 57 TGs |
  | via S-P VP's nearest interconnect | 0.93 | 0.95 | 1.91 | 0.12 | 0 |

- S-P VP's nearest interconnect = TG's nearest for 1,040/1,269 TGs (82%); 229 (18%) differ.

### The condition: RTT must vary with VP distance
- Per-target Spearman ρ(VP distance, VP RTT), median: C1 0.95, C2 0.86, C3 −0.04. ρ < 0.3 for 4.4% of TGs (all of C3, 5.2% of C2, none of C1).
- VPs within 1 ms of the lowest RTT, median: 3 / 3 / 15. RTT range (p90 − min): 60 / 56 / 18 ms.
- C3's lowest RTT ≈ 1.5× its own floor (~60 ms at ~3,900 km): not inflated; the ranking is a near-tie.

### Clusters [v5: plot-pni-gap --layout pooled, Ward]
- k = 3 (silhouette 0.746; k-means identical, ARI 1.0; 0 negative-silhouette points).
- C1 near interconnect: 643 TGs (51%), 33 sites (51%); d 1–41 km; Δ 0–119 km; smallest RTT median 1.5 ms [v5: plot-pni-cluster-rtt].
- C2 far: 601 TGs (47%), 31 sites (48%); d 123–1,081 km; Δ 0–3,922 km; smallest RTT median 10.5 ms.
- C3 AS01 pair (SeaTac + 5 San Jose IPs): 25 TGs (2%), 2 sites (3%); d 19 km; Δ 3,295–3,848 km; smallest RTT median 59.3 ms.
- Before the AS03 list was completed, a 4th cluster (AS03 sites 340–875 km from any listed PNI with Δ = 0) existed; it was an artefact of missing peering locations.

### Ceiling and floor
- Ceiling = d_geo; reached iff Δ_VP = 0. Δ_VP < 100 km: lowest RTT median 1.5 ms, p90 2.4 ms.
- Floor set by Δ_VP, not coverage: largest errors 3,300–3,900 km with nearest VP 2.5–24 km away; worst-covered TG (374 km) errs 920–930 km.
- Δ_VP > 100 km (638 TGs): nearest VP median 17.6 km away but 20.9 ms; S-P VP 10.5 ms (ratio 1.76×).
- S-P RTT / floor beyond 100 km: median 1.98×, IQR 1.72–2.52×.

### The four nearest-interconnect floor violations
| run | site | TGs | nearest interconnect | S-P VP beside |
|---|---|---|---|---|
| AS02 | western NC (35.82, −81.61) | 20 | Atlanta 342 km | Ashburn |
| AS02 | Denver (39.86, −104.67) | 11 | Dallas 1,053 km | Chicago |
| AS03 | Kansas City (39.10, −94.57) | 20 | Chicago 662 km | Dallas |
| AS03 | Pittsburgh (40.49, −80.23) | 6 | Ashburn 286 km | Chicago |

Each RTT rules out the nearest interconnect and fits the one beside the S-P VP. Candidate evidence for the routing-policy conjecture; not proof.

## Conclusions

<Pending: finalise once Figs. B and the v5 report exist and the paper's red placeholders are cleared.>
