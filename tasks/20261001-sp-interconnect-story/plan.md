# S-P Interconnect Story — Plan

## Background

The S-P subsection of `cbg-benchmark-paper/sections/eval-on-error-distance.tex`
("Understanding the Capability Ceiling and Floor of each Method → Shortest
Ping") started as "S-P error follows distance to the operator's PNIs" and has
been reframed through a long exploratory session. The surviving story:

> **Given a VP at every interconnect of the target network, S-P locates the
> interconnect the target's traffic crosses — provided the measured latency
> still reflects propagation delay (RTT varies across VPs with their distance).**
> Its ceiling and floor then follow from where the target sits relative to that
> interconnect.

Several numbers in the reframed paragraph come from scratchpad scripts, not
from reproducible v5 artifacts. They are marked red in the .tex. This task
turns each into a v5 figure/manifest and then clears the red.

## Context

**Substrate.** `pro-as01-mesh`, `pro-as02-mesh`, `pro-as03-mesh`, pooled:
1,269 TGs at 65 sites (sites keyed on `(run_id, lat, lon)`), 134 VPs per mesh.
Each RTT is the p5 of a week of continuous measurement.

**Interconnect lists.** `datasets/pni/as0{1,2,3}-us-pni.approx.csv`, declared in
each config's `analysis.common.pni_csv`. They must contain **both** private
business interconnects (PNIs) **and** settlement-free peering locations. The
original lists held PNIs only; AS03 gained two peering locations (Hillsboro,
New York `60hudson`) confirmed with the operator on 2026-10-01. Counts now:
AS01 14, AS02 9, AS03 11. CSVs are gitignored (`*.csv`), so list provenance
lives in manifests (sha256), not git.

**Existing v5 machinery.**
- `plot-vp-distance-cdf` → `figs/vp_distance_cdf.png` (Fig. A).
- `plot-pni-gap --layout pooled` → Δ_VP vs d(TG, nearest interconnect)
  scatter, Ward clusters (Fig. C). Output:
  `outputs/analysis/v5/_cross/pni-gap/3-runs-379a99/`.
- `plot-pni-cluster-rtt` → smallest-RTT boxes per cluster (to be dropped from
  the paper; numbers go in the text).
- Driver: `scripts/analysis/v5/create_analysis_artifacts.sh` runs both per run
  and pooled.

**Scratchpad demos (not reproducible yet).** `outputs/analysis/v5/_demo/`:
`rtt_vs_vp_distance_vppni_demo.png` (Fig. B prototype),
`rtt_vs_vp_distance_demo.png`, `dsp_vs_pni_demo.png`,
`vp_distance_cdf_rtt_demo.png`. Their scripts are copied into
`demo_scripts/` here (run from the repo root with `.venv/bin` on PATH, output
path as argv[1]); `rtt_vs_dist_vppni_demo.py` is the Fig. B prototype.

## Goals

1. Fig. B as a v5 command: RTT of the S-P VP vs (a) direct d(S-P VP, TG) and
   (b) path through the interconnect nearest the S-P VP; floor and 2× floor;
   manifest with ρ, RTT/floor ratios, floor violations.
2. A v5 report (manifest/CSV, no figure needed) for the claims in ¶3:
   S-P-VP-at-interconnect share vs random-VP baseline at 25/50/100 km; the
   per-target ρ(VP distance, VP RTT) criterion; VP coverage of every
   interconnect.
3. Fig. C relabelled PNI → interconnect and re-exported to the paper repo.
4. Every red number/figure in the S-P subsection replaced by a value from a v5
   artifact, or the sentence removed.

## Approach

- Extend `pni_gap.py` (it already owns interconnect loading, `run_population`,
  pooling guards) with a per-TG "S-P VP's interconnect" frame and the
  per-target ρ; add `figure_sp_interconnect_rtt.py` for Fig. B, following the
  `figure_pni_gap` pattern (per-run + pooled layouts, manifest, privacy tests).
- Keep the PNI-lists-as-config contract (`labels.declared_pni_csv`); refuse a
  pooled run whose list is undeclared.
- Rename display strings only ("interconnect"); keep code identifiers
  (`pni_*`) to avoid churn.

## Caveats

- **By construction is the point, not a flaw.** Fig. B routes through the
  interconnect next to the S-P VP, so its fit partly follows from the choice.
  The non-trivial evidence is the 92.3% vs 31.5% baseline and the zero floor
  violations; Fig. C (nearest interconnect, chosen without the S-P result) is
  the independent test.
- **Condition, not "inflation".** C3's lowest RTT is ~1.5× its own propagation
  floor — ordinary stretch. Say "RTT no longer varies with VP distance", never
  "RTT is inflated".
- **Routing policy is unconfirmed** — stays red in the paper.
- **List provenance.** The AS03 list was revised after outliers surfaced.
  Frame the additions as completing the list with settlement-free peering
  (confirmed with the operator), not as tuning.
- **Open question:** the new AS03 row `pni-nyc-60hudson` has `pni_asn = 20940`
  (AS01's ASN); the other AS03 rows have 2906. Loader ignores the column; ask
  whether it is intended.
- 65 sites; C3 is 2 sites. Quote site counts beside TG counts.
