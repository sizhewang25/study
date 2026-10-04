# Technical Context

## Key Findings (before this task)
- Pooled LOSO (`_cross/loso-delta/6-runs-197fee`): OCT-H cell acc 73.5 -> 57.8%
  (-15.8 pp, site CI -24.3..-7.8), p50 33 -> 118 km; OCT-S 69.1 -> 60.2%.
- Site-clustered effective n is ~66-99, not ~1.3k TGs: always bootstrap by site.

## Code References
- Benchmark outputs: `outputs/benchmark/v2/<run>/generic_csv/anchors_to_probes/fold_N/<combo>/{targets.parquet,run.json}`
- v5 answer spaces already built: `outputs/analysis/v5/<run>/answer-space/`
- `classify.load_method_frame`, `classify.score_method`: per-TG `cell_label`, ring offset
- `loso_delta.site_bootstrap`, `sites.site_key`: site-clustered paired bootstrap
- v5 `methods.py` labels: `octant_cbg_hull_geo` = OCT-H-GEO, `octant_cbg_spl_geo` = OCT-S-GEO

## Decisions Made
- The `_geo` combos stay out of every figure for now (user, 2026-10-04).
- Networks are AS-A/B/C in anything paper-facing (pro-as01/02/03).

## Open Questions
- Does the CTR swap change the LOSO drop, or only shift both regimes equally?
