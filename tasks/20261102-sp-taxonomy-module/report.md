# S-P Prediction Taxonomy Module — Report

**Status**: In Progress
**Created**: 2026-11-02
**Last Updated**: 2026-11-02

## Summary

Task created from the 2026-11-02 review of §4.2. The prototype in `prototype/` produced the
taxonomy below. The module still has to be built.

## Findings

Prototype, pooled `pro-paper.seen`, 1,269 TGs (counts are internal; the paper uses shares):

| Side | Cause | % all TGs | % of side | Networks |
|---|---|---|---|---|
| has-X | correct | 49.1 | 93.3 | A, B, C |
| has-X | A inflated | 2.0 | 3.7 | A |
| has-X | C near-tie | 1.6 | 3.0 | A |
| no-X | correct | 0.1 | 0.2 | B (measurement gap) |
| no-X | A inflated | 1.6 | 3.3 | A |
| no-X | B X_in ≠ X proven | 4.4 | 9.3 | B, C |
| no-X | C near-tie | 1.5 | 3.2 | B, C |
| no-X | D X_in ≠ X by fit | 3.7 | 7.8 | B, C |
| no-X | E at X (expected) | 35.9 | 75.9 | A, B, C |
| no-X | F residual | 0.2 | 0.3 | B |

Evidence medians:

- **Inflated:** min RTT ≈ 59–63 ms, ρ ≈ 0, about 15 VPs within 1 ms across 6 cells.
- **Near-tie (has-X):** min RTT 2.7 ms, ρ = 0.98; the 1 ms tie spans 2 cells, including the
  own cell. Correct has-X TGs never have a tie spanning more than one cell.

Low-coverage TGs: 26 VPs (AS-B, no-X, correct) and 80 VPs (AS-A, has-X, correct). Every
other TG has ≥130 of 134.

## Conclusions

<To be filled when the task completes.>

## 2026-11-02 — min-VPs-per-TG filter landed

- Config: `configs/groups/pro-paper.yaml` → `analysis.common.min_vps_per_tg: 130`.
- Code: `modules/coverage.py` (count = distinct VPs per TG in the edge CSV), `labels.declared_min_vps_per_tg`,
  `classify.score_rung/score_for_run(min_vps=)`, `pni_gap.run_population/compute_runs(min_vps=)`,
  `figure_pni_gap.build_for_runs(min_vps=)`, CLI `classify` / `plot-pni-gap`; `figure_vp_proximity.load` keeps scored TGs only.
- Tests: `tests/test_coverage.py` (13 pass). Full v5 suite: 1,163 pass, 3 fail — all 3 fail on the stashed tree too (pre-existing).
- Sandbox rebuild (scratchpad v5sb): AS-A 398, AS-B 411, AS-C 458 = 1,267 TGs. S-P no-X correct 0.17% → 0%;
  has-X 93.3% → 93.3% (622/667); has-X misses unchanged (25 inflated + 20 near-tie);
  S-P in interconnect cell 99.92% → 100%; ρ 0.857 → 0.858 and 0.381 → 0.380.
