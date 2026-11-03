# S-P Prediction Taxonomy Module — Todo

## Phase 0: Setup & Discovery
- [ ] Re-run `prototype/` on `pro-paper.seen` and save the cause table as the regression fixture
- [x] Decide the min-VP filter's form: absolute `min_vps_per_tg: 130` (user, 2026-11-02)
- [x] Hook points: `classify.score_rung` (all `*_tgs.parquet` readers) + `pni_gap.run_population` (all cluster readers)
- [ ] Confirm `FLOOR_KM_PER_MS`, `TIE_MS`, `low_rho` are importable constants in `figure_sp_interconnect`

## Phase 1: min-VPs-per-TG filter
- [x] `analysis.common.min_vps_per_tg` (`labels.declared_min_vps_per_tg`, `modules/coverage.py`); default off
- [x] Applied in classify + pni_gap; `figure_vp_proximity.load` restricted to scored TGs; answer space untouched
- [x] Dropped TGs recorded in classify manifest and pni_gap manifest (`min_vps_per_tg`)
- [ ] `report-dataset`: say the floor and its drops (still counts scored TGs only)
- [x] Set once in `configs/groups/pro-paper.yaml` (6 member runs); sandbox rebuild: tg-ecaeb32 (AS-A, 80) + tg-1391b0a (AS-B, 26) drop, 1,267 TGs
- [ ] Rebuild the real tree: `scripts/analysis/v5/create_paper_artifacts.sh`

## Phase 2: Taxonomy module
- [ ] `modules/sp_taxonomy.py`: per-TG features (tie cells, own-cell tie, ρ(RTT, dist), per-interconnect floor feasibility and path ρ, excess over X floor)
- [ ] Pure `cause()` rule A–F as in plan.md; flag "no listed interconnect fits" separately
- [ ] `sp_taxonomy_tgs.csv` + `sp_taxonomy.report.json` (side × cause, overall and per run; TG and site shares; median evidence)
- [ ] CLI `report-sp-taxonomy` with `--run-id` (per run) and `--group` (pooled)
- [ ] `report-paper` §4 block `tab:sp-taxonomy` (percentages only) and a call in `create_paper_artifacts.sh` §4.2

## Phase 3: Verification
- [ ] Unit tests: one synthetic TG per cause, plus the precedence between A/B/C/D/E
- [ ] Regression: pooled table matches the prototype, apart from the filtered TGs
- [ ] Assert the inflation gap (no TG excess in (14.4, 59) ms) in the report rather than silently thresholding
- [ ] Re-run `report-paper`; diff §4/§5 numbers before and after the filter and list what the paper must update
