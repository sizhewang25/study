# S-P Prediction Taxonomy Module — Plan

## Background

§4.2 of the paper explains Shortest Ping (S-P) by the has-X / no-X split. Its explanation of
the deviations (has-X misses, where no-X answers land) was read by hand from
`sp_baseline.manifest.json`, whose `has_x_misses` rule is a residual ("inflated if min RTT
above every correct has-X TG; **else near-tie**"). The 2026-11-02 review checked these by
hand with a scratchpad prototype (kept in `prototype/`). Every wrong S-P prediction got a
tested cause, and the result changed the paper's story:

- has-X misses (6.7% of has-X) = **3.7 pp inflated + 3.0 pp near-tie**, all AS-A. X_in ≠ X: 0%.
- X_in ≠ X shows up only in no-X TGs (≈17% of no-X, AS-B/AS-C). It moves *where* a wrong
  answer lands, never whether it is right: a no-X cell holds no interconnect.
- The single correct no-X TG is a measurement gap (26 of 134 VPs measured it; the three
  winning VPs at X are missing), not a no-X success.

A number the paper quotes must come from a v5 artifact (`create_paper_artifacts.sh` rule).
This task turns the prototype into a module, `report-sp-taxonomy`, that runs per run or
pooled. It also adds the analysis-side min-VPs-per-TG filter that removes the
measurement-gap TGs.

## Context

- **Prototype** (`prototype/`, run from the repo root with `PYTHONPATH=. .venv/bin/python`):
  - `sp_causes.py` builds per-TG features: own-cell VPs, 1 ms tie set and the cells it spans,
    ρ(RTT, VP–TG distance), paths via X and via K.
  - `ingress.py` runs per-interconnect tests over every VP: floor feasibility at 100 km/ms,
    Spearman ρ of RTT against the VP→I→TG path, excess of min RTT over X's best-case floor.
  - `final_causes.py` assigns causes in order and prints the table.
  - `q1.py` digs into the single correct no-X TG.
- **Reuse, don't re-derive:**
  - `sp_pni_cells.load_runs` / `tg_frame`: cells, X, K, correct, has-X, the check against
    the clusters.
  - `figure_sp_interconnect.load_edges`, `TIE_MS`, `FLOOR_KM_PER_MS` (100), `low_rho` (0.3).
  - `pni_gap.checked_source_csv`, `pooled_output_dir`, `cross.use_group`.
- **Output location:** beside the clusters, `_cross/pni-gap/<group>.seen/` (pooled) or the
  per-run pni-gap dir, like `sp_pni_cells`.
- **Taxonomy (first match wins; thresholds as constants in the module):**

  | Code | Cause | Test |
  |---|---|---|
  | 0 | correct | classify `cell_label == correct` and solved |
  | A | inflated (shared delay) | min RTT − best-case RTT via X > 30 ms **and** ρ(RTT, dist) < 0.3 |
  | B | X_in ≠ X, proven | some VP below the floor via X **and** ≥1 other interconnect feasible |
  | C | near-tie | VPs within 1 ms of the min span > 1 cell, S-P not in X's cell |
  | D | X_in ≠ X, by fit | S-P not in X's cell; best-ρ interconnect ≠ X; S-P in its cell |
  | E | expected no-X | S-P in X's cell, X outside the TG's own cell |
  | F | residual | anything else |

- **Prototype result on `pro-paper.seen`** (the regression target, % of all TGs):
  - has-X: 0 = 49.1, A = 2.0, C = 1.6
  - no-X: 0 = 0.1, A = 1.6, B = 4.4, C = 1.5, D = 3.7, E = 35.9, F = 0.2
  - 20 AS-A TGs (2 sites) have one VP below the floor via *every* listed interconnect.
    They land in E. Report them as a separate flag (list gap or VP location), not as B.

## Goals

1. `report-sp-taxonomy --run-id R` (per run) and `--group G` (pooled) write
   `sp_taxonomy_tgs.csv` and `sp_taxonomy.report.json`. The JSON has shares per side × cause,
   overall and per run, with TG and site shares and the median evidence columns.
2. The pooled numbers reproduce the prototype table above exactly, apart from TGs the min-VP
   filter removes, which are reported.
3. An analysis-side filter, `analysis.common.min_vps_per_tg` (or a fleet fraction), applied
   once where v5 loads per-TG rows so that every method and figure sees the same TGs. Every
   manifest records the dropped TGs.
4. `report-paper` §4 emits a `tab:sp-taxonomy` block, and `create_paper_artifacts.sh` calls
   the command in §4.2.
5. Tests cover each cause on a synthetic fixture, and the pooled regression test passes.

## Approach

- Build `modules/sp_taxonomy.py` on `sp_pni_cells.load_runs`, which already gives edges,
  PNIs, seeds and the TG frame per run. Add the interconnect-path matrix per TG (vectorized
  per run, not a Python loop per TG if avoidable).
- Write the cause rule as one pure function over the per-TG frame, so tests can feed rows
  directly.
- Keep the confidentiality rules: the CSV and JSON carry no coordinates, VP ids or
  interconnect ids, and the paper block prints percentages only.
- **min-VP filter:** apply it in v5 (`classify` / `bench_inputs.eval_observations`), not in
  the benchmark `source_kwargs.min_obs`, because:
  - `min_obs` counts raw CSV rows, not distinct VPs;
  - it changes fold composition;
  - it forces a rerun of every combo.

  In v5 the benchmark outputs stay as they are and the filter only decides which TGs are
  scored. On the meshes, any threshold from 81 to 130 drops exactly 2 TGs (26 and 80 VPs; the
  next lowest TG has 130). Pick 130, or 0.9 × fleet, so the threshold carries over to
  LOSO/RIPE.

## Caveats

- **The floor test (B)** assumes great-circle paths at 2/3 c and a complete interconnect
  list. "No listed interconnect fits" means a list gap or a wrong VP location, not an
  ingress.
- **D is weaker than B.** On correct has-X TGs the best-ρ interconnect is X only 78% of the
  time, because nearby interconnects fit almost as well. Call D "by fit" in the paper.
- **The 30 ms inflation threshold** sits in a gap in the data: no TG falls between 14.4 ms and
  59 ms, and the correct has-X maximum is 8.8 ms. Assert that the gap exists rather than
  trusting the constant on new data.
- **About 20 replicas share a site.** Report site shares next to TG shares; A and C are each
  1–3 sites.
- **S-P tie-break:** 5 TGs tie exactly and `classify` breaks the tie differently from
  `plot-pni-gap`. Use classify's S-P, as `sp_pni_cells` does.
- **The min-VP filter shifts every pooled percentage** a little (1,269 → 1,267 TGs).
  Re-run `report-paper` and re-check the quoted numbers once it lands.
- **Calibration leakage:** the dropped TGs still served as calibration TGs in other folds.
  The filter is scoring-only, so state that in the methodology.
