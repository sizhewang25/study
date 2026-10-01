# S-P Interconnect Story — Todo

## Phase 0: Inputs and decisions
- [x] Confirm `pni-nyc-60hudson` `pni_asn = 20940` in the AS03 list is intended (other AS03 rows: 2906)
- [x] Decide final figure set with the author: A (CDF), B (RTT vs direct/via-interconnect), C (Δ_VP vs d(TG, nearest interconnect)); drop the cluster RTT boxplot and the RTT-under-CDF panel
- [x] Decide whether the paper keeps Ward cluster names (C1–C3) or describes the three groups by geometry only

## Phase 1: Reproducible numbers (v5)
- [x] `pni_gap`: per-TG frame with S-P VP's nearest interconnect, d(S-P VP, it), whether it equals the TG's nearest
- [x] Report: share of TGs whose S-P VP is within R km of an interconnect vs random-VP baseline, R ∈ {25, 50, 100} (50 km: 92.3% vs 31.5%)
- [x] Report: VP coverage of every interconnect (all 34 within 50 km)
- [x] Report: per-target Spearman ρ(VP distance, VP RTT) by cluster (C1 0.95, C2 0.86, C3 −0.04); share with ρ < 0.3 (4.4%); VPs within 1 ms of the lowest (3 / 3 / 15)
- [x] Report: nearest-interconnect floor violations (4 sites, 57 TGs) vs S-P-VP-interconnect violations (0); share of TGs whose S-P VP's interconnect ≠ TG's nearest (18%, 229/1,269)
- [x] Report: floor-paragraph numbers — S-P RTT / floor beyond 100 km (median 2.0×, IQR 1.7–2.5×); nearest VP when Δ_VP > 100 km (median 18 km, 20.9 ms, 1.8× lowest)

## Phase 2: Figures (v5)
- [x] Fig. B: `figure_sp_interconnect.py` + CLI `plot-sp-interconnect` (per-run + pooled), linear x, floor + 2× floor, manifest with ρ / ratios / violations
- [x] Fig. B tests: floor-violation count, path construction, privacy (no coordinates, no interconnect names)
- [x] Fig. C: relabel axis/legend "PNI" → "interconnect" in `figure_pni_gap.py`; regenerate pooled
- [x] Wire Fig. B into `create_analysis_artifacts.sh` (per-run + pooled, gated on declared lists)
- [x] Export A, B, C into `cbg-benchmark-paper/figs/` with stable names

## Phase 3: Paper
- [x] Replace each red placeholder in the S-P subsection with the v5 value (or cut the sentence)
- [ ] Write captions for Figs. A, B, C (TODO now)
- [ ] Add Ward 1963 + Rousseeuw 1987 citations if clusters stay
- [x] Keep the routing-policy conjecture red until confirmed

## Phase 4: Verification
- [x] Rerun all v5 tests (`python -m pytest scripts/analysis/v5/tests -q`)
- [x] Re-derive every number quoted in the subsection from the v5 artifacts once more
- [x] Driver run on the default group: 0 failures, pooled PNI figures produced
