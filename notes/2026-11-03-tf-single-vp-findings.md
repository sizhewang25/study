# Traffic-filtered (TF) runs: sparse TGs, the SPO single-VP rule, and where methods differ (2026-11-03)

TF runs (`pro-tf` group: `pro-as0{1,2,3}-weighted`), seen sites (K-fold), cell (unbounded)
accuracy, the paper's six methods. A TG's **VP count** is `n_obs`, the number of observations
left after the traffic filter. The filter acts on the eval side only; LTDs are still fitted on
the full mesh.

The TF data stays on the TF machine. Every number below was read off the figures, at whole
percent. "≈" marks a segment too small to carry a label, estimated from its height. Shares
only; no counts.

## 0. What changed before these numbers

- **SPO answered too few TGs.** The old SPO (`gaussian_density`) refused any TG with fewer
  than 3 constraints (`MIN_CONSTRAINTS=3`). That was the 18% "no answer" in the first TF
  outcome bars: the targets left with 1–2 VPs after filtering.
- **New rule (15c65c1).** SPO answers with the VP's location when every constraint shares one
  centre: one VP, or several co-located VPs. Two VPs go through the density surface as
  before. This applies to SPO only; the other methods are unchanged. See memory
  `project_spo_single_vp_rule`.
- **SPO was rerun on the TF machine.** Its `spotter_cbg` folds were deleted and rebuilt
  (`--forcerun summarize`), then `create_tf_artifacts.sh` was run.
- **Analysis added:**
  - 3d75b8f: `classify` writes the 1-VP split into `accuracy.csv`: `single_vp_rate`,
    `accuracy_cell_correct_1vp` / `_2plus_vp`, `contrib_1vp`, `agree_sp_1vp`.
  - 136e2b3: `plot-vp-obs-hist`, `plot-vp-obs-has-x`, and outcome bars `--mode vp`.
  - c20180b: the `vp` mode split into 1 / 2 / ≥3 VPs, plus `two_vp_rate` and `*_2vp` /
    `*_3plus_vp`.

## 1. How sparse the filter leaves the TGs

- **VP count per TG:**

  | VPs | 1 | 2 | 3 | 4–5 | 6–10 | 11–20 | 21–50 | >50 |
  |---|---|---|---|---|---|---|---|---|
  | pooled | 11% | 7% | 3% | 9% | 24% | 34% | 12% | ≈0% |
  | AS-A | 17% | 12% | 4% | 10% | 38% | 18% | 2% | 0% |
  | AS-B | 8% | 3% | 4% | 8% | 5% | 45% | 25% | 1% |
  | AS-C | ≈0% | 2% | ≈0% | 5% | 17% | 57% | 18% | 0% |

  The pooled 1-VP share is mostly AS-A, so quote it per network as well.
- **Compared with the mesh:** every mesh TG has more than 100 VPs, and the synthetic
  randweight TF fixture has at least 64. Real traffic is much sparser than either.
- **Sparse TGs and no-X TGs overlap heavily:**

  | | share of TGs | 1 VP | 2 VPs | 3–10 VPs | >10 VPs |
  |---|---|---|---|---|---|
  | has-X | 89% | 6% | ≈5% | 37% | 52% |
  | no-X | 11% | 51% | 25% | 24% | 0% |

  - No no-X TG has more than 10 VPs.
  - The 1-VP TGs split about evenly by side: ≈5.3% of all TGs are has-X and ≈5.6% are no-X.

## 2. Outcomes, pooled

- **Cell outcome per method:**

  | | OCT-H | SOI | S-P | OCT-S | VAN | SPO |
  |---|---|---|---|---|---|---|
  | correct | 90% | 90% | 89% | 87% | 83% | 81% |
  | wrong | 10% | 10% | 11% | 13% | 11% | 19% |
  | no answer | 0% | 0% | 0% | 0% | 6% | 0% |

- **SPO's 18% "no answer" became 19% wrong, not correct.** SPO now answers every TG.
- **Correct and wrong split by VP count** (share of all TGs):

  | | OCT-H | SOI | S-P | OCT-S | VAN | SPO |
  |---|---|---|---|---|---|---|
  | correct, 1 VP | 6% | 6% | 6% | 4% | 6% | 6% |
  | correct, 2 VPs | 5% | 6% | 6% | 4% | 6% | 6% |
  | correct, ≥3 VPs | 79% | 78% | 78% | 80% | 72% | 69% |
  | wrong, 1 VP | 5% | 5% | 5% | 7% | 5% | 5% |
  | wrong, 2 VPs | ≈1–2% | ≈1% | ≈1% | ≈2% | ≈1% | ≈1% |
  | wrong, ≥3 VPs | ≈3% | 4% | 4% | ≈4% | 4% | 13% |

## 3. Per network

- **Cell correct per method:**

  | | OCT-H | SOI | S-P | OCT-S | VAN | SPO |
  |---|---|---|---|---|---|---|
  | AS-A | 89% | 90% | 90% | **80%** | 89% | 90% |
  | AS-B | 86% | 84% | 83% | **92%** | 74% | **63%** |
  | AS-C | 97% | 99% | 99% | 97% | 83% | 86% |

- **Correct on ≥3-VP TGs** (share of all TGs in the network): AS-B: OCT-S 87%, OCT-H 81%,
  SOI and S-P 80%, VAN 71%, SPO 59%. AS-C: SOI and S-P 98%, OCT-H and OCT-S 96%, SPO 85%,
  VAN 82%.
- **Wrong on ≥3-VP TGs:**
  - SPO: AS-B 30%, AS-C 14%.
  - Every other method: AS-B 8–9%, AS-C 1–3%.
- **AS-A's sparse TGs:**
  - Methods other than OCT-S: correct 10% on 1 VP and 9–11% on 2 VPs.
  - OCT-S: correct 6% on 1 VP and 6% on 2 VPs; wrong 11% on 1 VP and 6% on 2 VPs.
- **VAN's no answers:** AS-B 9%, AS-C 15%, AS-A ≈0%.

## 4. Findings

1. **On 1–2-VP TGs every method answers like S-P, except OCT-S.**
   - Pooled, the 1-VP tier is 6% correct and 5% wrong for OCT-H, SOI, S-P, VAN and SPO, so
     1-VP accuracy is ≈55%. OCT-S: 4% / 7%, ≈36%.
   - On 2-VP TGs, every method but OCT-S is ≈80–85% correct, with S-P's share.
   - So these TGs add about the same accuracy to every method; they do not reorder them.
   - Check: `agree_sp_1vp` in `accuracy.csv` should be ≈1.0 for VAN, SOI and SPO by
     construction, and lower for OCT-H and OCT-S.
2. **OCT-S's AS-A deficit (80% vs 89–90%) comes entirely from sparse TGs.**
   - It loses ≈4 pp on 1 VP and ≈5 pp on 2 VPs, and ties the others on ≥3 VPs (69%).
   - Mechanism, not yet verified: with one or two rings, the spline LTD's annulus (inner
     radius > 0) puts the medoid outside the VP's cell. OCT-H's medoid also moves off the VP
     but stays in the same cells.
3. **SPO's deficit is on well-measured TGs, not sparse ones.**
   - This refutes the two-VP "mirror point" hypothesis: SPO's 2-VP share correct (6%) equals
     S-P's.
   - The excess sits in the ≥3-VP wrong group (13% vs 3–4%), almost all in AS-B (30%) and
     AS-C (14%).
   - On AS-A, SPO ties the best method (90%). On AS-B and AS-C it is wrong on TGs that S-P
     gets right with many VPs: a problem in SPO's density map (LTD/MTL), not in coverage.
4. **CBG beats S-P only on ≥3-VP TGs, and only on AS-B.**
   - On AS-B: OCT-S +7 pp over S-P on ≥3 VPs (87 vs 80) and +9 pp overall (92 vs 83);
     OCT-H +3 pp.
   - On AS-A every method but OCT-S ties.
   - On AS-C, S-P and SOI are at 99% and no CBG method beats them.
   - The ranking flips by network, consistent with the regime-dependent findings on the mesh.
5. **1-VP accuracy depends on the network.**
   - AS-A ≈60% (10 of 17 pp correct); AS-B ≈35–40% (≈3 of 8 pp); AS-C has no 1-VP TGs.
   - Hypothesis, not verified: a 1-VP TG is correct exactly when it is has-X, i.e. its cell
     holds the interconnect, as S-P's cell accuracy does on the mesh (memory
     `finding_sp_cell_accuracy_is_pni_cell`). The pooled 1-VP split, 6 / 5 pp, matches the
     has-X / no-X split, 5.3 / 5.6 pp. The appendix script tests it.
6. **VAN's 6% no answers are refusals on ≥3-VP TGs** (AS-B 9%, AS-C 15%), not sparse TGs.
   VAN always answers a 1-VP TG. On answered TGs, VAN's wrong share is the same as the
   others'.
7. **Distance ranks the methods differently from cell accuracy** (error CDF, d/D ×10⁻³):
   - OCT-H and OCT-S: median ≈1.5.
   - S-P and SOI: median ≈10, with S-P's step shape; ≈15% of TGs near 0.
   - VAN: close to S-P; its CDF stops at 0.94 (the no answers).
   - SPO: median ≈25–30, and ≲5% of TGs below 5.
   - So OCT-H ties S-P on cells but is ≈6× closer at the median. SPO is fourth on cells and
     last on distance: the far-but-correct-cell pattern seen on randweight. Never quote one
     axis alone.

## 5. For the paper

- Suggested sentence: "After traffic filtering, 11% of TGs are measured by a single VP
  (AS-A 17%, AS-B 8%, AS-C ≈0%). On these every method but OCT-S answers in S-P's cells, so
  they add the same ≈6 pp to each method's accuracy; method differences arise among TGs with
  three or more VPs, and differ by network."
- Report the 1-VP share per network, not only pooled.
- SPO's TF accuracy is post-rule (15c65c1). The rule needs one line in the method
  description: a one-centre target is answered with the VP, the ring's mean.

## 6. Open

- [ ] Run the appendix crosstab to test finding 5 (1-VP correct ≈ has-X).
- [ ] Read `agree_sp_1vp` per method from `accuracy.csv` (finding 1).
- [ ] SPO on AS-B / AS-C ≥3-VP TGs where SPO is wrong and S-P is right: SPO's answer
      distance from the nearest VP, against that VP's RTT and the VPs' spread. Is the
      density flattened by a few distant traffic VPs, or is AS-B's LTD fit off?
- [ ] OCT-S on AS-A 1–2-VP TGs: confirm the annulus-inner-radius mechanism (finding 2).

## Appendix: VP-bin × has-X crosstab

Pooled share of TGs per method × VP bin (1 / 2 / ≥3) × has-X side × cell label. Needs
`classify` and `report-sp-pni-cells` outputs for each run (`create_tf_artifacts.sh` builds
both). Usage: `.venv/bin/python tf_vp_crosstab.py outputs/analysis/v5`.

```python
"""Pooled cell-label shares by VP bin (1 / 2 / >=3) and has-X, per method."""
import glob, sys
import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "outputs/analysis/v5"
runs = sys.argv[2:] or ["pro-as01-weighted", "pro-as02-weighted", "pro-as03-weighted"]
methods = ["shortest_ping", "million_scale_cbg", "vanilla_cbg",
           "octant_cbg_hull", "octant_cbg_spl", "spotter_cbg"]
frames = []
for run in runs:
    hx = pd.read_csv(glob.glob(f"{root}/{run}/pni-gap/*/sp_pni_cells_tgs.csv")[0])
    hx = hx[["tg_id", "tg_cell_holds_x"]].drop_duplicates("tg_id")
    for m in methods:
        df = pd.read_parquet(f"{root}/{run}/classify/healpix-128/{m}_tgs.parquet")
        df = df.merge(hx, on="tg_id", how="left", validate="many_to_one")
        df["run"], df["method"] = run, m
        frames.append(df[["run", "method", "tg_id", "n_obs", "cell_label", "tg_cell_holds_x"]])
d = pd.concat(frames)
d["vp_bin"] = pd.cut(d["n_obs"], [0, 1, 2, float("inf")], labels=["1", "2", ">=3"])
d["side"] = d["tg_cell_holds_x"].map({True: "has-X", False: "no-X"})
n = d.groupby("method").size().iloc[0]
t = (d.groupby(["method", "vp_bin", "side", "cell_label"], observed=True).size() / n * 100).round(1)
print(t.unstack("cell_label").fillna(0).to_string())
```

For a per-network table, pass one run id at a time.
