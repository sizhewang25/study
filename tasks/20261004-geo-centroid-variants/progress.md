# Progress Log

## 2026-10-04 16:30 - Task created
Geo-combo benchmark finished 16:13 with no errors. Every fold of
pro-as0{1,2,3}-{mesh,loso} has `run.json` for both `_geo` combos. Dispatched a
subagent to compute the paired differences and write `report.md`.

---

## 2026-10-04 16:20 - Paired GEO vs original comparison (subagent)
Scored OCT-H-GEO / OCT-S-GEO in memory with the v5 scorer against the existing
nside-128 answer spaces (nothing written under outputs/). Re-scoring the
originals reproduces the stored `_tgs.parquet` exactly. Invariants pass: all
folds present for both arms (5 and 20/22/23), identical tg_id sets per fold,
run.json identical apart from ctr, status identical (all SUCCESS, 0% unanswered).
Pooled Δ cell (GEO − orig, site CI): OCT-H K-fold +0.32 [−0.16, +1.10],
LOSO −0.08 [−0.24, 0]; OCT-S K-fold 0.00, LOSO −1.18 [−4.23, +0.70] (one
AS-B no-X site straddling a seed boundary). Δp50 within ±1.2 km everywhere, CI
spans 0. LOSO-drop diff-in-diff: OCT-H −0.39 pp [−1.18, +0.16], OCT-S −1.18
[−4.23, +0.70]. Median prediction displacement 0.4-0.9 km. CTR 470 ms → 0.4 ms.
Verdict: not separable; recommend appendix one-liner, not a §6.5 ablation.
Backing script `compare_geo.py`, tables in `tables/`. The harness refused the
subagent's write of report.md, so the caller has the report text.

---

## 2026-10-04 16:45 - report.md saved
The main session saved the subagent's report text as `report.md` after
checking it against `tables/pooled.csv` and `tables/invariants.json`. Next:
review with the user; the recommendation is an appendix line, not §6.5.

---
