# Paper Numbers Reader — Report

**Status**: In Progress (§4 reader built; report awaiting review)
**Created**: 2026-10-07
**Last Updated**: 2026-10-07

## Summary

The statistics inventory (plan.md) lists every number the paper quotes as of
cbg-benchmark-paper `258afd7`, in section order: Intro, §3–§7, Appendices A–B. Each
entry carries its artifact source and proposed format. The §4 reader is built (below).

## Findings

- Motivation, Discussion and Appendix C contain no statistics; Related Work is not input.
- Known ⚠ values carried over from the 2026-10-05 sweep:
  - §3: M.6
  - §4: E.8, E.17, E.19, E.21, E.23, E.27, E.30
  - §5: R.3, R.4, R.12
  - §6: O.11, O.13
  - §7: S.7, S.9, S.10
  - Appendices: A.1, B.3
- One statistic has no artifact yet: R.42, a red note.

### 2026-10-07: §4 reader

- `report-paper --group pro-paper --section 4` writes
  `outputs/analysis/v5/_cross/paper/pro-paper/paper_numbers.s4.{md,json}`.
- Code: `scripts/analysis/v5/paper/{fmt,core,s4_error_distance}.py`; tests:
  `tests/test_paper_reader.py`.
- 30 statistics + the percentile table. The reader needs nothing from the paper project:
  the `.tex` check was removed (user, 2026-10-07), and comparing the report with the text is
  the user's step. Values that differ from the current text, found during the build:
  - E.8: 6.7–6.8×
  - E.19: 3.13
  - E.21: 8.4×
  - E.23: 25.9
  - E.27: +0.81
  - E.30: 4.26
- E.14, E.15, E.17, E.25 and E.26 report the data behind a qualitative sentence.
- `pni_gap` manifest now keeps ρ and the cluster stats at 6 dp; they were 3 dp, the source
  of the double rounding.

## Conclusions

<To be filled when the reader is complete.>
