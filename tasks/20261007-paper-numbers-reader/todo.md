# Paper Numbers Reader — Todo

## Phase 0: Inventory (confirm with user, section by section)
- [ ] Confirm the formatting rules (e3, %, pp, ×, ms/s, ranges)
- [ ] Confirm Intro stats (I.1–I.4)
- [ ] Confirm §3 Methodology stats (M.1–M.13)
- [ ] Confirm §4 Error distance stats (E.1–E.33)
- [ ] Confirm §5 Region classification stats (R.1–R.65)
- [ ] Confirm §6 Overhead stats (O.1–O.14)
- [ ] Confirm §7 Comprehensive stats (S.1–S.10)
- [ ] Confirm Appendix A/B stats (A.1–A.2, B.1–B.7)

## Phase 1: Framework
- [x] Shared formatter module (rules from Phase 0, unrounded inputs only)
- [x] Claim registry: ID, section, template, reader; artifact paths from the group's pooled names
- [x] `report-paper --group <id> [--section N]` command; `.md` + `.json` output under `_cross/paper/<group>/`
- [x] ~~Optional `--check` against the `.tex`~~ dropped: the reader must not depend on the paper project

## Phase 2: Section readers (data tables per figure / table; no ratios, no sentences)
- [x] §4 (rebuilt as data tables)
- [x] §3 (Intro is a recap: nothing of its own)
- [x] §5
- [x] §6, §7
- [x] Appendices A, B
- [ ] Missing artifact: R.42 (SPO-only peripheral / >2 px share)

## Phase 3: Verification
- [x] Tests: formatter rules, pinned rows per section
- [x] Run against the real tree for every section
- [ ] Document in the v5 README and the paper-commands note
