# S-P Prediction Taxonomy Module — Lessons

## 2026-11-02

- **A residual label is not a test.** `sp_baseline.manifest.json` calls every non-inflated
  has-X miss "near-tie". The cross-cell tie test happened to agree, but only after it was
  run.
- **"K ≠ X" alone does not mean X_in ≠ X.** When a 1 ms tie spans 6 cells, S-P's VP (and so
  K) is effectively random. The inflated has-X misses all have K ≠ X, yet no evidence names
  a far ingress (best-fit ρ only 0.21–0.55). Recasting them as X_in ≠ X was wrong.
- **Check the floor test against every interconnect, not only X.** When nothing fits, the
  cause is a list gap or a VP location, not an ingress.
- **Check measurement coverage before explaining an outlier.** The one correct no-X TG came
  from a replica that 108 of 134 VPs never measured.
