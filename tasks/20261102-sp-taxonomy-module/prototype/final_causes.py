import sys
import numpy as np, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
t = pd.read_csv(sys.argv[1])
t["best_is_x"] = t.best_idx == t.x_idx
t["pred_in_best"] = t.pred_seed_id == t.best_seed
def cause(r):
    if r.correct:
        return "0 correct"
    if r.excess_over_x_floor_ms > 30 and r.rho_tg < 0.3:
        return "A inflated: shared delay, RTT ignores distance"
    if not r.x_feasible and r.n_feasible > 0:
        return "B X_in!=X proven: RTT impossible via X, another X fits"
    if r.tied_cells > 1 and not r.pred_in_x_cell:
        return "C near-tie: <1 ms tie spans cells"
    if not r.pred_in_x_cell and not r.best_is_x and r.pred_in_best:
        return "D X_in!=X by fit: S-P lands at best-fitting X"
    if r.pred_in_x_cell:
        return "E S-P at X; X outside own cell (expected no-X)"
    return "F residual"
t["cause"] = t.apply(cause, axis=1)
t["side"] = np.where(t.has_x, "has-X", "no-X")
N = len(t)
def tab(df, by):
    g = df.groupby(by)
    side_n = t.groupby("side").size()
    out = pd.DataFrame({
        "TGs": g.size(), "sites": g.site_key.nunique(),
        "%all": (g.size() / N * 100).round(1),
        "runs": g.run_id.agg(lambda s: ",".join(sorted({"ABC"[int(x[7]) - 1] for x in s}))),
        "minRTT_ms": g.min_rtt_ms.median().round(1),
        "excess_ms": g.excess_over_x_floor_ms.median().round(1),
        "rho_dist": g.rho_tg.median().round(2),
        "tie_cells": g.tied_cells.median(),
        "rho_X": g.rho_x.median().round(2), "rho_best": g.rho_best.median().round(2),
        "VPs<floorX": g.n_vps_break_x.median(),
        "pred_in_X_cell": (g.pred_in_x_cell.mean() * 100).round(0),
        "d_sp_km": g.d_sp_km_classify.median().round(0), "d_X_km": g.d_tg_x_km.median().round(0),
    })
    if by[0] == "side":
        out.insert(3, "%side", (out.TGs / side_n.reindex(out.index.get_level_values(0)).values * 100).round(1))
    return out
print(tab(t, ["side", "cause"]).to_string())
print()
print(tab(t[~t.correct], ["cause"]).to_string())
t.to_csv(sys.argv[2], index=False)
