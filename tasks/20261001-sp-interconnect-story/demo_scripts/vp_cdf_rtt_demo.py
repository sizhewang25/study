import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as ticker
from scripts.analysis.v5.modules.paths import resolve_run
from scripts.analysis.v5.modules import figure_vp_distance_cdf as C
from scripts.analysis.v5.modules.figure_vp_proximity import vp_distances
runs = ["pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh"]
d = pd.concat([vp_distances(resolve_run(r)).assign(run_id=r) for r in runs], ignore_index=True)
d = d.rename(columns={"geo_vp_dist_to_tg_km": C.GEO, "sping_vp_dist_to_tg_km": C.SPING})
d[C.GAP] = d[C.SPING] - d[C.GEO]
print(len(d), "TGs; zero gap", round((d[C.GAP] == 0).mean() * 100, 1))
# S-P RTT vs the gap. Zero gaps fold onto the axis floor, as the gap curve does.
from scipy.stats import spearmanr
gx = d[C.GAP].where(d[C.GAP] > 0, C.X_MIN_KM)
r_all = spearmanr(d[C.GAP], d.sping_vp_rtt_ms); r_pos = spearmanr(d[C.GAP][d[C.GAP] > 0], d.sping_vp_rtt_ms[d[C.GAP] > 0])
site = d.assign(gx=gx).groupby(["run_id", C.GEO, C.SPING], as_index=False).agg(gx=("gx", "first"), rtt=("sping_vp_rtt_ms", "median"), n=("tg_id", "size"))
r_site = spearmanr(site.gx, site.rtt)
print(f"spearman gap vs S-P RTT: all TGs {r_all.statistic:.3f}, gap>0 {r_pos.statistic:.3f}, per (site,gap) point {r_site.statistic:.3f} (n={len(site)})")
edges = np.r_[C.X_MIN_KM * 0.9, np.logspace(0, 4, 13)]
b = pd.cut(gx, edges)
g = d.assign(gx=gx).groupby(b, observed=True).sping_vp_rtt_ms.agg(n="size", p25=lambda x: x.quantile(.25), p50="median", p75=lambda x: x.quantile(.75))
g["x"] = [C.X_MIN_KM if i.left < 1 else np.sqrt(i.left * i.right) for i in g.index]
print(g.round(1).to_string())

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(4.6, 3.4), sharex=True, gridspec_kw={"height_ratios": [2.4, 1], "hspace": 0.08})
for key, label, colour, ls in C._SERIES:
    if key == C.GAP:
        x, y, z = C._gap_curve(d[C.GAP].to_numpy())
    else:
        x, y = C._ecdf(d[key].to_numpy())
    ax.step(x, y, where="post", color=colour, ls=ls, lw=1.4, label=label)
ax.plot([C.X_MIN_KM], [z], marker="o", ms=3.5, color=C._GAP_HUE, clip_on=False, zorder=5)
ax.text(C.X_MIN_KM * 1.18, z + 3, f"{z:.1f}%", fontsize=7, color=C._GAP_HUE)
ax.set_ylim(0, 100); ax.set_ylabel("share of TGs (%)", fontsize=8)
ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
ax.legend(loc="lower right", fontsize=7, frameon=False)
# bottom panel: S-P RTT against the gap -- same x as the dashed curve
ax2.fill_between(g.x, g.p25, g.p75, color=C._GAP_HUE, alpha=0.18, lw=0)
ax2.plot(g.x, g.p50, color=C._GAP_HUE, lw=1.2, ls="--")
ax2.scatter(site.gx, site.rtt, s=4 + site.n * 0.6, color=C._GAP_HUE, alpha=0.45, lw=0, clip_on=False, zorder=3)
ax2.text(0.02, 0.92, f"Spearman $\\rho$ = {r_site.statistic:.2f} over {len(site)} (site, gap) points",
         transform=ax2.transAxes, fontsize=6.5, color="0.35", va="top")
# Physical floor: RTT_sp >= 2 d_sp / (2/3 c) = d_sp / 100 km/ms, and d_sp >= gap.
km = np.logspace(-1, 4, 60); ax2.plot(km, km / 100, color="0.45", lw=0.8, ls=":")
ax2.text(2400, 9, "floor: gap / 100", fontsize=6, color="0.4", ha="left", va="top")
over = (site.rtt - site.gx.where(site.gx > C.X_MIN_KM, 0) / 100)
print("RTT above the floor, per point: median", round(over.median(), 1), "ms; ratio RTT/floor for gap>100km:",
      (site.rtt / (site.gx / 100))[site.gx > 100].describe()[["min", "50%", "max"]].round(2).to_dict())
ax2.set_ylabel("S-P RTT (ms)", fontsize=8); ax2.set_ylim(0, 70); ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax2.set_xscale("log"); ax2.set_xlim(C.X_MIN_KM, C.X_MAX_KM)
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax2.set_xlabel("distance (km)   [bottom panel: x = gap; gap 0 at the left edge]", fontsize=8)
for a in (ax, ax2):
    a.tick_params(labelsize=7.5, length=3); a.grid(alpha=0.25, lw=0.4)
    for s in ("top", "right"): a.spines[s].set_visible(False)
fig.savefig(sys.argv[1], dpi=200, bbox_inches="tight")
