import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from scipy.stats import spearmanr
from scripts.analysis.v5.modules.paths import resolve_run
from scripts.analysis.v5.modules import edges, pni_gap as P
from scripts.analysis.v5.modules.labels import declared_pni_csv
from scripts.analysis.v5.modules.geodesy import haversine_km
from scripts.libs.canonical.schema import load_canonical_csv

R = 50.0   # "at a PNI" = S-P VP within this many km of it
rows = []
for rid in ["pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh"]:
    run = resolve_run(rid); pn = P.load_pnis(declared_pni_csv(rid))
    e = load_canonical_csv(edges.resolve_source_csv(run)).groupby(["target_id", "vp_id"], as_index=False).agg(
        rtt=("rtt_ms", "min"), vla=("vp_lat", "first"), vlo=("vp_lon", "first"), tla=("target_lat", "first"), tlo=("target_lon", "first"))
    e["d"] = haversine_km(e.vla, e.vlo, e.tla, e.tlo); g = e.groupby("target_id")
    geo = e.loc[g.d.idxmin()].set_index("target_id"); sp = e.loc[g.rtt.idxmin()].set_index("target_id")
    t = pd.DataFrame({"tla": sp.tla, "tlo": sp.tlo, "d_geo": geo.d, "d_sp": sp.d, "rtt": sp.rtt, "svla": sp.vla, "svlo": sp.vlo})
    D = haversine_km(t.tla.values[:, None], t.tlo.values[:, None], pn.pni_lat.values[None], pn.pni_lon.values[None])
    k = D.argmin(1); t["d_pni"] = D.min(1)
    t["d_vp_pni"] = haversine_km(t.svla, t.svlo, pn.pni_lat.values[k], pn.pni_lon.values[k])
    # S-P VP -> the PNI nearest the S-P VP -> TG
    V = haversine_km(t.svla.values[:, None], t.svlo.values[:, None], pn.pni_lat.values[None], pn.pni_lon.values[None])
    j = V.argmin(1)
    t["d_via"] = V.min(1) + D[np.arange(len(t)), j]
    t["via_is_tg_nearest"] = j == k
    t["d_vp_anypni"] = haversine_km(t.svla.values[:, None], t.svlo.values[:, None], pn.pni_lat.values[None], pn.pni_lon.values[None]).min(1)
    t["run"] = rid; rows.append(t)
t = pd.concat(rows); t["gap"] = (t.d_sp - t.d_geo).round(3)
pt = t.groupby(["run", "tla", "tlo", "gap"], as_index=False).agg(
    n=("rtt", "size"), rtt=("rtt", "median"), d_sp=("d_sp", "first"), d_via=("d_via", "first"),
    via_is_tg_nearest=("via_is_tg_nearest", "first"),
    d_pni=("d_pni", "first"), d_vp_pni=("d_vp_pni", "first"), d_vp_anypni=("d_vp_anypni", "first"))
pt["where"] = np.select([pt.d_vp_pni <= R, pt.d_sp <= R, pt.d_vp_anypni <= R],
                        ["at the TG's nearest PNI", "near the TG, not at a PNI", "at another PNI"], "elsewhere")
print(f"{int(pt.n.sum())} TGs, {len(pt)} (site, gap) points")
for col, name in (("d_sp", "direct  d(VP,TG)"), ("d_via", "via S-P VP's PNI")):
    ratio = pt.rtt / (pt[col] / 100)
    big = pt[col] > 100
    lr = np.log10(ratio[big])
    print(f"{name:30s} rho={spearmanr(pt[col], pt.rtt).statistic:.3f}  d>100km: rho={spearmanr(pt[col][big], pt.rtt[big]).statistic:.3f} "
          f"RTT/floor median={ratio[big].median():.2f} spread(IQR of log10 ratio)={np.subtract(*np.percentile(lr,[75,25])):.2f}  "
          f"below floor: {(ratio < 1).sum()} pts ({int(pt.n[ratio < 1].sum())} TGs)")
print(pt.groupby("where").agg(points=("n", "size"), tgs=("n", "sum"),
      rtt_over_direct=("rtt", lambda r: np.median(r / (pt.loc[r.index, 'd_sp'] / 100))),
      rtt_over_via=("rtt", lambda r: np.median(r / (pt.loc[r.index, 'd_via'] / 100)))).round(2).to_string())
print("S-P VP's nearest PNI is also the TG's nearest PNI: %d of %d points (%d of %d TGs)" % (
    pt.via_is_tg_nearest.sum(), len(pt), pt.n[pt.via_is_tg_nearest].sum(), pt.n.sum()))
print("below the via-PNI floor:\n", pt[pt.rtt < pt.d_via / 100][["run", "n", "d_pni", "d_sp", "d_via", "rtt", "where"]].round(1).to_string(index=False))

CATS = (("at the TG's nearest PNI", "#2a78d6", "o"), ("at another PNI", "#eb6834", "s"),
        ("near the TG, not at a PNI", "#1baf7a", "^"), ("elsewhere", "#eda100", "D"))
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3), sharey=True, gridspec_kw={"wspace": 0.08})
xs = (("d_sp", r"$d$(S-P VP, TG) (km)", "direct"),
      ("d_via", r"$d$(S-P VP, PNI$_\mathrm{VP}$) + $d$(PNI$_\mathrm{VP}$, TG) (km)", "via the S-P VP's nearest PNI"))
XMAX = float(np.ceil(max(pt.d_sp.max(), pt.d_via.max()) / 1000) * 1000)
km = np.linspace(0, XMAX, 200)
for ax, (col, xl, title) in zip(axes, xs):
    for lab, colour, marker in CATS:
        b = pt[pt["where"] == lab]
        ax.scatter(b[col], b.rtt, s=8 + 2.0 * b.n, c=colour, marker=marker, alpha=0.75, edgecolor="white", lw=0.5,
                   clip_on=False, zorder=3, label=f"S-P VP {lab} ({len(b)} pts, {int(b.n.sum())} TGs)")
    ax.plot(km, km / 100, color="0.4", lw=0.8, ls=":", zorder=2)
    ax.plot(km, 2 * km / 100, color="0.65", lw=0.6, ls=":", zorder=2)
    ax.text(XMAX * 0.97, XMAX / 100 * 0.97 - 3, "floor", fontsize=6, color="0.4", ha="right", va="top")
    ax.text(3500 * 0.97, 70 - 1.5, "2× floor", fontsize=6, color="0.55", ha="right", va="top")
    ax.set_xlim(0, XMAX)
    ax.xaxis.set_major_locator(mt.MultipleLocator(1000))
    ax.xaxis.set_minor_locator(mt.MultipleLocator(500))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_xlabel(xl, fontsize=7.5); ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=7); ax.grid(alpha=0.25, lw=0.4)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("RTT of the S-P VP (ms)", fontsize=8); axes[0].set_ylim(0, 70)
axes[0].yaxis.set_major_locator(mt.MultipleLocator(10))
leg = axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.3, frameon=False, borderaxespad=0)
for h in leg.legend_handles: h.set_sizes([24])
fig.suptitle(f"pro-as01/02/03 pooled: {int(pt.n.sum())} TGs, {len(pt)} (site, gap) points; floor = distance / 100 km per ms", fontsize=7.5, y=1.0)
fig.savefig(sys.argv[1], dpi=200, bbox_inches="tight")
pt.to_csv(sys.argv[1].replace(".png", ".csv"), index=False)
