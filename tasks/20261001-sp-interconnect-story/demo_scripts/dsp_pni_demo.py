import sys; sys.path.insert(0, ".")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.ticker as mt
from scipy.stats import spearmanr
from scripts.analysis.v5.modules.paths import resolve_run
from scripts.analysis.v5.modules import edges, pni_gap as P
from scripts.analysis.v5.modules.labels import declared_pni_csv
from scripts.analysis.v5.modules.geodesy import haversine_km
from scripts.libs.canonical.schema import load_canonical_csv

VP_AT_PNI_KM = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
rows = []
for rid in ["pro-as01-mesh", "pro-as02-mesh", "pro-as03-mesh"]:
    run = resolve_run(rid); pnis = P.load_pnis(declared_pni_csv(rid))
    src = load_canonical_csv(edges.resolve_source_csv(run))
    e = src.groupby(["target_id", "vp_id"], as_index=False).agg(rtt=("rtt_ms", "min"), vla=("vp_lat", "first"), vlo=("vp_lon", "first"),
                                                                tla=("target_lat", "first"), tlo=("target_lon", "first"))
    e["d"] = haversine_km(e.vla, e.vlo, e.tla, e.tlo)
    g = e.groupby("target_id")
    geo = e.loc[g.d.idxmin()].set_index("target_id"); sp = e.loc[g.rtt.idxmin()].set_index("target_id")
    t = pd.DataFrame({"tla": geo.tla, "tlo": geo.tlo, "d_geo": geo.d, "d_sp": sp.d, "rtt_sp": sp.rtt, "svla": sp.vla, "svlo": sp.vlo})
    D = haversine_km(t.tla.values[:, None], t.tlo.values[:, None], pnis.pni_lat.values[None], pnis.pni_lon.values[None])
    k = D.argmin(1); t["d_pni"] = D.min(1)
    t["plat"], t["plon"] = pnis.pni_lat.values[k], pnis.pni_lon.values[k]
    # is there a measuring VP at the target's nearest PNI?
    vps = e.drop_duplicates("vp_id")[["vla", "vlo"]].to_numpy()
    pv = haversine_km(pnis.pni_lat.values[:, None], pnis.pni_lon.values[:, None], vps[None, :, 0], vps[None, :, 1]).min(1)
    t["d_pni_vp"] = pv[k]
    # where did the S-P VP sit relative to that PNI?
    t["d_spvp_pni"] = haversine_km(t.svla, t.svlo, t.plat, t.plon)
    # ...and relative to ANY of this operator's PNIs
    t["d_spvp_anypni"] = haversine_km(t.svla.values[:, None], t.svlo.values[:, None],
                                      pnis.pni_lat.values[None], pnis.pni_lon.values[None]).min(1)
    t["run_id"] = rid
    rows.append(t.reset_index())
t = pd.concat(rows, ignore_index=True)
t["gap"] = (t.d_sp - t.d_geo).round(3)
pt = t.groupby(["run_id", "tla", "tlo", "gap"], as_index=False).agg(
    n=("target_id", "size"), d_pni=("d_pni", "first"), d_sp=("d_sp", "first"), d_geo=("d_geo", "first"),
    d_pni_vp=("d_pni_vp", "first"), d_spvp_pni=("d_spvp_pni", "first"),
    d_spvp_anypni=("d_spvp_anypni", "first"), rtt=("rtt_sp", "median"))
R = VP_AT_PNI_KM
pt["where"] = np.select(
    [pt.d_spvp_pni <= R, pt.d_sp <= R, pt.d_spvp_anypni <= R],
    ["at the TG's nearest PNI", "near the TG, not at a PNI", "at another PNI"], "elsewhere")
print(pt.groupby("where").agg(points=("n", "size"), tgs=("n", "sum"), sites=("tla", "size")).to_string())
pt["vp_at_pni"] = pt.d_pni_vp <= VP_AT_PNI_KM
print(f"{len(t)} TGs, {len(pt)} points, {pt[['run_id','tla','tlo']].drop_duplicates().shape[0]} sites")
print("PNIs with a VP within %.0f km: " % VP_AT_PNI_KM, end="")
print(pt.groupby("vp_at_pni").size().to_dict(), "(points)")
print("spearman d_sp vs d_pni, all points: %.3f" % spearmanr(pt.d_pni, pt.d_sp).statistic)
for flag, b in pt.groupby("vp_at_pni"):
    print(f"  vp_at_pni={flag}: n={len(b)} rho={spearmanr(b.d_pni, b.d_sp).statistic:.3f}  "
          f"median |log10(d_sp/d_pni)|={np.median(np.abs(np.log10(b.d_sp.clip(lower=0.1) / b.d_pni.clip(lower=0.1)))):.2f}  "
          f"S-P VP within 50 km of that PNI: {(b.d_spvp_pni <= 50).mean():.0%}")
far = pt[pt.d_pni > 100]
print(f"d_pni>100 km: n={len(far)}  S-P VP within 50 km of the TG's nearest PNI: {(far.d_spvp_pni<=50).mean():.0%}  "
      f"median d_sp/d_pni = {np.median(far.d_sp/far.d_pni):.2f}")
print(pt.assign(ratio=(pt.d_sp/pt.d_pni).round(2))[pt.d_pni>100].sort_values("d_pni")[["run_id","n","d_pni","d_sp","d_geo","d_pni_vp","d_spvp_pni","rtt","ratio"]].round(1).to_string(index=False))

# ---- figure
fig, ax = plt.subplots(figsize=(5.6, 3.8))
CATS = (("at the TG's nearest PNI", "#2a78d6", "o"), ("at another PNI", "#eb6834", "s"),
        ("near the TG, not at a PNI", "#1baf7a", "^"), ("elsewhere", "#eda100", "D"))
for lab, colour, marker in CATS:
    b = pt[pt["where"] == lab]
    if b.empty: continue
    ax.scatter(b.d_pni, b.d_sp, s=10 + 2.5 * b.n, c=colour, marker=marker, alpha=0.75, edgecolor="white", lw=0.6,
               clip_on=False, zorder=3, label=f"S-P VP {lab}  ({len(b)} pts, {int(b.n.sum())} TGs)")
km = np.r_[0, np.logspace(-1, np.log10(4000), 200)]
ax.plot(km, km, color="0.45", lw=0.8, ls=":", zorder=1)
ax.text(2500, 1900, "$d_\\mathrm{sp} = d_\\mathrm{pni}$", fontsize=6.5, color="0.4", ha="right", rotation=45)
for setter in (ax.set_xscale, ax.set_yscale):
    setter("symlog", linthresh=P.LINTHRESH_KM, linscale=P.LINSCALE)
ax.set_xlim(0, 4000); ax.set_ylim(0, 4000); ax.set_aspect("equal")
for a in (ax.xaxis, ax.yaxis):
    a.set_major_locator(mt.FixedLocator([0, 25, 50, 75, 100, 1000, 4000]))
    a.set_minor_locator(mt.FixedLocator([200, 300, 400, 500, 600, 700, 800, 900, 2000, 3000]))
    a.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}"))
ax.axvline(100, color="0.6", lw=0.5, ls=":"); ax.axhline(100, color="0.6", lw=0.5, ls=":")
ax.set_xlabel(r"$d_\mathrm{pni}$ = $d$(TG, nearest PNI) (km)", fontsize=8)
ax.set_ylabel(r"$d_\mathrm{sp}$ = S-P error (km)", fontsize=8)
ax.tick_params(labelsize=7.5); ax.grid(alpha=0.25, lw=0.4)
for s in ("top", "right"): ax.spines[s].set_visible(False)
leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6.5, frameon=False, borderaxespad=0)
for h in leg.legend_handles: h.set_sizes([28])
ax.set_title(f"pro-as01/02/03 pooled, {len(t)} TGs, {len(pt)} (site, gap) points; 'at' = within {R:.0f} km", fontsize=7)
fig.tight_layout(pad=0.4); fig.savefig(sys.argv[1], dpi=200, bbox_inches="tight")
pt.to_csv(sys.argv[1].replace(".png", ".csv"), index=False)
