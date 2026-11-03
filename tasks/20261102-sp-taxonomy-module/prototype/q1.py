import pandas as pd
from scripts.analysis.v5.modules import pni_gap as P, sp_pni_cells as SPC, cross
from scripts.analysis.v5.modules.labels import load_group, declared_pni_csv
from scripts.analysis.v5.modules.paths import DEFAULT_OUTPUTS_ROOT as ROOT
from scripts.analysis.v5.modules.figure_sp_interconnect import load_edges
from scripts.analysis.v5.modules.map_answer_space import load_rung
from scripts.analysis.v5.cli import resolve_run
cross.use_group(load_group("pro-paper"))
RUNS=["pro-as01-mesh","pro-as02-mesh","pro-as03-mesh"]
rid="pro-as02-mesh"; run=resolve_run(rid,ROOT); pc=declared_pni_csv(rid,ROOT)
_,meta=P.read_clusters(P.pooled_output_dir(RUNS),run_ids=RUNS)
rec={r["run_id"]:r for r in meta["runs"]}
e=load_edges(P.checked_source_csv(run,pc,rec[rid],None))
seeds=load_rung(run,SPC.SOURCE_NSIDE).seeds
vps=e.drop_duplicates("vp_id"); s,_=SPC.nearest_seed(vps.vp_lat,vps.vp_lon,seeds); vs=dict(zip(vps.vp_id,s))
e["cell"]=e.vp_id.map(vs)
t=pd.read_csv(__import__("sys").argv[1]); site=t[t.site_key=="pro-as02-mesh|45.6185,-121.167"]
x=e[e.tg_id.isin(site.tg_id)]
x9=x[x.cell==9].pivot_table(index="vp_id",columns="tg_id",values="rtt_ms")
print("VPs per TG at this site:", x.groupby("tg_id").size().describe().to_dict())
print("TGs measured by each VP at this site (of 20):", x.groupby("vp_id").size().value_counts().sort_index().to_dict())
best=x.sort_values("rtt_ms").groupby("tg_id").head(3)
win=x.loc[x.groupby("tg_id").rtt_ms.idxmin()]
print("distinct S-P VPs:", win.vp_id.value_counts().to_dict())
odd="tg-1391b0a"
w=win.vp_id[win.tg_id!=odd].unique()
print("did the usual winners measure the odd TG?", {v: bool(((x.tg_id==odd)&(x.vp_id==v)).any()) for v in w})
print("odd TG best 5:"); print(x[x.tg_id==odd].nsmallest(5,"rtt_ms")[["vp_id","rtt_ms","cell","d_km"]].to_string())
print("a typical replica best 5:"); print(x[x.tg_id=="tg-0d30c08"].nsmallest(5,"rtt_ms")[["vp_id","rtt_ms","cell","d_km"]].to_string())
