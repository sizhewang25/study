import numpy as np, pandas as pd
R=6371.0
def hav(a1,o1,a2,o2):
    a1,o1,a2,o2=map(np.radians,(a1,o1,a2,o2))
    h=np.sin((a2-a1)/2)**2+np.cos(a1)*np.cos(a2)*np.sin((o2-o1)/2)**2
    return 2*R*np.arcsin(np.sqrt(h))
M=['shortest_ping','million_scale_cbg','vanilla_cbg','octant_cbg_hull','spotter_cbg']
T=dict(zip(M,['S-P','SOI','VAN','OCT-H','SPO']))
out=[]
for r in ['01','02','03']:
    run=f'pro-as{r}-mesh'; base=f'outputs/analysis/v5/{run}/classify/healpix-128'
    fr={m:pd.read_parquet(f'{base}/{m}_tgs.parquet').set_index('tg_id') for m in M}
    pni=pd.read_csv(f'datasets/pni/as{r}-us-pni.approx.csv')
    e=pd.read_csv(f'datasets/final/as{r}-20260728-20260802.mainland.sanitized.csv')
    e['d_vl']=hav(e.vp_lat,e.vp_lon,e.target_lat,e.target_lon)
    e['slack']=e.rtt_ms-e.d_vl/100.0   # 2/3c: RTT floor = 2d/200 km/ms
    for tg in fr['shortest_ping'].index:
        g=fr['shortest_ping'].loc[tg]; L=(g.tg_lat,g.tg_lon)
        P=[(T[m],fr[m].loc[tg,'pred_lat'],fr[m].loc[tg,'pred_lon']) for m in M if fr[m].loc[tg,'cell_label']!='unanswered']
        lat=np.array([p[1] for p in P]); lon=np.array([p[2] for p in P])
        D=hav(lat[:,None],lon[:,None],lat[None],lon[None]); k=D.sum(1).argmin()
        ml,mo=lat[k],lon[k]
        dm=D[k]
        ee=e[e.target_id==tg]
        # feasibility of the label and of the consensus point
        dvp=hav(ee.vp_lat.values,ee.vp_lon.values,ml,mo)
        out.append(dict(run=run,tg_id=tg,site=f'{run}|{L[0]:.3f},{L[1]:.3f}',tg_lat=L[0],tg_lon=L[1],
            med_lat=ml,med_lon=mo,n_pred=len(P),
            n_agree100=int((dm<=100).sum()),spread_med=float(np.median(dm)),
            d_label_med=float(hav(L[0],L[1],ml,mo)),
            d_med_pni=float(hav(ml,mo,pni.pni_lat.values,pni.pni_lon.values).min()),
            d_label_pni=float(hav(L[0],L[1],pni.pni_lat.values,pni.pni_lon.values).min()),
            n_correct=sum(fr[m].loc[tg,'cell_label']=='correct' for m in M),
            n_edges=len(ee),min_rtt=ee.rtt_ms.min(),
            label_minslack=ee.slack.min(),label_nviol=int((ee.slack<0).sum()),
            cons_minslack=float((ee.rtt_ms.values-dvp/100).min()),cons_nviol=int(((ee.rtt_ms.values-dvp/100)<0).sum()),
            d_vp_label_min=ee.d_vl.min(), d_vp_cons_min=float(dvp.min())))
d=pd.DataFrame(out); d['af']=d.n_correct==0
d.to_csv('notes/assets/2026-10-04-allfail-cohort/gt_check.csv',index=False)
