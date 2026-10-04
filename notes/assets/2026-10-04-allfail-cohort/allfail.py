import pandas as pd
M=['shortest_ping','million_scale_cbg','vanilla_cbg','octant_cbg_hull','spotter_cbg']
T={'shortest_ping':'S-P','million_scale_cbg':'SOI','vanilla_cbg':'VAN','octant_cbg_hull':'OCT-H','spotter_cbg':'SPO'}
allrows=[]
for run in ['pro-as01-mesh','pro-as02-mesh','pro-as03-mesh']:
    base=f'outputs/analysis/v5/{run}/classify/healpix-128'
    fr={m:pd.read_parquet(f'{base}/{m}_tgs.parquet').set_index('tg_id') for m in M}
    lab=pd.DataFrame({T[m]:fr[m]['cell_label'] for m in M})
    err=pd.DataFrame({T[m]:fr[m]['pred_dist_to_tg_km'].where(fr[m]['cell_label']!='unanswered') for m in M})
    ring=pd.DataFrame({T[m]:fr[m]['pred_dist_to_tg_grid'].where(fr[m]['cell_label']!='unanswered') for m in M})
    g=fr['shortest_ping']
    df=pd.DataFrame({'run':run,'lat':g.tg_lat.round(3),'lon':g.tg_lon.round(3)})
    df['n_correct']=(lab=='correct').sum(1)
    df['n_unans']=(lab=='unanswered').sum(1)
    df['best_km']=err.min(1); df['best_ring']=ring.min(1)
    df['sp_km']=err['S-P']
    allrows.append(df)
d=pd.concat(allrows)
d['site']=d.run+'|'+d.lat.astype(str)+','+d.lon.astype(str)
af=d[d.n_correct==0]
print('TGs',len(d),'all-wrong',len(af),f'{len(af)/len(d):.1%}', 'sites',d.site.nunique(),'all-wrong sites',af.site.nunique())
print(af.groupby('run').size().to_dict(), (d.groupby('run').size()).to_dict())
print('unanswered count dist among all-wrong:',af.n_unans.value_counts().sort_index().to_dict())
print('best_km quantiles', af.best_km.quantile([.1,.25,.5,.75,.9]).round(1).to_dict())
print('best_ring dist', af.best_ring.value_counts().sort_index().to_dict())
s=d.groupby('site').agg(n=('n_correct','size'),n_af=('n_correct',lambda x:(x==0).sum()),lat=('lat','first'),lon=('lon','first'),best_med=('best_km','median'))
s=s[s.n_af>0].assign(frac=lambda x:x.n_af/x.n).sort_values('n_af',ascending=False)
print(s.to_string())
print('-----')
p=pd.read_csv('outputs/analysis/v5/_cross/pni-gap/3-runs-379a99/sp_pni_cells_tgs.csv')
d2=d.reset_index().rename(columns={'index':'tg_id'})
d2=d2.merge(p[['run_id','tg_id','cluster','tg_cell_holds_x','d_tg_x_km','gap_km']],left_on=['run','tg_id'],right_on=['run_id','tg_id'],how='left')
d2['af']=d2.n_correct==0
print(pd.crosstab(d2.tg_cell_holds_x,d2.af,margins=True))
print(pd.crosstab(d2.cluster,d2.af))
a=d2[d2.af]
print(a.groupby('site').agg(n=('af','size'),holds_x=('tg_cell_holds_x','mean'),d_tg_x=('d_tg_x_km','median'),cluster=('cluster',lambda x:x.mode().iat[0])).sort_values('n',ascending=False).round(1).to_string())
