from pathlib import Path
import numpy as np,pandas as pd,json
from scipy.stats import wilcoxon
R=Path(__file__).parent;E=R.parent/'expanded'
def interval(x):
 rng=np.random.default_rng(20261014);return np.quantile(rng.choice(x,(5000,len(x)),replace=True).mean(axis=1),[.025,.975])
s=pd.read_csv(R/'synthetic_raw.csv');summ=[]
for study in ('grid','stress'):
 sub=s[s.study==study];keys=['ir','separation','noise_rate','mechanism'] if study=='grid' else ['family','value','n']
 for key,g in sub.groupby(keys):
  p=g.pivot(index='rep',columns='variant',values='gmean');d=(p.adaptive-p.fixed).to_numpy();lo,hi=interval(d);row=dict(study=study,**dict(zip(keys,key)),adaptive=p.adaptive.mean(),fixed=p.fixed.mean(),difference=d.mean(),ci_low=lo,ci_high=hi)
  for metric in ('synthetic','eligible_sources','adequacy_fraction','fdw_min','endpoint_contamination','base_size'):
   for v in ('adaptive','fixed'):row[v+'_'+metric]=g[g.variant==v][metric].mean()
  summ.append(row)
pd.DataFrame(summ).to_csv(R/'synthetic_paired_summary.csv',index=False)
# Regression check against previous adaptive results by exact task keys.
checks={}
for study,fname,keys in [('grid','synthetic_grid_raw.csv',['rep','ir','separation','noise_rate','mechanism']),('stress','stress_raw.csv',['rep','family','value','n'])]:
 old=pd.read_csv(E/fname);old=old[old.method=='proposed'];new=s[(s.study==study)&(s.variant=='adaptive')];m=old.merge(new,on=keys,suffixes=('_old','_new'));assert len(m)==len(old);err=float(abs(m.gmean_old-m.gmean_new).max());assert err<1e-12;checks[study]=dict(rows=len(m),max_gmean_difference=err)
f=pd.read_csv(R/'benchmark_fixed_raw.csv');a=pd.read_csv(E/'benchmark_raw.csv');a=a[a.method=='proposed'].copy();a['method']='adaptive';b=pd.concat([a,f],ignore_index=True);b.to_csv(R/'benchmark_raw.csv',index=False)
means=b[b.status=='ok'].groupby(['dataset','noise_rate','classifier','method']).agg(gmean=('gmean','mean'),count=('gmean','size')).reset_index();means.to_csv(R/'benchmark_dataset_means.csv',index=False)
rows=[]
for (rate,clf),g in means.groupby(['noise_rate','classifier']):
 g=g[g['count']==15];p=g.pivot(index='dataset',columns='method',values='gmean').dropna();d=(p.adaptive-p.fixed).to_numpy();nz=d[d!=0];pv=wilcoxon(nz).pvalue if len(nz) else 1.;rows.append(dict(noise_rate=rate,classifier=clf,datasets=len(d),mean_difference=d.mean(),median_difference=np.median(d),raw_p=pv))
q=pd.DataFrame(rows);order=np.argsort(q.raw_p);adj=np.maximum.accumulate((len(q)-np.arange(len(q)))*q.raw_p.to_numpy()[order]);q['holm_p']=0.;q.loc[order,'holm_p']=np.minimum(adj,1);q.to_csv(R/'benchmark_statistics.csv',index=False)
(R/'verification.json').write_text(json.dumps(checks,indent=2));print(checks);print(q.to_string(index=False));print(pd.DataFrame(summ).groupby('study')[['adaptive','fixed','difference']].mean())
