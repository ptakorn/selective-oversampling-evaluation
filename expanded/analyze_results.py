from pathlib import Path
import pandas as pd,numpy as np,json
from scipy.stats import wilcoxon,friedmanchisquare
ROOT=Path(__file__).parent
syn=pd.read_csv(ROOT/'synthetic_grid_raw.csv');ok=syn[syn.status=='ok'];keys=['ir','separation','noise_rate','mechanism']
rows=[];rng=np.random.default_rng(908172)
for key,group in ok.groupby(keys):
 pv=group.pivot(index='rep',columns='method',values='gmean')
 for baseline in ['none','smote','borderline','adasyn','no_downweight']:
  delta=(pv.proposed-pv[baseline]).dropna().to_numpy();boot=delta[rng.integers(0,len(delta),(5000,len(delta)))].mean(axis=1)
  rows.append(dict(zip(keys,key),baseline=baseline,mean_delta=delta.mean(),lower=np.quantile(boot,.025),upper=np.quantile(boot,.975),pairs=len(delta)))
pd.DataFrame(rows).to_csv(ROOT/'synthetic_paired_intervals.csv',index=False)
if not (ROOT/'benchmark_raw.csv').exists():raise SystemExit('Synthetic analysis complete; benchmark still running')
raw=pd.read_csv(ROOT/'benchmark_raw.csv');ok=raw[raw.status=='ok'];agg=ok.groupby(['dataset','noise_rate','classifier','method']).agg(gmean=('gmean','mean'),f1=('f1','mean'),outer_successes=('fold','size')).reset_index();agg.to_csv(ROOT/'benchmark_dataset_means.csv',index=False)
# Require all 15 outer folds in a paired dataset comparison; no invented failed scores.
complete=agg[agg.outer_successes==15];pairs=[];omnibus=[]
for (rate,clf),g in complete.groupby(['noise_rate','classifier']):
 pv=g.pivot(index='dataset',columns='method',values='gmean')
 for method in sorted(set(pv.columns)-{'proposed'}):
  both=pv[['proposed',method]].dropna();delta=(both.proposed-both[method]).to_numpy();nonzero=delta[delta!=0]
  if len(delta)==0: continue
  p=wilcoxon(nonzero,alternative='two-sided',method='auto').pvalue if len(nonzero) else 1.
  pairs.append(dict(noise_rate=rate,classifier=clf,baseline=method,datasets=len(delta),nonzero_pairs=len(nonzero),mean_delta=delta.mean(),median_delta=np.median(delta),p_raw=p))
 allmethods=pv.dropna()
 if len(allmethods)>=3:
  test=friedmanchisquare(*[allmethods[c].to_numpy() for c in allmethods]);omnibus.append(dict(noise_rate=rate,classifier=clf,datasets=len(allmethods),statistic=test.statistic,p_raw=test.pvalue))
p=pd.DataFrame(pairs);order=np.argsort(p.p_raw.to_numpy());adjusted=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p.p_raw.to_numpy()[order]));arr=np.empty(len(p));arr[order]=adjusted;p['p_holm']=arr;p.to_csv(ROOT/'benchmark_paired_statistics.csv',index=False);pd.DataFrame(omnibus).to_csv(ROOT/'benchmark_friedman.csv',index=False)
print(raw.groupby(['method','status']).size().to_string());print(agg.groupby(['noise_rate','classifier','method']).gmean.mean().round(4).to_string())
