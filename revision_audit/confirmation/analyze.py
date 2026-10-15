from pathlib import Path
import numpy as np,pandas as pd,json
R=Path(__file__).resolve().parent;d=pd.read_csv(R/'raw.csv');assert len(d)==5400 and (d.status=='ok').all()
keys=['geometry','rep','corruption','classifier','rule'];p=d.pivot(index=keys,columns='method',values='synthetic');assert (p.complete==p.smote_matched_weights).all()
for metric in ('original_weight_sum','synthetic_weight_sum'):
 q=d.pivot(index=keys,columns='method',values=metric);assert np.max(abs(q.complete-q.smote_matched_weights))<1e-8
summary=d.groupby(['geometry','corruption','classifier','method','rule'])[['gmean','auc','ap','synthetic','fdw_min','dw_recall']].mean().reset_index();summary.to_csv(R/'summary.csv',index=False)
rows=[]
for key,g in d.groupby(['geometry','corruption','classifier','rule']):
 for metric in ('gmean','auc','ap'):
  p=g.pivot(index='rep',columns='method',values=metric)
  for base in ('none','smote','no_downweight','smote_matched_weights'):
   delta=(p.complete-p[base]).to_numpy();rng=np.random.default_rng(90120261015);boot=rng.choice(delta,size=(5000,len(delta)),replace=True).mean(axis=1);lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(zip(['geometry','corruption','classifier','rule'],key),metric=metric,baseline=base,mean_difference=delta.mean(),ci_low=lo,ci_high=hi,pairs=len(delta)))
pd.DataFrame(rows).to_csv(R/'paired_intervals.csv',index=False)
# Independent replications within each scenario, descriptive intervals only, no pooled setting inference.
q=pd.DataFrame(rows);g=q[(q.metric=='gmean')&(q.rule=='selected')];print(g[['geometry','corruption','classifier','baseline','mean_difference','ci_low','ci_high']].to_string(index=False));print('SUMMARY',summary.groupby(['classifier','method','rule'])[['gmean','auc','ap']].mean().to_string())
(R/'verification.json').write_text(json.dumps(dict(models=2700,decision_rule_evaluations=5400,matched_counts_and_weight_sums_verified=True,seed_namespace=[20261015,901],inference='Individual paired bootstrap intervals within scenario; no simultaneous inference or cross-dataset significance claims'),indent=2))
