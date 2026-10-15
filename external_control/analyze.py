from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
R=Path(__file__).parent
new=pd.read_csv(R/'new_raw.csv');old=pd.read_csv(R/'inputs/previous_targeted_raw.csv');old=old[old.method.isin(['none','smote'])].copy()
assert len(new)==3240 and len(old)==2160
keys=['geometry','rep','corruption','method','classifier','rule'];assert not new.duplicated(keys).any()
assert hashlib.sha256((R/'inputs/previous_targeted_raw.csv').read_bytes()).hexdigest()==json.loads((R/'config.json').read_text())['source_raw_sha256']
d=pd.concat([old,new],ignore_index=True);d.to_csv(R/'all_raw.csv',index=False)
ok=d[d.status=='ok'];au=pd.read_csv(R/'intervention_audit.csv')
assert len(au)==270 and au.count_match_exact.all() and au.retention_match_exact.all()
assert au.synthetic_retained.between(0,au.full_synthetic).all()
assert au.original_retained.between(1,1000).all()
assert (ok.groupby(['geometry','rep','corruption','method','classifier']).size()==2).all()
# Ranking metrics must be identical between rules for a given fitted model.
for metric in ('auc','ap'):
 q=ok.pivot(index=['geometry','rep','corruption','method','classifier'],columns='rule',values=metric);assert np.array_equal(q.default.values,q.selected.values)
summary=ok.groupby(['geometry','corruption','classifier','method','rule'])[['gmean','auc','ap']].agg(['mean','std','count']);summary.to_csv(R/'summary_with_sd.csv')
means=ok.groupby(['geometry','corruption','classifier','method','rule'])[['gmean','auc','ap']].mean().reset_index();means.to_csv(R/'summary.csv',index=False)
rows=[]
for key,g in ok.groupby(['geometry','corruption','classifier','rule']):
 for metric in ('gmean','auc','ap'):
  p=g.pivot(index='rep',columns='method',values=metric)
  for base in ('none','smote','smote_matched_count','smote_matched_retention'):
   pair=p[['smoteenn',base]].dropna();delta=(pair.smoteenn-pair[base]).to_numpy()
   if len(delta)!=30:continue
   rng=np.random.default_rng(90120261016);boot=rng.choice(delta,size=(5000,len(delta)),replace=True).mean(axis=1);lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(zip(['geometry','corruption','classifier','rule'],key),metric=metric,baseline=base,pairs=len(delta),mean_difference=delta.mean(),ci_low=lo,ci_high=hi))
ci=pd.DataFrame(rows);ci.to_csv(R/'paired_intervals.csv',index=False)
selected=ci[(ci.metric=='gmean')&(ci.rule=='selected')];selected.to_csv(R/'selected_gmean_intervals.csv',index=False)
ci['direction']=np.where(ci.ci_low>0,'above_zero',np.where(ci.ci_high<0,'below_zero','includes_zero'))
ci[ci.metric=='gmean'].groupby(['classifier','rule','baseline','direction']).size().unstack(fill_value=0).to_csv(R/'interval_directions.csv')
ci.groupby(['classifier','rule','metric','baseline']).mean_difference.mean().to_csv(R/'descriptive_mean_differences.csv')
au.groupby(['geometry','corruption']).mean(numeric_only=True).to_csv(R/'intervention_summary.csv')
means.groupby(['classifier','method','rule'])[['gmean','auc','ap']].mean().to_csv(R/'overall_descriptive_means.csv')
verification=dict(new_models=1620,new_evaluations=len(new),reused_evaluations=len(old),new_failure_rows=int((new.status!='ok').sum()),structural_audits=len(au),count_matching_verified=True,original_retention_matching_verified=True,ranking_rule_invariance_verified=True,paired_intervals=len(ci),selected_gmean_intervals=len(selected),warnings=int(new.get('warnings',pd.Series(dtype=str)).fillna('').ne('').sum()),inference='Exploratory individual paired percentile bootstrap intervals; descriptive setting averages; no general superiority or noise-identifiability claim')
(R/'verification.json').write_text(json.dumps(verification,indent=2));print(json.dumps(verification,indent=2));print('MEAN DIFFERENCES\n',ci[ci.metric=='gmean'].groupby(['classifier','rule','baseline']).mean_difference.mean().to_string());print('SELECTED DIRECTIONS\n',ci[(ci.metric=='gmean')&(ci.rule=='selected')].groupby(['classifier','baseline','direction']).size().unstack(fill_value=0).to_string());print('INTERVENTION\n',au.groupby('corruption')[['original_deleted','synthetic_retained','removal_precision','removal_recall','valid_minority_removed','ecr_before','ecr_after']].mean().to_string())
