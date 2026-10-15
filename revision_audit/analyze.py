from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
R=Path(__file__).resolve().parent;S=R.parent
sources=['expanded/benchmark_raw.csv','expanded/synthetic_grid_raw.csv','expanded/stress_raw.csv','expanded/threshold_control_raw.csv','pilot/pilot_raw.csv']
(R/'source_hashes.json').write_text(json.dumps({f:hashlib.sha256((S/f).read_bytes()).hexdigest() for f in sources},indent=2))
b=pd.read_csv(S/sources[0]);ok=b[b.status=='ok'];keys=['dataset','noise_rate','classifier','method'];means=ok.groupby(keys).agg(nfolds=('gmean','size'),gmean=('gmean','mean'),auc=('auc','mean'),ap=('ap','mean')).reset_index();means.to_csv(R/'benchmark_dataset_metrics.csv',index=False);complete=means[means.nfolds==15]
summary=complete.groupby(['noise_rate','classifier','method']).agg(datasets=('dataset','size'),gmean=('gmean','mean'),auc=('auc','mean'),ap=('ap','mean')).reset_index();summary.to_csv(R/'benchmark_all_baselines.csv',index=False)
rows=[]
for (rate,kind),g in complete.groupby(['noise_rate','classifier']):
 for metric in ('auc','ap'):
  p=g.pivot(index='dataset',columns='method',values=metric)
  for base in sorted(set(p.columns)-{'proposed'}):
   q=p[['proposed',base]].dropna();d=(q.proposed-q[base]).to_numpy();nz=d[d!=0];rows.append(dict(noise_rate=rate,classifier=kind,metric=metric,baseline=base,datasets=len(d),mean_difference=d.mean(),median_difference=np.median(d),wins=int((d>0).sum()),raw_p=float(wilcoxon(nz).pvalue) if len(nz) else 1.))
stats=pd.DataFrame(rows);order=stats.raw_p.sort_values().index;stats['holm_p_84']=np.nan;pv=stats.loc[order,'raw_p'].to_numpy();stats.loc[order,'holm_p_84']=np.minimum(1,np.maximum.accumulate((len(pv)-np.arange(len(pv)))*pv));stats.to_csv(R/'ranking_paired_statistics_exploratory.csv',index=False)
grid=pd.read_csv(S/sources[1]);stress=pd.read_csv(S/sources[2]);gm=grid.groupby(['ir','separation','noise_rate','mechanism','method'])[['gmean','auc','ap','synthetic']].mean().reset_index();gm.to_csv(R/'synthetic_setting_metrics.csv',index=False);synsummary=gm.groupby(['mechanism','method'])[['gmean','auc','ap','synthetic']].mean().reset_index();synsummary.to_csv(R/'synthetic_all_baselines.csv',index=False)
stress.groupby(['family','method'])[['gmean','auc','ap','synthetic','fdw_min']].mean().reset_index().to_csv(R/'stress_all_baselines.csv',index=False)
# Paired realized augmentation counts; these are observed counts, not matched-weight controls.
budgets=[]
for study,df,k in [('grid',grid,['rep','ir','separation','noise_rate','mechanism']),('stress',stress,['rep','family','value','n']),('benchmark',ok,['dataset','rep','fold','noise_rate','classifier'])]:
 p=df.pivot(index=k,columns='method',values='synthetic')[['proposed','smote']].dropna();d=p.proposed-p.smote;budgets.append(dict(study=study,pairs=len(p),equal=int((d==0).sum()),proposed_fewer=int((d<0).sum()),proposed_more=int((d>0).sum()),min_difference=float(d.min()),max_difference=float(d.max())))
pd.DataFrame(budgets).to_csv(R/'realized_budget_audit.csv',index=False)
# Explicit data-generation seeds, not evidence of historical human parameter choices.
seeds=[]
for rep in range(30):
 for stage,role,seed in [('development','train',1000+rep),('development','test',2000+rep),('gaussian_threshold','train',2700000+rep),('gaussian_threshold','validation',2800000+rep),('gaussian_threshold','test',2900000+rep)]:seeds.append(dict(stage=stage,role=role,setting='base',rep=rep,seed=seed))
 for ir in (5,10,20):
  for sep in (1,2,3):
   seed=100000+rep+ir*1000+sep*10000
   for role,v in [('train',seed),('test',seed+500000)]:seeds.append(dict(stage='grid',role=role,setting=f'IR={ir},sep={sep}',rep=rep,seed=v))
for family,value,n in [('subgroup',g,n) for g in (.05,.1,.2) for n in (500,1000,2000)]+[('isolation',.05,n) for n in (500,1000,2000)]+[('feature_outliers',v,2000) for v in (.05,.1)]+[('dimensions',v,2000) for v in (2,10,50)]:
 for rep in range(30):
  seed=1300000+rep+int(float(value)*1000)+n
  for role,v in [('train',seed),('test',seed+500000)]:seeds.append(dict(stage='stress',role=role,setting=f'{family},value={value},n={n}',rep=rep,seed=v))
seeds=pd.DataFrame(seeds);seeds.to_csv(R/'data_generation_seeds.csv',index=False)
dev=set(seeds[seeds.stage=='development'].seed);ev=set(seeds[seeds.stage!='development'].seed);collisions=seeds[seeds.stage=='grid'].groupby(['role','seed']).filter(lambda x:len(x)>1);collisions.to_csv(R/'grid_seed_reuse.csv',index=False)
seedreport=dict(development_evaluation_seed_intersection=sorted(dev&ev),grid_reused_role_seed_groups=int(collisions.groupby(['role','seed']).ngroups),note='Disjoint RNG seeds do not make grid conditions independent. Noise levels share data deliberately; some distinct IR/separation cells also reuse RNG seeds. Different configurations transform those streams differently. No claim of preregistration follows from this audit.')
(R/'seed_audit.json').write_text(json.dumps(seedreport,indent=2))
def tab(df):
 return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(f'{v:.6f}' if isinstance(v,float) else str(v) for v in row)+' |' for row in df.itertuples(index=False,name=None)])
# Compact LaTeX tables preserve dataset completion counts, so failures cannot silently change denominators.
names={'none':'No intervention','ros':'Random oversampling','smote':'SMOTE','borderline':'Borderline-SMOTE1','adasyn':'ADASYN','proposed':'Complete','no_downweight':'No downweighting','wrnd_author':'WRND adapter'}
lines=[]
for (rate,kind),g in summary.groupby(['noise_rate','classifier']):
 lines += [r'\begin{table}[t]',f"\\caption{{All Baselines: {kind.upper()}, {'Supplied Labels' if rate==0 else '10\\% Fitting-Label Corruption'}}}",f"\\label{{tab:all_baselines_{kind}_{int(rate*100)}}}",r'\centering\small',r'\begin{tabular}{lrrrr}',r'\hline',r'Method & $D$ & G-mean & ROC-AUC & AP\\',r'\hline']
 for method in names:
  q=g[g.method==method]
  if q.empty:continue
  a=q.iloc[0];lines.append(f"{names[method]} & {int(a.datasets)} & {a.gmean:.4f} & {a.auc:.4f} & {a.ap:.4f} "+chr(92)*2)
 lines += [r'\hline',r'\end{tabular}',r'\par\smallskip\footnotesize $D$: datasets with all 15 outer folds completed. Means weight datasets equally. Different $D$ values preclude direct interpretation as a common-dataset ranking; use paired results for comparisons.',r'\end{table}','']
(R/'all_baselines_tables.tex').write_text('\n'.join(lines))
report='# Evidence audit before reframing\n\n## Existing benchmark results (complete datasets only)\n\n'+tab(summary)+'\n\n## Synthetic ranking metrics\n\n'+tab(synsummary)+'\n\n## Realized sample budgets\n\n'+tab(pd.DataFrame(budgets))+'\n\n## Seed audit\n\n'+json.dumps(seedreport,indent=2)+'\n\n## Inference limits\n\nSix nonzero pairs give a minimum exact two-sided Wilcoxon p=2/2^6=0.03125 when all differences favor one method and there are no ties. This cannot pass the first Holm threshold in families of 6, 12 or 42 (0.00833, 0.00417, 0.00119). Minimum nonzero dataset counts permitting any rejection in those families under ideal conditions are 8, 9 and 11. These are feasibility bounds, not power or sample-size targets. Increasing datasets must broaden a predefined target population, not chase significance. Bayesian analysis does not create evidence and also needs assumptions and sensitivity analysis.\n\nThe new ROC-AUC/AP analysis is exploratory, with one conservative 84-comparison Holm family. Do not imply it was a prespecified primary family.\n\n## What existing evidence does not answer\n\nThe separate Gaussian threshold control covers separation 2, IR about 10 and 10% corruption only; it cannot show that the 39/63 grid wins generally disappear after threshold tuning. Raw-count equality does not equal matched effective training weights. Corruption concentrated at class means is a structured stress test, not an established realistic noise model. Dropping downweighting now is a post-result method revision and requires fresh confirmation. Development/test historical access cannot be proven from RNG code alone.\n'
(R/'report.md').write_text(report)
print(pd.DataFrame(budgets).to_string(index=False));print(seedreport);print(stats[(stats.baseline=='smote')][['noise_rate','classifier','metric','mean_difference','holm_p_84']].to_string(index=False));print(synsummary[synsummary.method.isin(['none','smote','proposed'])].to_string(index=False))
