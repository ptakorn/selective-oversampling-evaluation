from pathlib import Path
import json,hashlib,re
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;S=R.parent
raw=pd.read_csv(R/'inputs/targeted_raw.csv');oldci=pd.read_csv(R/'inputs/previous_paired_intervals.csv')
rows=[]
# Recompute exact paired percentile intervals, retaining the prespecified bootstrap count and seed used previously.
for (geo,noise,clf,rule),g in raw.groupby(['geometry','corruption','classifier','rule']):
 for metric in ('gmean','auc','ap'):
  p=g.pivot(index='rep',columns='method',values=metric).sort_index();assert len(p)==30 and not p.isna().any().any()
  for base in ('none','smote','no_downweight','smote_matched_weights'):
   d=(p.complete-p[base]).to_numpy();rng=np.random.default_rng(90120261015);boot=rng.choice(d,(5000,len(d)),replace=True).mean(axis=1);lo,hi=np.quantile(boot,[.025,.975]);rows.append(dict(geometry=geo,corruption=noise,classifier=clf,rule=rule,metric=metric,baseline=base,mean_difference=d.mean(),ci_low=lo,ci_high=hi,pairs=len(d)))
ci=pd.DataFrame(rows);keys=['geometry','corruption','classifier','rule','metric','baseline'];check=ci.merge(oldci,on=keys,suffixes=('_new','_old'));assert len(check)==432
for col in ('mean_difference','ci_low','ci_high'):assert np.max(abs(check[col+'_new']-check[col+'_old']))<1e-12
ci.to_csv(R/'all_paired_intervals.csv',index=False);focus=ci[(ci.rule=='selected')&(ci.metric=='gmean')&ci.baseline.isin(['smote','smote_matched_weights'])].copy();focus['direction']=np.where(focus.ci_low>0,'positive',np.where(focus.ci_high<0,'negative','includes_zero'));focus.to_csv(R/'selected_gmean_intervals.csv',index=False)
# Plot every requested setting, not a favorable subset. Individual intervals only.
settings=[(g,n) for g in ('gaussian','moons','circles') for n in ('clean','dispersed','boundary')];colors={'smote':'#2166ac','smote_matched_weights':'#b35806'};labels={'smote':'SMOTE','smote_matched_weights':'Count/weight-matched SMOTE'}
fig,axs=plt.subplots(1,2,figsize=(11.8,5.9),sharey=True,sharex=True)
for ax,clf in zip(axs,('lr','svm')):
 for base,offset in [('smote',-.14),('smote_matched_weights',.14)]:
  q=focus[(focus.classifier==clf)&(focus.baseline==base)].set_index(['geometry','corruption']).loc[settings];yy=np.arange(9)+offset
  ax.errorbar(q.mean_difference,yy,xerr=np.vstack([q.mean_difference-q.ci_low,q.ci_high-q.mean_difference]),fmt='o',markersize=4,capsize=2,lw=1.2,color=colors[base],label=labels[base])
 ax.axvline(0,color='#555555',lw=1,ls='--');ax.set_title('Logistic regression' if clf=='lr' else 'RBF support vector machine');ax.set_xlabel('Complete minus comparator: selected-threshold G-mean');ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
axs[0].set_yticks(range(9),[f'{g.capitalize()} / {n}' for g,n in settings]);axs[0].invert_yaxis();handles,legendlabels=axs[0].get_legend_handles_labels();fig.legend(handles,legendlabels,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.58,.015));fig.suptitle('Targeted synthetic extension: paired mean differences and individual 95% intervals',fontsize=12);fig.tight_layout(rect=[0,.09,1,.95]);fig.savefig(R/'paired_gmean_intervals.png',dpi=250,bbox_inches='tight');fig.savefig(R/'paired_gmean_intervals.svg',bbox_inches='tight');plt.close(fig)
# Complete baseline data and fair paired comparisons.
b=pd.read_csv(R/'inputs/benchmark_raw.csv');assert len(b)==4320;ok=b[b.status=='ok'];means=ok.groupby(['dataset','noise_rate','classifier','method']).agg(folds=('gmean','size'),gmean=('gmean','mean'),auc=('auc','mean'),ap=('ap','mean')).reset_index();means.to_csv(R/'benchmark_dataset_all_methods.csv',index=False);full=means[means.folds==15];summ=full.groupby(['noise_rate','classifier','method']).agg(datasets=('dataset','size'),gmean=('gmean','mean'),auc=('auc','mean'),ap=('ap','mean')).reset_index();summ.to_csv(R/'benchmark_all_methods_summary.csv',index=False)
fail=b.groupby(['method','status']).size().unstack(fill_value=0).reset_index();fail.to_csv(R/'benchmark_completion.csv',index=False)
fair=[]
for (rate,clf),g in full.groupby(['noise_rate','classifier']):
 a=g[g.method=='proposed'].set_index('dataset')
 for base in sorted(set(g.method)-{'proposed'}):
  z=g[g.method==base].set_index('dataset');ids=a.index.intersection(z.index);q=dict(noise_rate=rate,classifier=clf,baseline=base,datasets=len(ids))
  for metric in ('gmean','auc','ap'):
   q['complete_'+metric]=a.loc[ids,metric].mean();q['baseline_'+metric]=z.loc[ids,metric].mean();q['delta_'+metric]=(a.loc[ids,metric]-z.loc[ids,metric]).mean()
  fair.append(q)
fair=pd.DataFrame(fair);fair.to_csv(R/'benchmark_common_dataset_pairs.csv',index=False)
# LaTeX additions, input body can be used with journal template; standalone wrapper provided below.
names={'none':'No intervention','ros':'Random oversampling','smote':'SMOTE','borderline':'Borderline-SMOTE1','adasyn':'ADASYN','proposed':'Complete','no_downweight':'No downweighting','wrnd_author':'WRND adapter'};lines=[r'\section*{Supplementary Evidence}',r'This supplement reports existing fitted-model results. No new model fits were performed for these tables. Synthetic intervals use paired replications; benchmark comparisons use dataset means, not folds as independent observations.',r'\subsection*{S1. Paired intervals for the targeted extension}',r'Differences are Complete minus comparator. Each setting has 30 paired replications. Percentile intervals use 5000 paired bootstrap resamples. They are exploratory individual 95\% intervals, with no simultaneous-coverage or multiplicity-adjusted significance claim. Full default-rule, ranking and additional-comparator intervals are included in the accompanying CSV.']
for clf in ('lr','svm'):
 lines += [r'\begin{table}[htbp]',f"\\caption{{Selected-threshold G-mean differences: {clf.upper()}}}",r'\centering\small',r'\begin{tabular}{llrr}',r'\hline',r'Geometry / noise & Comparator & Mean difference & Individual 95\% CI\\',r'\hline']
 for geo,noise in settings:
  for base in ('smote','smote_matched_weights'):
   a=focus[(focus.geometry==geo)&(focus.corruption==noise)&(focus.classifier==clf)&(focus.baseline==base)].iloc[0];lines.append(f"{geo.capitalize()} / {noise} & {'SMOTE' if base=='smote' else 'Matched SMOTE'} & {a.mean_difference:+.5f} & $[{a.ci_low:+.5f}, {a.ci_high:+.5f}]$ "+chr(92)*2)
 lines += [r'\hline',r'\end{tabular}',r'\end{table}',r'\clearpage']
lines += [r'\subsection*{S2. All benchmark baselines}',r'Only datasets with all 15 outer folds completed for a method enter its mean. $D$ counts such datasets. Different $D$ values indicate different populations; these method-wise averages must not be interpreted as a common-dataset ranking. WRND denotes the documented author-code adapter.']
for (rate,clf),g in summ.groupby(['noise_rate','classifier']):
 lines += [r'\begin{table}[htbp]',f"\\caption{{All methods: {clf.upper()}, {'supplied labels' if rate==0 else '10 percent fitting-label corruption'}}}",r'\centering\small',r'\begin{tabular}{lrrrr}',r'\hline',r'Method & $D$ & G-mean & ROC-AUC & AP\\',r'\hline']
 for method in names:
  a=g[g.method==method]
  if len(a):
   a=a.iloc[0];lines.append(f'{names[method]} & {int(a.datasets)} & {a.gmean:.4f} & {a.auc:.4f} & {a.ap:.4f} '+chr(92)*2)
 lines += [r'\hline',r'\end{tabular}',r'\end{table}']
lines += [r'\clearpage',r'\subsection*{S3. Comparisons on common complete datasets}',r'Each row restricts Complete and its comparator to the same datasets, with 15 complete outer folds for both. Differences are Complete minus comparator. No new significance test is introduced by these descriptive tables.']
for (rate,clf),g in fair.groupby(['noise_rate','classifier']):
 lines += [r'\begin{table}[htbp]',f"\\caption{{Paired means on common datasets: {clf.upper()}, {'supplied labels' if rate==0 else 'corrupted fitting labels'}}}",r'\centering\small',r'\begin{tabular}{lrrrr}',r'\hline',r'Comparator & $D$ & $\Delta$ G-mean & $\Delta$ ROC-AUC & $\Delta$ AP\\',r'\hline']
 for _,a in g.iterrows():lines.append(f'{names[a.baseline]} & {int(a.datasets)} & {a.delta_gmean:+.5f} & {a.delta_auc:+.5f} & {a.delta_ap:+.5f} '+chr(92)*2)
 lines += [r'\hline',r'\end{tabular}',r'\end{table}']
lines += [r'\clearpage',r'\subsection*{S4. Completed and failed benchmark entries}',r'Counts refer to classifier--outer-fold--condition entries, not necessarily distinct resampling events. Failures are retained and are not assigned scores.',r'\begin{table}[htbp]',r'\caption{Benchmark Completion and Failure Counts}',r'\centering\begin{tabular}{lrr}',r'\hline',r'Method & Successful & Failed\\',r'\hline']
for _,a in fail.iterrows():lines.append(f"{names[a.method]} & {int(a.get('ok',0))} & {int(a.get('failed',0))} "+chr(92)*2)
lines += [r'\hline',r'\end{tabular}\end{table}'];(R/'supplement_body.tex').write_text('\n'.join(lines)+'\n');(R/'supplement.tex').write_text(r'\documentclass[10pt]{article}'+'\n'+r'\usepackage[a4paper,margin=20mm]{geometry}'+'\n'+r'\usepackage{amsmath,graphicx}'+'\n'+r'\renewcommand{\thetable}{S\arabic{table}}'+'\n'+r'\begin{document}'+'\n'+r'\input{supplement_body.tex}'+'\n'+r'\end{document}'+'\n')
# One main-text figure insertion; no need to put all supplementary tables in the 19-page article.
(R/'main_text_insert.tex').write_text(r'''\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{paired_gmean_intervals.png}
\caption{Paired Complete-minus-comparator G-mean after validation-based threshold selection in the targeted extension. Bars are individual 95\% paired percentile bootstrap intervals from 30 replications and 5000 resamples per setting. They are exploratory and do not provide simultaneous coverage. Matched SMOTE preserves Complete's synthetic count, original weights, and synthetic-weight multiset, with synthetic weights independently permuted over SMOTE-generated observations.}
\label{fig:targeted_paired_intervals}
\end{figure*}

Figure~\ref{fig:targeted_paired_intervals} shows the uncertainty of the setting-specific differences. Favorable and adverse outcomes remain after threshold selection, and the matched control does not reveal a consistent advantage for Complete. Supplementary Tables S1--S2 report the numerical intervals; further supplementary tables provide all baseline means, comparisons on common complete datasets, and fitting-failure counts. These comparisons retain the original evaluation protocols and do not introduce new model fits.
''')
counts=focus.groupby(['classifier','baseline','direction']).size().unstack(fill_value=0).reset_index();counts.to_csv(R/'interval_direction_summary.csv',index=False);report=dict(new_model_fits=0,bootstrap_resamples=5000,paired_replications=30,focus_intervals=36,all_intervals=432,previous_intervals_reproduced=True,benchmark_rows=len(b),successful=int((b.status=='ok').sum()),failed=int((b.status!='ok').sum()),baseline_methods=int(b.method.nunique()),common_dataset_comparisons=len(fair),source_hashes={str(p.relative_to(S)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'inputs/benchmark_raw.csv',R/'inputs/targeted_raw.csv']});(R/'validation.json').write_text(json.dumps(report,indent=2));print(counts.to_string(index=False));print(fail.to_string(index=False));print(report)
