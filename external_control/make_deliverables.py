from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).parent;ci=pd.read_csv(R/'paired_intervals.csv');means=pd.read_csv(R/'summary.csv');au=pd.read_csv(R/'intervention_audit.csv')
order=[(g,c) for g in ['gaussian','moons','circles'] for c in ['clean','dispersed','boundary']]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9})
fig,axes=plt.subplots(1,2,figsize=(12,5.7),sharex=True,sharey=True)
for ax,clf,title in zip(axes,['lr','svm'],['Logistic regression','RBF support vector machine']):
 for base,color,offset,label in [('smote','#2471a3',-.12,'SMOTE'),('smote_matched_retention','#ca6f1e',.12,'Count + original-retention matched SMOTE')]:
  q=ci[(ci.classifier==clf)&(ci.rule=='selected')&(ci.metric=='gmean')&(ci.baseline==base)].set_index(['geometry','corruption']).loc[order]
  ax.errorbar(q.mean_difference,np.arange(9)+offset,xerr=np.c_[q.mean_difference-q.ci_low,q.ci_high-q.mean_difference].T,fmt='o',markersize=3,capsize=2,color=color,lw=1,label=label)
 ax.axvline(0,ls='--',lw=.9,color='#555');ax.grid(axis='x',alpha=.2);ax.set_title(title);ax.set_xlabel('SMOTE-ENN minus comparator: selected-threshold G-mean');ax.spines[['top','right']].set_visible(False)
axes[0].set_yticks(range(9),[g.title()+' / '+c for g,c in order]);axes[0].invert_yaxis();handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,bbox_to_anchor=(.55,.005),frameon=False)
fig.suptitle('External-method follow-up: individual 95% paired bootstrap intervals',y=.99);fig.tight_layout(rect=[0,.06,1,.96]);fig.savefig(R/'external_control_intervals.png',dpi=300,bbox_inches='tight');fig.savefig(R/'external_control_intervals.svg',bbox_inches='tight')
# Mean differences summarize settings equally, not independent-dataset inference.
protocol=r'''\subsection{External Hybrid-Method Control}
\label{subsec:external_method_protocol}
A follow-up experiment evaluates SMOTE-ENN as an external hybrid oversampling-and-cleaning method\cite{batista2004balancing}, using imbalanced-learn version 0.14.2\cite{lemaitre2017imblearn}. This is the explicitly configured software implementation, not a reproduction of every setting in the original publication. WRND was not used for this additional confirmation because full algorithm-to-code reconciliation remained unresolved; earlier adapter results remain reported separately.

The experiment reuses the targeted extension's Gaussian, moons, and circles observations, clean/dispersed/boundary-near corruption, 30 replications, fitting-only standardization, and fixed LR and RBF-SVM configurations. Each replication contains 1000 fitting, 2000 independent trusted-validation, and 5000 independent test observations. SMOTE uses a target ratio of 1, five generation neighbors, and the same recorded resampling seed as the earlier extension. ENN uses three neighbors, \texttt{kind\_sel=all}, and \texttt{sampling\_strategy=all}. All resulting classifier weights are one.

The SMOTE stage is followed by ENN editing of both original and synthetic observations. Index provenance identifies which originals and synthetics survive. A count-matched SMOTE control generates exactly the number of surviving SMOTE-ENN synthetic observations on the full original fitting data. A second control uses these same count-matched synthetics but retains exactly the original observations retained by SMOTE-ENN. It applies no additional editing to the control synthetics. Thus, count matching alone leaves original deletion uncontrolled, whereas the second control matches original retention and both original and synthetic training mass. The latter comparison changes the retained synthetic coordinates and endpoint composition together; it is not a pure causal estimate of location alone.

Default and validation-selected decisions are evaluated on the same fitted models, with distinct descending validation scores and the first G-mean maximum resolving ties. Test observations do not enter any selection. No intervention and ordinary SMOTE results are reused from the identical prior targeted runs; the three new configurations require 1620 new model fits. Fitting-only preflight checks establish exact output agreement with the uninstrumented library implementation before comparative held-out results are examined. The protocol is frozen after these checks, but the study remains a follow-up designed after earlier results.

Paired percentile 95\% intervals use 5000 resamples of the 30 replications within each setting (seed 90120261016). These are exploratory individual intervals without simultaneous coverage or multiplicity-adjusted significance claims. Cross-setting means are descriptive. Original-removal precision/recall, valid minority removal, and corrupted-endpoint use before and after editing are recorded; endpoint contamination does not identify the true label of each synthetic observation.
'''
(R/'protocol_insert.tex').write_text(protocol)
rows=[]
labels={'none':'No intervention','smote':'SMOTE','smote_matched_count':'Count-matched SMOTE','smote_matched_retention':'Count + retention matched'}
for clf in ['lr','svm']:
 for base in labels:
  q=ci[(ci.classifier==clf)&(ci.metric=='gmean')&(ci.baseline==base)];qd=q[q.rule=='default'];qt=q[q.rule=='selected'];above=(qt.ci_low>0).sum();below=(qt.ci_high<0).sum();cross=9-above-below
  rows.append(f"{'LR' if clf=='lr' else 'SVM'} & {labels[base]} & {qd.mean_difference.mean():+.5f} & {qt.mean_difference.mean():+.5f} & {above}/{below}/{cross} \\\\")
results=r'''\subsection{External Hybrid-Method Results}
\label{subsec:external_method_results}
All 1620 additional model fits complete without fitting warnings, giving 3240 decision-rule evaluations. Provenance checks verify exact synthetic-count matching and original-retention matching in all 270 geometry--corruption--replication combinations. The original and copied generators agree exactly in 270 data-role checks. No-intervention and ordinary SMOTE results are reused from the earlier targeted extension.

Table~\ref{tab:external_control_means} reports equally weighted means of the nine setting-specific differences. SMOTE-ENN's default-rule advantage over no intervention is $0.2436$ for LR and $0.0555$ for SVM; after validation-based threshold selection, the corresponding differences are $-0.0026$ and $-0.0030$. Against ordinary SMOTE, selected-threshold mean differences are $-0.0003$ and $-0.0026$. Thus, the operating-point explanation is relevant to an external hybrid method as well, but does not explain every pairwise outcome.

\begin{table*}[t]
\caption{External Control: Descriptive Mean SMOTE-ENN-minus-Comparator G-mean Differences}
\label{tab:external_control_means}
\centering\small
\begin{tabular}{llrrc}
\hline
Model & Comparator & Default & Selected & Selected interval directions\\
\hline
'''+ '\n'.join(rows)+r'''
\hline
\end{tabular}
\par\smallskip\footnotesize Each mean weights nine settings equally. Direction counts are above zero / below zero / including zero for individual 95\% paired intervals, not adjusted significance decisions. Count matching uses post-editing synthetic counts; retention matching additionally uses the exact surviving original observations. Intervals are setting-specific; no pooled interval over the nine dependent settings is claimed.
\end{table*}

Count matching alone gives selected-threshold mean differences of $-0.0003$ for LR and $-0.0048$ for SVM. Matching original retention as well changes these to $+0.0004$ and $+0.0058$. For SVM with dispersed corruption, SMOTE-ENN exceeds the retention-matched control in Gaussian, moons, and circles settings, with respective mean differences of $0.01018$, $0.01601$, and $0.03062$; their individual paired intervals lie above zero. Boundary-near corruption instead produces adverse SVM differences for Gaussian and circles. These outcomes do not establish uniform superiority.

Editing also changes intervention quality. Averaged across geometries under dispersed corruption, corrupted-endpoint use declines from $0.5741$ before editing to $0.3656$ among retained synthetics. However, approximately $14.2\%$ of correctly labeled minority originals are removed, and original-removal precision is approximately $14.2\%$. Under boundary-near corruption, endpoint contamination changes little ($0.5944$ to $0.5864$). These summaries distinguish label-related intervention errors from classification outcomes and do not establish synthetic-label correctness.

% Optional figure: copy external_control_intervals.png into the manuscript figures folder.
\begin{figure*}[t]
\def\xfigwd{\columnwidth}
\centering
\includegraphics[width=\textwidth]{external_control_intervals.png}
\caption{External hybrid-method follow-up. SMOTE-ENN-minus-comparator G-mean after validation-based threshold selection; bars are individual exploratory 95\% paired bootstrap intervals based on 30 replications and 5000 resamples. Matching original retention is additional to matching synthetic count. These intervals do not provide simultaneous coverage.}
\label{fig:external_control_intervals}
\end{figure*}
'''
(R/'results_insert.tex').write_text(results)
(R/'discussion_insert.tex').write_text(r'''\subsection{Evidence beyond the Case-Study Framework}
The external SMOTE-ENN experiment extends the control analysis to an established hybrid cleaning method, rather than another variant of the proposed framework. Its default-rule gains over no intervention largely disappear on average after validation-based threshold selection. This corroborates the relevance of operating-point assessment beyond the case-study intervention within the examined synthetic conditions.

Original deletion introduces an additional distinction. A control matching only the surviving synthetic count can give a different conclusion from one that also matches the retained originals. The retention-matched SVM comparisons retain benefits under dispersed corruption in all three geometries, while boundary-near conditions include adverse outcomes. Thus, the evidence does not support a universal explanation that threshold shifts or sample counts account for all hybrid-intervention effects. It instead shows why original retention, synthetic selection, and decision rules require separate controls.

Removal of corrupted observations or endpoints can coexist with removal of valid minority originals. These measurements should accompany predictive metrics when a hybrid method is interpreted as noise handling. The additional experiment uses one configured external hybrid method and the same nine synthetic settings; it neither verifies WRND nor establishes a general result across modern noise-aware oversamplers or real-world datasets. Its design follows earlier findings and requires broader independent confirmation.
''')
# Full standalone numerical source; no conflict with existing supplementary S1--S15 numbering.
lines=[r'\section*{External Hybrid-Method Follow-up}',r'This separate addendum uses E-numbered tables. The configured external method is SMOTE-ENN from imbalanced-learn 0.14.2. All differences are SMOTE-ENN minus comparator. Bootstrap intervals are individual exploratory 95\% intervals, not simultaneous intervals or adjusted hypothesis tests.']
for clf in ['lr','svm']:
 for rule in ['default','selected']:
  lines += [r'\begin{table}[htbp]',f'\\caption{{G-mean: {clf.upper()}, {rule} decision rule}}',r'\centering\small',r'\begin{tabular}{llrr}',r'\hline',r'Setting & Comparator & Mean $\Delta$ & Individual 95\% CI\\',r'\hline']
  for g,c in order:
   for base in labels:
    q=ci[(ci.geometry==g)&(ci.corruption==c)&(ci.classifier==clf)&(ci.rule==rule)&(ci.metric=='gmean')&(ci.baseline==base)].iloc[0]
    short={'none':'None','smote':'SMOTE','smote_matched_count':'Count','smote_matched_retention':'Count + retention'}[base]
    lines.append(f'{g.title()} / {c} & {short} & {q.mean_difference:+.5f} & $[{q.ci_low:+.5f}, {q.ci_high:+.5f}]$ \\\\')
  lines += [r'\hline',r'\end{tabular}',r'\end{table}',r'\clearpage']
for clf in ['lr','svm']:
 for rule in ['default','selected']:
  lines += [r'\begin{table}[htbp]',f'\\caption{{All metrics: {clf.upper()}, {rule} decision rule}}',r'\centering\small',r'\begin{tabular}{llrrr}',r'\hline',r'Setting & Method & G-mean & ROC-AUC & AP\\',r'\hline']
  for g,c in order:
   for name,lab in [('none','None'),('smote','SMOTE'),('smoteenn','SMOTE-ENN'),('smote_matched_count','Count'),('smote_matched_retention','Count + retention')]:
    q=means[(means.geometry==g)&(means.corruption==c)&(means.classifier==clf)&(means.rule==rule)&(means.method==name)].iloc[0];lines.append(f'{g.title()} / {c} & {lab} & {q.gmean:.4f} & {q.auc:.4f} & {q.ap:.4f} \\\\')
  lines += [r'\hline',r'\end{tabular}',r'\end{table}',r'\clearpage']
(R/'external_tables.tex').write_text('\n'.join(lines))
(R/'external_supplement.tex').write_text(r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage{amsmath,graphicx}
\renewcommand{\thetable}{E\arabic{table}}
\begin{document}
\input{external_tables.tex}
\end{document}
''')
(R/'bibitem_insert.tex').write_text(r'''% Insert in thebibliography in first-citation order.
\bibitem{batista2004balancing}
G. E. A. P. A. Batista, R. C. Prati, and M. C. Monard,
``A study of the behavior of several methods for balancing machine learning training data,''
\emph{ACM SIGKDD Explorations Newsletter}, vol. 6, no. 1, pp. 20--29, 2004.
[Online]. Available: \url{https://www.kdd.org/exploration_files/batista.pdf}
''')
print('Figure, three manuscript inserts, bibliography entry, and E1--E8 table sources written')
