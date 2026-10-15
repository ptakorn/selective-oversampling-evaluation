"""Read-only checks of new evidence; no classifier fitting or result rewriting."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,io,re,zipfile,hashlib
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
R=Path(__file__).resolve().parent
old=pd.read_csv(R/'revision_audit/confirmation/raw.csv');assert len(old)==5400 and old.status.eq('ok').all()
keys=['geometry','rep','corruption','method','classifier','rule'];assert not old.duplicated(keys).any()
assert old.groupby(keys[:-1]).size().eq(2).all()
for col in ('synthetic','original_weight_sum','synthetic_weight_sum'):
 p=old.pivot(index=['geometry','rep','corruption','classifier','rule'],columns='method',values=col)
 assert np.max(abs(p.complete-p.smote_matched_weights))<1e-8
new=pd.read_csv(R/'external_control/new_raw.csv');allraw=pd.read_csv(R/'external_control/all_raw.csv');audit=pd.read_csv(R/'external_control/intervention_audit.csv')
assert len(new)==3240 and new.status.eq('ok').all() and len(audit)==270
assert not new.duplicated(keys).any();assert new.groupby(keys[:-1]).size().eq(2).all();assert new.warnings.fillna('').eq('').all();assert len(allraw)==5400
reused=allraw[allraw.method.isin(['none','smote'])].merge(old,on=keys,suffixes=('_added','_prior'));assert len(reused)==2160
for col in ('gmean','auc','ap'):assert np.max(abs(reused[col+'_added']-reused[col+'_prior']))<1e-12
for col in ('auc','ap'):
 p=new.pivot(index=keys[:-1],columns='rule',values=col);assert np.array_equal(p.default.values,p.selected.values)
for frame,path,target,seed in [(old,R/'evidence_supplement/all_paired_intervals.csv','complete',90120261015),(allraw,R/'external_control/paired_intervals.csv','smoteenn',90120261016)]:
 ci=pd.read_csv(path);assert len(ci)==432
 for row in ci.itertuples():
  g=frame[(frame.geometry==row.geometry)&(frame.corruption==row.corruption)&(frame.classifier==row.classifier)&(frame.rule==row.rule)]
  p=g.pivot(index='rep',columns='method',values=row.metric).sort_index();delta=(p[target]-p[row.baseline]).to_numpy();assert len(delta)==30
  boot=np.random.default_rng(seed).choice(delta,(5000,30),replace=True).mean(axis=1);lo,hi=np.quantile(boot,[.025,.975]);assert max(abs(delta.mean()-row.mean_difference),abs(lo-row.ci_low),abs(hi-row.ci_high))<1e-12
sys.path.insert(0,str(R/'external_control'));from protocol import fitting,GEO
indexed=audit.set_index(['geometry','rep','corruption'])
with zipfile.ZipFile(R/'external_control/provenance.zip') as z:
 assert len(z.namelist())==270
 for member in z.namelist():
  m=re.fullmatch(r'g(\d)_r(\d+)_(clean|dispersed|boundary)\.npz',member);assert m
  g,rep,kind=int(m[1]),int(m[2]),m[3];a=np.load(io.BytesIO(z.read(member)));x,y,true=fitting(g,rep,kind);scaled=StandardScaler().fit_transform(x);assert hashlib.sha256(scaled.tobytes()+y.tobytes()).hexdigest()==str(a['input_sha256'])
  keep=a['original_keep'];sk=a['synthetic_keep'];ep=a['endpoints'];q=indexed.loc[(GEO[g],rep,kind)]
  assert len(keep)==q.original_retained and len(sk)==q.synthetic_retained and len(ep)==q.full_synthetic;assert len(np.unique(keep))==len(keep) and len(np.unique(sk))==len(sk)
  assert keep.min()>=0 and keep.max()<1000 and ep.min()>=0 and ep.max()<1000;assert np.all(y[ep]==1)
  removed=np.ones(1000,bool);removed[keep]=False;bad=y!=true;valid=(y==true)&(true==1);ec=bad[ep].any(axis=1)
  values={'original_deleted':removed.sum(),'valid_minority_removed':removed[valid].mean(),'ecr_before':ec.mean(),'ecr_after':ec[sk].mean() if len(sk) else np.nan,'removal_precision':bad[removed].mean() if removed.any() else np.nan,'removal_recall':removed[bad].mean() if bad.any() else np.nan}
  for col,v in values.items():assert (pd.isna(v) and pd.isna(q[col])) or abs(v-q[col])<1e-12,(member,col)
for copy,canonical in [('evidence_supplement/inputs/targeted_raw.csv','revision_audit/confirmation/raw.csv'),('evidence_supplement/inputs/benchmark_raw.csv','expanded/benchmark_raw.csv'),('external_control/inputs/previous_targeted_raw.csv','revision_audit/confirmation/raw.csv')]:assert (R/copy).read_bytes()==(R/canonical).read_bytes()
assert (R/'reports/supplement_body.tex').read_text().count('\\begin{table}')==23
print('PASS: 2700 targeted models, 1620 external models, 2160 reused evaluations, 864 paired intervals, 270 endpoint/retention audits, canonical raw copies and 23 supplementary tables. No classifiers fitted.')
