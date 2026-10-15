import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,time,hashlib,warnings
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score,f1_score
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel,delayed
from protocol import fitting,generate,rng,threshold,GEO,KINDS
from resampling import prepare
R=Path(__file__).parent

def metrics(y,p,s):
 tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel();return dict(gmean=float(np.sqrt(tp/(tp+fn)*tn/(tn+fp))),auc=float(roc_auc_score(y,s)),ap=float(average_precision_score(y,s)),recall=float(tp/(tp+fn)),precision=float(tp/(tp+fp)) if tp+fp else 0.,f1=float(f1_score(y,p)))
def ratio(a,b):return float(a/b) if b else None

def task(g,rep,kind):
 dest=R/'checkpoints'/f'g{g}_r{rep}_{kind}.json'
 if dest.exists():return json.loads(dest.read_text())
 x,y,true=fitting(g,rep,kind);xv,yv=generate(2000,g,rep,2);xt,yt=generate(5000,g,rep,3)
 sc=StandardScaler().fit(x);z=sc.transform(x);zv=sc.transform(xv);zt=sc.transform(xt);seed=int(rng(g,rep,20).integers(0,2**31-1));start=time.perf_counter()
 try:variants,a=prepare(z,y,seed)
 except Exception as ex:
  rows=[dict(geometry=GEO[g],rep=rep,corruption=kind,method=m,classifier=c,rule=rule,status='failed',error=repr(ex)) for m in ('smoteenn','smote_matched_count','smote_matched_retention') for c in ('lr','svm') for rule in ('default','selected')]
  dest.write_text(json.dumps(rows));return rows
 resampling_seconds=time.perf_counter()-start
 keep=a['original_keep'];deleted=np.ones(len(y),bool);deleted[keep]=False;bad=y!=true;validminor=(y==true)&(true==1)
 ec=np.any(bad[a['endpoints']],axis=1);sk=a['synthetic_keep'];ns=a['synthetic_retained']
 audit=dict(geometry=GEO[g],rep=rep,corruption=kind,original=len(y),original_retained=len(keep),original_deleted=int(deleted.sum()),retained_0=int((y[keep]==0).sum()),retained_1=int((y[keep]==1).sum()),full_synthetic=a['full_synthetic'],synthetic_retained=ns,synthetic_removed=a['full_synthetic']-ns,removal_precision=ratio((deleted&bad).sum(),deleted.sum()),removal_recall=ratio((deleted&bad).sum(),bad.sum()),valid_minority_removed=ratio((deleted&validminor).sum(),validminor.sum()),ecr_before=float(ec.mean()),ecr_after=float(ec[sk].mean()) if ns else None,count_match_exact=True,retention_match_exact=True)
 np.savez_compressed(R/'provenance'/f'g{g}_r{rep}_{kind}.npz',original_keep=keep,synthetic_keep=sk,endpoints=a['endpoints'],input_sha256=np.array(hashlib.sha256(z.tobytes()+y.tobytes()).hexdigest()))
 rows=[]
 for name,(zz,yy) in variants.items():
  for clf in ('lr','svm'):
   model=LogisticRegression(C=1,max_iter=2000) if clf=='lr' else SVC(C=1,kernel='rbf',gamma=1/zz.shape[1])
   try:
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always');model.fit(zz,yy)
    fitsec=time.perf_counter()-start;sv=model.decision_function(zv);st=model.decision_function(zt);t=threshold(yv,sv)
    assert np.isfinite(st).all() and np.isfinite(sv).all()
    for rule,pred in [('default',model.predict(zt)),('selected',(st>=t).astype(int))]:
     rows.append(dict(geometry=GEO[g],rep=rep,corruption=kind,method=name,classifier=clf,rule=rule,status='ok',threshold=t,synthetic=ns,original_retained=(len(y) if name=='smote_matched_count' else len(keep)),fit_seconds=fitsec,resampling_seconds=resampling_seconds,warnings=' | '.join(str(w.message) for w in caught),**metrics(yt,pred,st)))
   except Exception as ex:
    rows.extend([dict(geometry=GEO[g],rep=rep,corruption=kind,method=name,classifier=clf,rule=rule,status='failed',error=repr(ex)) for rule in ('default','selected')])
 (R/'checkpoints'/f'audit_g{g}_r{rep}_{kind}.json').write_text(json.dumps(audit));dest.write_text(json.dumps(rows));return rows

if __name__=='__main__':
 cfg=(R/'config.json').read_bytes();print('Frozen protocol SHA256',hashlib.sha256(cfg).hexdigest(),flush=True)
 jobs=[(g,rep,kind) for g in range(3) for kind in KINDS for rep in range(30)]
 out=Parallel(n_jobs=4,verbose=10)(delayed(task)(*j) for j in jobs)
 df=pd.DataFrame([r for rows in out for r in rows]);df.to_csv(R/'new_raw.csv',index=False)
 au=[json.loads(p.read_text()) for p in sorted((R/'checkpoints').glob('audit_*.json'))];pd.DataFrame(au).to_csv(R/'intervention_audit.csv',index=False)
 print(df.groupby(['method','status']).size().to_string(),flush=True);print('Expected 1620 new model fits / 3240 decision-rule evaluations',flush=True)
