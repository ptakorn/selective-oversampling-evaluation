import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json
from pathlib import Path
R=Path(__file__).resolve().parent;E=R.parent/'expanded';sys.path.insert(0,str(E))
import numpy as np,pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from joblib import Parallel,delayed
from run_benchmarks import clf,resample,noise,metrics
OLD=pd.read_csv(E/'benchmark_raw.csv');OLD=OLD[(OLD.status=='ok')&OLD.method.isin(['none','smote','proposed'])].set_index(['dataset','rep','fold','noise_rate','method','classifier'])

def choose_threshold(ys,ss):
 y=np.concatenate(ys);s=np.concatenate(ss);fid=np.concatenate([np.full(len(v),i) for i,v in enumerate(ys)]);order=np.argsort(-s,kind='stable');y=y[order];s=s[order];fid=fid[order];ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
 values=[]
 for v in range(3):
  tp=np.cumsum((fid==v)&(y==1))[ends];fp=np.cumsum((fid==v)&(y==0))[ends];pos=(ys[v]==1).sum();neg=(ys[v]==0).sum();assert pos and neg;values.append(np.sqrt((tp/pos)*(1-fp/neg)))
 mean=np.mean(values,axis=0);j=int(np.argmax(mean));return float(s[ends[j]]),float(mean[j])

def task(name,rep,fold,tr,te,rate):
 dest=R/'checkpoints'/f'{name}_r{rep}_f{fold}_noise{rate}.json'
 if dest.exists():return json.loads(dest.read_text())
 dat=np.load(E/'data'/f'{name}.npz');X,Y=dat['X'],dat['y'];x,y=X[tr],Y[tr];gx=np.unique(x,axis=0,return_inverse=True)[1];inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=41000+rep).split(x,y,gx));prepared=[]
 for it,iv in inner:
  assert not set(gx[it])&set(gx[iv]);sel=VarianceThreshold().fit(x[it]);sc=StandardScaler().fit(sel.transform(x[it]));prepared.append((sc.transform(sel.transform(x[it])),noise(y[it],51000+rep*100+fold,rate),sc.transform(sel.transform(x[iv])),y[iv]))
 sel=VarianceThreshold().fit(x);sc=StandardScaler().fit(sel.transform(x));z=sc.transform(sel.transform(x));test=sc.transform(sel.transform(X[te]));yn=noise(y,61000+rep*100+fold,rate);rows=[];saved={}
 for method in ('none','smote','proposed'):
  cache=[(*resample(a,b,method,71000+rep*100+fold)[:3],c,d) for a,b,c,d in prepared];final=resample(z,yn,method,81000+rep*100+fold)[:3]
  for kind in ('lr','svm','rf'):
   previous=OLD.loc[(name,rep,fold,rate,method,kind)];selected=previous.selected;ss=[];ys=[]
   for a,b,w,c,d in cache:
    model=clf(kind,selected if kind!='rf' else int(selected),a.shape[1],91000+rep*100+fold).fit(a,b,sample_weight=w);score=model.predict_proba(c)[:,1] if kind=='rf' else model.decision_function(c);ss.append(score);ys.append(d)
   threshold,inner_score=choose_threshold(ys,ss);a,b,w=final;model=clf(kind,selected if kind!='rf' else int(selected),a.shape[1],91000+rep*100+fold).fit(a,b,sample_weight=w);score=model.predict_proba(test)[:,1] if kind=='rf' else model.decision_function(test);prefix=method+'_'+kind
   saved[prefix+'_test_scores']=score;saved[prefix+'_oof_scores']=np.concatenate(ss);saved[prefix+'_oof_labels']=np.concatenate(ys);saved[prefix+'_oof_fold']=np.concatenate([np.full(len(v),i) for i,v in enumerate(ys)])
   for rule,pred in [('default',model.predict(test)),('selected',(score>=threshold).astype(int))]:
    result=metrics(Y[te],pred,score)
    if rule=='default':assert abs(result['gmean']-previous.gmean)<1e-12
    rows.append(dict(dataset=name,rep=rep,fold=fold,noise_rate=rate,method=method,classifier=kind,rule=rule,status='ok',selected=selected,threshold=threshold,inner_threshold_gmean=inner_score,synthetic=len(b)-len(y),**result))
 saved['test_labels']=Y[te];saved['test_indices']=te;np.savez_compressed(dest.with_suffix('.npz'),**saved);dest.write_text(json.dumps(rows));return rows

def main():
 jobs=[]
 for m in json.loads((E/'data/manifest.json').read_text()):
  name=m['name'];dat=np.load(E/'data'/f'{name}.npz');X,Y=dat['X'],dat['y'];groups=np.unique(X,axis=0,return_inverse=True)[1]
  for rep in range(3):
   for fold,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=31000+rep).split(X,Y,groups)):
    assert not set(groups[tr])&set(groups[te])
    for rate in (0.,.1):jobs.append((name,rep,fold,tr,te,rate))
 out=Parallel(n_jobs=int(os.environ.get("EXPERIMENT_JOBS","4")),verbose=10)(delayed(task)(*j) for j in jobs);pd.DataFrame([r for group in out for r in group]).to_csv(R/'raw.csv',index=False);print('Completed 1620 final models; 3240 decision-rule evaluations',flush=True)
if __name__=='__main__':main()
