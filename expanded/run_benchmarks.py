import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from methods import proposed,PARAMS
from pathlib import Path
import numpy as np,pandas as pd,json,time,random,sys
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix,f1_score,average_precision_score,roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE,BorderlineSMOTE,ADASYN,RandomOverSampler
from joblib import Parallel,delayed
ROOT=Path(__file__).parent
METHODS=tuple(os.environ.get('BENCHMARK_METHODS','none,ros,smote,borderline,adasyn,proposed,no_downweight,wrnd_author').split(','))

def noise(y,seed,rate):
 rng=np.random.default_rng(seed); out=y.copy()
 for c in (0,1):
  ids=np.flatnonzero(y==c); picked=rng.choice(ids,int(np.floor(len(ids)*rate)),replace=False);out[picked]=1-out[picked]
 return out

def resample(z,y,method,seed):
 start=time.perf_counter()
 if method=='none': zz,yy,w=z,y,np.ones(len(y))
 elif method in ('proposed','no_downweight'):
  zz,yy,w,_=proposed(z,y,PARAMS,seed,'full' if method=='proposed' else 'no_downweight')
 elif method=='wrnd_author':
  sys.path.insert(0,str(ROOT/'vendor/WRND'))
  try:
   from nan_zhou import Natural_Neighbor
   from wrnd_generation import generate_samples_Lm
  except ImportError as ex:
   raise RuntimeError('WRND source is not bundled. Run python expanded/setup_wrnd.py first.') from ex
  random.seed(seed);np.random.seed(seed);obj=Natural_Neighbor(z,y);obj.algorithm();obj.RelativeDensity(1,0)
  weight=np.array(obj.relative_cox,float); idx=np.flatnonzero(weight>0); budget=int((y==0).sum()-(y==1).sum())
  if budget<=0:zz,yy=z,y
  elif len(idx)==0:zz,yy=z,y
  else:
   mapping={int(v):j for j,v in enumerate(idx)};neighbors=[[mapping[j] for j in obj.nan[int(i)] if j in mapping] for i in idx]
   syn,sy=generate_samples_Lm(z[idx],1,z[idx],neighbors,budget,weight[idx])
   zz=np.vstack([z,syn]);yy=np.r_[y,sy]
  w=np.ones(len(yy))
 else:
  klass={'ros':RandomOverSampler,'smote':SMOTE,'borderline':BorderlineSMOTE,'adasyn':ADASYN}[method]
  kw=dict(random_state=seed,sampling_strategy=1.)
  if method in ('smote','borderline'):kw['k_neighbors']=min(5,int((y==1).sum())-1)
  if method=='adasyn':kw['n_neighbors']=min(5,int((y==1).sum())-1)
  zz,yy=klass(**kw).fit_resample(z,y);w=np.ones(len(yy))
 assert np.isfinite(zz).all() and np.isfinite(w).all()
 return zz,yy.astype(int),w,time.perf_counter()-start

def clf(kind,value,d,seed):
 if kind=='lr':return LogisticRegression(C=value,max_iter=2000)
 if kind=='svm':return SVC(C=value,kernel='rbf',gamma=1/d)
 return RandomForestClassifier(n_estimators=300,min_samples_leaf=value,n_jobs=1,random_state=seed)

def metrics(y,p,s):
 tn,fp,fn,tp=confusion_matrix(y,p,labels=[0,1]).ravel(); rec=tp/(tp+fn);spec=tn/(tn+fp)
 return dict(gmean=float(np.sqrt(rec*spec)),recall=rec,precision=tp/(tp+fp) if tp+fp else 0.,f1=f1_score(y,p),ap=average_precision_score(y,s),auc=roc_auc_score(y,s))

def fit_task(name,rep,fold,tr,te,rate):
 dest=ROOT/'results_v2'/f'{name}_r{rep}_f{fold}_noise{rate}.json'
 if dest.exists():return json.loads(dest.read_text())
 dat=np.load(ROOT/'data'/f'{name}.npz');X=dat['X'];Y=dat['y'];x,y=X[tr],Y[tr];gx=np.unique(x,axis=0,return_inverse=True)[1]
 inner=list(StratifiedGroupKFold(3,shuffle=True,random_state=41000+rep).split(x,y,gx))
 prepared=[]
 for it,iv in inner:
  sel=VarianceThreshold().fit(x[it]);sc=StandardScaler().fit(sel.transform(x[it]));z=sc.transform(sel.transform(x[it]));val=sc.transform(sel.transform(x[iv]));yn=noise(y[it],51000+rep*100+fold,rate)
  prepared.append((z,yn,val,y[iv]))
 sel=VarianceThreshold().fit(x);sc=StandardScaler().fit(sel.transform(x));z=sc.transform(sel.transform(x));test=sc.transform(sel.transform(X[te]));yn=noise(y,61000+rep*100+fold,rate)
 rows=[]
 for method in METHODS:
  cache=[];errors=[]
  for a,b,c,d in prepared:
   try: cache.append((*resample(a,b,method,71000+rep*100+fold),c,d))
   except Exception as ex:errors.append(type(ex).__name__+': '+str(ex));break
  try: final=resample(z,yn,method,81000+rep*100+fold) if not errors else None
  except Exception as ex:errors.append(type(ex).__name__+': '+str(ex));final=None
  for kind,candidates in [('lr',[.01,.1,1,10,100]),('svm',[.1,1,10]),('rf',[1,3,5])]:
   common=dict(dataset=name,rep=rep,fold=fold,noise_rate=rate,method=method,classifier=kind,ntrain=len(tr),ntest=len(te))
   if errors: rows.append(dict(**common,status='failed',error=' | '.join(errors)));continue
   vals=[]
   for value in candidates:
    scores=[]
    for a,b,w,t,c,d in cache:
     model=clf(kind,value,a.shape[1],91000+rep*100+fold);model.fit(a,b,sample_weight=w);pp=model.predict(c)
     scores.append(metrics(d,pp,model.decision_function(c) if kind!='rf' else model.predict_proba(c)[:,1])['gmean'])
    vals.append(np.mean(scores))
   selected=candidates[int(np.argmax(vals))];a,b,w,t=final;start=time.perf_counter();model=clf(kind,selected,a.shape[1],91000+rep*100+fold).fit(a,b,sample_weight=w);fitsec=time.perf_counter()-start
   pp=model.predict(test);ss=model.decision_function(test) if kind!='rf' else model.predict_proba(test)[:,1]
   rows.append(dict(**common,status='ok',selected=float(selected),inner_gmean=max(vals),synthetic=len(b)-len(y),intervention_seconds=t,fit_seconds=fitsec,**metrics(Y[te],pp,ss)))
 dest.write_text(json.dumps(rows));return rows

def main():
 (ROOT/'results_v2').mkdir(exist_ok=True);manifest=json.loads((ROOT/'data/manifest.json').read_text());jobs=[]
 for m in manifest:
  name=m['name'];data=np.load(ROOT/'data'/f'{name}.npz');X,Y=data['X'],data['y'];groups=np.unique(X,axis=0,return_inverse=True)[1]
  for rep in range(3):
   splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=31000+rep).split(X,Y,groups))
   for fold,(tr,te) in enumerate(splits):
    assert set(groups[tr]).isdisjoint(set(groups[te]))
    for rate in (0.,.10):jobs.append((name,rep,fold,tr,te,rate))
 print('Starting',len(jobs),'paired outer-fold tasks; 24 final models per task',flush=True)
 results=Parallel(n_jobs=int(os.environ.get("EXPERIMENT_JOBS","4")),verbose=10)(delayed(fit_task)(*job) for job in jobs)
 df=pd.DataFrame([r for group in results for r in group]);df.to_csv(ROOT/'benchmark_raw.csv',index=False)
 ok=df[df.status=='ok'];ok.groupby(['dataset','noise_rate','classifier','method'])[['gmean','recall','precision','f1','ap']].agg(['mean','std']).to_csv(ROOT/'benchmark_summary.csv')
 print(df.groupby(['method','status']).size().to_string(),flush=True)
if __name__=='__main__':main()
