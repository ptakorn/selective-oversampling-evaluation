import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from run_benchmarks import ROOT,resample,metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel,delayed
import numpy as np,pandas as pd,json

def generate(n,family,value,seed):
 rng=np.random.default_rng(seed);n1=n//11;y=np.r_[np.zeros(n-n1,int),np.ones(n1,int)];d=int(value) if family=='dimensions' else 2
 X=rng.normal(size=(n,d));X[y==1,0]+=2 if family=='dimensions' else 3;component=np.zeros(n,int)
 if family in ('subgroup','isolation'):
  gamma=float(value) if family=='subgroup' else .05;rare=np.flatnonzero(y==1)[rng.random(n1)<gamma];component[rare]=1;X[rare,:2]=rng.normal([0,2 if family=='subgroup' else 6],.3,size=(len(rare),2))
 perm=rng.permutation(n);return X[perm],y[perm],component[perm]

def task(rep,family,value,n):
 seed=1300000+rep+int(float(value)*1000)+n;X,y,component=generate(n,family,value,seed);xt,yt,ct=generate(10000,family,value,seed+500000)
 if family=='feature_outliers':
  rng=np.random.default_rng(seed+200000)
  for c in (0,1):
   ids=np.flatnonzero(y==c);chosen=rng.choice(ids,int(float(value)*len(ids)),replace=False);X[chosen]+=rng.normal(0,5,size=(len(chosen),X.shape[1]))
 sc=StandardScaler().fit(X);z=sc.transform(X);zt=sc.transform(xt);rows=[]
 for method in ('none','smote','borderline','adasyn','proposed','no_downweight'):
  common=dict(rep=rep,family=family,value=value,n=n,method=method)
  try:
   zz,yy,w,t=resample(z,y,method,seed+900000);model=LogisticRegression(C=1,max_iter=2000).fit(zz,yy,sample_weight=w);p=model.predict(zt);sg=(yt==1)&(ct==1)
   valid=y==1;rows.append(dict(**common,status='ok',synthetic=len(yy)-len(y),mean_min_weight=w[:len(y)][valid].mean(),fdw_min=(w[:len(y)][valid]<1).mean(),subgroup_recall=float(p[sg].mean()) if sg.any() else None,**metrics(yt,p,model.decision_function(zt))))
  except Exception as e:rows.append(dict(**common,status='failed',error=type(e).__name__+': '+str(e)))
 return rows
if __name__=='__main__':
 settings=[('subgroup',g,n) for g in (.05,.1,.2) for n in (500,1000,2000)]+[('isolation',.05,n) for n in (500,1000,2000)]+[('feature_outliers',z,2000) for z in (.05,.1)]+[('dimensions',d,2000) for d in (2,10,50)]
 out=Parallel(n_jobs=int(os.environ.get("EXPERIMENT_JOBS","4")),verbose=10)(delayed(task)(r,*setting) for setting in settings for r in range(30));df=pd.DataFrame([row for group in out for row in group]);df.to_csv(ROOT/'stress_raw.csv',index=False);df[df.status=='ok'].groupby(['family','value','n','method'])[['gmean','f1','fdw_min','subgroup_recall']].agg(['mean','std']).to_csv(ROOT/'stress_summary.csv');print('Completed',len(df),'stress-method evaluations')
