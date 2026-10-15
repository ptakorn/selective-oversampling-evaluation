import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from run_benchmarks import resample,metrics,ROOT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from joblib import Parallel,delayed
import numpy as np,pandas as pd,json

def generate(n,ir,sep,seed):
 rng=np.random.default_rng(seed);n1=n//(1+ir);y=np.r_[np.zeros(n-n1,int),np.ones(n1,int)];X=rng.normal(size=(n,2));X[y==1,0]+=sep;p=rng.permutation(n);return X[p],y[p]

def task(rep,ir,sep,rate,mechanism):
 dest=ROOT/'synthetic_results'/f'r{rep}_ir{ir}_s{sep}_e{rate}_{mechanism}.json'
 if dest.exists():return json.loads(dest.read_text())
 seed=100000+rep+ir*1000+sep*10000;x,true=generate(2000,ir,sep,seed);xt,yt=generate(10000,ir,sep,seed+500000)
 rng=np.random.default_rng(seed+700000);y=true.copy()
 for c in (0,1):
  ids=np.flatnonzero(true==c);count=int(np.floor(len(ids)*rate))
  chosen=rng.choice(ids,count,replace=False) if mechanism=='dispersed' else ids[np.argsort(((x[ids]-np.array([sep*c,0]))**2).sum(axis=1),kind='stable')[:count]]
  y[chosen]=1-y[chosen]
 sc=StandardScaler().fit(x);z=sc.transform(x);zt=sc.transform(xt);rows=[]
 for method in ('none','smote','borderline','adasyn','proposed','no_downweight'):
  common=dict(rep=rep,ir=ir,separation=sep,noise_rate=rate,mechanism=mechanism,method=method)
  try:
   zz,yy,w,t=resample(z,y,method,seed+900000);model=LogisticRegression(C=1,max_iter=2000).fit(zz,yy,sample_weight=w)
   rows.append(dict(**common,status='ok',synthetic=len(yy)-len(y),**metrics(yt,model.predict(zt),model.decision_function(zt))))
  except Exception as ex:rows.append(dict(**common,status='failed',error=type(ex).__name__+': '+str(ex)))
 dest.write_text(json.dumps(rows));return rows
if __name__=='__main__':
 (ROOT/'synthetic_results').mkdir(exist_ok=True)
 jobs=[(r,ir,s,e,m) for ir in (5,10,20) for s in (1,2,3) for e,m in [(0.,'clean')]+[(e,m) for e in (.05,.10,.20) for m in ('dispersed','localized')] for r in range(30)]
 print('Starting',len(jobs),'independent synthetic scenario/replication tasks',flush=True)
 out=Parallel(n_jobs=int(os.environ.get("EXPERIMENT_JOBS","4")),verbose=10)(delayed(task)(*j) for j in jobs)
 df=pd.DataFrame([row for group in out for row in group]);df.to_csv(ROOT/'synthetic_grid_raw.csv',index=False)
 df[df.status=='ok'].groupby(['ir','separation','noise_rate','mechanism','method'])[['gmean','recall','precision','f1','ap']].agg(['mean','std']).to_csv(ROOT/'synthetic_grid_summary.csv')
 print('Completed',len(df),'method evaluations',flush=True)
