import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent; EXP=ROOT.parent/'expanded';sys.path.insert(0,str(EXP))
import numpy as np,pandas as pd
from joblib import Parallel,delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import run_benchmarks as bench
from run_synthetic_grid import generate
from run_stress import generate as stress_generate
from ablation_methods import proposed,PARAMS

def intervention(z,y,seed,variant):return proposed(z,y,PARAMS,seed,neighborhood=variant)
def diagnostics(y,true,info,w):
 bad=y!=true;valid=(y==true)&(true==1);ends=info['endpoints'];cont=bad[ends].any(axis=1) if len(ends) else np.array([])
 return dict(eligible_sources=len(info['sources']),adequacy_fraction=float(info['g'].mean()),fdw_min=float(info['dw'][valid].mean()) if valid.any() else None,endpoint_contamination=float(cont.mean()) if len(cont) else None,mean_min_weight=float(w[:len(y)][valid].mean()) if valid.any() else None,base_size=float(info['sizes'][1].mean()))
def task(rep,ir=None,sep=None,rate=None,mechanism=None,family=None,value=None,n=None):
 key=f'grid_{rep}_{ir}_{sep}_{rate}_{mechanism}' if family is None else f'stress_{rep}_{family}_{value}_{n}'
 dest=ROOT/'checkpoints'/f'{key}.json'
 if dest.exists():return json.loads(dest.read_text())
 if family is None:
  seed=100000+rep+ir*1000+sep*10000;x,true=generate(2000,ir,sep,seed);xt,yt=generate(10000,ir,sep,seed+500000);rng=np.random.default_rng(seed+700000);y=true.copy()
  for c in (0,1):
   ids=np.flatnonzero(true==c);count=int(np.floor(len(ids)*rate));chosen=rng.choice(ids,count,replace=False) if mechanism=='dispersed' else ids[np.argsort(((x[ids]-np.array([sep*c,0]))**2).sum(axis=1),kind='stable')[:count]];y[chosen]=1-y[chosen]
  common=dict(study='grid',rep=rep,ir=ir,separation=sep,noise_rate=rate,mechanism=mechanism)
 else:
  seed=1300000+rep+int(float(value)*1000)+n;x,y,component=stress_generate(n,family,value,seed);true=y.copy();xt,yt,ct=stress_generate(10000,family,value,seed+500000)
  if family=='feature_outliers':
   rng=np.random.default_rng(seed+200000)
   for c in (0,1):
    ids=np.flatnonzero(y==c);chosen=rng.choice(ids,int(float(value)*len(ids)),replace=False);x[chosen]+=rng.normal(0,5,size=(len(chosen),x.shape[1]))
  common=dict(study='stress',rep=rep,family=family,value=value,n=n)
 sc=StandardScaler().fit(x);z=sc.transform(x);zt=sc.transform(xt);rows=[]
 for variant in ('adaptive','fixed'):
  zz,yy,w,info=intervention(z,y,seed+900000,variant);model=LogisticRegression(C=1,max_iter=2000).fit(zz,yy,sample_weight=w);pred=model.predict(zt)
  row=dict(**common,variant=variant,status='ok',synthetic=len(yy)-len(y),**diagnostics(y,true,info,w),**bench.metrics(yt,pred,model.decision_function(zt)))
  if family is not None:
   sg=(yt==1)&(ct==1);row['subgroup_recall']=float(pred[sg].mean()) if sg.any() else None
  rows.append(row)
 dest.write_text(json.dumps(rows));return rows

def benchmark_task(*args):
 original=bench.resample
 def replacement(z,y,method,seed):
  start=time.perf_counter();zz,yy,w,_=intervention(z,y,seed,'fixed');return zz,yy,w,time.perf_counter()-start
 bench.resample=replacement;bench.METHODS=('fixed',);bench.ROOT=ROOT/'benchmark_work'
 # Dataset bytes are original; checkpoint directory is separate.
 try:return bench.fit_task(*args)
 finally:bench.resample=original

def main():
 (ROOT/'checkpoints').mkdir(exist_ok=True); jobs=[dict(rep=r,ir=ir,sep=s,rate=e,mechanism=m) for ir in (5,10,20) for s in (1,2,3) for e,m in [(0.,'clean')]+[(e,m) for e in (.05,.1,.2) for m in ('dispersed','localized')] for r in range(30)]
 settings=[('subgroup',g,n) for g in (.05,.1,.2) for n in (500,1000,2000)]+[('isolation',.05,n) for n in (500,1000,2000)]+[('feature_outliers',v,2000) for v in (.05,.1)]+[('dimensions',d,2000) for d in (2,10,50)]
 jobs += [dict(rep=r,family=f,value=v,n=n) for f,v,n in settings for r in range(30)]
 out=Parallel(n_jobs=int(os.environ.get("EXPERIMENT_JOBS","4")),verbose=10)(delayed(task)(**j) for j in jobs);pd.DataFrame([r for group in out for r in group]).to_csv(ROOT/'synthetic_raw.csv',index=False)
 print('Synthetic completed',flush=True)
 work=ROOT/'benchmark_work';work.mkdir(exist_ok=True);(work/'results_v2').mkdir(exist_ok=True)
 if not (work/'data').exists():
  import shutil
  shutil.copytree(EXP/'data',work/'data')
 jobs=[]
 for item in json.loads((EXP/'data/manifest.json').read_text()):
  name=item['name'];dat=np.load(EXP/'data'/f'{name}.npz');X,Y=dat['X'],dat['y'];groups=np.unique(X,axis=0,return_inverse=True)[1]
  for rep in range(3):
   for fold,(tr,te) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=31000+rep).split(X,Y,groups)):
    for rate in (0.,.1):jobs.append((name,rep,fold,tr,te,rate))
 out=Parallel(n_jobs=int(os.environ.get("EXPERIMENT_JOBS","4")),verbose=10)(delayed(benchmark_task)(*j) for j in jobs);pd.DataFrame([r for group in out for r in group]).to_csv(ROOT/'benchmark_fixed_raw.csv',index=False)
 print('Benchmark completed',flush=True)
if __name__=='__main__':main()
