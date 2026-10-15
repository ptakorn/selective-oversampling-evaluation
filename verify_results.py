from pathlib import Path
import zipfile,io
import numpy as np,pandas as pd
R=Path(__file__).resolve().parent
expected={'pilot/pilot_raw.csv':270,'ablation/ablation_raw.csv':450,'expanded/synthetic_grid_raw.csv':11340,'expanded/stress_raw.csv':3060,'expanded/benchmark_raw.csv':4320,'expanded/threshold_control_raw.csv':720,'neighborhood_ablation/synthetic_raw.csv':4800,'neighborhood_ablation/benchmark_fixed_raw.csv':540,'benchmark_threshold/raw.csv':3240}
for path,count in expected.items():
 df=pd.read_csv(R/path);assert len(df)==count,(path,len(df),count)
 if 'status' in df and path!='expanded/benchmark_raw.csv':assert (df.status=='ok').all(),path
old=pd.read_csv(R/'expanded/benchmark_raw.csv');assert int((old.status=='ok').sum())==4107
raw=pd.read_csv(R/'benchmark_threshold/raw.csv',float_precision='round_trip');keys=['dataset','rep','fold','noise_rate','method','classifier'];check=raw[raw.rule=='default'].merge(old,on=keys,suffixes=('_new','_old'));assert len(check)==1620 and np.max(abs(check.gmean_new-check.gmean_old))<1e-12
p=raw.pivot(index=keys,columns='rule',values='auc');assert (p.default==p.selected).all()
for study,fname,join in [('grid','synthetic_grid_raw.csv',['rep','ir','separation','noise_rate','mechanism']),('stress','stress_raw.csv',['rep','family','value','n'])]:
 a=pd.read_csv(R/'neighborhood_ablation/synthetic_raw.csv');a=a[(a.study==study)&(a.variant=='adaptive')];b=pd.read_csv(R/'expanded'/fname);b=b[b.method=='proposed'];m=a.merge(b,on=join,suffixes=('_new','_old'));assert len(m)==len(b) and np.max(abs(m.gmean_new-m.gmean_old))<1e-12
count=0
with zipfile.ZipFile(R/'benchmark_threshold/audit_scores.zip') as z:
 for member in z.namelist():
  name,task=member.split('_r');rep=int(task.split('_f')[0]);fold=int(task.split('_f')[1].split('_noise')[0]);rate=float(task.split('_noise')[1].removesuffix('.npz'));a=np.load(io.BytesIO(z.read(member)));y=a['test_labels'];subset=raw[(raw.dataset==name)&(raw.rep==rep)&(raw.fold==fold)&(raw.noise_rate==rate)]
  for _,row in subset.iterrows():
   prefix=row.method+'_'+row.classifier;s=a[prefix+'_test_scores'];threshold=row.threshold if row.rule=='selected' else (.5 if row.classifier=='rf' else 0);pred=s>=threshold if row.rule=='selected' else s>threshold;gm=np.sqrt(pred[y==1].mean()*(~pred[y==0]).mean());assert abs(gm-row.gmean)<1e-12,(member,row.method,row.classifier,row.rule)
   if row.rule=='selected':
    ys=a[prefix+'_oof_labels'];ss=a[prefix+'_oof_scores'];vf=a[prefix+'_oof_fold'];gs=[]
    for v in range(3):
     q=vf==v;pp=ss[q]>=threshold;yy=ys[q];gs.append(np.sqrt(pp[yy==1].mean()*(~pp[yy==0]).mean()))
    assert abs(np.mean(gs)-row.inner_threshold_gmean)<1e-12
   count+=1
assert count==3240
print('PASS: row counts, original failures, 1620 default reproductions, 2400 adaptive reproductions, 3240 audited G-means and selected validation objectives.')
