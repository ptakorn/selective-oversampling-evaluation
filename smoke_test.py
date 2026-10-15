from pathlib import Path
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'expanded'));from methods import PARAMS,proposed
sys.path.insert(0,str(R/'neighborhood_ablation'));from ablation_methods import proposed as variant
rng=np.random.default_rng(20261014);x=rng.normal(size=(80,2));y=np.r_[np.zeros(60,int),np.ones(20,int)];x[y==1,0]+=2;z=StandardScaler().fit_transform(x)
a=proposed(z,y,PARAMS,123);b=variant(z,y,PARAMS,123,neighborhood='adaptive');assert all(np.array_equal(a[i],b[i]) for i in range(3))
f=variant(z,y,PARAMS,123,neighborhood='fixed');assert np.array_equal(f[3]['sizes'],np.repeat(np.array([[5],[10],[20]]),len(y),axis=1));assert np.array_equal(f[0][:80],z) and np.array_equal(f[1][:80],y) and np.all(f[2]>0)
from run_benchmarks import resample
out=resample(z,y,'smote',123);assert len(out[1])==120
print('PASS: adaptive equality, fixed counts, preserved originals, positive weights, SMOTE availability. WRND is not required.')
