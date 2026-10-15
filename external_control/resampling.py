import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours

class TracedSMOTE(SMOTE):
 """Observe public implementation's generation indices without changing RNG or coordinates."""
 def _generate_samples(self,X,nn_data,nn_num,rows,cols,steps,y_type=None,y=None):
  self.endpoint_local_=np.c_[rows,nn_num[rows,cols]].copy()
  return super()._generate_samples(X,nn_data,nn_num,rows,cols,steps,y_type,y)

def prepare(z,y,seed):
 n=len(y)
 sm=TracedSMOTE(sampling_strategy=1.0,k_neighbors=5,random_state=seed)
 enn=EditedNearestNeighbours(sampling_strategy='all',n_neighbors=3,kind_sel='all',n_jobs=1)
 method=SMOTEENN(smote=sm,enn=enn,random_state=seed)
 ze,ye=method.fit_resample(z,y)
 # Recreate SMOTE stage for provenance; identical seeded generation, no new decisions.
 zs,ys=method.smote_.fit_resample(z,y)
 ids=np.asarray(method.enn_.sample_indices_,int)
 assert np.array_equal(ze,zs[ids]) and np.array_equal(ye,ys[ids])
 keep=ids[ids<n];synids=ids[ids>=n]-n;ns=len(synids)
 assert len(np.unique(ids))==len(ids)
 if ns:
  matched=SMOTE(sampling_strategy={1:int((y==1).sum()+ns)},k_neighbors=5,random_state=seed)
  zm,ym=matched.fit_resample(z,y)
 else:zm,ym=z.copy(),y.copy()
 zr=np.vstack([z[keep],zm[n:]]);yr=np.r_[y[keep],ym[n:]]
 assert len(zm)-n==ns and len(zr)==len(ze)
 assert np.array_equal(yr,ye) # ENN class ordering: all originals sorted by class, then synthetics in class 1
 assert np.isfinite(ze).all() and np.isfinite(zr).all()
 assert set(np.unique(ye))=={0,1}
 minor=np.flatnonzero(y==1);endpoints=minor[method.smote_.endpoint_local_]
 variants={'smoteenn':(ze,ye),'smote_matched_count':(zm,ym),'smote_matched_retention':(zr,yr)}
 return variants,dict(original_keep=keep,synthetic_keep=synids,endpoints=endpoints,full_synthetic=len(zs)-n,synthetic_retained=ns)
