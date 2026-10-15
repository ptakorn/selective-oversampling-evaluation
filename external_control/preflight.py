import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import numpy as np,json,hashlib,inspect,sys
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from protocol import fitting,rng,GEO,KINDS
from resampling import prepare
p=Path(__file__).parent;rows=[]
for g in range(3):
 for kind in KINDS:
  x,y,_=fitting(g,0,kind);z=StandardScaler().fit_transform(x);seed=int(rng(g,0,20).integers(0,2**31-1))
  v,a=prepare(z,y,seed)
  plain=SMOTEENN(smote=SMOTE(sampling_strategy=1.,k_neighbors=5,random_state=seed),enn=EditedNearestNeighbours(sampling_strategy='all',n_neighbors=3,kind_sel='all',n_jobs=1),random_state=seed)
  zz,yy=plain.fit_resample(z,y)
  assert np.array_equal(zz,v['smoteenn'][0]) and np.array_equal(yy,v['smoteenn'][1])
  rows.append(dict(geometry=GEO[g],corruption=kind,status='passed',original_retained=len(a['original_keep']),synthetic_retained=a['synthetic_retained'],plain_implementation_exact=True))
(p/'preflight.json').write_text(json.dumps({'scope':'Fitting-only structural checks; no validation/test performance inspected','checks':rows},indent=2));print('9 fitting-only preflight checks passed')
