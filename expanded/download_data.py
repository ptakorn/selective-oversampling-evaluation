from pathlib import Path
import urllib.request,zipfile,io,json,hashlib
import numpy as np,pandas as pd
from sklearn.datasets import load_breast_cancer
root=Path(__file__).parent/'data';root.mkdir(exist_ok=True)
manifest=[]
b=load_breast_cancer(); np.savez_compressed(root/'wdbc.npz',X=b.data,y=(b.target==0).astype(int));manifest.append(dict(name='wdbc',uci_id=17,n=569,d=30,positive='malignant',n1=212,n0=357))
for name,uid,filename in [('banknote',267,'data_banknote_authentication.txt'),('spambase',94,'spambase.data'),('ionosphere',52,'ionosphere.data'),('ecoli_imu',39,'ecoli.data'),('glass7',42,'glass.data')]:
 url=f'https://archive.ics.uci.edu/static/public/{uid}/'+{267:'banknote+authentication',94:'spambase',52:'ionosphere',39:'ecoli',42:'glass+identification'}[uid]+'.zip'
 # Numeric legacy ZIP endpoints are also accepted by UCI.
 try: raw=urllib.request.urlopen(url,timeout=40).read()
 except Exception: raw=urllib.request.urlopen(f'https://archive.ics.uci.edu/static/public/{uid}/{name.split("_")[0]}.zip',timeout=40).read()
 (root/(name+'_source.zip')).write_bytes(raw)
 z=zipfile.ZipFile(io.BytesIO(raw));member=next(k for k in z.namelist() if k.endswith(filename));content=z.read(member)
 frame=pd.read_csv(io.BytesIO(content),header=None,sep=r'\s+' if name=='ecoli_imu' else ',')
 if name=='ecoli_imu': X=frame.iloc[:,[1,2,5,6,7]].to_numpy(float);y=(frame.iloc[:,-1]=='imU').to_numpy(int);pos='imU vs rest; binary lip/chg removed'
 elif name=='glass7': X=frame.iloc[:,1:-1].to_numpy(float);y=(frame.iloc[:,-1]==7).to_numpy(int);pos='class 7 (headlamps) vs rest; ID removed'
 elif name=='ionosphere': X=frame.iloc[:,2:-1].to_numpy(float);y=(frame.iloc[:,-1]=='b').to_numpy(int);pos='bad radar return; first binary and second constant features removed'
 else: X=frame.iloc[:,:-1].to_numpy(float);y=frame.iloc[:,-1].to_numpy(int);pos='class 1' if name=='banknote' else 'spam'
 assert np.isfinite(X).all() and y.sum()<len(y)/2
 np.savez_compressed(root/(name+'.npz'),X=X,y=y)
 manifest.append(dict(name=name,uci_id=uid,n=len(y),d=X.shape[1],positive=pos,n1=int(y.sum()),n0=int((y==0).sum()),sha256=hashlib.sha256(raw).hexdigest(),url=url))
 print(name,len(y),X.shape[1],np.bincount(y),flush=True)
for m in manifest:m['ir']=m['n0']/m['n1']
(root/'manifest.json').write_text(json.dumps(manifest,indent=2));pd.DataFrame(manifest).to_csv(root/'manifest.csv',index=False)
