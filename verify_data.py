from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parent/'expanded/data';expected=json.loads((R/'expected_array_hashes.json').read_text())
for name,value in expected.items():
 p=R/(name+'.npz')
 if not p.exists():raise SystemExit('Run python expanded/download_data.py first.')
 a=np.load(p);h=hashlib.sha256();h.update(a['X'].astype('<f8').tobytes());h.update(a['y'].astype('<i8').tobytes());assert h.hexdigest()==value,name
print('PASS: all six X/y array hashes match the experiment inputs.')
