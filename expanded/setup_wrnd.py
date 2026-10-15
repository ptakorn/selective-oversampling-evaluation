"""Fetch the authors' code; do not redistribute their implementation in this bundle."""
from pathlib import Path
import subprocess,ast
ROOT=Path(__file__).parent
repo=ROOT/'vendor/WRND';repo.parent.mkdir(exist_ok=True)
if not repo.exists():subprocess.run(['git','clone','https://github.com/dream-lm/WRND_framework.git',str(repo)],check=True)
subprocess.run(['git','-C',str(repo),'checkout','f69a5c71fdff7dd3ff9b52c9d96801c3a931cc99'],check=True)
src=(repo/'all_smote_v5.py').read_text();node=next(n for n in ast.parse(src).body if isinstance(n,ast.FunctionDef) and n.name=='generate_samples_Lm')
(ROOT/'wrnd_generation.py').write_text('import numpy as np\nimport math,random\nfrom sklearn.preprocessing import scale\n'+ast.get_source_segment(src,node)+'\n')
print('Retrieved pinned author code and generated compatibility import')
