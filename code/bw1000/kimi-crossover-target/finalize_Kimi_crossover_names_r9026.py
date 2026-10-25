from pathlib import Path
import json,hashlib,ast
r=Path(__file__).resolve().parent;b=r/'Kimi-crossover-Model-private-r9026'
rename={'modern_Kimi_crossover_Model_r9008.py':'modern_Kimi_crossover_Model_r9026.py','kimi_crossover_r9008.py':'kimi_crossover_r9026.py','Kimi_crossover_binding_r9008.py':'Kimi_crossover_binding_r9026.py'}
for old,new in rename.items():
 p=b/old;s=p.read_text()
 if old.startswith('modern_'):
  s=s.replace('Kimi-crossover-manifest-r9008.json','Kimi-crossover-manifest-r9026.json').replace('from kimi_crossover_r9008 import','from kimi_crossover_r9026 import').replace('from Kimi_crossover_binding_r9008 import','from Kimi_crossover_binding_r9026 import').replace("f'kimi-crossover-r9008-","f'kimi-crossover-r9026-")
 (b/new).write_bytes(s.encode());ast.parse(s)
for name in ['Kimi_crossover_Model_r9026_inside.sh','launch_Kimi_crossover_Model_r9026.py','audit_Kimi_crossover_Model_r9026.py']:
 p=b/name;s=p.read_text().replace('modern_Kimi_crossover_Model_r9008.py','modern_Kimi_crossover_Model_r9026.py').replace('Kimi-crossover-manifest-r9008.json','Kimi-crossover-manifest-r9026.json').replace("f'kimi-crossover-r9008-","f'kimi-crossover-r9026-");p.write_bytes(s.encode())
 if name.endswith('.py'):ast.parse(s)
m=json.loads((b/'Kimi-crossover-manifest-r9008.json').read_bytes());m['runtime_files']=[rename.get(n,n) for n in m['runtime_files']];m['source_sha256']={n:hashlib.sha256((b/n).read_bytes()).hexdigest() for n in m['runtime_files']};m['preparation_erratum']='Initial stage aborted before launch due shared incoming binding basename conflict; preserve prior file, use unique r9026 Python modules/manifest. No Model or GPU submitted.'
(b/'Kimi-crossover-manifest-r9026.json').write_bytes((json.dumps(m,indent=2)+'\n').encode())
stage=r/'stage_Kimi_crossover_Model_r9026.py';s=stage.read_text().replace('Kimi-crossover-manifest-r9008.json','Kimi-crossover-manifest-r9026.json');stage.write_bytes(s.encode())
print('Unique r9026 Python modules and manifest sealed; native ABI r9008 unchanged')
