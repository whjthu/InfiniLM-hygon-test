import ast,hashlib,json
from pathlib import Path
root=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-minicpm-hygon-processor-r2/cpu/runtime-site/transformers')
paths={'qwen':'models/qwen3_vl/modeling_qwen3_vl.py','rope':'modeling_rope_utils.py'}
out={}
for k,n in paths.items():
 p=root/n
 raw=p.read_bytes()
 assert len(raw)<=262144
 sha=hashlib.sha256(raw).hexdigest()
 if k=='qwen':assert sha=='f39b3621fea3f757a01d2e00ebd3a2be3516b7df3048d862a365988aba9fa994'
 tree=ast.parse(raw)
 funcs=[]
 for parent in tree.body:
  methods=[parent] if isinstance(parent,ast.FunctionDef) else parent.body if isinstance(parent,ast.ClassDef) else []
  for f in methods:
   if not isinstance(f,ast.FunctionDef) or f.name in ('__init__','__repr__'):continue
   matches=[]
   for n in ast.walk(f):
    if isinstance(n,ast.Attribute) and isinstance(n.ctx,(ast.Store,ast.Del)) and isinstance(n.value,ast.Name) and n.value.id=='self':
     matches.append((n.lineno,n.attr))
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='self' and n.func.attr in ('register_buffer','update','_update_causal_mask'):
     matches.append((n.lineno,n.func.attr))
   if matches:funcs.append(dict(class_name=parent.name if isinstance(parent,ast.ClassDef) else None,method=f.name,locations=sorted(set(matches))))
 out[k]=dict(path=paths[k],SHA256=sha,bytes=len(raw),self_assignments_and_buffer_writes=funcs)
print(json.dumps(out))
