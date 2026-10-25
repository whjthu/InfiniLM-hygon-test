import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
ops=r/'candidates/rebased-5f91ad7-r3668/Ops/Kimi/InfiniOps'
out={'CPU_only':True,'files':[]}
for rel in ['src/native/cuda/hygon/ops/rms_norm','src/native/cuda/ops/rms_norm','src/ops/rms_norm']:
 b=ops/rel
 if b.exists():
  for p in b.rglob('*'):
   if p.is_file() and p.suffix in ['.h','.hpp','.cuh','.cu','.cpp'] and p.stat().st_size<50000:out['files'].append({'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':p.read_text()})
print(json.dumps(out))
