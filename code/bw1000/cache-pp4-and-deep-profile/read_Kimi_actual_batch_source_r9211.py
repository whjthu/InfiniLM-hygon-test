import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');p=r/'candidates/Kimi-speculative-batch-Model-r9122c/CPU-build.json';d=json.loads(p.read_bytes());paths=set()
def walk(v):
 if isinstance(v,str)and v.startswith('/refactor/')and v.endswith(('.cpp','.hpp','.h','.cuh')):paths.add(v)
 elif isinstance(v,dict):
  for k,x in v.items():walk(k);walk(x)
 elif isinstance(v,list):
  for x in v:walk(x)
walk(d);out={'receipt_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'native_SHA256':d.get('native_SHA256'),'keys':list(d),'sources':[]};total=0
for v in sorted(paths):
 q=r/pathlib.PurePosixPath(v).relative_to('/refactor')
 if q.is_file()and q.stat().st_size<650000:
  total+=q.stat().st_size;assert total<2200000;b=q.read_bytes();out['sources'].append({'container_path':v,'SHA256':hashlib.sha256(b).hexdigest(),'text':b.decode()})
print(json.dumps(out))
