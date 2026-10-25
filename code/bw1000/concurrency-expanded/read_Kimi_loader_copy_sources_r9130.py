import pathlib,json,hashlib
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');base=r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/csrc/runtime';q={'sources':{},'cgroup_reports':{}}
for p in base.rglob('*'):
 if p.is_file()and p.suffix in ['.cpp','.cc','.hpp']and p.stat().st_size<2000000:
  t=p.read_text()
  if 'modern_cpu_copy_stats_r2515'in t or 'void TensorImpl::copy_from'in t:q['sources'][str(p)]={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'text':t}
for n in range(4):
 p=r/'results'/f'Kimi-batch-Model-cgroup-r9122b-batch-r9122b2-222b154b669f9165-node{n}.json'
 if p.exists():q['cgroup_reports'][str(p)]={'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_text())}
print(json.dumps(q))
