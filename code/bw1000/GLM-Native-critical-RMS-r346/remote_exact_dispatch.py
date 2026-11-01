import pathlib,json,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';cp=r/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json';assert hashlib.sha256(cp.read_bytes()).hexdigest()=='6aa62f788aad6080027d665f30d7c9824c07cb1866de905397fef15c56b26529';q=json.loads(cp.read_bytes());out={'CPU_only':True,'GPU_initialized':False,'CPU_receipt_SHA256':hashlib.sha256(cp.read_bytes()).hexdigest(),'files':{}}
for field in ['parent_actual_dependency_SHA256','dependency_SHA256']:
 for name,pin in q[field].items():
  if not any(s in name for s in ['submit.hpp','/operator.h','/dispatcher.h','rmsnorm.hpp','/base/rms_norm.h','/base/fused_add_rms_norm.h','runtime/runtime.hpp']):continue
  p=pathlib.Path(name.replace('/refactor',str(r),1));raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();assert h==pin,(name,h,pin);out['files'][name]={'SHA256':h,'field':field,'text':raw.decode(errors='replace')}
for rel in ['candidates/glm-transient-layer-r5681','candidates/glm-vector-model-r8542','candidates/GLM-B32-Model-r9183']:
 p=r/rel;out[rel]=[x.name for x in p.iterdir() if x.suffix in ['.d','.json']][:40]
print(json.dumps(out))
