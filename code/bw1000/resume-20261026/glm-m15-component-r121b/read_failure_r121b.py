import pathlib,json,hashlib,re,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');b=r/'candidates/GLM-M15-component-r121';q={'files':{},'known_dependencies':{},'CPU_only':True,'GPU':0,'dlopen':0}
def meta(p,text=False):
 if not p.is_file():return {'exists':False}
 raw=p.read_bytes();v={'exists':True,'SHA256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'resolved':str(p.resolve())}
 if text:assert len(raw)<300000;v['text']=raw.decode('utf8')
 return v
for name in ['ldd.log','dynamic.log','exports.log','mockrun.log','CPU-build.json','glm_fp8_cache_m15_r119.cpp.o','glm_m15_component_r121.cpp.o','glm_fp8_cache_m15_r119.cpp.d','glm_m15_component_r121.cpp.d','libGLM_M15_component_r121.so']:
 q['files'][name]=meta(b/name,text=name.endswith(('.log','.json','.d')))
q['failed_receipt']=json.loads(q['files']['CPU-build.json']['text'])
missing=re.findall(r'^\s*(\S+)\s+=>\s+not found',q['files']['ldd.log']['text'],re.M);q['missing']=missing
known=['candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python','candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib','install/hygon-glm-5f91ad7-r3672/lib','candidates/glm-grouped-native-model-r5117/helpers','containers/dtk26-sandbox-r1/usr/local/lib/python3.10/dist-packages/torch/lib','containers/dtk26-sandbox-r1/opt/dtk/cuda/cuda/lib64','containers/dtk26-sandbox-r1/opt/dtk/cuda/lib64']
for folder in known:
 for name in missing:
  assert '/' not in name and name.startswith('lib')
  p=r/folder/name
  if p.is_file():q['known_dependencies'][str(p.relative_to(r))]=meta(p)
for rel in ['install/hygon-glm-5f91ad7-r3672/lib/libinfiniops.so','install/hygon-glm-5f91ad7-r3672/lib/libinfinirt.so','candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so']:
 q['known_dependencies'][rel]=meta(r/rel)
q['nm_fresh']=subprocess.check_output(['nm','-D','--defined-only',str(b/'libGLM_M15_component_r121.so')],text=True,timeout=15)
q['dynamic_fresh']=subprocess.check_output(['readelf','-d',str(b/'libGLM_M15_component_r121.so')],text=True,timeout=15)
print(json.dumps(q,separators=(',',':')))
