import pathlib,json,hashlib,subprocess
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');out={'files':{},'receipts':{},'no_GPU_or_dlopen':True}
def take(rel,text=False,limit=180000):
 p=r/rel
 if not p.is_file():out['files'][rel]={'exists':False};return
 assert p.resolve().is_relative_to(r.resolve());raw=p.read_bytes();q={'exists':True,'SHA256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 if text:
  assert len(raw)<=limit,(rel,len(raw));q['text']=raw.decode('utf8')
 out['files'][rel]=q
for rel in ['candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so','candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_cache_r807.so','install/hygon-glm-5f91ad7-r3672/lib/libinfiniops.so','install/hygon-glm-5f91ad7-r3672/lib/libinfinirt.so','candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so']:
 take(rel)
for rel in ['candidates/glm-grouped-native-model-r5117/build.json','results/model-latest-r2856-GLM-build.json','results/glm-fp8-batched-r776-build.json','candidates/glm-fp8-batched-r776/build.json','candidates/glm-deferred-attention-model-5f91ad7-r5096/objects-manifest.json','candidates/GLM-down-row2-Model-r9284/CPU-build.json']:
 p=r/rel
 if p.is_file():
  raw=p.read_bytes();q=json.loads(raw);out['receipts'][rel]={'SHA256':hashlib.sha256(raw).hexdigest(),'fields':{k:v for k,v in q.items() if k in ['completed','library','library_sha256','library_SHA256','fresh_helpers','fresh_helpers_sources','helper_sha256','cache_library_sha256','sources_sha256','source_sha256','flags','commands','args','compiled','entries','source_SHA256','helper_sources_sha256','SDK_libraries_SHA256','reused_objects_SHA256']}}
 else:out['receipts'][rel]={'exists':False}
for folder in ['candidates/glm-grouped-native-model-r5117/helpers','candidates/glm-grouped-native-model-r5117/src','incoming','candidates/glm-fp8-batched-r776']:
 for name in ['glm_fp8_batched_r776.cu','glm_fp8_cache_r807.hpp']:
  p=r/folder/name
  if p.is_file():take(str(p.relative_to(r)),True)
common='candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniLM/csrc/'
for name in ['runtime/runtime.hpp','runtime/tensor.hpp','runtime/device.hpp','execution/submit.hpp']:
 take(common+name,True)
sdk='install/hygon-glm-5f91ad7-r3672/include/'
for name in ['infini/ops/fp8_block_linear.h','infini/ops/fp8_block_linear.hpp','infini/ops.h','infini/ops.hpp','infini/rt/device_guard.h']:
 take(sdk+name,True)
# Inventory only fp8 operator public headers in the exact active SDK include tree.
inc=r/sdk
for p in inc.rglob('*fp8*'):
 if p.is_file() and p.suffix in ['.h','.hpp']:take(str(p.relative_to(r)),True)
for name in ['adapters/InfiniOps/src/native/cuda/hygon/ops/fp8_block_linear/kernel.h','candidates/rebased-5f91ad7-r3668/Common/GLM/InfiniCore/src/ops/fp8_block_linear/hygon/fp8_block_linear_hygon.cc']:
 take(name,True)
lib=r/'candidates/glm-grouped-native-model-r5117/helpers/libglm_fp8_batched_r776.so'
out['776_exports']=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True,timeout=15).splitlines()
print(json.dumps(out,separators=(',',':')))
