import pathlib,json,hashlib,re,datetime
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
b=r/'build/ops-kimi-5f91ad7-r3672'
s=r/'candidates/rebased-5f91ad7-r3668/Ops/Kimi/InfiniOps'
sdk=r/'install/hygon-kimi-5f91ad7-r3672'
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'no_compile_dlopen_Runtime_Slurm_weights':True,'files':{},'generated_TUs':[],'mapping':[]}
budget=0
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(1048576),b''):h.update(z)
 return h.hexdigest()
def read(p,full=True,limit=1500000):
 global budget
 p=pathlib.Path(p)
 if not p.exists():q['files'][str(p)]={'exists':False};return None
 n=p.stat().st_size
 a={'bytes':n,'SHA256':digest(p)};q['files'][str(p)]=a
 if not full:return a
 if n>limit:a['text_omitted']='bounded_limit';return None
 budget+=n
 if budget>10000000:raise RuntimeError('bounded read budget')
 a['text']=p.read_text(errors='replace');return a['text']
read(sdk/'lib/libinfiniops.so',False)
for rel in ['results/latest-sdk-r3672-Kimi-build.json','results/latest-sdk-r3672-Kimi-configure.json']:
 p=r/rel
 if p.exists():
  raw=read(p);d=json.loads(raw);q['files'][str(p)]['text']=json.dumps({k:v for k,v in d.items() if k in ['completed','library_sha256','configure_sha256','source','build','install','pins','configuration']})
for p in [b/'CMakeCache.txt',b/'src/CMakeFiles/infiniops.dir/link.txt',b/'src/CMakeFiles/infiniops.dir/DependInfo.cmake',b/'src/CMakeFiles/infiniops.dir/flags.make',b/'src/CMakeFiles/infiniops.dir/build.make',b/'build.ninja',b/'compile_commands.json']:
 text=read(p,limit=5000000)
 if text:
  ls=text.splitlines();hits=[i for i,x in enumerate(ls) if any(t in x for t in ['operator_call_instantiations','fused_moe_mxfp4','libinfiniops.so','WITH_HYGON:','WITH_TORCH:','CMAKE_HOME_DIRECTORY:'])]
  indices=sorted(set(j for i in hits for j in range(max(0,i-1),min(len(ls),i+3))))
  q['files'][str(p)]['text']='\n'.join(ls[j] for j in indices)[:45000]
for p in sorted((s/'generated/src').glob('operator_call_instantiations_*.cc')):
 if p.stat().st_size>1500000:continue
 text=p.read_text(errors='replace')
 if 'FusedMoeMxfp4' not in text:continue
 read(p);q['generated_TUs'].append(str(p))
 # Exact CMake object path of this named generated source; no recursive inventory.
 for obj in [b/'src/CMakeFiles/infiniops.dir/__/generated/src'/(p.name+'.o'),b/'src/CMakeFiles/infiniops.dir/__/generated/src'/(p.name+'.o.d'),b/'src/CMakeFiles/infiniops.dir/__/generated/src'/(p.name+'.o.d.internal')]:
  read(obj,full=not obj.name.endswith('.o'),limit=1500000)
  if obj.exists() and obj.name.endswith('.o'):q['mapping'].append({'source':str(p),'object':str(obj),'source_SHA256':digest(p),'object_SHA256':digest(obj)})
for rel in ['src/base/fused_moe_mxfp4.h','src/base/moe_compute_common.h','src/handle.h','src/operator.h','src/native/cuda/hygon/ops/fused_moe_mxfp4/kernel.h','src/native/cuda/hygon/ops/fused_moe_mxfp4/kimi_mxfp4_dispatch_r330.cuh','src/native/cuda/ops/fused_moe_mxfp4/kernel.cuh']:
 read(s/rel)
for rel in ['include/base/fused_moe_mxfp4.h','include/handle.h','include/operator.h']:
 read(sdk/rel)
manifest=r/'candidates/kimi-host-object-cache-5f91ad7-r4462/manifest.json'
if manifest.exists():
 q['cache_manifest_SHA256']=digest(manifest);data=json.loads(manifest.read_text());q['grouped_entries']=[]
 for entry in data.get('entries',[]):
  if 'grouped' in str(entry.get('source','')).lower():
   q['grouped_entries'].append(entry)
   # Only helper sources actually named by its build receipt; never model payloads.
   name=entry.get('source','')
   if name.startswith('@LM/'):
    p=r/'candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM'/name[4:]
    read(p)
read(r/'candidates/rebased-5f91ad7-r3668/Model/Kimi/InfiniLM/csrc/execution/submit.hpp')
for rel in ['csrc/runtime/runtime.cc','csrc/runtime/memory.cc']:
 read(r/'candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM'/rel)
q['text_bytes_read']=budget
print(json.dumps(q,ensure_ascii=True))
