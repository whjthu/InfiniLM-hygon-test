from pathlib import Path
import ast,hashlib,json,os,shlex,subprocess,sys,time,traceback
C=Path(__file__).resolve().parent;D=C.parent/'split-v128-source-v1';B=Path('/refactor/candidates/Kimi-split-V128-CPU-v2')
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').split(':',1)[1]))
def main():
 assert not B.exists(),'existing candidate, reconcile neverrepeat';B.mkdir();start=time.monotonic()
 q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Model=False,Native_dlopen=False,SDK_device_execution=False,SDK_metadata_header_only=True,new_HOST_TUs=4,new_GPU_TUs=1,pure_CPU_TUs=2,commands=[],logs={})
 def save():(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(args,label,cap=90):
  left=270-(time.monotonic()-start);assert left>0;q['phase']=label;q['commands'].append(args);save()
  try:z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=min(left,cap))
  except subprocess.TimeoutExpired as e:
   text=b''.join(v.encode()if isinstance(v,str)else(v or b'')for v in [e.stdout,e.stderr]);(B/(label+'.log')).write_bytes(text);raise
  p=B/(label+'.log');p.write_text(z.stdout+z.stderr);q['logs'][label]=dict(rc=z.returncode,bytes=p.stat().st_size,SHA256=H(p));save()
  if z.returncode:raise RuntimeError(label+' failed; original log retained')
  return z.stdout
 try:
  assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  assert H(C/'SOURCE_MANIFEST.json')==os.environ['KIMI_SPLITV_REPAIR_SHA256']
  repair=json.loads((C/'SOURCE_MANIFEST.json').read_bytes())
  for p,h in repair['files_SHA256'].items():assert H(C/p)==h,p
  assert H(D/'SOURCE_MANIFEST.json')==repair['production_source_manifest_SHA256']==os.environ['KIMI_SPLITV_SOURCE_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
  for p,h in m['files_SHA256'].items():assert H(D/p)==h,p
  q.update(actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),CPU_repair_manifest_SHA256=H(C/'SOURCE_MANIFEST.json'))
  c=json.loads((D/'BUILD_CONTRACT.json').read_bytes());pin=c['parent_CPU622'];assert H(pin['path'])==pin['SHA256'];base=json.loads(Path(pin['path']).read_bytes())
  assert base['passed']and base['native_SHA256']==c['Native_b8_SHA256']==H(c['Native_b8'])
  for p,h in base['all_linked_object_SHA256'].items():assert H(p)==h,p
  for v in base['actual_dependency_library_SHA256'].values():assert H(v['path'])==v['SHA256']
  q['actual_dependency_library_SHA256']=base['actual_dependency_library_SHA256'];q['parent_CPU622']=pin
  from apply_caller import prepare
  q['actual_generated_sources']=prepare(D/'baseline-inputs/allocate.cpp',D/'baseline-inputs/mla.cpp',D/'baseline-inputs/pair.cpp',B/'native');assert all(v['private_SHA256']==H(D/'draft-native'/(tag+'.cpp'))for tag,v in q['actual_generated_sources'].items());save()
  body=(D/'cache.hpp').read_text();assert 'Tensor::empty_parameter'in body and body.count('inline infinilm::runtime::Tensor allocate')==1
  lines=[l for l in body.splitlines()if not l.startswith(('#include','#pragma'))]
  recording=B/'record_cache_body.hpp';recording.write_text('#include "record_cache_prefix.hpp"\n'+'\n'.join(lines)+'\n')
  q['production_cache_body_SHA256']=H(D/'cache.hpp')
  sdk='/refactor/install/hygon-kimi-5f91ad7-r3672/include'
  for name in ['CPU_contract','CPU_cache']:
   planning_includes=[v[2:]for v in c['HOST_prefix']if v.startswith('-I')]
   assert '/opt/dtk/cuda/cuda/include'in planning_includes and sdk in planning_includes
   q['planning_includes_from_verified_HOST_prefix']=planning_includes
   args=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-I'+str(D),'-I'+str(B)]+[a for inc in planning_includes for a in ['-isystem',inc]]+[str(D/(name+'.cpp')),'-o',str(B/name)]
   run(args,name+'-compile',25);dyn=run(['readelf','-d',str(B/name)],name+'-ELF',10)
   assert not any(s in dyn for s in ['libcuda','libinfinirt','libinfiniops','libtorch'])
   value=json.loads(run([str(B/name)],name+'-consumer',30));assert value['passed'];q[name]=value
  q['binding_recording']=json.loads(run([sys.executable,'-B',str(D/'check_binding.py')],'new4arg-binding-recording',10));assert q['binding_recording']['passed']
  units=[('allocate',B/'native/allocate.cpp'),('mla',B/'native/mla.cpp'),('pair',B/'native/pair.cpp'),('fa_backend',D/'fa_backend.cpp'),('fa_kernels',D/'fa_kernels.cu')];objects={};detail={}
  for name,source in units:
   prefix=(c['GPU_prefix']if name=='fa_kernels'else c['HOST_prefix'])+['-I'+str(D)]
   # Private headers first; parent flags/ABI unchanged.
   prefix=[prefix[0],'-I'+str(D),*prefix[1:]]
   before=B/(name+'.before.d');after=B/(name+'.d');obj=B/(name+'.o')
   run(prefix+['-M','-MF',str(before),'-MT',str(obj),str(source)],name+'-before-MD',25);snapshot={p:H(p)for p in sorted(deps(before))}
   run(prefix+['-MD','-MF',str(after),'-c',str(source),'-o',str(obj)],name+'-compile',85)
   assert deps(after)==set(snapshot)and all(H(p)==h for p,h in snapshot.items())
   objects[name]=str(obj);detail[name]=dict(source=str(source),source_SHA256=H(source),object=str(obj),object_SHA256=H(obj),all_MD=snapshot,MD_SHA256=H(after));q['compiled']=detail;save()
  link=c['baseline_link_argv'][:];old={p for p in link if p.endswith('.o')};new=set(base['all_linked_object_SHA256']);removed=old-new;added=new-old
  assert len(removed)==len(added)==1 and next(iter(removed)).endswith('/worker.o')and next(iter(added)).endswith('/worker.o')
  link=[next(iter(added))if p in removed else p for p in link]
  replacements={v:objects[k]for k,v in c['replacement_objects'].items()};assert set(replacements)<=set(link)
  link=[replacements.get(p,p)for p in link];link[link.index('-o')+1]=str(B/'libKimi_split_v128_private.so')
  link=[('-Wl,-Map,'+str(B/'link.map'))if p.startswith('-Wl,-Map,')else p for p in link];link[link.index('-o'):link.index('-o')]=[objects['fa_backend'],objects['fa_kernels']]
  run(link,'private-Native-link',45);lib=B/'libKimi_split_v128_private.so'
  reused={p:h for p,h in base['all_linked_object_SHA256'].items()if p not in replacements};assert len(reused)==31 and all(p in link for p in reused)
  def strong(path,label):return{l.split()[-1]for l in run(['nm','-D','--defined-only',str(path)],label,15).splitlines()if len(l.split())>=3 and l.split()[-2]in ['T','D','B','R']}
  old=strong(c['Native_b8'],'parent-symbols');now=strong(lib,'new-symbols');assert old<=now
  required={'modern_kimi_engine_create_k192_v128_private','modern_kimi_k192_v128_capabilities_private'};assert required<=now
  dynamic=run(['readelf','-d',str(lib)],'new-ELF',10);assert 'SYMBOLIC'in dynamic
  run(['ldd','-r',str(lib)],'relocation-trace-no-entry',25);ldd=(B/'relocation-trace-no-entry.log').read_text();assert 'not found'not in ldd and 'undefined symbol'not in ldd
  q['project_maps']=[line for line in Path('/proc/self/maps').read_text().splitlines()if any(x in line for x in ['libKimi','libinfinilm','libinfiniops','libcudart'])];assert not q['project_maps'];q.update(passed=True,native=str(lib),native_SHA256=H(lib),new_CAPI=sorted(required),all_linked_object_SHA256={p:H(p)for p in link if p.endswith('.o')},reused_objects_SHA256=reused,old_strong_exports_retained=len(old),old_CAPI_retained=len([s for s in old if s.startswith('modern_')]),actual_Class_Context_fields_changed=False,qualification='only source/pureCPU metadata+LSE algebra and recording/candidate compile/link; no actual FA operator/weights/Runtime/device/fit/speed qualification')
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:q.update(completed=True,elapsed_s=time.monotonic()-start,Torch_imported='torch'in sys.modules);save();print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','phase','error','elapsed_s','native_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
