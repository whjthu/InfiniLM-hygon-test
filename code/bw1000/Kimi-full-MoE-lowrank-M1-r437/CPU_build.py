"""Two necessary HOST TUs; original33objects + compiled432GPU reused. No dlopen."""
from pathlib import Path
import datetime,hashlib,json,os,shlex,subprocess,sys,sysconfig,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();started=time.monotonic()
q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Native_dlopen=False,weights=0,Runtime_created=False,new_HOST_TUs=2,new_GPU_TUs=0,old_objects_recompiled=False,commands=[],compiled=[])
def run(args,label,cap=90):
 left=260-(time.monotonic()-started);assert left>0,'437 original CPU260s inside bound'
 z=subprocess.run(args,capture_output=True,text=True,timeout=min(cap,left));p=B/(label+'.log');p.write_text(z.stdout+z.stderr);q['commands'].append(dict(argv=args,label=label,rc=z.returncode,log=str(p),SHA256=sha(p)))
 if z.returncode:raise RuntimeError(label+' failed: '+z.stderr[-2500:])
 return z.stdout
def deps(path):return set(shlex.split(path.read_text().replace('\\\n',' ').split(':',1)[1]))
def main():
 assert not B.exists(),'Existing CPU artifact: follow original, never retry';B.mkdir()
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI_R437_CPU_SOURCE_SHA'];m=json.loads(mf.read_bytes());q.update(actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=sha(mf))
  for name,h in m['files_SHA256'].items():assert sha(D/name)==h,name
  c=json.loads((D/'BUILD_CONTRACT.json').read_bytes());n=c['parent_native'];assert sha(n['path'])==n['SHA256'];native=json.loads(Path(n['path']).read_bytes());assert sha(native['native'])==native['native_SHA256']==n['native_SHA256']
  objects=dict(native['all_linked_object_SHA256']);assert len(objects)==33 and objects[c['parent_moe']['object']]==c['parent_moe']['object_SHA256']
  for p,h in objects.items():assert sha(p)==h,p
  for x in native['actual_dependency_library_SHA256'].values():assert sha(x['path'])==x['SHA256']
  for label in ['parent_moe','GPU_CPU_receipt']:
   v=c[label];assert sha(v['path'])==v['SHA256'],label
  assert sha(c['old_GPU_object']['object'])==c['old_GPU_object']['object_SHA256']
  assert sha(c['old_GPU_object']['MD'])==c['old_GPU_object']['MD_SHA256']
  assert sha(c['Common_extension']['path'])==c['Common_extension']['SHA256']
  # Current exact old project headers across r201/869/native/real TP16 adapter.
  references={}
  def remember(md):
   for p,h in md.items():
    if p.startswith('/refactor/'):
     key=str(Path(p).resolve());assert key not in references or references[key]==h,('conflicting pinned header',p);references[key]=h
  remember(c['source_dependency_reference'])
  for v in json.loads(Path(c['GPU_CPU_receipt']['path']).read_bytes())['compiled']:remember(v['dependency_SHA256'])
  legacy=Path('/refactor/results/Kimi-I192-CPU-r8669y.json');assert sha(legacy)=='e596e714cc9aff69a01810a27028193af402bb75bc70d4a17cda41b8d1cd2916'
  for md in json.loads(legacy.read_bytes())['fresh_MD'].values():remember(md)
  for p,h in references.items():assert sha(p)==h,p
  q['source_reverse_contract']=json.loads(run([sys.executable,'-B',str(D/'check_source.py')],'reverse-qualified-methods',15))
  q['actual_payload_FFI_and_retained_input']=json.loads(run([sys.executable,'-B',str(D/'consumer.py'),'--check-contract','--fixture-root','/refactor'],'actual-shared-consumer-payload',15))
  model='/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc'
  flags=c['HOST_prefix'][:6]+['-pthread','-I'+str(D),'-I'+model,'-I'+model+'/models/kimi_k3','-I/refactor/candidates/kimi-N4-tile-CPU-r8174','-I/refactor/incoming/Kimi-long-scale-exact-r201']+c['HOST_prefix'][6:]
  for name in ['private_moe.cpp','component.cpp']:
   obj=B/(name+'.o');before=B/(name+'.M');md=B/(name+'.d')
   run(flags+['-M','-MF',str(before),'-MT',str(obj),str(D/name)],name+'-dependencies',25)
   before_set=deps(before);pin={p:sha(p)for p in before_set}
   for p,h in pin.items():
    if p.startswith('/refactor/')and not Path(p).resolve().is_relative_to(D.resolve()):assert references.get(str(Path(p).resolve()))==h,('unpinned project header',p)
   run(flags+['-MD','-MF',str(md),'-c',str(D/name),'-o',str(obj)],name+'-compile',90)
   assert deps(md)==before_set and all(sha(p)==h for p,h in pin.items())
   q['compiled'].append(dict(source=str(D/name),source_SHA256=sha(D/name),object=str(obj),object_SHA256=sha(obj),MD=str(md),MD_SHA256=sha(md),dependency_SHA256=pin))
  py=Path(sysconfig.get_config_var('LIBDIR'))/sysconfig.get_config_var('LDLIBRARY');assert py.is_file()
  lib=B/'libKimi_full_MoE_lowrank_M1_r437.so';link=['g++','-shared','-pthread','-Wl,-Bsymbolic','-Wl,--no-undefined','-Wl,-Map,'+str(B/'link.map'),'-o',str(lib),*[v['object']for v in q['compiled']],*objects,c['old_GPU_object']['object'],'-Wl,--no-as-needed',c['Common_extension']['path'],str(py),'-Wl,--as-needed']
  root='/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib';sdk='/refactor/install/hygon-kimi-5f91ad7-r3672/lib'
  link+=['-L'+root,'-linfinilm_runtime','-L'+sdk,'-linfiniops','-linfinirt','-linfiniccl','-L/opt/dtk/cuda/cuda/lib64','-lcudart','-ldl','-Wl,-rpath,'+sdk+':'+root]
  run(link,'isolated-complete-original-plus-private-link',50)
  nm=run(['nm','--defined-only',str(lib)],'all-defined',20);dyn=run(['nm','-D','--defined-only',str(lib)],'dynamic-defined',20)
  symbols=lambda s:{line.split()[-1]:line.split()[-2]for line in s.splitlines()if len(line.split())>=3}
  full=symbols(nm);export=symbols(dyn);old=symbols(run(['nm','-D','--defined-only',native['native']],'old-Native-CABI',15));oldC={k:v for k,v in old.items()if k.startswith('modern_')and v in ['T','D','B','R']}
  assert oldC and all(k in export and export[k]==v for k,v in oldC.items()),'Old strong Native CABI missing'
  moe=symbols(run(['nm','-g','--defined-only',c['parent_moe']['object']],'exact-original-moe-object',15));strong={k for k,v in moe.items()if v in ['T','D','B','R']}
  assert strong<=set(full),'Original strong MoE functions lost'
  for name in ['modern_kimi_lowrank_full_moe_r437','modern_kimi_lowrank_full_moe_result_r437','kimi_lowrank_gemv_r432']:assert export.get(name)=='T',name
  assert any('KimiK3MoER437' in k and 'forward' in k for k in full)
  dynamic=run(['readelf','--wide','-d',str(lib)],'actual-dynamic',15);assert 'SYMBOLIC'in dynamic
  ldd=run(['ldd','-r',str(lib)],'actual-ldd-relocations',25);assert 'not found'not in ldd and 'undefined symbol'not in ldd
  for v in native['actual_dependency_library_SHA256'].values():assert sha(v['path'])==v['SHA256']
  q.update(completed=True,passed=True,library=str(lib),library_SHA256=sha(lib),original_33_objects_SHA256=objects,reused_GPU_object=c['old_GPU_object'],old_strong_CABI_preserved=len(oldC),original_moe_strong_preserved=len(strong),private_Bsymbolic=True,actual_TP16_owner_or_numeric=False,Model_or_device_qualification=False,PP1_old_padded_scale_scope=True)
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:
  q.update(elapsed_s=time.monotonic()-started,Torch_imported='torch'in sys.modules);(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['actual_CPU','completed','passed','error','elapsed_s','library_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
