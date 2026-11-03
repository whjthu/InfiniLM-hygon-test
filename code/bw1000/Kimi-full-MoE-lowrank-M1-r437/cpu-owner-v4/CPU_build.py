"""Known757 affinity seam: one changed HOST consumer + pure sched contract, no SDK execution."""
from pathlib import Path
import hashlib,json,os,shlex,subprocess,time,traceback
D=Path(__file__).resolve().parent;P=D.parent;B=Path('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU-owner-v4');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q=dict(passed=False,completed=False,GPU=0,Native_dlopen=False,Runtime_created=False,weights=0,new_production_HOST_TUs=1,pure_sched_contract_HOST_TUs=1,new_GPU_TUs=0,old_successful_TUs_recompiled=False,commands=[])
def run(a,n,cap=45):
 left=260-(time.monotonic()-start);assert left>0
 z=subprocess.run(a,capture_output=True,text=True,timeout=min(left,cap));p=B/(n+'.log');p.write_text(z.stdout+z.stderr);q['commands'].append(dict(argv=a,label=n,rc=z.returncode,log=str(p),SHA256=sha(p)))
 if z.returncode:raise RuntimeError(n+': '+z.stderr[-2500:])
 return z.stdout
def symbols(s):return {v.split()[-1]:v.split()[-2]for v in s.splitlines()if len(v.split())>=3}
def main():
 assert not B.exists();B.mkdir()
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert sha(D/'SOURCE_MANIFEST.json')==os.environ['KIMI_R437_OWNER_V4_SHA']
  for p,h in m['files_SHA256'].items():assert sha(D/p)==h,p
  q.update(actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],entry_affinity=sorted(os.sched_getaffinity(0)),source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'));assert len(q['entry_affinity'])>=32
  oldpaths={'883':('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU/CPU-build.json','dfa142029c9fc168cddc3a7cffa133bbbad68a9ebdd8149f433402968680cb65'),'885':('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU-link-v2/CPU-build.json','8d664b80d9f4d069cfc19200af3da4a8d81205c03dfeaa4ce084c6bef663b84b'),'887':('/refactor/candidates/Kimi-full-MoE-lowrank-M1-r437-CPU-qualified-v3/CPU-build.json','00d74be5558d141d87e7cc598e7eaa10ef60e0d8ade1bb6f9e5387940c6571c2')};receipts={}
  for n,(p,h)in oldpaths.items():assert sha(p)==h;receipts[n]=json.loads(Path(p).read_bytes())
  old=receipts['883'];qualified=receipts['887'];assert qualified['passed']and sha(qualified['library'])==qualified['library_SHA256']
  references={}
  for v in old['compiled']:
   assert sha(v['object'])==v['object_SHA256']and sha(v['MD'])==v['MD_SHA256']
   for p,h in v['dependency_SHA256'].items():assert sha(p)==h;references[p]=h
  localized=qualified['localized_private_object'];assert sha(localized['object'])==localized['object_SHA256']
  c=json.loads((P/'BUILD_CONTRACT.json').read_bytes());native=c['parent_native'];assert sha(native['path'])==native['SHA256']and sha(native['native'])==native['native_SHA256']
  for p,h in native['all_linked_object_SHA256'].items():assert sha(p)==h
  g=c['old_GPU_object'];assert sha(g['object'])==g['object_SHA256']and sha(g['MD'])==g['MD_SHA256']
  for p,h in g['dependency_SHA256'].items():assert sha(p)==h
  assert sha(P/'component.cpp')==json.loads((D/'DERIVATION.json').read_bytes())['component_parent_SHA256']
  # Pure POSIX consumer is the SAME header used by the changed production TU.
  exe=B/'check_owner';run(['g++','-std=c++17','-O2','-pthread','-I'+str(D),str(D/'check_owner.cpp'),'-o',str(exe)],'pure-sched-contract-compile',20)
  q['actual_8thread_contract']=json.loads(run([str(exe)],'actual-8thread-pin-recording-FFI-restore',15));assert q['actual_8thread_contract']['passed']
  compile_old=next(v['argv']for v in old['commands']if v['label']=='component.cpp-compile');prefix=compile_old[:compile_old.index('-MD')];prefix.insert(1,'-I'+str(D))
  obj=B/'component.cpp.o';md=B/'component.cpp.d';run(prefix+['-MD','-MF',str(md),'-c',str(D/'component.cpp'),'-o',str(obj)],'changed-consumer-only-compile',90)
  deps=set(shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]));pins={p:sha(p)for p in deps}
  for p,h in pins.items():
   if p.startswith('/refactor/')and not Path(p).resolve().is_relative_to(D.resolve()):assert references.get(p)==h,('New unpinned nonprivate dependency',p)
  q['compiled']=dict(source=str(D/'component.cpp'),source_SHA256=sha(D/'component.cpp'),object=str(obj),object_SHA256=sha(obj),MD=str(md),MD_SHA256=sha(md),dependency_SHA256=pins)
  link=next(v['argv']for v in receipts['885']['commands']if v['label']=='private-complete-link-with-retained-provider')[:];assert link.count(old['compiled'][1]['object'])==1;link[link.index(old['compiled'][1]['object'])]=str(obj)
  lib=B/'libKimi_full_MoE_lowrank_M1_r437_owner_v4.so';link[link.index('-o')+1]=str(lib);link=[('-Wl,-Map,'+str(B/'link.map'))if v.startswith('-Wl,-Map,')else v for v in link];run(link,'isolated-link-reuse-old-clone-GPU-33objects',45)
  dyn=symbols(run(['nm','-D','--defined-only',str(lib)],'new-symbols'));prior=symbols(run(['nm','-D','--defined-only',qualified['library']],'prior-symbols'));strong={k:v for k,v in prior.items()if v in ['T','D','B','R']};assert all(dyn.get(k)==v for k,v in strong.items()),'Old strong/CABI removed'
  assert dyn.get('modern_kimi_owner_affinity_preflight_r437')=='T';dynamic=run(['readelf','-d',str(lib)],'new-dynamic');assert 'SYMBOLIC'in dynamic
  ldd=run(['ldd','-r',str(lib)],'new-ldd-r');assert 'not found'not in ldd and 'undefined symbol'not in ldd
  for name,expected in native['actual_dependency_library_SHA256'].items():
   rows=[v.split()for v in ldd.splitlines()if v.split()and v.split()[0]==name];assert len(rows)==1 and rows[0][1]=='=>'and sha(rows[0][2])==expected['SHA256']
  helper=qualified['helper_binding']['path'];assert qualified['helper_SHA256']==sha(helper);rows=[v.strip()for v in ldd.splitlines()if helper in v];assert len(rows)==1 and rows[0].split()[0]==helper and '(NEEDED)'in dynamic and helper in dynamic
  q.update(passed=True,completed=True,library=str(lib),library_SHA256=sha(lib),old_strong_symbols_preserved=len(strong),old887_reused=oldpaths['887'],old_GPU757_FAIL_preserved=True,new_device_or_math_qualification=False,unchanged_private_object=localized,reused_original33=native['all_linked_object_SHA256'],reused_GPU=g,GPU_GO=False,Model_GO=False)
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['passed','actual_CPU','error','elapsed_s','library_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
