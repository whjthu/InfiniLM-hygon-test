from pathlib import Path
import ast,hashlib,json,os,shlex,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Kimi-mixed-prefill-CPU-v2')
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def deps(p):return set(shlex.split(Path(p).read_text().replace('\\\n',' ').split(':',1)[1]))
def main():
 assert not B.exists(),'existing candidate: reconcile never repeat';B.mkdir();start=time.monotonic()
 q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Model=False,Native_dlopen=False,SDK_device_execution=False,new_HOST_TUs=1,new_GPU_TUs=0,commands=[])
 def save():(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n')
 def run(args,label,limit=90):
  left=270-(time.monotonic()-start);assert left>0
  q['phase']=label;q['commands'].append(args);save()
  z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=min(limit,left))
  p=B/(label+'.log');p.write_text(z.stdout+z.stderr);q.setdefault('logs',{})[label]=dict(rc=z.returncode,bytes=p.stat().st_size,SHA256=H(p));save()
  if z.returncode:raise RuntimeError(label+' failed; original log retained')
  return z.stdout
 try:
  assert all(os.environ.get(k)==''for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  assert not os.environ.get('LD_PRELOAD')and 'torch'not in sys.modules
  m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert H(D/'SOURCE_MANIFEST.json')==os.environ['KIMI_MIXED_SOURCE_SHA256']
  q.update(source_manifest_SHA256=H(D/'SOURCE_MANIFEST.json'),actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],entry_mask=sorted(os.sched_getaffinity(0)))
  for p,h in m['files_SHA256'].items():assert H(D/p)==h,p
  for p in D.glob('*.py'):ast.parse(p.read_bytes())
  c=json.loads((D/'BUILD_CONTRACT.json').read_bytes())
  for p,h in c['libraries_SHA256'].items():assert H(p)==h,p
  for p,h in c['source_pins'].items():assert H(p)==h,p
  q['exact_dependency_libraries']=c['libraries_SHA256'];q['backend_source_pins']=c['source_pins']
  parent=json.loads((D/'PARENT.json').read_bytes());pin=parent['CPU1062_receipt'];assert H(pin['path'])==pin['SHA256']
  old=json.loads(Path(pin['path']).read_bytes());assert old['passed']and old['actual_CPU']=='910407.1062'and not old['SDK_device_execution']
  assert old['recording_consumer']['run_failure']['driver']['missing_arms']==11 and old['recording_consumer']['expired_before_create']['driver']['missing_arms']==12
  assert H(D/'mixed_prefill.hpp')==parent['unchanged_helper_SHA256']
  q['reused_1062_negative_boundaries']=dict(receipt=pin,executed_again=False)

  source=D/'component.cpp';obj=B/'component.o';before=B/'before.d';after=B/'actual.d';prefix=c['HOST_prefix']+['-I'+str(D)]
  run(prefix+['-M','-MF',str(before),'-MT',str(obj),str(source)],'before-MD',30)
  snapshot={p:H(p)for p in sorted(deps(before))};assert str(D/'mixed_prefill.hpp')in snapshot and str(D/'contract.hpp')in snapshot
  run(prefix+['-MD','-MF',str(after),'-c',str(source),'-o',str(obj)],'component-HOST-compile',120)
  assert deps(after)==set(snapshot)and all(H(p)==h for p,h in snapshot.items())
  q['compiled']=dict(source_SHA256=H(source),object=str(obj),object_SHA256=H(obj),before_MD_SHA256=H(before),actual_MD_SHA256=H(after),all_MD=snapshot,unchanged_before_after=True);save()
  lib=B/'libKimi_mixed_prefill_r461.so'
  runtime='/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib'
  sdk='/refactor/install/hygon-kimi-5f91ad7-r3672/lib';cuda='/opt/dtk/cuda/cuda/lib64'
  args=['g++','-shared','-pthread','-Wl,-Bsymbolic','-Wl,--no-undefined',str(obj),'-o',str(lib),'-L'+runtime,'-linfinilm_runtime','-L'+sdk,'-linfiniops','-linfinirt','-L'+cuda,'-lcudart','-Wl,-rpath,'+runtime+':'+sdk+':'+cuda]
  run(args,'private-HOST-link',45)
  syms=run(['nm','-D','--defined-only',str(lib)],'symbols',15)
  required=['kimi_mixed_'+s+'_r461'for s in ['create','prepare','run','trim','counts','destroy','error','inspect']]
  for s in required:assert any(l.split()[-2:]==['T',s]for l in syms.splitlines()),s
  dynamic=run(['readelf','--wide','-d',str(lib)],'dynamic',15);assert 'SYMBOLIC'in dynamic
  run(['ldd','-r',str(lib)],'relocation-trace-no-entry',25);ldd=(B/'relocation-trace-no-entry.log').read_text();assert 'not found'not in ldd and 'undefined symbol'not in ldd
  text=run([sys.executable,'-B',str(D/'check_entry.py')],'actual-driver-libffi-recording',45)
  result=json.loads(text);assert result['passed']and result['GPU']==0 and not result['Native_dlopen']and not result['SDK_execution']
  assert len(result['success']['driver']['inspections'])==4 and result['reused_1062_negative_boundaries_not_rerun'];q['recording_consumer']=result
  q['project_maps']=[l for l in Path('/proc/self/maps').read_text().splitlines()if any(t in l for t in ['libKimi','libinfinilm','libinfiniops','libcudart'])];assert not q['project_maps']
  q.update(passed=True,library=str(lib),library_SHA256=H(lib),CAPI=required,qualified='only one real HOST TU/MD/link/ELF and typed recording entry. No Runtime create, SDK device call, weights, Model, numerical or memory fit qualification.')
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:q.update(completed=True,elapsed_s=time.monotonic()-start,Torch_imported='torch'in sys.modules);save();print(json.dumps({k:q.get(k)for k in ['actual_CPU','passed','phase','error','elapsed_s','library_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
