"""Compile one thin HOST wrapper using the successful661 recipe; no Native rebuild."""
from pathlib import Path
import hashlib,json,os,re,shlex,subprocess,time,traceback,sys
R=Path('/refactor');D=R/'incoming/Kimi-MoE-prefill512-micro64-r391';B=R/'candidates/Kimi-MoE-prefill512-micro64-r391'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q=dict(completed=False,passed=False,CPU_only=True,Native_dlopen=False,GPU_initialized=False,Runtime_created=False,weights_opened=False,HOST_TUs_compiled=0,GPU_TUs_compiled=0,Native_rebuilt=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],commands=[])
def run(args,name,limit=120):
 q['commands'].append(args);left=250-(time.monotonic()-start);assert left>0
 p=subprocess.run(args,text=True,capture_output=True,timeout=min(limit,left));(B/(name+'.log')).write_text(p.stdout+p.stderr)
 if p.returncode:raise RuntimeError(name+': '+p.stderr[-5000:])
 return p.stdout
def main():
 mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI391_SOURCE_SHA256'];m=json.loads(mf.read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items());B.mkdir(exist_ok=False)
 q['Python_entry_contract']=json.loads(run(['python3','-B',str(D/'check_CPU.py')],'actual-Python-glue'));assert q['Python_entry_contract']['passed']
 oldpath=R/'candidates/Kimi-MoE-batch-profile-r9125/CPU-build.json';assert sha(oldpath)=='f81e8a83008873224e5d6159e5355ed5d4290c7f63618b03a07c58ac7dfc1a3a';old=json.loads(oldpath.read_bytes());assert old['completed']and old['CPU_only']and not old['dlopen']
 basepath=R/'candidates/Kimi-append512-r247/CPU-build.json';assert sha(basepath)=='21fc8ed69c6db1ec3a3c8925e61dc5f73112b5fea596a8caef977008da57046b';base=json.loads(basepath.read_bytes());native=Path(base['native']);assert base['completed']and not base['Native_dlopen']and sha(native)==base['native_SHA256']=='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
 for p,h in base['all_linked_object_SHA256'].items():assert sha(p)==h,p
 projectlibs={v['path']:v['SHA256']for v in base['actual_dependency_library_SHA256'].values()}
 for p,h in projectlibs.items():assert sha(p)==h,p
 r201path=R/'candidates/Kimi-long-scale-exact-r201/CPU-build.json';assert sha(r201path)==m['r201_CPU_receipt_SHA256'];r201=json.loads(r201path.read_bytes());originalmd=r201['all_new_MD']['moe']
 command=next(list(c)for c in old['commands']if '-shared'in c and any(s.endswith('/kimi_TP16_batch_runner_r9125.cpp')for s in c));source=next(s for s in command if s.endswith('/kimi_TP16_batch_runner_r9125.cpp'));library=B/'libKimi_MoE512_micro64_r391.so';md=B/'adapter.d';command=[str(D/'adapter.cpp')if s==source else str(native)if s==old['native']else str(md)if s.endswith('/adapter.d')else str(library)if s==old['library']else s for s in command]
 run(command,'compile-link-one-HOST',180);q['HOST_TUs_compiled']=1
 deps=shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]);dependency={p:sha(p)for p in deps};critical={}
 for suffix,pin in [('models/kimi_k3/kimi_k3_moe.hpp','3f415dcb1585d2236397bde988bdbbec4a0d73dc48fe2e2b371310b706ac67fd'),('layers/moe/router/topk_router.hpp','0e1af6b8c603a65e138faf01d6367c010b9ddfc7f24c59bf598172209b87c4c6')]:
  paths=[p for p in deps if p.endswith(suffix)];assert paths
  for p in paths:assert dependency[p]==pin and originalmd[p]==pin;critical[p]=pin
 # Same public Runtime/weights/Comm headers from the already-successful adapter.
 prior_deps=old['dependencies']['adapter.d'];checked=0
 for p,h in dependency.items():
  if p in prior_deps:assert prior_deps[p]==h,p;checked+=1
 assert checked>100
 dynamic=run(['readelf','-d',str(library)],'DT-needed',20);assert str(native)in dynamic and old['native']not in dynamic
 unresolved=run(['ldd','-r',str(library)],'container-ldd-r',45);assert 'not found'not in unresolved and'undefined symbol'not in unresolved
 symbols=run(['nm','-D','--defined-only',str(library)],'wrapper-exports',20);assert all(name in symbols for name in ['kimi_moe512_micro64_r391','kimi_moe512_micro64_result_r391'])
 paths=set(re.findall(r'=>\s+(/\S+)',unresolved));resolved={p:sha(p)for p in paths};assert resolved[str(native)]==base['native_SHA256']
 for p,h in projectlibs.items():
  if p in resolved:assert resolved[p]==h
 assert 'torch'not in sys.modules
 q.update(completed=True,passed=True,library=str(library),library_SHA256=sha(library),native=str(native),native_SHA256=sha(native),source_manifest_SHA256=sha(mf),source_SHA256=sha(D/'adapter.cpp'),MD_SHA256=sha(md),dependency_SHA256=dependency,critical_current_MoE_header_SHA256=critical,original_successful_adapter_receipt_SHA256=sha(oldpath),current_native_receipt_SHA256=sha(basepath),reused_all_Native_objects_SHA256=base['all_linked_object_SHA256'],actual_dynamic_dependency_SHA256=resolved,old_adapter_shared_header_bytes_checked=checked,scope='One HOST wrapper compile/link/actual Python payload glue recording only. No Native dlopen/Runtime/Model/GPU/numerical or performance qualification.')
if B.exists():raise SystemExit('Immutable output exists; no resubmit')
try:main()
except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
finally:
 B.mkdir(exist_ok=True);q['elapsed_s']=time.monotonic()-start;(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_CPU','HOST_TUs_compiled','library_SHA256','elapsed_s','error']}))
 if not q['passed']:raise SystemExit(1)
