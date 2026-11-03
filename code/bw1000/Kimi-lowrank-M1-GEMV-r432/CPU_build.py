"""Exactly one private GPU TU and one thin production HOST; old libraries untouched."""
from pathlib import Path
import datetime,hashlib,json,os,shlex,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Kimi-lowrank-M1-GEMV-r432-CPU')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();start=time.monotonic()
q=dict(completed=False,passed=False,CPU_only=True,GPU=0,Model=False,Native_dlopen=False,checkpoint_weight_payload=False,old_Native_or_SDK_rebuilt=False,commands=[],compiled=[])
def run(args,label,limit=120):
 left=260-(time.monotonic()-start);assert left>0,'CPU inside260s original budget'
 z=subprocess.run(args,capture_output=True,text=True,timeout=min(left,limit));p=B/(label+'.log');p.write_text(z.stdout+z.stderr)
 q['commands'].append(dict(argv=args,label=label,rc=z.returncode,log=str(p),SHA256=sha(p)))
 if z.returncode:raise RuntimeError(label+' failed; raw log retained: '+z.stderr[-2000:])
 return z.stdout
def deps(p):return set(shlex.split(p.read_text().replace('\\\n',' ').split(':',1)[1]))
def main():
 assert not B.exists(),'Existing artifact: reconcile, never repeat';B.mkdir()
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])and not os.environ.get('LD_PRELOAD')
  mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI_R432_SOURCE_SHA'];m=json.loads(mf.read_bytes());q.update(actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],source_manifest_SHA256=sha(mf))
  for p,h in m['files_SHA256'].items():assert sha(D/p)==h,p
  c=json.loads((D/'BUILD_CONTRACT.json').read_bytes())
  for p,h in c['libraries_SHA256'].items():assert sha(p)==h,p
  for v in [c['Common_extension'],c['qualified_parent']]:assert sha(v['path'])==v['SHA256']
  for p,h in c['source_pins'].items():assert sha(p)==h,p
  q['source_pins']=c['source_pins'];q['library_pins']={**c['libraries_SHA256'],c['Common_extension']['path']:c['Common_extension']['SHA256']}
  for name,prefix in [('component.cpp',c['HOST_prefix']),('gemv.cu',c['GPU_prefix'])]:
   obj=B/(name+'.o');md=B/(name+'.d');before=B/(name+'.M');prefix=prefix+['-I'+str(D)]
   run(prefix+['-M','-MF',str(before),'-MT',str(obj),str(D/name)],name+'-deps',30);pin={p:sha(p)for p in deps(before)}
   run(prefix+['-MD','-MF',str(md),'-c',str(D/name),'-o',str(obj)],name+'-compile',120)
   assert deps(md)==set(pin)and all(sha(p)==h for p,h in pin.items())
   q['compiled'].append(dict(source=str(D/name),source_SHA256=sha(D/name),object=str(obj),object_SHA256=sha(obj),MD=str(md),MD_SHA256=sha(md),dependency_SHA256=pin))
  # The real shared geometry contract; no fake Runtime, loader or SDK device run.
  run(['g++','-std=c++17','-O2','-I'+str(D),str(D/'CPU_contract.cpp'),'-o',str(B/'CPU_contract')],'necessary-pure-geometry-contract-compile',15)
  run([str(B/'CPU_contract')],'same-production-geometry-contract',10)
  import sysconfig
  py=Path(sysconfig.get_config_var('LIBDIR'))/sysconfig.get_config_var('LDLIBRARY');assert py.is_file()
  lib=B/'libKimi_lowrank_M1_r432.so'
  link=['g++','-shared','-pthread','-Wl,-Bsymbolic','-Wl,--no-undefined','-o',str(lib),str(B/'component.cpp.o'),str(B/'gemv.cu.o'),'-Wl,--no-as-needed',c['Common_extension']['path'],str(py),'-Wl,--as-needed']
  link+=['-L/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib','-linfinilm_runtime','-L/refactor/install/hygon-kimi-5f91ad7-r3672/lib','-linfiniops','-linfinirt','-L/opt/dtk/cuda/cuda/lib64','-lcudart','-Wl,-rpath,/refactor/install/hygon-kimi-5f91ad7-r3672/lib:/refactor/candidates/rebased-5f91ad7-r3668/Common/Kimi/InfiniLM/python/infinilm/_lib']
  run(link,'private-isolated-link',40);nm=run(['nm','-D','--defined-only',str(lib)],'actual-symbols',15)
  expected=['kimi_lowrank_create_r432','kimi_lowrank_run_r432','kimi_lowrank_counts_r432','kimi_lowrank_destroy_r432','kimi_lowrank_error_r432','kimi_lowrank_gemv_r432','kimi_lowrank_layout_r432']
  for name in expected:assert any(line.split()[-2:]==['T',name]for line in nm.splitlines()),name
  dynamic=run(['readelf','--wide','-d',str(lib)],'actual-dynamic',15);assert 'SYMBOLIC'in dynamic
  needed=run(['ldd','-r',str(lib)],'actual-ldd-relocations',25);assert 'not found'not in needed and 'undefined symbol'not in needed
  q.update(completed=True,passed=True,library=str(lib),library_SHA256=sha(lib),private_CABI=expected,HOST_production_TUs=1,GPU_TUs=1,pure_geometry_contract_TUs=1,geometry_valid=4,geometry_negative=9,qualified_parent=c['qualified_parent'],additional_Python_library=dict(path=str(py),SHA256=sha(py)),actual_device_NUMERIC_or_speed=False,qualification='CPU compile/link/source/dependency and actual shared geometry contract only. No library dlopen, Runtime/device, checkpoints or GEMV numerical execution.')
 except BaseException as error:q.update(error=repr(error),traceback=traceback.format_exc())
 finally:
  q.update(elapsed_s=time.monotonic()-start,Torch_imported='torch'in sys.modules);(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['actual_CPU','completed','passed','error','elapsed_s','library_SHA256']}),flush=True)
 return 0 if q['passed']else 1
if __name__=='__main__':raise SystemExit(main())
