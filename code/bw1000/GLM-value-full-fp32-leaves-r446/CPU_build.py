"""Prepared only: one private HOST+one private GPU TU; never execute linked SDK.

The pure bitfield/reference executable is standalone and links no SDK/Runtime.
This source is not a CPU submission or a device qualification.
"""
from pathlib import Path
import json,hashlib,os,shlex,subprocess,time,re,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/GLM-value-full-fp32-leaves-r446-CPU');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 start=time.monotonic();B.mkdir(exist_ok=False)
 q=dict(completed=False,passed=False,producer=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=0,Native_dlopen=False,Runtime_executed=False,weights_opened=False,commands=[],HOST_TUs_compiled=0,GPU_TUs_compiled=0)
 def run(argv,name):
  left=260-(time.monotonic()-start);assert left>0
  q['commands'].append(argv);p=subprocess.run(argv,capture_output=True,text=True,timeout=left);(B/(name+'.log')).write_text(p.stdout+p.stderr)
  assert p.returncode==0,dict(phase=name,returncode=p.returncode,log=str(B/(name+'.log')))
  return p.stdout+p.stderr
 try:
  assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  assert sha(D/'SOURCE_MANIFEST.json')==os.environ['GLM446_SOURCE_SHA256']
  m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
  pin=json.loads((D/'DEPENDENCIES.json').read_bytes());p=Path(pin['parent_receipt']['path']);assert sha(p)==pin['parent_receipt']['SHA256'];a=json.loads(p.read_bytes());old=a['compiled']
  assert a['passed']and a['producer']=='910407.879'and sha(a['library'])==pin['parent_library']['SHA256']
  for name,h in old['dependencies_SHA256'].items():assert sha(name)==h,name
  assert sha(old['source'])==old['source_SHA256']and sha(old['object'])==old['object_SHA256']and sha(old['MD_path'])==old['MD_SHA256']
  assert set(shlex.split(Path(old['MD_path']).read_text().replace('\\\n',' ').split(':',1)[1]))==set(old['dependencies_SHA256'])
  for name,h in a['reused_objects_SHA256'].items():assert sha(name)==h,name
  assert len(a['reused_objects_SHA256'])==29
  prefix=json.loads((D/'ACTUAL_COMPILER_PREFIX.json').read_bytes());actual=Path(prefix['receipt']['path']);assert sha(actual)==prefix['receipt']['SHA256']
  proven=json.loads(actual.read_bytes());assert proven['returncode']==0 and not proven['GPU_initialized']and proven['CPU_only']
  run(['g++','-std=c++17','-O2','-fno-fast-math','-ffp-contract=off','-Wall','-Wextra','-Werror','-I'+str(D),str(D/'check_bits_reference.cpp'),'-o',str(B/'check_bits')],'bits.compile')
  q['pure_contract']=json.loads(run([str(B/'check_bits')],'bits.run'));assert q['pure_contract']['passed']
  obj=B/'attention_component_r446.o';md=B/'attention_component_r446.d'
  argv=next(c[:]for c in a['commands']if isinstance(c,list)and'-c'in c and'-o'in c)
  assert argv[argv.index('-c')+1]==old['source']and argv[argv.index('-o')+1]==old['object']
  argv[1:1]=['-I'+str(D)];argv[argv.index('-c')+1]=str(D/'attention_component_r446.cpp');argv[argv.index('-o')+1]=str(obj);argv[argv.index('-MF')+1]=str(md)
  run(argv,'HOST.compile');q['HOST_TUs_compiled']=1
  fresh=set(shlex.split(md.read_text().replace('\\\n',' ').split(':',1)[1]));shared=fresh&set(old['dependencies_SHA256']);assert len(shared)>850 and all(sha(p)==old['dependencies_SHA256'][p]for p in shared)
  for suffix in ['runtime/runtime.hpp','runtime/tensor.hpp','glm_moe_dsa_attention_prepare.hpp','glm_moe_dsa_attention_output.hpp']:
   found=[p for p in fresh if p.endswith('/'+suffix)];assert len(found)==1 and found[0]in shared,suffix
  gpu=B/'fp32_leaves_r446.o';gpu_md=B/'fp32_leaves_r446.d'
  # DTK nvcc frontend/target inherited from actual GLM341 and r438 build paths.
  # Newly requested strict IEEE flags must be accepted, never silently dropped.
  proven_argv=proven['command'];assert proven_argv[:4]==['/opt/dtk/cuda/cuda/bin/nvcc','-std=c++17','-O3','-arch=gfx936']
  assert proven_argv[4:6]==['-shared','-Xcompiler=-fPIC']
  gpu_argv=proven_argv[:4]+['-Xcompiler=-fPIC','-fno-fast-math','-ffp-contract=off','-fdenormal-fp-math=ieee','-I'+str(D),'-MD','-MF',str(gpu_md),'-c',str(D/'fp32_leaves_r446.cu'),'-o',str(gpu)]
  text=run(gpu_argv,'GPU.compile');assert not re.search('unused.command.line.argument|argument unused|unused argument|unknown.argument|unsupported.option|ignoring.unknown|option.*ignored',text,re.I)
  q['GPU_TUs_compiled']=1;gpu_deps=set(shlex.split(gpu_md.read_text().replace('\\\n',' ').split(':',1)[1]))
  from inspect_gpu import inspect
  q['actual_new_GPU_object_inspection']=inspect(gpu,B,run)
  link=next(c[:]for c in a['commands']if isinstance(c,list)and'-shared'in c);assert link.count(old['object'])==1
  lib=B/'libGLM_value_full_fp32_leaves_r446.so';link=[str(obj)if v==old['object']else v for v in link];link.insert(link.index('-o'),str(gpu));link[link.index('-o')+1]=str(lib)
  run(link,'library.link')
  before=run(['nm','-D','--defined-only',a['library']],'parent.nm');after=run(['nm','-D','--defined-only',str(lib)],'candidate.nm')
  exports=lambda text:{v.split()[-1]:v.split()[-2]for v in text.splitlines()if len(v.split())>=3}
  x,y=exports(before),exports(after)
  assert all(y.get(n)==t for n,t in x.items()if t not in ['W','V','u','w','v']or not n.startswith('_Z'))
  for n in ['glm_value_cast_r446','glm_value_gemv_r446','glm_attention_value_fullfp32_r446']:assert y[n]=='T',n
  parent_ldd=run(['ldd','-r',a['library']],'parent.ldd');new_ldd=run(['ldd','-r',str(lib)],'candidate.ldd')
  assert 'not found'not in new_ldd and 'undefined symbol:'not in new_ldd
  paths=lambda text:set(re.findall(r'(?:=>\s+|^\s*)(/[^\s()]+)',text,re.M))
  p0,p1=paths(parent_ldd),paths(new_ldd);assert {p:sha(p)for p in p0}=={p:sha(p)for p in p1}
  q.update(passed=True,source_manifest_SHA256=sha(D/'SOURCE_MANIFEST.json'),parent_receipt_SHA256=sha(p),library=str(lib),library_SHA256=sha(lib),compiled_HOST=dict(source=str(D/'attention_component_r446.cpp'),source_SHA256=sha(D/'attention_component_r446.cpp'),object=str(obj),object_SHA256=sha(obj),MD=str(md),MD_SHA256=sha(md),dependencies_SHA256={p:sha(p)for p in sorted(fresh)}),compiled_GPU=dict(source=str(D/'fp32_leaves_r446.cu'),source_SHA256=sha(D/'fp32_leaves_r446.cu'),object=str(gpu),object_SHA256=sha(gpu),MD=str(gpu_md),MD_SHA256=sha(gpu_md),dependencies_SHA256={p:sha(p)for p in sorted(gpu_deps)}),reused_objects_SHA256=a['reused_objects_SHA256'],final_objects_SHA256={**a['reused_objects_SHA256'],str(obj):sha(obj),str(gpu):sha(gpu)},actual_dynamic_dependencies_SHA256={p:sha(p)for p in sorted(p1)},default_OFF=True,device_numeric=False,Model_READY=False,denormal_hardware_behavior='UNKNOWN pending first actual complete-call typed cases; no GPU executed')
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 q.update(completed=True,seconds=time.monotonic()-start);(B/'CPU-build.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['passed','producer','HOST_TUs_compiled','GPU_TUs_compiled','library_SHA256','seconds','error']}));raise SystemExit(0 if q['passed']else 1)
if __name__=='__main__':main()
