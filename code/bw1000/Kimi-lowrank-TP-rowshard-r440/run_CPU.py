"""Bounded CPU syntax/interface and algebra checks; never dlopen project SDK."""
import hashlib,json,os,pathlib,subprocess,sys,time,traceback
D=pathlib.Path(__file__).resolve().parent
R=pathlib.Path('/refactor')
B=R/'candidates/Kimi-lowrank-TP-rowshard-r440-CPU-v1'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
start=time.monotonic();B.mkdir(exist_ok=False)
q={'completed':False,'passed':False,'GPU':0,'Native_dlopen':False,'Runtime_execution':False,'Model':False,'commands':[]}
def run(argv,name,limit):
 left=255-(time.monotonic()-start);assert left>0
 t=time.monotonic();p=subprocess.run(argv,capture_output=True,text=True,timeout=min(left,limit))
 (B/(name+'.stdout')).write_text(p.stdout);(B/(name+'.stderr')).write_text(p.stderr)
 q['commands'].append({'name':name,'argv':argv,'rc':p.returncode,'wall_s':time.monotonic()-t,'stdout_SHA256':sha(B/(name+'.stdout')),'stderr_SHA256':sha(B/(name+'.stderr'))})
 assert p.returncode==0,{'operation':name,'rc':p.returncode,'stderr':p.stderr[-2500:]}
 return p.stdout
try:
 q['actual_CPU']=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
 q['visible_env']={k:os.environ.get(k)for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']};assert set(q['visible_env'].values())=={'-1'}
 q['source_manifest_SHA256']=sha(D/'SOURCE_MANIFEST.json');assert q['source_manifest_SHA256']==os.environ['KIMI_R440_SOURCE_SHA256']
 for name,h in json.loads((D/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(D/name)==h,name
 c=json.loads((D/'SOURCE_INTERFACES.json').read_bytes())
 for p,h in c['project_source_pins'].items():assert sha(p)==h,p
 mask=sorted(os.sched_getaffinity(0));assert len(mask)>=4;os.sched_setaffinity(0,set(mask[:4]));q['actual_CPU_affinity']=sorted(os.sched_getaffinity(0))
 model='/refactor/candidates/kimi-serving-MoE-5f91ad7-r4665/InfiniLM/csrc'
 flags=c['HOST_prefix'][:6]+['-I'+str(D),'-I'+model]+c['HOST_prefix'][6:]
 source=D/'compile_contract.cpp';obj=B/'compile_contract.o';md=B/'compile_contract.d';pre=B/'compile_contract.pre.d'
 run(flags+['-M','-MF',str(pre),'-MT',str(obj),str(source)],'production-pre-MD',40)
 def deps(path):
  return {str(pathlib.Path(p).resolve())for p in path.read_text().replace('\\\n',' ').split(':',1)[1].split()}
 before={p:sha(p)for p in deps(pre)}
 run(flags+['-MD','-MF',str(md),'-c',str(source),'-o',str(obj)],'production-actual-Dense-and-AllGather-compile-only',100)
 assert deps(md)==set(before)and all(sha(p)==h for p,h in before.items())
 q['actual_production_compile']={'object_SHA256':sha(obj),'MD_SHA256':sha(md),'dependency_SHA256':before,'executed':False,'linked_or_dlopen':False}
 run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror',str(D/'check_geometry.cpp'),'-o',str(B/'check_geometry')],'pure-geometry-compile',30)
 q['geometry']=json.loads(run([str(B/'check_geometry')],'pure-geometry-run',10))
 q['CPU_math_reference']=json.loads(run([sys.executable,'-B',str(D/'check_math_CPU.py')],'CPU-Torch-layout-and-reduction-reference',80))
 assert q['geometry']['passed']and q['CPU_math_reference']['passed']
 maps=pathlib.Path('/proc/self/maps').read_text();q['project_Native_maps']=[s for s in maps.splitlines()if any(v in s for v in ['libinfinilm','_infinilm.cpython','libinfiniops','libinfinirt','libKimi','libDeep','libGLM'])];assert not q['project_Native_maps']
 q.update(passed=True,device_collective_or_precision_qualified=False,performance_qualified=False,implementation_ready_for_Model=False)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
finally:
 q.update(completed=True,wall_s=time.monotonic()-start);(B/'result.json').write_text(json.dumps(q,indent=2)+'\n')
 print(json.dumps({k:q.get(k)for k in ['actual_CPU','completed','passed','error','wall_s']}),flush=True)
if not q['passed']:raise SystemExit(1)
