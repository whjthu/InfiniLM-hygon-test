"""Actual derived50object metadata, publicCLI and changed owner-mode CPU glue."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
D=Path(__file__).resolve().parent;B=Path('/refactor/candidates/Deep454-fixed-profile-CPU-v1')
started=time.monotonic();B.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(completed=False,passed=False,Native_dlopen=False,Runtime_execution=False,Model_loaded=False,GPU=0,new_HOST_TUs=0,new_GPU_TUs=0,commands=[])

def run(argv,name,env):
 left=250-(time.monotonic()-started);assert left>0
 p=subprocess.run(argv,env=env,capture_output=True,text=True,timeout=min(left,110))
 (B/(name+'.stdout')).write_text(p.stdout);(B/(name+'.stderr')).write_text(p.stderr)
 q['commands'].append(dict(name=name,argv=argv,rc=p.returncode,stdout_SHA256=sha(B/(name+'.stdout')),stderr_SHA256=sha(B/(name+'.stderr'))))
 assert p.returncode==0,(name,p.stderr[-5000:],p.stdout[-2500:])
 return json.loads(p.stdout)

try:
 q['actual_CPU']=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID']
 assert all(os.environ.get(k)=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
 mask=sorted(os.sched_getaffinity(0));assert len(mask)>=4;os.sched_setaffinity(0,set(mask[:4]));q['affinity']=sorted(os.sched_getaffinity(0))
 mf=sha(D/'SOURCE_MANIFEST.json');assert mf==os.environ['DEEP454_SOURCE_SHA256'];q['source_manifest_SHA256']=mf
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
 for name,h in m['files_SHA256'].items():assert sha(D/name)==h,name
 env=dict(os.environ,PYTHONPATH=':'.join(map(str,[D,D.parents[1]/'public-Deep-native-provider-r427/source-v3',Path('/old/offline-deps')])) )
 q['worker_control']=run([sys.executable,'-B',str(D/'check_profile_CPU.py')],'changed-fullpre-worker-and-typed-CABI',env)
 assert q['worker_control']['passed']and q['worker_control']['recording_processor_HTTP_Native_leaves']and not q['worker_control']['actual_Native_or_GPU']
 code="from public_deep_model.native_binding import binding;import json,os,pathlib;b,r,n=binding();maps=[x for x in pathlib.Path('/proc/self/maps').read_text().splitlines()if any(v in x for v in ['libinfinilm','libinfiniops','libinfiniccl','libinfinirt','_infinilm.cpython','libflash_attn'])];assert not maps;print(json.dumps({'library_SHA256':r['library_SHA256'],'objects':len(n['link']['full']['objects']),'guard_retained':n['original_guard_proof_preserved'],'Native_dlopen':False,'project_maps':maps,'hidden':{k:os.environ[k]for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']}}))"
 q['actual_binding']=run([sys.executable,'-B','-c',code],'actual50object-derivation-NO-dlopen',env)
 pin=json.loads((D/'MODEL_PIN.json').read_bytes());a=q['actual_binding'];assert a['library_SHA256']==pin['library_SHA256']and a['objects']==50 and a['guard_retained']and not a['Native_dlopen']and not a['project_maps']
 args=[sys.executable,'-B','-m','public_deep_model','--config',str(D/'config.template.json'),'--config-sha',sha(D/'config.template.json'),'--source-sha',mf,'--run','CPU.source-only','--actual-step',q['actual_CPU'],'--gpu-start-epoch','0']
 q['actual_whole_CLI_default']=run(args,'actual-CLI-default-OFF-NOT-serving',env)
 assert q['actual_whole_CLI_default']==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False,experimental_profile=None)
 q['actual_whole_CLI_explicit']=run(args+['--experimental-profile','deep-hc0-fullpre-c1'],'actual-CLI-explicit-profile-NOT-serving',env)
 assert q['actual_whole_CLI_explicit']==dict(source_and_schema_valid=True,Model_READY=False,Native_dlopen=False,GPU=False,experimental_profile='deep-hc0-fullpre-c1')
 assert not any(n in sys.modules for n in ['torch','_infinilm','flash_attn'])
 q.update(passed=True,GPU_GO=False,Model_GO=False,performance='UNMEASURED',r366_fixed=0,default_mode=0,explicit_engine_profile='deep-hc0-fullpre-c1',request_mode_control=False,old_math_or_kernel_suites_repeated=False)
except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
finally:
 q.update(completed=True,wall_s=time.monotonic()-started);(B/'result.json').write_text(json.dumps(q,indent=2)+'\n')
 print(json.dumps({k:q.get(k)for k in ['actual_CPU','completed','passed','wall_s','error']}))
if not q['passed']:raise SystemExit(1)
