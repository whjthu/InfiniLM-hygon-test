"""Only new device session/observer/controller source-only glue; no device."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback
from support import D,R,PACKAGE,sha,verify,actual
B=Path('/refactor/candidates/Deep458-device-glue-CPU-v1')

def main():
 B.mkdir(exist_ok=False);start=time.monotonic();q=dict(completed=False,passed=False,actual_CPU=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],GPU=0,Native_dlopen=False,SDK_execution=False,weights=False,new_compiles=0,old967_cases_replayed=False,old945_encoding_replayed=False)
 try:
  assert all(os.environ[k]=='-1'for k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES'])
  digest=os.environ['DEEP458_GLUE_SOURCE_SHA256'];m=verify(digest);q['source_manifest_SHA256']=digest
  original=actual('/refactor/candidates/Deep-a314-fixed-cohort-C4-r458-CPU-v1/CPU-result.json')
  cpu=json.loads(original.read_bytes());packet=actual(cpu['actual_packet_handoff']['path'])
  assert sha(packet)==cpu['actual_packet_handoff']['SHA256']
  prepared=json.loads(packet.read_bytes());assert [r['ordinal']for r in prepared['records']]==[0,1,128,129]
  from check_glue_CPU import run
  q['whole_changed_session']=run(prepared);assert q['whole_changed_session']['passed']
  env=dict(os.environ,DEEP458_C4_CONTROLLER_SHA=digest)
  z=subprocess.run([sys.executable,'-B',str(D/'worker.py'),'--source-only'],env=env,capture_output=True,text=True,timeout=60)
  (B/'worker-source-only.stdout').write_text(z.stdout);(B/'worker-source-only.stderr').write_text(z.stderr);assert z.returncode==0,(z.stdout,z.stderr)
  q['actual_worker_main_source_only']=json.loads(z.stdout);assert not q['actual_worker_main_source_only']['GPU']and not q['actual_worker_main_source_only']['Model_READY']
  args=[sys.executable,'-B',str(D/'launch_once.py'),'--manifest-sha',digest,'--publication-commit',os.environ['DEEP458_GLUE_PUBLICATION'],'--source-only']
  z=subprocess.run(args,env=env,capture_output=True,text=True,timeout=60)
  (B/'launcher-source-only.stdout').write_text(z.stdout);(B/'launcher-source-only.stderr').write_text(z.stderr);assert z.returncode==0,(z.stdout,z.stderr)
  q['actual_launcher_main_source_only']=json.loads(z.stdout);assert q['actual_launcher_main_source_only']['Slurm']==q['actual_launcher_main_source_only']['GPU']==0
  for p in D.rglob('*.py'):compile(p.read_bytes(),str(p),'exec')
  for p in D.glob('*.sh'):subprocess.run(['bash','-n',str(p)],check=True,timeout=10)
  maps=[s for s in Path('/proc/self/maps').read_text().splitlines()if any(x in s for x in ['libDeep_','libinfinilm_runtime','libinfiniops','libinfiniccl','libinfinirt','_infinilm.cpython'])]
  assert not maps and 'torch'not in sys.modules
  q.update(passed=True,actual_changed_session=True,source_only_no_job=True,project_SDK_maps=maps,GPU_GO=False,Model_READY=False,
    original_CPU967_receipt_SHA256=sha(original),original_packet_SHA256=sha(packet),recording_scope='One whole sixcohort session only; actual public parser/scorer; Native/libffi,weights,memory,affinity/provider leaves recorded')
 except BaseException as e:q.update(error=repr(e),traceback=traceback.format_exc())
 finally:
  q.update(completed=True,elapsed_s=time.monotonic()-start);(B/'CPU-result.json').write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:q.get(k)for k in ['passed','completed','actual_CPU','elapsed_s','error']}),flush=True)
 return 0 if q['passed']else 1

if __name__=='__main__':raise SystemExit(main())
