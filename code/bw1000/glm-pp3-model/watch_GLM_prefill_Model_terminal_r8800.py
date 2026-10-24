"""Terminal CPU observer. Model launch/stream/weights untouched. No restart."""
import datetime,hashlib,json,pathlib,subprocess,sys,time,traceback,re
from glm_crossrequest_environment_r5742 import network_log_gate
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/GLM-transient-Model-watch-r8800.json';assert not out.exists()
side=json.loads((r/'incoming/GLM-transient-Model-observer-manifest-r8800.json').read_text())
for n,h in side['source_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
launch=json.loads((r/'glm-transient-Model-r8800-launch.json').read_text());run=launch['run']
state={'state':'watching','run':run,'bound_seconds':5000,'Model_submission':False,'restart':False,'observer_manifest_SHA256':hashlib.sha256((r/'incoming/GLM-transient-Model-observer-manifest-r8800.json').read_bytes()).hexdigest()}
def save():
 state['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(out)
def postidle():
 # Unconditional terminal finally: failure of audit/json/log parsing cannot skip.
 try:
  p=subprocess.run(['srun','--jobid=910376','--overlap','--exact','-N3','-n3','--ntasks-per-node=1','-c1','-w','a02r4n12,a02r4n13,a02r4n14','--gres=none','--mem=512M','-t','00:02:00','--job-name=glm-transient-postidle-r8800','python3',str(r/'incoming/verify_gpu_idle_r5819.py')],capture_output=True,text=True,timeout=150)
  state['postKFD_returncode']=p.returncode;state['postKFD_stderr_tail']=p.stderr[-6000:];state['postKFD_stdout']=p.stdout
  if p.returncode==0:
   rows=[json.loads(v)for v in p.stdout.splitlines()if v.startswith('{')]
   assert sorted(v['node']for v in rows)==['a02r4n12','a02r4n13','a02r4n14']and all(v['kfd_pids']==[]for v in rows)
   state['postKFD']=rows;state['postKFD_passed']=True
 except BaseException as e:state.update(postKFD_passed=False,postKFD_error=repr(e),postKFD_traceback=traceback.format_exc())
 finally:save()
def terminal(job):
 audit=None;state['producer']=job[0]
 try:
  logs=[]
  for node in (0,1,2):
   p=r/'logs'/f'glm-transient-Model-r8800-{job[0]}-{node}.log';text=p.read_text(errors='replace')if p.exists()else'';lines=text.splitlines()
   warnings=[{'line':i+1,'text':v,'nearby':lines[max(0,i-4):i+5]}for i,v in enumerate(lines)if re.search(r'(?i)(?:error|status|code)\s*[:=(]?\s*709\b|\b709\b.*(?:rocblas|hipError)',v) or 'rocblas'in v.lower()and('error'in v.lower()or'warning'in v.lower())]
   logs.append({'file':str(p),'exists':p.exists(),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest()if p.exists()else None,'warnings709_and_rocBLAS':warnings,'strict_errors':[{'line':i+1,'text':v[:1500]}for i,v in enumerate(lines)if re.search(r'Traceback|AssertionError|RuntimeError|Segmentation fault|NCCL WARN|HIP error|CUDA error|rocblas_status',v)],'high_speed_network_evidence':[v for v in lines if any(s in v for s in ['Using network IBext','NET/IBext','NET/SHCA','NET/Socket','GLM_NETWORK_R3573'])][-50:]})
  for item in logs:item['network_gate']=network_log_gate(pathlib.Path(item['file']).read_text(errors='replace'))if item['exists']else{'passed':False}
  state['logs']=logs;save()
  p=subprocess.run([sys.executable,str(r/'incoming/audit_GLM_prefill_Model_r8800.py')],capture_output=True,text=True,timeout=1200)
  state['audit_returncode']=p.returncode;state['audit_stderr_tail']=p.stderr[-12000:];save()
  if p.returncode==0:
   audit=json.loads(p.stdout);target=r/'results/GLM-transient-Model-terminal-audit-r8800.json'
   with target.open('x')as f:json.dump(audit,f,indent=2)
   state['audit_SHA256']=hashlib.sha256(target.read_bytes()).hexdigest();state['ABBA']=audit['ABBA'];save()
 except BaseException as e:state.update(audit_error=repr(e),audit_traceback=traceback.format_exc());save()
 finally:postidle()
 state['warning_review_required']=any(x['warnings709_and_rocBLAS']or x['strict_errors']for x in state.get('logs',[]))
 state['qualified']=bool(audit and audit['qualified']and state.get('postKFD_passed')and job[2:4]==['COMPLETED','0:0']and not state['warning_review_required']and all(item['network_gate']['passed']for item in state.get('logs',[])))
 state['state']='terminal_audited'if audit and state.get('postKFD_passed')else'terminal_requires_review';save()
save();begin=time.monotonic()
try:
 while time.monotonic()-begin<5000:
  try:
   text=subprocess.check_output(['sacct','-S','2026-10-23','-j','910376','-n','-P','-o','JobID,JobName%80,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=30)
  except (subprocess.TimeoutExpired,subprocess.CalledProcessError)as exc:
   state['sacct_transient_errors']=state.get('sacct_transient_errors',0)+1
   state['sacct_last_read_error']={'type':type(exc).__name__,'message':str(exc),'same_handle':state.get('bound_actual_handle')}
   save();time.sleep(30);continue
  rows=[v.split('|')for v in text.splitlines()]
  jobs=[v for v in rows if v[1]=='glm-transient-Model-r8800'and v[0].split('.')[-1].isdigit()]
  assert len(jobs)<=1,('duplicate Model step',jobs)
  if jobs:
   if state.get('bound_actual_handle')is None:state['bound_actual_handle']=jobs[0][0]
   else:assert state['bound_actual_handle']==jobs[0][0],('actual numeric handle changed',state['bound_actual_handle'],jobs[0][0])
  state['jobs']=jobs;save()
  if jobs and jobs[0][2].split()[0].split('+')[0]in {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','DEADLINE','PREEMPTED','REVOKED'}:
   terminal(jobs[0]);break
  time.sleep(60)
 else:state['state']='watch_bound_expired_no_restart';save()
except BaseException as e:
 state.update(state='watch_error_no_restart',error=repr(e),traceback=traceback.format_exc());save();raise
