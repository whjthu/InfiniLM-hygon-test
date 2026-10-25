import pathlib,json,subprocess,os,time,hashlib,sys,traceback,datetime,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming/GLM-real-concurrency-r9132';out=r/'results/GLM-G98A-r9132-watch.json';sys.path.insert(0,str(r/'incoming'));assert not out.exists();launch=json.loads((r/'GLM-G98A-r9132-launch.json').read_text());state={'state':'watching','run':launch['run'],'Model_submission':False,'restart':False,'manifest_SHA256':launch['manifest_SHA256']};env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_'))and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
def save():state['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(out)
def post():
 try:
  a=['srun','--jobid=910376','--overlap','--exact','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','-w','a02r4n12,a02r4n13','--gres=none','--mem=512M','-t','00:02:00','--job-name=GLM-G98A-post-r9132','python3',str(r/'incoming/verify_gpu_idle_r5738.py')];z=subprocess.run(a,env=env,capture_output=True,text=True,timeout=150);state['post_returncode']=z.returncode;state['post_stderr']=z.stderr[-2000:];rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];state['post_KFD']=rows;state['post_passed']=z.returncode==0 and sorted(x['node']for x in rows)==['a02r4n12','a02r4n13']and all(x['kfd_pids']==[]for x in rows)
 except BaseException as e:state['post_error']=repr(e);state['post_passed']=False
 save()
save();began=time.monotonic()
try:
 while time.monotonic()-began<4400:
  rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-25','-j','910376','-n','-P','-o','JobID,JobName%100,State,ExitCode,Elapsed,NodeList%100'],env=env,text=True,timeout=30).splitlines()if x.split('|')[1]=='glm-G98A-Model-r9132'];assert len(rows)<=1
  if rows:
   row=rows[0];old=state.get('actual_handle');assert old is None or old==row[0];state['actual_handle']=row[0];state['Slurm']=row;save()
   if row[2].split('+')[0]in ['COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL','PREEMPTED']:
    try:
     z=subprocess.run([sys.executable,str(inc/'audit.py')],env=env,capture_output=True,text=True,timeout=120);state['audit_returncode']=z.returncode;state['audit_stderr']=z.stderr[-6000:]
     if z.returncode==0:
      audit=json.loads(z.stdout);target=r/'results/GLM-G98A-r9132-audit.json';target.write_text(json.dumps(audit,indent=2));state['audit_path']=str(target);state['audit_SHA256']=hashlib.sha256(target.read_bytes()).hexdigest();state['audit_qualified']=audit['qualified']
     from glm_crossrequest_environment_r9059 import network_log_gate
     state['logs']=[]
     for node in [0,1]:
      p=r/'logs'/f'GLM-G98A-r9132-{row[0]}-{node}.log';text=p.read_text(errors='replace');state['logs'].append({'path':str(p),'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'network_gate':network_log_gate(text),'errors':[x for x in text.splitlines()if re.search(r'Traceback|AssertionError|RuntimeError|Segmentation fault|HIP error|CUDA error|rocblas_status',x)]})
    except BaseException as e:state.update(audit_error=repr(e),audit_traceback=traceback.format_exc())
    finally:post()
    state['qualified']=bool(state.get('audit_qualified')and state.get('post_passed')and row[2:4]==['COMPLETED','0:0']and all(not x['errors']and x['network_gate']['passed']for x in state.get('logs',[])));state['state']='terminal_audited';save();break
  time.sleep(45)
 else:state['state']='watch_bound_expired_no_restart';save()
except BaseException as e:state.update(state='watch_error_no_restart',error=repr(e),traceback=traceback.format_exc());save();raise
