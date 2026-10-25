import pathlib,json,subprocess,datetime,time,os,socket,threading,hashlib
base=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');src=base/'incoming/Kimi-draft-truebatch-r9142';out=base/'results/Kimi-draft-truebatch-r9142-controller.json';assert not out.exists();step=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];assert step.startswith('910407.');label='Kimi-draft-truebatch-r9142';env=dict(os.environ)
for k in list(env):
 if((k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])or k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']):env.pop(k)
q={'CPU_step':step,'host':socket.gethostname(),'PID':os.getpid(),'claim_published_commit':'71c6fb147e89c3d2db34e88edbdb3791992fe00d','scope':'onepinneddraftcomponent noTargetModel','state':'preflight','watch_ready_before_GPU':False,'numeric_attempts':0,'post_attempts':0};guard=threading.Lock();stop=threading.Event();start=time.monotonic()
def save():
 with guard:
  q['heartbeat_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=out.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2));p.replace(out)
def heartbeat():
 while not stop.wait(5):save()
threading.Thread(target=heartbeat,daemon=True).start();save()
def acct():return [v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-25','-j','910371','-n','-P','-o','JobID,JobName%100,State%40,ExitCode,NodeList%100'],env=env,text=True,timeout=20).splitlines()]
def state(v):return v.split()[0].rstrip('+')
def idle(tag):
 a=['srun','--jobid=910371','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','--mem=256M','--gres=none','-w','a02r3n07','-t','00:02:00','--job-name='+label+'-'+tag,'python3',str(base/'incoming/Kimi_Q0_idle_r8066.py')];z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=150);rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];assert z.returncode==0 and len(rows)==1 and rows[0]['passed']and not rows[0]['kfd_pids'];return rows[0]
proc=None
try:
 rows=acct();assert not any(v[1]==label for v in rows),'uniquecomponent alreadyexists';busy=[]
 for v in rows:
  if v[0].startswith('910371.')and v[0].split('.')[-1].isdigit()and state(v[2])in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:
   nodes=subprocess.check_output(['scontrol','show','hostnames',v[4]],env=env,text=True,timeout=10).splitlines()
   if'a02r3n07'in nodes:busy.append(v)
 assert not busy,busy;q['fresh_ownership_rows']=rows;q['before_KFD']=idle('before');save()
 # Thiscontroller is the sole ready watcher and follows exactJobName/thenactualhandle.
 acct();q['watch_ready_before_GPU']=True;q['state']='watch_ready';save()
 a=['srun','--jobid=910371','--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c4','--mem=32768M','--gres=dcu:1','-w','a02r3n07','-t','00:12:00','--job-name='+label,'--output='+str(base/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=Kimi-draft-truebatch-r9142/fixture_inside.sh','bash',str(base/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];q['numeric_attempts']=1;f=(base/'logs'/('Kimi-draft-truebatch-r9142-srun.log')).open('x');proc=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT);q['state']='SUBMITTED_FOLLOW_SAME_HANDLE';q['srun_PID']=proc.pid;save();terminal=None
 while time.monotonic()-start<850:
  matches=[v for v in acct()if v[1]==label];assert len(matches)<=1,matches
  if matches:
   row=matches[0]
   if'actual_handle'not in q:q['actual_handle']=row[0]
   assert q['actual_handle']==row[0];q['actual_Slurm']=row;save()
   if state(row[2])not in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:terminal=row;break
  time.sleep(5)
 assert terminal is not None,'boundedcontrollerdeadline_followactual';q['terminal']=terminal;q['srun_returncode']=proc.wait(timeout=30);q['state']='TERMINAL_AUDIT';save();report=base/'results'/('Kimi-draft-truebatch-fixture-r9142-'+terminal[0]+'.json');raw=report.read_bytes();data=json.loads(raw);q['report_SHA256']=hashlib.sha256(raw).hexdigest();q['report_path']=str(report);q['report_compact']={k:v for k,v in data.items()if k not in ['loader_receipt']};log=base/'logs'/(label+'-'+terminal[0]+'.log');lines=log.read_text(errors='replace').splitlines();q['normal_marker_count']=sum('KIMI_TRUEBATCH_R9142_NORMAL'in x for x in lines);q['GPU_log_SHA256']=hashlib.sha256(log.read_bytes()).hexdigest();q['audit_pass']=state(terminal[2])=='COMPLETED'and terminal[3]=='0:0'and data.get('completed')and data.get('normal_close')and len(data['cases'])==36 and all(v['finite']for v in data['cases'])and q['normal_marker_count']==1
except BaseException as e:q['error']=repr(e);q['state']='CONTROLLER_EXCEPTION_NO_REPLAY';save()
finally:
 if q.get('terminal'):
  try:q['post_attempts']=1;q['post_KFD']=idle('post');q['post_pass']=True
  except BaseException as e:q['post_error']=repr(e)
 q['qualified_component']=bool(q.get('audit_pass')and q.get('post_pass'));q['state']='CLOSED'if q.get('terminal')else q['state'];save();stop.set()
print(json.dumps(q))
