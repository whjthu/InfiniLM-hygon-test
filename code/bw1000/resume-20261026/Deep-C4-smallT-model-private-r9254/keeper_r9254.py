"""One bounded Slurm CPU supervisor; exact owned target step, no retry/reload."""
import sys,pathlib,json,os,subprocess,socket,time,datetime,hashlib,traceback
assert len(sys.argv)==2,'exact reviewed manifest SHA required'
EXPECTED_MANIFEST_SHA=sys.argv[1]
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/Deep-C4-smallT-model-r9254';LABEL='deep-c4-smallt-model-r9254';PARENT='910374';NODE='a02r4n05';ACTIVE={'RUNNING','PENDING','CONFIGURING','COMPLETING'};H=R/'results/Deep-C4-smallT-Model-control-r9254.json';assert not H.exists();state={'label':LABEL,'CPU_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'hostname':socket.gethostname(),'PID':os.getpid(),'GPU_submitted':False,'completed':False,'numeric_policy_changed':False,'restart':False};env=dict(os.environ)
for k in list(env):
 if(k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])or k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']:env.pop(k)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save():
 state['heartbeat_UTC']=now();p=H.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2),encoding='utf-8');p.replace(H)
def account(name):
 raw=subprocess.check_output(['sacct','-S','2026-10-25','-j',PARENT,'-n','-P','-o','JobIDRaw,JobName%100,State%35,ExitCode,NodeList%100,Elapsed'],env=env,text=True,timeout=25);rows=[s.split('|')for s in raw.splitlines()if len(s.split('|'))>=6 and s.split('|')[1]==name];assert len(rows)<=1,(name,rows);return rows[0]if rows else None
def terminal(name,handle):
 for _ in range(70):
  v=account(name)
  if v:
   assert v[0]==handle
   if v[2].split()[0].rstrip('+')not in ACTIVE:return v
  save();time.sleep(1)
 raise RuntimeError('authoritative terminal timeout '+handle)
def run_cpu(name,inside=None,idle=False):
 a=['srun','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c'+('1'if idle else'32'),'-w',NODE,'--gres=none','--mem='+('512M'if idle else'2048M'),'-t','00:04:00','--job-name='+name]
 if idle:a+=['python3',str(R/'incoming/Kimi_Q0_idle_r8066.py')]
 else:a+=['--export=ALL,WAVE_INSIDE_R2323=Deep-C4-smallT-model-r9254/'+inside,'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=265);rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];v={'args':a,'returncode':z.returncode,'rows':rows,'stdout_tail':z.stdout[-15000:],'stderr_tail':z.stderr[-10000:]};state[name]=v;save();assert z.returncode==0,(name,v)
 if idle:assert len(rows)==1 and rows[0]['passed']and not rows[0]['kfd_pids'];handle=rows[0]['step']
 else:
  checks=[x for x in rows if x.get('preflight')=='actualfinalfrontend beforetorch/CDLL/nativeAPI'];aff=[x for x in rows if x.get('check')=='actual32CPUbeforeGPU'];assert len(checks)==len(aff)==1 and checks[0]['passed']and aff[0]['passed']and aff[0]['affinity_count']>=32;handle=aff[0]['step']
 v['actual_terminal']=terminal(name,handle);assert v['actual_terminal'][2:4]==['COMPLETED','0:0'];save();return v
save();gpu=None
try:
 state['phase']='validate_source_and_ownership';save();m=json.loads((S/'manifest.json').read_bytes());state['manifest_SHA256']=hashlib.sha256((S/'manifest.json').read_bytes()).hexdigest();save();assert hashlib.sha256((S/'manifest.json').read_bytes()).hexdigest()==EXPECTED_MANIFEST_SHA
 for n,h in m['incoming_source_SHA256'].items():assert hashlib.sha256((S/n).read_bytes()).hexdigest()==h,n
 raw=subprocess.check_output(['scontrol','show','job',PARENT,'-o'],env=env,text=True,timeout=25);assert'JobState=RUNNING'in raw and'UserId=aclsylqidf('in raw;ns=next(x.split('=',1)[1]for x in raw.split()if x.startswith('NodeList='));assert NODE in subprocess.check_output(['scontrol','show','hostnames',ns],env=env,text=True,timeout=25).splitlines();assert account(LABEL)is None,'never repeat owned Model label'
 fresh=subprocess.check_output(['sacct','-S','2026-10-25','-j',PARENT,'-n','-P','-o','JobIDRaw,State%35,NodeList%100'],env=env,text=True,timeout=25);state['fresh_sacct']=fresh
 for line in fresh.splitlines():
  v=line.split('|')
  if len(v)>=3 and '.'in v[0]and v[0].split('.')[-1].isdigit()and v[1].split()[0].rstrip('+')in ACTIVE:
   hosts=subprocess.check_output(['scontrol','show','hostnames',v[2]],env=env,text=True,timeout=25).splitlines();assert NODE not in hosts,('activeGPUparentNode',v)
 state['phase']='CPU_preflight';save();run_cpu('Deep-C4-smallT-Model-preflight-r9254','actual_preflight_inside.sh');state['phase']='fresh_before';save();run_cpu('Deep-C4-smallT-Model-before-r9254',idle=True)
 state['watch_ready_before_GPU']=True;state['phase']='ready_before_unique_GPU';save()
 a=['srun','--jobid='+PARENT,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c32','-w',NODE,'--gres=dcu:8','--mem=128000M','-t','01:30:00','--job-name='+LABEL,'--export=ALL,WAVE_INSIDE_R2323=Deep-C4-smallT-model-r9254/GPU_inside.sh','--output='+str(R/'logs'/(LABEL+'-%J.log')),'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];fo=(R/'logs/Deep-C4-smallT-model-r9254-srun-client.log').open('x');gpu=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=fo,stderr=subprocess.STDOUT);state.update(GPU_submitted=True,phase='following_same_unique_GPU',GPU_args=a,GPU_client_PID=gpu.pid);save();start=time.monotonic()
 while gpu.poll()is None:
  v=account(LABEL)
  if v:
   assert v[0].startswith(PARENT+'.')and v[0].split('.')[-1].isdigit();assert state.get('GPU_handle',v[0])==v[0];state['GPU_handle']=v[0];state['GPU_actual']=v
  assert time.monotonic()-start<5550,'GPU srun exceeded own hard bound';save();time.sleep(15)
 fo.close();state['GPU_srun_returncode']=gpu.returncode;v=account(LABEL);assert v and v[0].startswith(PARENT+'.');state['GPU_handle']=v[0];state['GPU_terminal']=terminal(LABEL,v[0]);result=R/'results'/(LABEL+'-'+v[0]+'.json');state['result']=str(result);state['phase']='terminal_audit_cleanup';save()
 try:
  assert result.exists();state['result_SHA256']=hashlib.sha256(result.read_bytes()).hexdigest();z=subprocess.run(['python3',str(S/'audit_Model_r9254.py'),str(result)],env=env,text=True,capture_output=True,timeout=60);state.update(audit_returncode=z.returncode,audit_stdout=z.stdout,audit_stderr=z.stderr)
 finally:state['post_KFD']=run_cpu('Deep-C4-smallT-Model-post-r9254',idle=True);save()
 state['qualified_terminal']=state['GPU_terminal'][2:4]==['COMPLETED','0:0']and state.get('audit_returncode')==0;state['completed']=True;state['phase']='completed';save()
except BaseException as e:
 state['error']=repr(e);state['traceback']=traceback.format_exc();state['phase']='failed';save()
 # Never signal a parent or unrelated step. Native/model shell retains own hard timeout.
 if gpu is not None and gpu.poll()is not None and not state.get('post_KFD'):
  try:state['post_KFD']=run_cpu('Deep-C4-smallT-Model-exception-post-r9254',idle=True);save()
  except BaseException as x:state['post_error']=repr(x);save()
 raise
