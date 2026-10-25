import pathlib,json,time,os,socket,subprocess,datetime,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-eos-compaction-r9130e';out=r/(label+'-supervisor.json');assert not out.exists()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):
 try:return json.loads(p.read_bytes())
 except(OSError,ValueError):return None
def save(d):
 t=out.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(out)
state={'hostname':socket.gethostname(),'PID':os.getpid(),'CPU_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'created_UTC':now(),'status':'launching_once','restart':False,'heartbeats':0,'model_handle':None};save(state)
child_env={k:v for k,v in os.environ.items()if not(k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ('SLURM_CONF','SLURM_CONF_SERVER'))and k not in ('CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL')}
probe_name='Deep-eos-affinity-r9130e';args=['srun','--jobid=910374','--overlap','--exact','-N1','-n1','-c64','-w','a02r4n05','--cpu-bind=none','--gres=none','--mem=512M','-t','00:02:00','--job-name='+probe_name,'env','WAVE_INSIDE_R2323=CPU_affinity_Deep_eos_r9130e_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];probe=subprocess.run(args,capture_output=True,text=True,timeout=145,env=child_env);state.update(probe_returncode=probe.returncode,probe_stdout=probe.stdout,probe_stderr=probe.stderr);save(state);assert probe.returncode==0
checks=[json.loads(s)for s in probe.stdout.splitlines()if s.startswith('{')];checks=[v for v in checks if v.get('check')=='8owner_CPUmask_beforeload'];assert len(checks)==1 and checks[0]['passed'];handle=checks[0]['step'];state['actual_CPU_affinity_probe']=checks[0];save(state)
for _ in range(60):
 rows=[v.split('|')for v in subprocess.check_output(['sacct','-j',handle,'-n','-P','-o','JobID,State,ExitCode'],text=True,timeout=20,env=child_env).splitlines()];exact=[v for v in rows if v[0]==handle]
 if exact and exact[0][1]=='COMPLETED'and exact[0][2]=='0:0':state['probe_authoritative_terminal']=exact[0];save(state);break
 time.sleep(1)
else:raise RuntimeError('same affinity probe not authoritative completed; no GPU launch')
source=r/'incoming/launch_Deep_eos_compaction_r9130e.py';q=subprocess.run(['python3',str(source)],capture_output=True,text=True,timeout=90,env=child_env);state.update(launcher_returncode=q.returncode,launcher_stdout=q.stdout,launcher_stderr=q.stderr);save(state);assert q.returncode==0
start=time.monotonic()
while time.monotonic()-start<4000:
 launch=read(r/(label+'-launch.json'));watch=read(r/(label+'-watch.json'));state.update(UTC=now(),status='observing_same_attempt',heartbeats=state['heartbeats']+1,launch_state=launch.get('state')if launch else None,controller_PID=launch.get('controller_pid')if launch else None)
 if watch:
  state.update(model_handle=watch.get('actual_handle'),terminal=watch.get('terminal'),post_KFD=watch.get('post_KFD'),audit_returncode=watch.get('CPU_result_audit_returncode'),qualified=watch.get('qualified_terminal_result'))
  if watch.get('no_GPU_submitted'):state['status']='failed_before_GPU';save(state);break
  if watch.get('terminal')and(watch.get('post_KFD')is not None or watch.get('post_KFD_error')is not None):state['status']='terminal_observed_with_original_cleanup';save(state);break
 save(state);time.sleep(15)
else:state.update(status='supervisor_bound_reached_no_model_restart',UTC=now());save(state)
