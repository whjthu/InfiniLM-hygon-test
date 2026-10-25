import pathlib,json,time,os,socket,subprocess,datetime,hashlib
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='deep-eos-compaction-r9130d';out=r/(label+'-supervisor.json');assert not out.exists()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):
 try:return json.loads(p.read_bytes())
 except(OSError,ValueError):return None
def save(d):
 t=out.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(out)
state={'hostname':socket.gethostname(),'PID':os.getpid(),'CPU_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'created_UTC':now(),'status':'launching_once','restart':False,'heartbeats':0,'model_handle':None};save(state)
source=r/'incoming/launch_Deep_eos_compaction_r9130d.py';q=subprocess.run(['python3',str(source)],capture_output=True,text=True,timeout=90);state.update(launcher_returncode=q.returncode,launcher_stdout=q.stdout,launcher_stderr=q.stderr);save(state);assert q.returncode==0
start=time.monotonic()
while time.monotonic()-start<4000:
 launch=read(r/(label+'-launch.json'));watch=read(r/(label+'-watch.json'));state.update(UTC=now(),status='observing_same_attempt',heartbeats=state['heartbeats']+1,launch_state=launch.get('state')if launch else None,controller_PID=launch.get('controller_pid')if launch else None)
 if watch:
  state.update(model_handle=watch.get('actual_handle'),terminal=watch.get('terminal'),post_KFD=watch.get('post_KFD'),audit_returncode=watch.get('CPU_result_audit_returncode'),qualified=watch.get('qualified_terminal_result'))
  if watch.get('no_GPU_submitted'):state['status']='failed_before_GPU';save(state);break
  if watch.get('terminal')and(watch.get('post_KFD')is not None or watch.get('post_KFD_error')is not None):state['status']='terminal_observed_with_original_cleanup';save(state);break
 save(state);time.sleep(15)
else:state.update(status='supervisor_bound_reached_no_model_restart',UTC=now());save(state)
