"""Keep immutable launcher/controller/watcher inside one real CPU Slurm step."""
import pathlib,json,subprocess,datetime,time,socket,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';src=r/'incoming/Kimi-speculative-batch-Model-r9122b2/runtime';health=r/'results/Kimi-batch-control-r9122b2-health.json';assert not health.exists();q={'hostname':socket.gethostname(),'PID':os.getpid(),'CPU_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'CPU_job_name':os.environ.get('SLURM_JOB_NAME'),'frozen_launch_unmodified':True,'completed':False,'state':'starting_unique_launcher'}
def save():
 q['heartbeat_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=health.with_suffix('.tmp');p.write_text(json.dumps(q,indent=2));p.replace(health)
save();started=time.monotonic()
try:
 child_env=dict(os.environ);removed=[]
 for k in list(child_env):
  if ((k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])or k in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES']):removed.append(k);child_env.pop(k)
 q['clean_child_environment_removed_names']=removed;save()
 args=['srun','--jobid=910380','--overlap','--exact','-N4','-n4','--ntasks-per-node=1','-c1','-w','a14r2n08,a14r2n09,a14r2n10,a14r2n12','--gres=none','--mem=512M','-t','00:02:00','--job-name=Kimi-batch-crossalloc-r9122b2','python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')]
 z=subprocess.run(args,env=child_env,capture_output=True,text=True,timeout=160);rows=[json.loads(v)for v in z.stdout.splitlines()if v.startswith('{')];q['CPU_crossallocation_probe']={'returncode':z.returncode,'rows':rows,'stderr':z.stderr};save();assert z.returncode==0 and len(rows)==4 and all(v['passed']and not v['kfd_pids']for v in rows),(z.stdout,z.stderr)
 probe_deadline=time.monotonic()+90
 while True:
  lines=subprocess.check_output(['sacct','-S','2026-10-25','-j','910380','-n','-P','-o','JobID,JobName%90,State,ExitCode'],env=child_env,text=True).splitlines();own=[v.split('|')for v in lines if v.split('|')[1]=='Kimi-batch-crossalloc-r9122b2'];assert len(own)<=1
  if own:
   q['CPU_crossallocation_actual_Slurm']=own[0];save()
   if own[0][2].split()[0].rstrip('+')not in ['RUNNING','PENDING','COMPLETING','CONFIGURING']:assert own[0][2:4]==['COMPLETED','0:0'];break
  assert time.monotonic()<probe_deadline,'same crossallocationprobe terminal timeout';time.sleep(2)
 z=subprocess.run(['python3',str(src/'launch.py')],env=child_env,capture_output=True,text=True,timeout=180);q.update(launcher_returncode=z.returncode,launcher_stdout=z.stdout,launcher_stderr=z.stderr,state='following_same_unique_launch');save();assert z.returncode==0
 launcher_info=json.loads(z.stdout.strip().splitlines()[-1]);q['launcher_controller_PID']=launcher_info.get('controller_pid');save()
 while True:
  a=r/'Kimi-batch-TP16PP2-r9122b2-launch.json';b=r/'Kimi-batch-TP16PP2-r9122b2-watch.json';d=json.loads(a.read_bytes())if a.exists()else{};w=json.loads(b.read_bytes())if b.exists()else{}
  q.update(run=d.get('run'),controller_PID=d.get('controller_pid')or q.get('launcher_controller_PID'),watcher_PID=d.get('watcher_pid'),draft_handle=d.get('draft_handle'),target_handle=w.get('actual_handle'),launch_state=d.get('state'),target_state=w.get('last_observation'),watch_file_mtime_ns=b.stat().st_mtime_ns if b.exists()else None,draft_post=d.get('draft_post'),watch_terminal=w.get('terminal'),post_KFD_passed=w.get('post_KFD_passed'),audit_exit=w.get('audit_exit'));save()
  if d.get('state')=='launch_failed_before_Model'and(not d.get('draft_srun_pid')or d.get('draft_post')or d.get('draft_cleanup_error')):q['completed']=True;q['state']='launch_failed_before_Model';save();break
  if w.get('terminal')and' audit_exit'.strip()in w and(d.get('draft_post')or d.get('draft_cleanup_error')):q['completed']=True;q['state']='terminal_and_cleanup_observed';save();break
  assert time.monotonic()-started<5900,'CPU controller maximum budget'
  time.sleep(15)
except BaseException as e:q.update(error=repr(e),state='CPU_controller_error');save();raise
print(json.dumps(q))
