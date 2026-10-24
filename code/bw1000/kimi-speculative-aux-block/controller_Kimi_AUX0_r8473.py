"""Single-use controller and independent same-handle watcher; never resubmit."""
import pathlib,json,hashlib,subprocess,time,sys,os,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';i=r/'incoming'
mp=i/'Kimi-AUX0-runtime-manifest-r8473.json';m=json.loads(mp.read_text())
record=r/'Kimi-AUX0-r8473-launch.json';wp=r/'Kimi-AUX0-r8473-watch.json'
ready_path=r/'Kimi-AUX0-r8473-current-run.json'
parent='910374';node='a02r4n04';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def norm(x):return x.split()[0].split('+')[0]
def write(p,q):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(p)
def validate():
 assert m['resource_contract']['parent']==parent and m['resource_contract']['node']==node
 for name,h in m['sources_SHA256'].items():assert sha(i/name)==h,name
 for name,h in m['existing_dependencies_SHA256'].items():assert sha(r/name)==h,name
 assert sha(r/'candidates/kimi-K7-MoE-CPU-r8104/libmodern_Kimi_K7_MoE_r8104.so')==m['DSO_SHA256']
 for rank in [0,6]:assert sha(r/'results'/f'trained-features-r4809-trainedModal4809-f71640dc8e2f73073d20-call396/stage0-rank{rank}-layer20-ffn_norm-full-bf16.bin')==m['fixture_SHA256']
def accounting():
 text=subprocess.check_output(['sacct','-S','2026-10-23','-j','910374,910375,910377,910379,910381,910384','-n','-P','-o','JobID,JobName%110,State%40,ExitCode,NodeList%160'],text=True,timeout=30)
 return [dict(zip(['JobID','JobName','State','ExitCode','NodeList'],v.split('|')))for v in text.splitlines()if v]
def check_idle_slurm(rows):
 assert any(v['JobID']==parent and norm(v['State'])=='RUNNING'for v in rows)
 for v in rows:
  if '.'in v['JobID'] and v['JobID'].split('.')[-1].isdigit() and norm(v['State'])not in TERMINAL:
   occupied=subprocess.check_output(['scontrol','show','hostnames',v['NodeList']],text=True,timeout=20).splitlines()
   assert node not in occupied,('node_busy',v)
def idle(label):
 args=['srun','--jobid='+parent,'--overlap','--exact','-N1','-n1','-c1','--gres=none','--mem=512M','-w',node,'-t','00:02:00','--job-name='+label,'python3',str(i/'Kimi_Q0_idle_r8066.py')]
 z=subprocess.run(args,capture_output=True,text=True,timeout=150)
 rows=[json.loads(v)for v in z.stdout.splitlines()if v.startswith('{')]
 assert z.returncode==0 and len(rows)==1 and rows[0]['node']==node and rows[0]['passed'] and rows[0]['kfd_pids']==[],(z.stdout,z.stderr)
 return rows[0]
validate()
if len(sys.argv)==1:
 assert os.environ.get('KIMI_AUX0_ROOT_GO_R8473')=='1','Final source GO required; no upload/launch from CPU preparation'
 assert all(not p.exists()for p in [record,wp,ready_path,r/'results/Kimi-speculative-AUX0-r8473.run.lock'])
 d={'run':'AUX0r8473-'+os.urandom(8).hex(),'state':'intent','manifest_SHA256':sha(mp),'controller_SHA256':sha(pathlib.Path(__file__)),'numeric_attempts':0,'postKFD_attempts':0}
 with record.open('x')as f:json.dump(d,f,indent=2)
 with(r/'logs/Kimi-AUX0-r8473-controller.log').open('x')as f:p=subprocess.Popen([sys.executable,__file__,'controller'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps({'controller_pid':p.pid,'run':d['run'],'state':'intent'}));raise SystemExit
d=json.loads(record.read_text());assert sha(mp)==d['manifest_SHA256'] and sha(pathlib.Path(__file__))==d['controller_SHA256']
run=d['run'];jobname='Kimi-'+run
if sys.argv[1]=='watch':
 w={'run':run,'watch_owner':run,'ready':True,'watcher_pid':os.getpid(),'actual_handle':None,'terminal':None,'postKFD_attempts':0,'observation_errors':[],'no_resubmit':True}
 with wp.open('x')as f:json.dump(w,f,indent=2)
 start=time.monotonic()
 while time.monotonic()-start<1200:
  try:
   rows=[v for v in accounting()if v['JobName']==jobname];assert len(rows)<=1,rows
   if rows:
    row=rows[0]
    if w['actual_handle']is None:w['actual_handle']=row['JobID']
    assert w['actual_handle']==row['JobID'];w['last_observation']=row;write(wp,w)
    if norm(row['State'])in TERMINAL:
     w['terminal']=dict(row,State=norm(row['State']));write(wp,w)
     # One post KFD attempt, only after original numeric actual handle is terminal.
     w['postKFD_attempts']=1;write(wp,w)
     try:w['postKFD']=idle('Kimi-AUX0-post-r8473');w['postKFD_passed']=True
     except BaseException as exc:w['postKFD_passed']=False;w['postKFD_error']=repr(exc)
     output=r/'results/kimi-moe-AUX0-r8473-a02r4n04-L20-C396.json'
     if output.exists():
      q=json.loads(output.read_text());w['component_report_SHA256']=sha(output);w['component_phase']=q.get('phase');w['CPU_memory_guard']=q.get('CPU_memory_guard')
      w['component_completed']=q.get('completed',False);w['tracefree_ABBA']=q.get('tracefree_ABBA');w['differences']=q.get('comparisons');w['Model_performance_adopted']=False
      w['all_four_native_owner_closures']=len(q.get('arms',[]))==4 and all(a['producer_report'].get('normal_all8_Runtime_owners_closed_before_communicators')and a['producer_report'].get('communication_group_destroyed')for a in q.get('arms',[]))
     w['status']='TERMINAL_POSTKFD_VERIFIED'if w.get('postKFD_passed')else'UNKNOWN_POSTKFD';write(wp,w);break
   elif json.loads(record.read_text()).get('state')=='launch_failed_before_numeric':
    w['status']='NOT_RUN';w['no_numeric_launched']=True;write(wp,w);break
  except BaseException as exc:w['observation_errors'].append({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':repr(exc)});write(wp,w)
  time.sleep(10)
 else:w['status']='UNKNOWN_OBSERVATION_TIMEOUT_SAME_HANDLE_ONLY';write(wp,w)
 raise SystemExit
assert sys.argv[1]=='controller' and d['state']=='intent'
try:
 rows=accounting();check_idle_slurm(rows);pre_tick=time.monotonic();before=idle('Kimi-AUX0-pre-r8473')
 with(r/'logs/Kimi-AUX0-r8473-watcher.log').open('x')as f:watch=subprocess.Popen([sys.executable,__file__,'watch'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 until=time.monotonic()+30
 while not wp.exists():
  assert watch.poll()is None and time.monotonic()<until;time.sleep(.1)
 w=json.loads(wp.read_text());assert w['ready'] and w['run']==run and w['watcher_pid']==watch.pid and watch.poll()is None
 # Refresh Slurm after watcher readiness; immutable current-run receipt precedes Popen.
 fresh=accounting();check_idle_slurm(fresh);assert time.monotonic()-pre_tick<=60,'preKFD not fresh; no numeric launch';now=time.time()
 current={'job_id':parent,'node':node,'watch_owner':run,'watcher_pid':watch.pid,'fresh_Slurm_no_conflict':True,'Slurm_rows':fresh,'whole_node_KFD_empty':True,'preKFD':before,'watch_ready':True,'lease_seconds':600,'issued_unix':now,'expires_unix':now+600,'manifest_SHA256':sha(mp),'source_SHA256':m['sources_SHA256']}
 with ready_path.open('x')as f:json.dump(current,f,indent=2)
 args=['srun','--jobid='+parent,'--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c64','-w',node,'--gres=dcu:8','--mem=128000M','-t','00:10:00','--job-name='+jobname,'--output='+str(r/'logs'/(jobname+'-%J.log')),'env','KIMI_AUX0_ROOT_GO_R8473=1','KIMI_AUX0_WATCH_OWNER_R8473='+run,'KIMI_AUX0_READY_RECEIPT_R8473=/refactor/Kimi-AUX0-r8473-current-run.json','WAVE_INSIDE_R2323=Kimi_AUX0_r8473_inside.sh','bash',str(i/'resource_wave_local_r3097_outer.sh'),'probe']
 d.update(state='numeric_launch_intent',watcher_pid=watch.pid,current_run_SHA256=sha(ready_path),args=args);write(record,d);assert watch.poll()is None
 with(r/'logs/Kimi-AUX0-r8473-numeric.log').open('x')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d.update(state='numeric_submitted_follow_same_handle',numeric_attempts=1,numeric_pid=p.pid);write(record,d)
 d['numeric_returncode']=p.wait(timeout=670);d['state']='srun_returned_watch_same_handle';write(record,d)
except BaseException as exc:
 d['error']=repr(exc);d['state']='UNKNOWN_numeric_execution_watch_same_handle'if d.get('numeric_pid')else'launch_failed_before_numeric';write(record,d);raise
