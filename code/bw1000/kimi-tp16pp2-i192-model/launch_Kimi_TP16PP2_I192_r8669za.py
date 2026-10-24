"""Single-use controller + independently ready terminal watcher. No auto replay.

Run with no args ONLY after root source GO. Watcher ready precedes Model Popen;
explicit actual Slurm terminal -> postKFD on all4nodes even on failed Model.
Transient observation errors persist and follow SAME actual handle.
"""
import pathlib,json,hashlib,subprocess,datetime,sys,os,time,shutil
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
record=r/'Kimi-TP16PP2-I192-r8669za-launch.json';wp=r/'Kimi-TP16PP2-I192-r8669za-watch.json'
mp=r/'incoming/Kimi-TP16PP2-I192-manifest-r8669za.json';m=json.loads(mp.read_text());nodes=m['nodes'];parent='910375'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def write(p,d):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)
def validate():
 p=r/'results/Kimi-TP16-head-CPU-r8669z.json';assert p.is_file();assert sha(p)==m['CPU_receipt_SHA256'];b=json.loads(p.read_text());assert b['completed']and b['CPU_only']and not b['GPU']and not b['dlopen'];assert b['native_SHA256']==m['native_DSO_SHA256']
 for n,h in m['source_sha256'].items():assert sha(r/'incoming'/n)==h,n
 for n,h in m['existing_dependency_sha256'].items():assert sha(r/'incoming'/n)==h,n
 for n,h in m['remote_evidence_sha256'].items():assert sha(r/n)==h,n
def accounting():
 out=subprocess.check_output(['sacct','-S','2026-10-23','-j',parent,'-n','-P','-o','JobID,JobName%110,State%40,ExitCode,NodeList%110'],text=True,timeout=30)
 return [dict(zip(['JobID','JobName','State','ExitCode','NodeList'],v.split('|'))) for v in out.splitlines() if v]
def norm(state):return state.split()[0].split('+')[0]
def idle(label):
 args=['srun','--jobid='+parent,'--overlap','--exact','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(nodes),'-t','00:02:00','--job-name='+label,'python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')]
 z=subprocess.run(args,capture_output=True,text=True,timeout=160)
 rows=[json.loads(v) for v in z.stdout.splitlines() if v.startswith('{')]
 assert z.returncode==0 and len(rows)==4 and {v['node'] for v in rows}==set(nodes) and all(v['passed'] and not v['kfd_pids'] for v in rows),(z.stdout[-2000:],z.stderr[-2000:])
 return rows
if len(sys.argv)==1:
 validate();assert not record.exists() and not wp.exists()
 d=dict(run='TP16PP2-I1928669za-'+os.urandom(8).hex(),nonce=os.urandom(16).hex(),state='intent',nodes=nodes,manifest_SHA256=sha(mp),controller_SHA256=sha(pathlib.Path(__file__)))
 with record.open('x')as f:json.dump(d,f,indent=2)
 with(r/'logs/Kimi-TP16PP2-I192-r8669za-controller.log').open('x')as f:p=subprocess.Popen([sys.executable,__file__,'controller'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(run=d['run'],state=d['state'],nodes=nodes,controller_pid=p.pid)));raise SystemExit
d=json.loads(record.read_text());validate();assert sha(mp)==d['manifest_SHA256'] and sha(pathlib.Path(__file__))==d['controller_SHA256']
run=d['run'];jobname='Kimi-TP16PP2-I192-'+run
if sys.argv[1]=='watch':
 w=dict(ready=True,run=run,actual_handle=None,terminal=None,post_KFD_passed=False,observation_errors=0)
 with wp.open('x')as f:json.dump(w,f,indent=2)
 while True:
  try:
   try:rows=[v for v in accounting() if v['JobName']==jobname]
   except (subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError) as exc:
    w['observation_errors']+=1;w['last_observation_error']=repr(exc);write(wp,w);time.sleep(15);continue
   assert len(rows)<=1,rows
   if rows:
    row=rows[0]
    if w['actual_handle'] is None:w['actual_handle']=row['JobID']
    assert w['actual_handle']==row['JobID']
    w['last_observation']=row;write(wp,w)
    if norm(row['State']) in TERMINAL:
     w['terminal']=dict(row,State=norm(row['State']));write(wp,w)
     try:
      w['post_KFD']=idle('TP16PP2-I192-post-r8669z');w['post_KFD_passed']=True
     except BaseException as exc:w['post_KFD_error']=repr(exc)
     write(wp,w)
     try:
      z=subprocess.run([sys.executable,str(r/'incoming/audit_Kimi_TP16PP2_I192_r8669za.py')],capture_output=True,text=True,timeout=120)
      w['audit_exit']=z.returncode;w['audit_stdout']=z.stdout[-3000:];w['audit_stderr']=z.stderr[-3000:]
     except BaseException as exc:w['audit_error']=repr(exc)
     write(wp,w);break
   elif json.loads(record.read_text()).get('state')=='launch_failed_before_Model':
    w['no_Model_launched']=True;write(wp,w);break
  except BaseException as exc:
   w['fatal_observer_error']=repr(exc);write(wp,w);raise
  time.sleep(15)
 raise SystemExit
assert sys.argv[1]=='controller' and d['state']=='intent'
try:
 rows=accounting();assert any(v['JobID']==parent and norm(v['State'])=='RUNNING' for v in rows)
 for v in rows:
  if v['JobID'].startswith(parent+'.') and v['JobID'].split('.')[-1].isdigit() and norm(v['State']) not in TERMINAL:
   occupied=subprocess.check_output(['scontrol','show','hostnames',v['NodeList']],text=True,timeout=20).splitlines()
   assert not set(occupied)&set(nodes),v
 before=idle('TP16PP2-I192-before-r8669z');head=next(v['ib0_ipv4'][0] for v in before if v['node']==nodes[0]);port=19000+int(d['nonce'][:5],16)%500
 env='ALL,KIMI_USE_RAMCACHE_R3679=0,KIMI_REDUCE_POLICY=0,INFINILM_OWNER_CPU_WIDTH_R2392=4,INFINILM_DENSE_PREFILL_F32_R2475=1,INFINILM_MXFP4_TILE_TP_R2559=0,GLOBAL_OFFSET_R2354=0,GLOBAL_RUN_R2354='+run+',GLOBAL_PORT_R2354='+str(port)+',KIMI_ENGINE_HEAD='+head+',KIMI_ENGINE_NONCE='+d['nonce']+',KIMI_TP_NODES=2,KIMI_PP_SIZE=2,GLOBAL_MODEL_MODE_R2354=shca'
 def args(inside,label,limit,cpus):return ['srun','--export='+env+',WAVE_INSIDE_R2323='+inside,'--jobid='+parent,'--overlap','--exact','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c'+str(cpus),'-w',','.join(nodes),'--gres=dcu:8','--mem=240000M','-t',limit,'--job-name='+label,'--output='+str(r/'logs'/(label+'-%J-%t.log')),'bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 # One fresh candidate Model, inherited loader/cleanup; no second preengine load.
 with(r/'logs/Kimi-TP16PP2-I192-r8669za-watcher.log').open('x')as f:watch=subprocess.Popen([sys.executable,__file__,'watch'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 deadline=time.monotonic()+30
 while not wp.exists():
  assert time.monotonic()<deadline and watch.poll()is None;time.sleep(.1)
 ready=json.loads(wp.read_text());assert ready['ready'] and ready['run']==run and watch.poll() is None;d['watcher_pid']=watch.pid;d['state']='Model_launch_intent';write(record,d)
 a=args('Kimi_TP16PP2_I192_r8669za_inside.sh',jobname,'00:50:00',64)
 assert watch.poll() is None, 'same ready watcher exited before Model Popen'
 with(r/'logs/Kimi-TP16PP2-I192-r8669za-Model.log').open('x')as f:p=subprocess.Popen(a,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d.update(state='Model_submitted_follow_same_handle',Model_srun_pid=p.pid,Model_args=a);write(record,d)
 rc=p.wait(timeout=3110);d.update(state='srun_returned_watch_same_handle',Model_srun_returncode=rc);write(record,d)
except BaseException as exc:
 d['error']=repr(exc)
 d['state']='unknown_Model_execution_watch_same_handle' if d.get('Model_srun_pid') else 'launch_failed_before_Model'
 write(record,d);raise
