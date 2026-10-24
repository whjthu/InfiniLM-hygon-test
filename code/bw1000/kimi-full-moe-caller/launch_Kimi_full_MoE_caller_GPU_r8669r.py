"""One finite current fullMoE TP8 caller diagnosis. Root continuous GO."""
import pathlib,json,hashlib,subprocess,datetime,sys,time,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='Kimi-full-MoE-caller-r8669r';parent='910374';node='a02r4n01'
record=r/(label+'-launch.json');watchpath=r/(label+'-watch.json')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def accounting():return [v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobID,JobName%110,State%40,ExitCode,NodeList%150'],text=True,timeout=30).splitlines()]
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}
def norm(s):return s.split()[0].rstrip('+')
def idle(suffix):
 a=['srun','--jobid='+parent,'--overlap','--exact','-N1','-n1','-c1','-w',node,'--gres=none','--mem=256M','-t','00:02:00','--job-name='+label+'-'+suffix,'python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')]
 z=subprocess.run(a,capture_output=True,text=True,timeout=145);assert z.returncode==0,(z.stdout,z.stderr)
 rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert len(rows)==1 and rows[0]['passed'];return rows[0]
if len(sys.argv)==1:
 assert not record.exists() and not watchpath.exists()
 d=dict(created_utc=now(),state='intent',label=label,parent=parent,node=node,scope='CURRENT_FULL_MOE_TP8_CALLER_DIAGNOSTIC_NO_MODEL_OR_OPTIMIZATION')
 write(record,d)
 with(r/'logs'/(label+'-controller.log')).open('x')as f:p=subprocess.Popen([sys.executable,__file__,'controller'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps(dict(created_utc=d['created_utc'],controller_pid=p.pid,scope=d['scope'])));raise SystemExit
if sys.argv[1]=='watch':
 w=dict(ready=True,label=label,actual_handle=None,terminal=None,post_KFD=None,errors=0,created_utc=now());write(watchpath,w)
 while True:
  try:
   matching=[v for v in accounting()if v[1]==label]
   if matching:
    assert len(matching)==1;v=matching[0]
    if w['actual_handle']is None:w['actual_handle']=v[0]
    assert w['actual_handle']==v[0];w['actual']={'JobID':v[0],'State':v[2],'ExitCode':v[3],'NodeList':v[4]};write(watchpath,w)
    if norm(v[2])in terminal:
     w['terminal']=dict(w['actual'],observed_utc=now());write(watchpath,w)
     try:w['post_KFD']=idle('post')
     except BaseException as e:w['post_KFD_error']=repr(e)
     p=r/'results'/('Kimi-full-MoE-caller-r8669r-'+v[0]+'.json')
     if p.exists():w['result']=json.loads(p.read_text());w['result_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
     if 'result'in w:
      q=w['result'];audit={'scope':'CurrentfullMoEcallerTP8diagnostic','actual_handle':v[0],'terminal':w['terminal'],'normal_CUDA_cleanup':q.get('normal_CUDA_cleanup',False),'post_KFD_empty':bool(w.get('post_KFD',{}).get('passed')),'component_result':'PENDING_OFFLINE_ANALYSIS_NOT_MODEL' ,'Model_or_performance_claim':False,'result_SHA256':w['result_SHA256']};ap=r/'results'/(label+'-audit.json');assert not ap.exists();ap.write_text(json.dumps(audit,indent=2));w['audit']=audit
     write(watchpath,w);break
   elif json.loads(record.read_text())['state']=='failed_before_GPU':w['no_GPU_submitted']=True;write(watchpath,w);break
  except BaseException as e:w['errors']+=1;w['last_error']=repr(e);write(watchpath,w)
  time.sleep(10)
 raise SystemExit
assert sys.argv[1]=='controller';d=json.loads(record.read_text())
try:
 m=json.loads((r/'incoming/Kimi-full-MoE-caller-manifest-r8669r.json').read_text())
 for p,h in m['incoming_source_SHA256'].items():assert hashlib.sha256((r/'incoming'/p).read_bytes()).hexdigest()==h,p
 for p,h in m['existing_incoming_dependency_SHA256'].items():assert hashlib.sha256((r/'incoming'/p).read_bytes()).hexdigest()==h,p
 for p,h in m['existing_remote_SHA256'].items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
 rows=accounting();assert any(v[0]==parent and norm(v[2])=='RUNNING'for v in rows)
 while True:
  busy=[]
  for v in rows:
   if v[0].startswith(parent+'.')and v[0].split('.')[-1].isdigit()and norm(v[2])not in terminal:
    if node in subprocess.check_output(['scontrol','show','hostnames',v[4]],text=True,timeout=20).splitlines():busy.append(v[0])
  if not busy:break
  d.update(state='waiting_original_n01_busy_no_GPU_submitted',busy_handles=busy);write(record,d);time.sleep(60);rows=accounting()
 assert not any(v[1]==label for v in rows),'never replay same component'
 d['fresh_before_KFD']=idle('before');write(record,d)
 with(r/'logs'/(label+'-watcher.log')).open('x')as f:w=subprocess.Popen([sys.executable,__file__,'watch'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 limit=time.monotonic()+30
 while not watchpath.exists():assert time.monotonic()<limit and w.poll()is None;time.sleep(.1)
 ready=json.loads(watchpath.read_text());assert ready['ready']and ready['label']==label and w.poll()is None
 args=['srun','--jobid='+parent,'--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c64','-w',node,'--gres=dcu:8','--mem=128000M','-t','00:10:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=Kimi_full_MoE_caller_r8669r_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 assert w.poll()is None,'same ready watcher died before GPU Popen'
 with(r/'logs'/(label+'-srun.log')).open('x')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d.update(state='GPU_submitted_follow_same_handle',pid=p.pid,watcher_pid=w.pid,submitted_utc=now());write(record,d)
 rc=p.wait(timeout=750);d.update(state='GPU_srun_returned_watch_same_handle',srun_returncode=rc,returned_utc=now());write(record,d)
except BaseException as e:
 d['error']=repr(e);d['state']='unknown_GPU_follow_same_handle'if d.get('pid')else'failed_before_GPU';write(record,d);raise
