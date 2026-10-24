"""Single bounded finite component. Must only be invoked after root GPU GO."""
import pathlib,json,hashlib,subprocess,datetime,sys,time
r=pathlib.Path.home()/'hygon-inference-refactor-20261014'
label='deep-activation-r8842c';parent='910374';node='a02r4n05'
record=r/(label+'-launch.json');watchpath=r/(label+'-watch.json')
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def norm(s):return s.split()[0].rstrip('+')
def write(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def accounting():return [v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobID,JobName%100,State%40,ExitCode,NodeList%150'],text=True,timeout=30).splitlines()]
def idle(suffix):
 z=subprocess.run(['srun','--jobid='+parent,'--overlap','--exact','-N1','-n1','-c1','-w',node,'--gres=none','--mem=256M','-t','00:02:00','--job-name='+label+'-'+suffix,'python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')],capture_output=True,text=True,timeout=145)
 assert z.returncode==0,(z.stdout,z.stderr)
 rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert len(rows)==1 and rows[0]['passed'];return rows[0]
if len(sys.argv)==1:
 assert not record.exists()and not watchpath.exists()
 d={'created_utc':now(),'state':'intent','label':label,'parent':parent,'node':node,'scope':'OriginalGemmExmode2PACKONscatterONbankleaseON8independentTP0real591,activationTLS0vs1,ABBA+BAAB,noModelTPS'}
 with record.open('x')as f:json.dump(d,f,indent=2)
 with(r/'logs'/(label+'-controller.log')).open('x')as f:p=subprocess.Popen([sys.executable,__file__,'controller'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps({'controller_pid':p.pid,'label':label,'created_utc':d['created_utc']}));raise SystemExit
if sys.argv[1]=='watch':
 w={'ready':True,'label':label,'actual_handle':None,'terminal':None,'post_KFD':None,'read_errors':0};write(watchpath,w)
 while True:
  try:rows=accounting()
  except(subprocess.TimeoutExpired,subprocess.CalledProcessError)as e:
   w['read_errors']+=1;w['last_read_error']=repr(e);write(watchpath,w);time.sleep(10);continue
  matching=[v for v in rows if v[1]==label];assert len(matching)<=1,'duplicate actual handle'
  if matching:
   v=matching[0]
   if w['actual_handle']is None:w['actual_handle']=v[0]
   assert w['actual_handle']==v[0],'handle changed'
   w['actual']={'JobID':v[0],'State':v[2],'ExitCode':v[3],'NodeList':v[4]};write(watchpath,w)
   if norm(v[2])in terminal:
    w['terminal']=dict(w['actual'],observed_utc=now());write(watchpath,w)
    try:
     p=r/'results'/(label+'-'+v[0]+'.json');deadline=time.monotonic()+30
     while not p.exists()and time.monotonic()<deadline:time.sleep(1)
     w['terminal_report_wait_seconds']=30-max(0,deadline-time.monotonic());w['result_exists']=p.exists()
     if p.exists():
      z=subprocess.run([sys.executable,str(r/'incoming/verify_Deep_activation_r8842c.py'),str(p)],capture_output=True,text=True,timeout=60)
      w['CPU_result_audit_returncode']=z.returncode;w['CPU_result_audit_stdout']=z.stdout;w['CPU_result_audit_stderr']=z.stderr
      w['result_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
    except BaseException as e:w['audit_error']=repr(e)
    finally:
     try:w['post_KFD']=idle('post')
     except BaseException as e:w['post_KFD_error']=repr(e)
     log=r/'logs'/(label+'-'+v[0]+'.log')
     if log.exists():
      lines=log.read_text(errors='replace').splitlines();w['error_warning_lines']=[s for s in lines if any(k in s.lower()for k in ['warning','error','traceback','failed','709'])][-100:]
     w['qualified_terminal_result']=norm(v[2])=='COMPLETED'and v[3]=='0:0'and w.get('CPU_result_audit_returncode')==0 and bool(w.get('post_KFD',{}).get('passed'))
     write(watchpath,w)
    break
  elif json.loads(record.read_text())['state']=='failed_before_GPU':w['no_GPU_submitted']=True;write(watchpath,w);break
  time.sleep(10)
 raise SystemExit
assert sys.argv[1]=='controller';d=json.loads(record.read_text())
try:
 m=json.loads((r/'incoming/Deep-activation-manifest-r8842c.json').read_text())
 for group,base in [('incoming_source_SHA256',r/'incoming'),('existing_incoming_dependency_SHA256',r/'incoming'),('existing_remote_SHA256',r)]:
  for p,h in m[group].items():assert hashlib.sha256((base/p).read_bytes()).hexdigest()==h,p
 rows=accounting();assert any(v[0]==parent and norm(v[2])=='RUNNING'for v in rows)
 assert node in subprocess.check_output(['scontrol','show','hostnames',next(v[4]for v in rows if v[0]==parent)],text=True,timeout=20).splitlines()
 busy=[]
 for v in rows:
  if v[0].startswith(parent+'.')and v[0].split('.')[-1].isdigit()and norm(v[2])not in terminal:
   if node in subprocess.check_output(['scontrol','show','hostnames',v[4]],text=True,timeout=20).splitlines():busy.append(v[0])
 assert not busy,busy
 assert not any(v[1]==label for v in rows),'do not replay numeric'
 d['fresh_before_KFD']=idle('before');write(record,d)
 with(r/'logs'/(label+'-watcher.log')).open('x')as f:w=subprocess.Popen([sys.executable,__file__,'watch'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 limit=time.monotonic()+30
 while not watchpath.exists():assert time.monotonic()<limit and w.poll()is None;time.sleep(.1)
 ready=json.loads(watchpath.read_text());assert ready['ready']and ready['label']==label and w.poll()is None
 args=['srun','--jobid='+parent,'--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c32','-w',node,'--gres=dcu:8','--mem=64000M','-t','00:10:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=Deep_activation_r8842c_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 assert w.poll()is None
 with(r/'logs'/(label+'-srun.log')).open('x')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d.update(state='GPU_submitted_follow_same_handle',pid=p.pid,watcher_pid=w.pid,submitted_utc=now());write(record,d)
 rc=p.wait(timeout=800);d.update(state='GPU_srun_returned_watch_same_handle',srun_returncode=rc,returned_utc=now());write(record,d)
except BaseException as e:
 d.update(error=repr(e),state='unknown_GPU_follow_same_handle'if d.get('pid')else'failed_before_GPU');write(record,d);raise
