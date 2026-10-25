"""Single-use finite raw caller diagnosis; execution requires root GO/sourceclaim."""
import pathlib,json,hashlib,subprocess,sys,time,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='Kimi-raw-W13-egress-r8669aac';parent='910374';node='a02r4n01';record=r/(label+'-launch.json');wp=r/(label+'-watch.json');terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}
def write(p,q):tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(p)
def rows():return [v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobID,JobName%100,State%40,ExitCode,NodeList%100'],text=True,timeout=30).splitlines()]
def norm(s):return s.split()[0].rstrip('+')
def idle(suffix):
 z=subprocess.run(['srun','--jobid='+parent,'--overlap','--exact','-N1','-n1','-c1','-w',node,'--gres=none','--mem=256M','-t','00:02:00','--job-name='+label+'-'+suffix,'python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')],capture_output=True,text=True,timeout=145);v=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert z.returncode==0 and len(v)==1 and v[0]['passed']and not v[0]['kfd_pids'];return v
if len(sys.argv)==1:
 assert not record.exists()and not wp.exists();write(record,{'state':'intent','label':label,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 with(r/'logs'/(label+'-controller.log')).open('x')as f:p=subprocess.Popen([sys.executable,__file__,'controller'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps({'controller_pid':p.pid,'label':label,'state':'intent'}));raise SystemExit
if sys.argv[1]=='watch':
 w={'ready':True,'label':label,'actual_handle':None,'terminal':None};write(wp,w)
 while True:
  try:
   found=[v for v in rows()if v[1]==label]
   if found:
    assert len(found)==1;v=found[0]
    if w['actual_handle']is None:w['actual_handle']=v[0]
    assert w['actual_handle']==v[0];w['actual']={'JobID':v[0],'State':v[2],'ExitCode':v[3],'NodeList':v[4]};write(wp,w)
    if norm(v[2])in terminal:
     w['terminal']=w['actual'];write(wp,w)
     try:w['post_KFD']=idle('post')
     except BaseException as e:w['post_KFD_error']=repr(e)
     p=r/'results'/f'{label}-{v[0]}.json'
     if p.exists():
      raw=p.read_bytes();q=json.loads(raw);w['result_SHA256']=hashlib.sha256(raw).hexdigest();w['diagnostic_summary']={k:q.get(k)for k in ['completed','returncode','error','normal_owner_close','canary_intact','maxabs','RMSE','relativeRMS','first_largest','group_count']}
     w['audit']={'normal_owner1':bool(w.get('diagnostic_summary',{}).get('normal_owner_close')),'post_empty':bool(w.get('post_KFD')),'Model_or_performance_claim':False};write(wp,w);break
   elif json.loads(record.read_text()).get('state')=='failed_before_GPU':w['no_GPU_submitted']=True;write(wp,w);break
  except BaseException as e:w['observer_error']=repr(e);write(wp,w)
  time.sleep(5)
 raise SystemExit
d=json.loads(record.read_text())
try:
 m=json.loads((r/'incoming/Kimi-raw-W13-egress-manifest-r8669aac.json').read_text())
 for n,h in m['source_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
 assert hashlib.sha256((r/pathlib.Path(m['native']).relative_to('/refactor')).read_bytes()).hexdigest()==m['native_SHA256']
 a=rows();assert any(v[0]==parent and norm(v[2])=='RUNNING'for v in a);assert not any(v[1]==label for v in a)
 for v in a:
  if v[0].startswith(parent+'.')and v[0].split('.')[-1].isdigit()and norm(v[2])not in terminal:assert node not in subprocess.check_output(['scontrol','show','hostnames',v[4]],text=True,timeout=20).splitlines(),('busy',v[0])
 d['fresh_before_KFD']=idle('before');write(record,d)
 with(r/'logs'/(label+'-watcher.log')).open('x')as f:w=subprocess.Popen([sys.executable,__file__,'watch'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 deadline=time.monotonic()+30
 while not wp.exists():assert time.monotonic()<deadline and w.poll()is None;time.sleep(.1)
 ready=json.loads(wp.read_text());assert ready['ready']and ready['label']==label and w.poll()is None
 args=['srun','--jobid='+parent,'--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c8','-w',node,'--gres=dcu:1','--mem=16000M','-t','00:03:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=Kimi_raw_W13_egress_r8669aac_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 assert w.poll()is None
 with(r/'logs'/(label+'-srun.log')).open('x')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d.update(state='GPU_submitted_follow_same_handle',pid=p.pid,watcher_pid=w.pid);write(record,d);rc=p.wait(timeout=230);d.update(state='GPU_returned_watch_same_handle',returncode=rc);write(record,d)
except BaseException as e:d.update(error=repr(e),state='unknown_GPU_follow_same_handle'if d.get('pid')else'failed_before_GPU');write(record,d);raise
