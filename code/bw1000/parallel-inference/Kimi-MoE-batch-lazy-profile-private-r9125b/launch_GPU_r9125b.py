"""One finite current fullMoE TP8 caller diagnosis. Root continuous GO."""
import pathlib,json,hashlib,subprocess,datetime,sys,time,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';label='Kimi-MoE-batch-lazy-profile-r9125b';parent='910376';node='a02r4n14,a02r4n15';nodes=set(node.split(','))
record=r/(label+'-launch.json');watchpath=r/(label+'-watch.json')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def accounting():return [v.split('|')for v in subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobID,JobName%110,State%40,ExitCode,NodeList%150'],text=True,timeout=30).splitlines()]
terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED','BOOT_FAIL','DEADLINE'}
def norm(s):return s.split()[0].rstrip('+')
def idle(suffix):
 a=['srun','--jobid='+parent,'--overlap','--exact','-N2','-n2','--ntasks-per-node=1','-c1','-w',node,'--gres=none','--mem=256M','-t','00:02:00','--job-name='+label+'-'+suffix,'python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')]
 z=subprocess.run(a,capture_output=True,text=True,timeout=145);assert z.returncode==0,(z.stdout,z.stderr)
 rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert len(rows)==2 and all(x['passed']for x in rows);return rows
if len(sys.argv)==1:
 assert not record.exists() and not watchpath.exists()
 d=dict(created_utc=now(),state='intent',label=label,parent=parent,node=node,scope='TP16_FULL_MOE_I192_SPECIALIZATION_ABBA_COMPONENT_NO_MODEL')
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
     results=[]
     for n in range(2):
      p=r/'results'/(label+'-'+v[0]+'-node'+str(n)+'.json')
      if p.exists():results.append({'node':n,'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'data':json.loads(p.read_text())})
     w['results']=results;normal=len(results)==2 and all(z['data'].get('completed')and z['data'].get('normal_CUDA_cleanup')and z['data'].get('native',{}).get('all8normal_before_communicators')for z in results)
     w['audit']={'scope':'TP16fullMoEcomponentonly','terminal':w['terminal'],'normal_owners':16 if normal else None,'normal_CUDA_cleanup':normal,'post_KFD_empty':len(w.get('post_KFD',[]))==2 and all(x['passed']for x in w.get('post_KFD',[])),'component_numerical_difference':'OFFLINE_DIAGNOSTIC_NOT_TOKEN_ACCEPTANCE','Model_or_TPS_claim':False,'result_SHA256':[z['SHA256']for z in results]}
     ap=r/'results'/(label+'-audit.json');assert not ap.exists();ap.write_text(json.dumps(w['audit'],indent=2))
     write(watchpath,w);break
   elif json.loads(record.read_text())['state']=='failed_before_GPU':w['no_GPU_submitted']=True;write(watchpath,w);break
  except BaseException as e:w['errors']+=1;w['last_error']=repr(e);write(watchpath,w)
  time.sleep(10)
 raise SystemExit
assert sys.argv[1]=='controller';d=json.loads(record.read_text())
try:
 m=json.loads((r/'incoming/Kimi-MoE-lazy-profile-r9125b/manifest.json').read_text())
 for p,h in m['incoming_source_SHA256'].items():assert hashlib.sha256((r/'incoming/Kimi-MoE-lazy-profile-r9125b'/p).read_bytes()).hexdigest()==h,p
 for p,h in m['existing_incoming_dependency_SHA256'].items():assert hashlib.sha256((r/'incoming'/p).read_bytes()).hexdigest()==h,p
 for p,h in m['existing_remote_SHA256'].items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h,p
 from Kimi_full_MoE_input_r8669s import load_fixture
 checked_bytes,checked_info=load_fixture(r,m['fixture']);d['CPU_fixture_checked']=checked_info;del checked_bytes;write(record,d)
 rows=accounting();assert any(v[0]==parent and norm(v[2])=='RUNNING'for v in rows)
 while True:
  busy=[]
  for v in rows:
   if v[0].startswith(parent+'.')and v[0].split('.')[-1].isdigit()and norm(v[2])not in terminal:
    if nodes.intersection(subprocess.check_output(['scontrol','show','hostnames',v[4]],text=True,timeout=20).splitlines()):busy.append(v[0])
  if not busy:break
  d.update(state='waiting_original_n01_busy_no_GPU_submitted',busy_handles=busy);write(record,d);time.sleep(10);rows=accounting()
 assert not any(v[1]==label for v in rows),'never replay same component'
 contract_path=r/m['CPU_contract']['path'];assert hashlib.sha256(contract_path.read_bytes()).hexdigest()==m['CPU_contract']['SHA256'];contract=json.loads(contract_path.read_text());assert contract['passed']and not contract['GPU_initialized']
 prep_args=['srun','--jobid='+parent,'--overlap','--exact','-N2','-n2','--ntasks-per-node=1','-c2','-w',node,'--gres=none','--mem=16384M','-t','00:03:00','--job-name='+label+'-prewarm','env','WAVE_INSIDE_R2323=Kimi-MoE-lazy-profile-r9125b/prewarm_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 prep=subprocess.run(prep_args,capture_output=True,text=True,timeout=175);d['CPU_prewarm_returncode']=prep.returncode;d['CPU_prewarm_stdout']=prep.stdout;d['CPU_prewarm_stderr']=prep.stderr;write(record,d);assert prep.returncode==0,(prep.stdout,prep.stderr);witness=[json.loads(v)for v in prep.stdout.splitlines()if v.startswith('{')and 'producer'in v];assert len(witness)==2 and {v['node']for v in witness}==nodes;checks=[]
 for v in witness:
  path=r/pathlib.PurePosixPath(v['path']).relative_to('/refactor');data=json.loads(path.read_text());assert data['completed']and data['read_completed_bytes']==data['planned_page_read_bytes']and data['checkpoint_header_SHA256']in m['expected_header_SHA256'].values();checks.append({'path':str(path),'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'data':data})
 d['CPU_prewarm_qualified']=checks;write(record,d)
 d['fresh_before_KFD']=idle('before');master=next(v['ib0_ipv4'][0]for v in d['fresh_before_KFD']if v['node']=='a02r4n14');m['master_addr']=master;m['master_port']=41026;(r/'incoming/Kimi-MoE-lazy-profile-r9125b/manifest.json').write_text(json.dumps(m,indent=2));d['frozen_runtime_manifest_SHA256']=hashlib.sha256((r/'incoming/Kimi-MoE-lazy-profile-r9125b/manifest.json').read_bytes()).hexdigest();write(record,d)
 with(r/'logs'/(label+'-watcher.log')).open('x')as f:w=subprocess.Popen([sys.executable,__file__,'watch'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 limit=time.monotonic()+30
 while not watchpath.exists():assert time.monotonic()<limit and w.poll()is None;time.sleep(.1)
 ready=json.loads(watchpath.read_text());assert ready['ready']and ready['label']==label and w.poll()is None
 args=['srun','--jobid='+parent,'--overlap','--exact','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','-w',node,'--gres=dcu:8','--mem=128000M','-t','00:10:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=Kimi-MoE-lazy-profile-r9125b/GPU_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 assert w.poll()is None,'same ready watcher died before GPU Popen'
 with(r/'logs'/(label+'-srun.log')).open('x')as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d.update(state='GPU_submitted_follow_same_handle',pid=p.pid,watcher_pid=w.pid,submitted_utc=now());write(record,d)
 rc=p.wait(timeout=700);d.update(state='GPU_srun_returned_watch_same_handle',srun_returncode=rc,returned_utc=now());write(record,d)
except BaseException as e:
 d['error']=repr(e);d['state']='unknown_GPU_follow_same_handle'if d.get('pid')else'failed_before_GPU';write(record,d);raise
