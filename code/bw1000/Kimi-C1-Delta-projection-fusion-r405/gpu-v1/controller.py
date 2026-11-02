"""Reuse391/399 bounded lifecycle for one complete routedFFN geometry comparison; never replay."""
from pathlib import Path
import datetime,hashlib,json,os,re,signal,subprocess,time,traceback
NAME='Kimi-C1-Delta-projection-fusion-r405/gpu-v1';LABEL='Kimi405-Delta-projection-GPU-v1'
NODES=['a02r3n03','a02r3n07']
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean_env():return{k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
def gpu_command(root):
 root=Path(root);return['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','--distribution=block:block','-c32','-w',','.join(NODES),'--gres=dcu:8','--mem=49152M','-t','00:10:00','--job-name='+LABEL,'--output='+str(root/'logs'/(LABEL+'-%J-%t.log')),'--export=ALL','bash',str(root/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
def start(root,pin,popen=None):
 popen=popen or subprocess.Popen;args=gpu_command(root);env=clean_env();runtime=json.loads((Path(root)/'results/Kimi405-GPU-runtime-v1.json').read_bytes())
 env.update(WAVE_INSIDE_R2323=NAME+'/GPU_inside.sh',KIMI405_GPU_MANIFEST_SHA=pin,KIMI_ENGINE_HEAD=runtime['head_ib0'],KIMI_ENGINE_NONCE=runtime['nonce'],GLOBAL_PORT_R2354='23040',GLOBAL_MODEL_MODE_R2354='shca',GLOBAL_OFFSET_R2354='0')
 with(Path(root)/'logs'/(LABEL+'-srun.log')).open('x')as f:p=popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 return p,args

def verify(root,go):
 root=Path(root);assert go['execute_once']is True and re.fullmatch('[0-9a-f]{40}',go['publication_commit'])and go['publication_commit']!='0'*40
 d=root/'incoming'/NAME;mf=d/'GPU_MANIFEST.json';assert sha(mf)==go['GPU_source_manifest_SHA256'];m=json.loads(mf.read_bytes());assert m['GPU_GO']is True
 for n,h in m['files_SHA256'].items():assert sha(d/n)==h,n
 source=root/'incoming/Kimi-C1-Delta-projection-fusion-r405/cpu-v2';sf=source/'SOURCE_MANIFEST.json';assert sha(sf)==m['CPU_source_manifest_SHA256']
 for n,h in json.loads(sf.read_bytes())['files_SHA256'].items():assert sha(source/n)==h,n
 cp=root/'candidates/Kimi-C1-Delta-projection-fusion-r405/cpu-v2/CPU-build.json';assert sha(cp)==m['CPU_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['completed']and c['passed']and c['actual_CPU']=='910407.810'and not c['Native_dlopen']
 host=lambda p:root/str(p).removeprefix('/refactor/')
 assert sha(host(c['native']))==c['native_SHA256']==m['library_SHA256']
 assert sha(host(c['object']))==c['object_SHA256']
 for p,h in c['MD_source_and_dependencies_SHA256'].items():
  if p.startswith('/refactor/'):assert sha(host(p))==h,p
 for p,h in c['actual_resolved_dependency_library_SHA256'].items():
  if p.startswith('/refactor/'):assert sha(host(p))==h,p
 assert sha(root/'incoming/resource_wave_local_r3097_outer.sh')==m['namespace_wrapper_SHA256']
 return m,c

def visible(path,until):
 while True:
  try:
   q=json.loads(path.read_bytes())
   if q.get('completed')is True:return q
  except(FileNotFoundError,json.JSONDecodeError):pass
  if time.monotonic()>=until:raise TimeoutError('Artifact visibility unresolved; no GPU replay')
  time.sleep(1)
def run(root,go):
 root=Path(root);m,c=verify(root,go);watch=root/'results/Kimi405-GPU-watch-v1.json';q=dict(completed=False,passed=False,GPU_cards=16,publication_commit=go['publication_commit'],GPU_source_manifest_SHA256=go['GPU_source_manifest_SHA256'],submissions=[],actual_step=None,original_start=time.time(),profile=0,no_retry=True)
 with watch.open('x')as f:json.dump(q,f)
 p=None;actual=None;env=clean_env();deadline=time.monotonic()+900
 def save():
  q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=watch.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(watch)
 def command(args):return subprocess.check_output(args,env=env,text=True,timeout=30)
 def rows():return[line.split('|')for line in command(['sacct','-S','2026-11-01','-j','910371','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,NodeList%100']).splitlines()if line.strip()]
 def account():
  nonlocal actual
  matches=[r for r in rows()if r[1]==LABEL];assert len(matches)<=1,'Duplicate label never authorizes another step'
  if not matches:return None
  r=matches[0];assert re.fullmatch(r'910371\.\d+',r[0])and command(['scontrol','show','hostnames',r[4]]).splitlines()==NODES;assert actual in [None,r[0]];actual=r[0];q['actual_step']=actual;return r
 def probe(suffix,expected):
  args=['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','-w',','.join(NODES),'--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+suffix,'python3','-B',str(root/'incoming'/NAME/'resource_probe.py'),json.dumps(expected)]
  z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=150);v=[json.loads(l)for l in z.stdout.splitlines()if l.startswith('{')];assert z.returncode==0 and len(v)==2 and{a['node']for a in v}==set(NODES)and all(a['passed']for a in v),z.stderr[-1000:];return v
 def known_pids():
  found={n:[]for n in NODES}
  if actual:
   for node,n in enumerate(NODES):
    f=root/'results'/('Kimi405-C1-Delta-projection-'+actual+'-node'+str(node)+'.json')
    try:a=json.loads(f.read_bytes())
    except(FileNotFoundError,json.JSONDecodeError):a={}
    found[n]=sorted({w['PID']for w in a.get('workers',[])if w.get('PID')})
  return found

 try:
  parent=command(['scontrol','show','job','910371','-o']);assert'UserId=aclsylqidf('in parent and'JobState=RUNNING'in parent;v=re.search(r'(?:^| )NodeList=([^ ]+)',parent);assert v and set(NODES)<=set(command(['scontrol','show','hostnames',v.group(1)]).splitlines());assert account()is None
  for r in rows():
   if re.fullmatch(r'910371\.\d+',r[0])and r[2].split()[0].rstrip('+')not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE']:assert not set(NODES)&set(command(['scontrol','show','hostnames',r[4]]).splitlines()),'Foreign active step'
  q['before']=probe('before',{});save();q['original_GPU_start_unix']=time.time()
  execution=dict(head_ib0=next(a['ib0_IPv4']for a in q['before']if a['node']==NODES[0]),nonce=os.urandom(16).hex(),CPU_receipt='/refactor/candidates/Kimi-C1-Delta-projection-fusion-r405/cpu-v2/CPU-build.json',CPU_receipt_SHA256=m['CPU_receipt_SHA256'],library_SHA256=m['library_SHA256'],deadline_unix=q['original_GPU_start_unix']+540,publication_commit=go['publication_commit'],GPU_source_manifest_SHA256=go['GPU_source_manifest_SHA256'],source_manifest_SHA256=m['CPU_source_manifest_SHA256'])
  ep=root/'results/Kimi405-GPU-runtime-v1.json'
  with ep.open('x')as f:json.dump(execution,f,indent=2);f.flush();os.fsync(f.fileno())
  p,args=start(root,go['GPU_source_manifest_SHA256']);q['submissions'].append(dict(owned_srun_PID=p.pid,owned_original_PGID=p.pid,argv=args));save();unresolved=None
  while time.monotonic()<deadline-230:
   try:r=account()
   except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as error:
    rc=p.poll();q.setdefault('observation_errors',[]).append(dict(error=repr(error),monotonic=time.monotonic(),original_step=actual,srun_exit=rc));q['last_observation']=dict(row=None,observation_error=repr(error),srun_exit=rc);save()
    if rc not in [None,0]:raise RuntimeError('Original srun failed while accounting unavailable')from error
    time.sleep(15);continue
   rc=p.poll();q['last_observation']=dict(row=r,srun_exit=rc);save();terminal=r is not None and r[2].split()[0].rstrip('+')in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE']
   if rc not in [None,0]or(terminal and r[2:4]!=['COMPLETED','0:0']):raise RuntimeError('Original component failed; no replay')
   if terminal and rc==0:break
   if terminal or rc is not None:
    unresolved=unresolved or time.monotonic()
    if time.monotonic()-unresolved>90:raise TimeoutError('Accounting/srun propagation unresolved')
   else:unresolved=None
   time.sleep(15)
  else:raise TimeoutError('Original controller budget exhausted')
  until=min(deadline-150,time.monotonic()+90);owners=[]
  for node,n in enumerate(NODES):
   f=root/'results'/('Kimi405-C1-Delta-projection-'+actual+'-node'+str(node)+'.json');a=visible(f,until)
   assert a['completed']and a['normal_close']and a['owners']==8 and len(a['workers'])==8 and a['Math']==0 and not a['capture']
   for w in a['workers']:
    assert w['completed']and w['normal_close']and len(w['samples'])==320 and 'error'not in w
    assert all(t['passed']and t['call_completed']and t['counter_passed']and t['numerical']['passed']for t in w['samples'])
    assert w['projection_counts']['chosen']==160 and w['projection_counts']['consumer_counts']==[160]*4 and w['projection_counts']['fallback']==0
   owners.append(dict(node=n,path=str(f),SHA256=sha(f),PID=sorted({w['PID']for w in a['workers']})))
  q.update(owners=owners,normal16=True,srun_exit0=True,Slurm_COMPLETED0=True,Model_gain=False)
  q['post']=probe('post',known_pids());q.update(passed=True,performance_scope='Complete Delta C1 real TP16 component only; source fixture historical normalized input, not Model/HTTP.')

 except BaseException as error:
  q.update(error=repr(error),traceback=traceback.format_exc())
  if p is not None:
   try:
    r=account()
    if r and r[2].split()[0].rstrip('+')not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']:subprocess.run(['scancel','--signal=TERM',actual],env=env,check=True,timeout=25)
   except BaseException as e:q['scancel_error']=repr(e)
   for sig,seconds in [(signal.SIGTERM,20),(signal.SIGKILL,10)]:
    try:os.killpg(p.pid,sig)
    except ProcessLookupError:break
    try:p.wait(timeout=seconds)
    except subprocess.TimeoutExpired:continue
   try:q['failure_post']=probe('failure-post',known_pids())
   except BaseException as e:q['post_error']=repr(e)
 finally:q['completed']=True;save()
 return q
if __name__=='__main__':
 import sys
 root=Path.home()/'hygon-inference-refactor-20261014';go=json.loads(Path(sys.argv[1]).read_bytes());q=run(root,go);print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','error']}));raise SystemExit(0 if q['passed']else 1)
