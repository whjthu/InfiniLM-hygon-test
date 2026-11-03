"""One bounded current038d layer component step; never replay a label/watch."""
from pathlib import Path,PurePosixPath
import json,hashlib,subprocess,os,re,time,signal,traceback,datetime
NAME='Kimi-W2-Runtime-H2D-r439/gpu-v2';LABEL='Kimi-r439-W2-H2D-v2';GPU_SECONDS=300
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
def close_owned(label,proc,recorded,exact,cleanup_limit,term_s=20.,kill_s=20.):
 """TERM then KILL only this keeper's known srun PGID/Slurm step, bounded."""
 result=dict(label=label,srun_PID=proc.pid,recorded_step=recorded,signals=[],observations=[],closed=False)
 step=recorded;row=None
 def group_live():
  try:os.killpg(proc.pid,0);return True
  except ProcessLookupError:return False
 def observe():
  nonlocal step,row
  try:
   got=exact(label)
   if got:
    assert re.fullmatch(r'910371[.]\d+',got[0]) and(step is None or step==got[0]),'Owned cleanup step identity mismatch'
    step=got[0];row=got
   result['observations'].append(dict(step=step,row=got,srun_rc=proc.poll(),group_live=group_live()))
  except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError,AssertionError,ValueError)as error:
   result['observations'].append(dict(step=step,observation_error=repr(error),srun_rc=proc.poll(),group_live=group_live()))
  terminal=bool(row and row[2].split()[0].rstrip('+') in TERMINAL)
  # If accounting never yielded a step, absence is UNKNOWN, even if the local srun exits.
  return terminal and proc.poll() is not None and not group_live()
 def signal_owned(kind):
  if step is not None:
   assert re.fullmatch(r'910371[.]\d+',step),'Never signal an unknown or parent allocation'
   try:
    z=subprocess.run(['scancel','--signal='+kind,step],capture_output=True,text=True,timeout=min(10,max(1,cleanup_limit-time.monotonic())))
    result['signals'].append(dict(scope='known_Slurm_step',step=step,signal=kind,rc=z.returncode,stdout=z.stdout,stderr=z.stderr))
   except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as error:result['signals'].append(dict(scope='known_Slurm_step',step=step,signal=kind,error=repr(error)))
  try:os.killpg(proc.pid,getattr(signal,'SIG'+kind));result['signals'].append(dict(scope='owned_start_new_session_PGID',PGID=proc.pid,signal=kind))
  except ProcessLookupError:result['signals'].append(dict(scope='owned_start_new_session_PGID',PGID=proc.pid,signal=kind,already_gone=True))
 observe();signal_owned('TERM')
 stop=min(cleanup_limit,time.monotonic()+term_s)
 while time.monotonic()<stop:
  if observe():result['closed']=True;break
  time.sleep(min(1.,max(0,stop-time.monotonic())))
 if not result['closed']:
  signal_owned('KILL');stop=min(cleanup_limit,time.monotonic()+kill_s)
  while time.monotonic()<stop:
   if observe():result['closed']=True;break
   time.sleep(min(1.,max(0,stop-time.monotonic())))
 if not result['closed']:result['closed']=observe()
 result.update(actual_step=step,last_accounting=row,srun_returncode=proc.poll(),PGID_still_live=group_live(),terminal_unknown_or_unclosed=not result['closed'])
 return result

def clean_env():return{k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
def gpu_command(root):
 root=Path(root);return['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','--ntasks-per-node=1','-c4','-w','a02r3n16','--gres=dcu:1','--mem=8192M','-t','00:05:00','--job-name='+LABEL,'--output='+str(root/'logs'/(LABEL+'-%J.log')),'--export=ALL','bash',str(root/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
def start(root,pin,popen=None,start_unix=None):
 popen=popen or subprocess.Popen;args=gpu_command(root);env=clean_env();env.update(WAVE_INSIDE_R2323=NAME+'/GPU_inside.sh',R439_GPU_SOURCE_SHA256=pin,R439_GPU_START=str(start_unix if start_unix is not None else time.time()))
 with(Path(root)/'logs'/(LABEL+'-srun.log')).open('x')as f:p=popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 return p,args
def verify(root,go):
 assert go['execute_once']is True and re.fullmatch('[0-9a-f]{40}',go['publication_commit'])and go['publication_commit']!='0'*40
 d=Path(root)/'incoming'/NAME;mf=d/'GPU_SOURCE_MANIFEST.json';assert sha(mf)==go['GPU_source_manifest_SHA256'];m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(d/n)==h,n
 host=lambda value:Path(root).joinpath(*PurePosixPath(value).relative_to('/refactor').parts)
 receipt=host(m['CPU_receipt']);assert sha(receipt)==m['CPU_receipt_SHA256'];artifact=json.loads(receipt.read_bytes());assert artifact['completed']and artifact['passed']and artifact['actual_CPU']==m['CPU_actual']and artifact['Runtime_executions']==0 and artifact['HOST_compile']==1 and artifact['Runtime_relink']==0 and artifact['executable_links']==2
 source=Path(root)/'incoming/Kimi-W2-Runtime-H2D-r439/cpu-v2';assert sha(source/'SOURCE_MANIFEST.json')==m['CPU_source_manifest_SHA256']
 for n,h in json.loads((source/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(source/n)==h,n
 for v in artifact['executables'].values():assert sha(host(v['path']))==v['SHA256']and sha(host(v['Runtime']['path']))==v['Runtime']['SHA256']
 assert sha(host(artifact['compiled']['object']))==artifact['compiled']['object_SHA256']and sha(host(artifact['compiled']['MD']))==artifact['compiled']['MD_SHA256']
 assert sha(Path(root)/'incoming/resource_wave_local_r3097_outer.sh')==m['namespace_wrapper_SHA256']
 return m
def visible(path,until):
 while True:
  try:
   q=json.loads(path.read_bytes())
   if q.get('completed')is True:return q
  except(FileNotFoundError,json.JSONDecodeError):pass
  if time.monotonic()>=until:raise TimeoutError('Completed artifact visibility unresolved; no GPU replay')
  time.sleep(1)
def run(root,go):
 root=Path(root);m=verify(root,go);watch=root/'results/Kimi439-watch-v2.json';q=dict(completed=False,passed=False,GPU_cards=1,publication_commit=go['publication_commit'],GPU_source_manifest_SHA256=go['GPU_source_manifest_SHA256'],submissions=[],actual_step=None,original_start=time.time(),profile=0,no_retry=True)
 with watch.open('x')as f:json.dump(q,f)
 p=None;actual=None;env=clean_env();deadline=time.monotonic()+480
 def save():
  q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=watch.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(watch)
 def command(args):return subprocess.check_output(args,env=env,text=True,timeout=30)
 def rows():return[line.split('|')for line in command(['sacct','-S','2026-11-01','-j','910371','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,NodeList%100']).splitlines()if line.strip()]
 def account():
  nonlocal actual
  matches=[row for row in rows()if row[1]==LABEL];assert len(matches)<=1,'Duplicate label never authorizes another step'
  if not matches:return None
  row=matches[0];assert re.fullmatch(r'910371\.\d+',row[0])and command(['scontrol','show','hostnames',row[4]]).splitlines()==['a02r3n16'];assert actual in [None,row[0]];actual=row[0];q['actual_step']=actual;return row
 def probe(suffix,expected):
  args=['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a02r3n16','--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+suffix,'python3','-B',str(root/'incoming'/NAME/'resource_probe.py'),json.dumps(expected)]
  z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=150);q.setdefault('resource_probe_raw',[]).append(dict(suffix=suffix,argv=args,rc=z.returncode,stdout=z.stdout,stderr=z.stderr));save()
  records=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')];assert z.returncode==0 and len(records)==1 and records[0]['passed'],z.stderr[-1000:];return records[0]
 def known_pids():
  ids=[]
  if actual:
   for f in(root/'results').glob('Kimi439-'+actual+'-*.json'):
    try:a=json.loads(f.read_bytes())
    except json.JSONDecodeError:continue
    ids +=[a[k]for k in ['PID','parent_PID']if k in a]
    ids +=[v['PID']for v in a.get('records',[])if 'PID'in v]
  return sorted(set(ids))
 try:
  parent=command(['scontrol','show','job','910371','-o']);assert'UserId=aclsylqidf('in parent and'JobState=RUNNING'in parent;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',parent);assert nodes and'a02r3n16'in command(['scontrol','show','hostnames',nodes.group(1)]).splitlines()
  assert account()is None
  for row in rows():
   if re.fullmatch(r'910371\.\d+',row[0])and row[2].split()[0].rstrip('+')not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE']:
    assert'a02r3n16'not in command(['scontrol','show','hostnames',row[4]]).splitlines(),'Foreign active step on assigned node'
  q['before']=probe('before',[]);q['original_start']=time.time();save();p,args=start(root,go['GPU_source_manifest_SHA256'],start_unix=q['original_start']);q['submissions'].append(dict(owned_srun_PID=p.pid,owned_original_PGID=p.pid,argv=args));save();unresolved=None
  while time.monotonic()<deadline-100:
   try:row=account()
   except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as error:
    q.setdefault('transient_observation_errors',[]).append(dict(error=repr(error),actual_step=actual,original_start=q['original_start'],time=time.time()));save()
    if p.poll()not in [None,0]:raise RuntimeError('Known owned srun failed during temporary accounting loss')
    time.sleep(10);continue
   rc=p.poll();q['last_observation']=dict(row=row,srun_exit=rc);save()
   terminal=row is not None and row[2].split()[0].rstrip('+')in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE']
   if rc not in [None,0]or(terminal and row[2:4]!=['COMPLETED','0:0']):raise RuntimeError('Original component step failed; no replay')
   if terminal and rc==0:break
   if terminal or rc is not None:
    unresolved=unresolved or time.monotonic()
    if time.monotonic()-unresolved>90:raise TimeoutError('Original accounting/srun propagation unresolved')
   else:unresolved=None
   time.sleep(10)
  else:raise TimeoutError('Original component controller bound exhausted')
  until=min(deadline-75,time.monotonic()+45)
  f=root/'results'/('Kimi439-'+actual+'-main.json');a=visible(f,until)
  assert a['passed']and a['all_normal']and a['Runtime_processes_normal']==38 and len(a['records'])==38 and a['source_manifest_SHA256']==go['GPU_source_manifest_SHA256']and a['actual_step']==actual
  host=lambda value:root/str(value).removeprefix('/refactor/')
  for v in a['records']:
   assert v['rc']==0 and v['qualified']and v['Runtime_expired']
   for key in ['actual_result','stdout','stderr','phase_file']:assert sha(host(v[key]['path']))==v[key]['SHA256']
  assert len({v['device']['physical_PCI']for v in a['records']})==1
  assert a['records'][0]['device']['physical_PCI']in q['before']['physical_PCI']
  q.update(report=dict(path=str(f),SHA256=sha(f)),Runtime_processes_normal=38,paired=a['paired'],stable_component_gain=a['stable_component_gain'],Slurm_COMPLETED0=True,srun_exit0=True,Model_quality_qualified=False)
  q['post']=probe('post',known_pids());q.update(passed=True,actual_H2D_component_complete=True)
 except BaseException as error:
  q.update(error=repr(error),traceback=traceback.format_exc())
  if p is not None:
   try:
    closed=close_owned(LABEL,p,actual,lambda label:account(),deadline)
    q['owned_failure_cleanup']=closed;q['forced_cleanup']=bool(closed['signals'])
    if not closed['closed']:q['cleanup_terminal_unknown']=True
   except BaseException as e:q['cleanup_error']=repr(e)
   try:q['failure_post']=probe('failure-post',known_pids())
   except BaseException as e:q['post_error']=repr(e)
 finally:q['completed']=True;save()
 return q
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();root=Path.home()/'hygon-inference-refactor-20261014';go=json.loads(Path(a.execution_go).read_bytes());q=run(root,go);print(json.dumps({k:q.get(k)for k in ['completed','passed','actual_step','error']}));raise SystemExit(0 if q['passed']else 1)
