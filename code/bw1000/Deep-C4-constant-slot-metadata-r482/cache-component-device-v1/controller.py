"""One bounded one-GPU Deep cache trial; six sequential independent Runtime phases."""
from pathlib import Path,PurePosixPath
import json,hashlib,subprocess,os,re,time,signal,traceback,datetime
NAME='Deep-C4-constant-slot-metadata-r482/cache-component-device-v1';LABEL='Deep-r482-constant-slot-cache-component-v1';GPU_SECONDS=600
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
    assert re.fullmatch(r'910374[.]\d+',got[0]) and(step is None or step==got[0]),'Owned cleanup step identity mismatch'
    step=got[0];row=got
   result['observations'].append(dict(step=step,row=got,srun_rc=proc.poll(),group_live=group_live()))
  except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError,AssertionError,ValueError)as error:
   result['observations'].append(dict(step=step,observation_error=repr(error),srun_rc=proc.poll(),group_live=group_live()))
  terminal=bool(row and row[2].split()[0].rstrip('+') in TERMINAL)
  # If accounting never yielded a step, absence is UNKNOWN, even if the local srun exits.
  return terminal and proc.poll() is not None and not group_live()
 def signal_owned(kind):
  if step is not None:
   assert re.fullmatch(r'910374[.]\d+',step),'Never signal an unknown or parent allocation'
   try:
    z=subprocess.run(['scancel','--signal='+kind,step],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=min(10,max(1,cleanup_limit-time.monotonic())))
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

def clean_env():return{k:v for k,v in os.environ.items()if k not in ['BASH_ENV','ENV','PYTHONSTARTUP','PYTHONINSPECT','LD_PRELOAD'] and (not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_','INFINILM_','INFINICORE_','KIMI_','GLM_','DEEP_','GLOBAL_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER'])}
def gpu_command(root):
 root=Path(root);return['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','--ntasks-per-node=1','-c4','-w','a02r4n01','--gres=dcu:1','--mem=8192M','-t','00:10:00','--job-name='+LABEL,'--output='+str(root/'logs'/(LABEL+'-%J.log')),'--export=ALL','bash',str(root/'incoming'/NAME/'resource_wave_host_alias_outer.sh'),'probe']
def start(root,pin,popen=None,start_unix=None):
 popen=popen or subprocess.Popen;args=gpu_command(root);env=clean_env();env.update(WAVE_INSIDE_R2323=NAME+'/GPU_inside.sh',DEEP_SLOT_GPU_SOURCE_SHA=pin,DEEP_SLOT_GPU_START=str(start_unix if start_unix is not None else time.time()))
 with(Path(root)/'logs'/(LABEL+'-srun.log')).open('x')as f:p=popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 return p,args
def verify(root,go):
 root=Path(root);d=root/'incoming'/NAME
 assert Path(__file__).resolve().parent==d.resolve()
 assert go['execute_once']is True and re.fullmatch('[0-9a-f]{40}',go['publication_commit'])
 mf=d/'GPU_SOURCE_MANIFEST.json';assert sha(mf)==go['GPU_source_manifest_SHA256'];m=json.loads(mf.read_bytes())
 assert m['GPU_GO']and not m['Model_GO']
 for n,h in m['files_SHA256'].items():assert sha(d/n)==h,n
 def host(v):return root.joinpath(*PurePosixPath(v).relative_to('/refactor').parts)
 p=host(m['CPU_receipt']);assert sha(p)==m['CPU_receipt_SHA256'];a=json.loads(p.read_bytes());assert a['completed']and a['passed']and a['actual_CPU']=='910407.1116'and a['new_HOST']==1 and a['new_GPU']==0 and not a['Native_dlopen']and not a['Runtime']and a['SDK_explicit_calls']==0
 v=a['libraries']['probe'];assert v['path']==m['probe_path']and sha(host(v['path']))==v['SHA256']==m['probe_SHA256'];assert sha(host(m['Native_path']))==m['Native_SHA256'];assert v['NEEDED'].count(m['Native_path'])==1 and len(v['all_transitive_resolved_library_SHA256'])==69
 assert sha(host(a['compiled']['probe']['object']))==a['compiled']['probe']['object_SHA256'];assert a['all_fresh_M_before_compile']and a['all_fresh_M_equal_MD']and not a['unmatched_dependencies']
 for path,h in v['all_transitive_resolved_library_SHA256'].items():
  if path.startswith('/refactor/'):assert sha(host(path))==h,path
 claim=m['resource_claim'];assert json.loads((d/'GPU_RESOURCE_CLAIM.json').read_bytes())==claim;assert(claim['parent'],claim['node'],claim['cards'],claim['GPU_hard_s'],claim['worker_s'],claim['keeper_s'])==(910374,'a02r4n01',1,600,480,660)
 assert m['modes']==[0,1,1,0,0,1]and m['paired_units']==3 and m['planned_measured_gathers']==756 and m['planned_warm_gathers']==126 and claim['logical_context']==32768 and claim['compressed_slots']==8192 and claim['floor_bytes']==17179869184
 args=gpu_command(root);assert args[args.index('-w')+1]==claim['node']and '--jobid=910374'in args and '--gres=dcu:1'in args
 for p in d.glob('*.sh'):assert b'\r'not in p.read_bytes()
 return m

def visible(path,until):
 while True:
  try:
   raw=path.read_bytes();q=json.loads(raw)
   if q.get('completed')is True:return q,dict(path=str(path),bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
  except(FileNotFoundError,json.JSONDecodeError):pass
  if time.monotonic()>=until:raise TimeoutError('original completed artifact not visible; no GPU replay')
  time.sleep(.5)

def run(root,go):
 root=Path(root);m=verify(root,go);watch=root/'results/Deep482-constant-slot-cache-watch-v1.json'
 started=time.monotonic();deadline=started+660
 q=dict(completed=False,passed=False,numeric_agreement=None,GPU_cards=1,publication_commit=go['publication_commit'],GPU_source_manifest_SHA256=go['GPU_source_manifest_SHA256'],actual_step=None,keeper_started_unix=time.time(),original_start=None,submissions=[],no_retry=True,Model=False)
 with watch.open('x')as f:json.dump(q,f)
 env=clean_env();actual=None;p=None
 def save():
  q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=watch.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(watch)
 def command(args):return subprocess.check_output(args,env=env,universal_newlines=True,timeout=min(10,max(1,deadline-time.monotonic())))
 def rows():return[l.split('|')for l in command(['sacct','-S','2026-11-05','-j','910374','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,NodeList%100']).splitlines()if l.strip()]
 def account():
  nonlocal actual
  matches=[r for r in rows()if r[1]==LABEL];assert len(matches)<=1
  if not matches:return None
  r=matches[0];assert re.fullmatch(r'910374[.]\d+',r[0])and command(['scontrol','show','hostnames',r[4]]).splitlines()==['a02r4n01'];assert actual in [None,r[0]]
  actual=r[0];q['actual_step']=actual;return r
 def probe(suffix,expected):
  args=['srun','--jobid=910374','--overlap','--exact','--immediate=10','--cpu-bind=none','-N1','-n1','-c1','-w','a02r4n01','--gres=none','--mem=512M','-t','00:00:30','--job-name='+LABEL+'-'+suffix,'python3','-B',str(root/'incoming'/NAME/'resource_probe.py'),json.dumps(expected)]
  z=subprocess.run(args,env=env,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=min(40,max(1,deadline-time.monotonic())))
  records=[json.loads(l)for l in z.stdout.splitlines()if l.startswith('{')];assert z.returncode==0 and len(records)==1 and records[0]['passed'],z.stderr[-1000:];return records[0]
 def known_pids():
  if not actual:return []
  try:
   r=json.loads((root/'results'/('Deep-slot-r482-'+actual+'.json')).read_bytes());return [{'PID':r['PID'],'startticks':r['startticks']}]+[{'PID':v['PID'],'startticks':v['startticks']}for v in r.get('children',[])]
  except(FileNotFoundError,json.JSONDecodeError,KeyError):return []
 try:
  parent=command(['scontrol','show','job','910374','-o']);assert 'UserId=aclsylqidf('in parent and 'JobState=RUNNING'in parent
  found=re.search(r'(?:^| )NodeList=([^ ]+)',parent);assert found and 'a02r4n01'in command(['scontrol','show','hostnames',found.group(1)]).splitlines()
  assert account()is None
  for row in rows():
   if re.fullmatch(r'910374[.]\d+',row[0])and row[2].split()[0].rstrip('+')not in TERMINAL:
    assert 'a02r4n01'not in command(['scontrol','show','hostnames',row[4]]).splitlines(),'foreign actual active step on assigned node'
  q['before']=probe('before',[]);q['original_start']=time.time();save()
  p,args=start(root,go['GPU_source_manifest_SHA256'],start_unix=q['original_start']);q['submissions'].append(dict(owned_srun_PID=p.pid,owned_original_PGID=p.pid,argv=args));save()
  while time.monotonic()<deadline-65:
   row=account();q['last_accounting']=row;q['srun_rc']=p.poll();save()
   terminal=bool(row and row[2].split()[0].rstrip('+')in TERMINAL)
   if p.poll()not in [None,0]or(terminal and row[2:4]!=['COMPLETED','0:0']):raise RuntimeError('original GPU step failed; no replay')
   if terminal and p.poll()==0:break
   time.sleep(3)
  else:raise TimeoutError('one original component budget exhausted')
  owner,pin=visible(root/'results'/('Deep-slot-r482-'+actual+'.json'),min(deadline-40,time.monotonic()+15))
  assert owner['actual_step']==actual and owner['source_manifest_SHA256']==go['GPU_source_manifest_SHA256']
  assert owner['passed']and owner['planned']==756 and owner['completed_queries']==756 and owner['failed_queries']==0 and owner['missing_queries']==0 and owner['normal_Runtime_closes']==6 and len(owner['phases'])==6 and not owner['forced_child_cleanup']
  assert all(v['value']['normal_Runtime_close']and v['value']['close']['rc']==0 and v['value']['close']['values'][2]==1 and v['value']['sampled_floor_passed']and v['value']['schedule']['passed']for v in owner['phases'])
  assert owner['physical_PCI'].lower()in[p.lower()for p in q['before']['physical_PCI']]
  q.update(owner_report=pin,owner_PID=owner['PID'],numeric_agreement=True,normal_Runtime_destroy=True)
  q['post']=probe('post',known_pids());q['passed']=True
 except BaseException as error:
  q.update(error=repr(error),traceback=traceback.format_exc())
  if p is not None:
   try:
    try:os.killpg(p.pid,0);group_live=True
    except ProcessLookupError:group_live=False
    row=account()
    if p.poll()is not None and not group_live and row and row[2].split()[0].rstrip('+')in TERMINAL:
     q['owned_failure_cleanup']=dict(already_terminal=True,actual_step=actual,signals=[],group_gone=True,srun_rc=p.poll())
    else:q['owned_failure_cleanup']=close_owned(LABEL,p,actual,lambda label:account(),deadline,term_s=10,kill_s=10)
   except BaseException as e:q['cleanup_error']=repr(e)
   try:q['failure_post']=probe('failure-post',known_pids())
   except BaseException as e:q['post_error']=repr(e)
 finally:q['completed']=True;save()
 return q
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();root=Path.home()/'hygon-inference-refactor-20261014'
 q=run(root,json.loads(Path(a.execution_go).read_bytes()));print(json.dumps({k:q.get(k)for k in ['completed','passed','numeric_agreement','actual_step','error']}));raise SystemExit(0 if q['passed']else 1)
