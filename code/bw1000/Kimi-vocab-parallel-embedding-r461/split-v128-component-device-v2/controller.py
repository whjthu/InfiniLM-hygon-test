"""One bounded single Runtime/one-GPU Kimi split K192/V128 attention component."""
from pathlib import Path,PurePosixPath
import json,hashlib,subprocess,os,re,time,signal,traceback,datetime
NAME='Kimi-vocab-parallel-embedding-r461/split-v128-component-device-v2';LABEL='Kimi-r461-split-V128-component-v2';GPU_SECONDS=600
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
    assert re.fullmatch(r'910380[.]\d+',got[0]) and(step is None or step==got[0]),'Owned cleanup step identity mismatch'
    step=got[0];row=got
   result['observations'].append(dict(step=step,row=got,srun_rc=proc.poll(),group_live=group_live()))
  except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError,AssertionError,ValueError)as error:
   result['observations'].append(dict(step=step,observation_error=repr(error),srun_rc=proc.poll(),group_live=group_live()))
  terminal=bool(row and row[2].split()[0].rstrip('+') in TERMINAL)
  # If accounting never yielded a step, absence is UNKNOWN, even if the local srun exits.
  return terminal and proc.poll() is not None and not group_live()
 def signal_owned(kind):
  if step is not None:
   assert re.fullmatch(r'910380[.]\d+',step),'Never signal an unknown or parent allocation'
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

def clean_env():return{k:v for k,v in os.environ.items()if k not in ['BASH_ENV','ENV','PYTHONSTARTUP','PYTHONINSPECT','LD_PRELOAD'] and (not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_','INFINILM_','INFINICORE_','KIMI_','GLM_','GLOBAL_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER'])}
def gpu_command(root):
 root=Path(root);return['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','--ntasks-per-node=1','-c4','-w','a14r2n08','--gres=dcu:1','--mem=8192M','-t','00:10:00','--job-name='+LABEL,'--output='+str(root/'logs'/(LABEL+'-%J.log')),'--export=ALL','bash',str(root/'incoming'/NAME/'resource_wave_host_alias_outer.sh'),'probe']
def start(root,pin,popen=None,start_unix=None):
 popen=popen or subprocess.Popen;args=gpu_command(root);env=clean_env();env.update(WAVE_INSIDE_R2323=NAME+'/GPU_inside.sh',KIMI_SPLITV_GPU_SOURCE_SHA=pin,KIMI_SPLITV_GPU_START=str(start_unix if start_unix is not None else time.time()))
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
 p=host(m['CPU_receipt']);assert sha(p)==m['CPU_receipt_SHA256'];a=json.loads(p.read_bytes())
 assert a['actual_CPU']=='910407.1080'and a['completed']and a['passed']and not a['SDK_device_execution']
 assert sha(host(a['probe']))==a['probe_SHA256']==m['probe_SHA256'];assert sha(host(a['parent_Native']['path']))==a['parent_Native']['SHA256']==m['Native_SHA256']
 r=a['reused1076'];assert sha(host(r['object']))==r['object_SHA256'];assert sha(host(r['receipt']['path']))==r['receipt']['SHA256'];assert sha(host(str(PurePosixPath(r['object']).with_suffix('.d'))))==r['MD_SHA256']
 source=root/'incoming/Kimi-vocab-parallel-embedding-r461/split-v128-component-source-v1';assert sha(source/'SOURCE_MANIFEST.json')==m['component_source_manifest_SHA256']
 for n,h in json.loads((source/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(source/n)==h,n
 for v in a['actual_dependency_library_SHA256'].values():assert sha(host(v['path']))==v['SHA256']
 loader=json.loads((d/'LOADER_BINDING.json').read_bytes());lp=host(loader['receipt']);assert sha(lp)==loader['receipt_SHA256'];lc=json.loads(lp.read_bytes());assert lc['completed']and lc['passed']and lc['actual_CPU']==loader['actual_CPU']=='910407.1089'and lc['Torch_imported']is False
 assert lc['ENV_SHA256']==sha(d/'ENV.sh')==loader['ENV_SHA256']and lc['LD_LIBRARY_PATH']==loader['LD_LIBRARY_PATH']and lc['LD_PRELOAD']==loader['LD_PRELOAD']
 assert len(lc['all_transitive_resolved_library_SHA256'])==loader['resolved_libraries']==71
 claim=m['resource_claim'];assert json.loads((d/'GPU_RESOURCE_CLAIM.json').read_bytes())==claim;assert(claim['parent'],claim['node'],claim['cards'],claim['GPU_hard_s'],claim['worker_s'],claim['keeper_s'])==(910380,'a14r2n08',1,600,480,660)
 assert m['modes']==[0,1,1,0]and m['total_queries']==20 and m['querytokens_per_phase']==2049
 args=gpu_command(root);assert args[args.index('-w')+1]==claim['node']and '--jobid=910380'in args and '--gres=dcu:1'in args
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
 root=Path(root);m=verify(root,go);watch=root/'results/Kimi461-split-V128-watch-v2.json'
 started=time.monotonic();deadline=started+660
 q=dict(completed=False,passed=False,numeric_agreement=None,GPU_cards=1,publication_commit=go['publication_commit'],GPU_source_manifest_SHA256=go['GPU_source_manifest_SHA256'],actual_step=None,keeper_started_unix=time.time(),original_start=None,submissions=[],no_retry=True,Model=False)
 with watch.open('x')as f:json.dump(q,f)
 env=clean_env();actual=None;p=None
 def save():
  q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=watch.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(watch)
 def command(args):return subprocess.check_output(args,env=env,universal_newlines=True,timeout=min(10,max(1,deadline-time.monotonic())))
 def rows():return[l.split('|')for l in command(['sacct','-S','2026-11-05','-j','910380','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,NodeList%100']).splitlines()if l.strip()]
 def account():
  nonlocal actual
  matches=[r for r in rows()if r[1]==LABEL];assert len(matches)<=1
  if not matches:return None
  r=matches[0];assert re.fullmatch(r'910380[.]\d+',r[0])and command(['scontrol','show','hostnames',r[4]]).splitlines()==['a14r2n08'];assert actual in [None,r[0]]
  actual=r[0];q['actual_step']=actual;return r
 def probe(suffix,expected):
  args=['srun','--jobid=910380','--overlap','--exact','--immediate=10','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n08','--gres=none','--mem=512M','-t','00:00:30','--job-name='+LABEL+'-'+suffix,'python3','-B',str(root/'incoming'/NAME/'resource_probe.py'),json.dumps(expected)]
  z=subprocess.run(args,env=env,universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=min(40,max(1,deadline-time.monotonic())))
  records=[json.loads(l)for l in z.stdout.splitlines()if l.startswith('{')];assert z.returncode==0 and len(records)==1 and records[0]['passed'],z.stderr[-1000:];return records[0]
 def known_pids():
  if not actual:return []
  try:return [json.loads((root/'results'/('Kimi-split-V128-'+actual+'.json')).read_bytes())['PID']]
  except(FileNotFoundError,json.JSONDecodeError,KeyError):return []
 try:
  parent=command(['scontrol','show','job','910380','-o']);assert 'UserId=aclsylqidf('in parent and 'JobState=RUNNING'in parent
  found=re.search(r'(?:^| )NodeList=([^ ]+)',parent);assert found and 'a14r2n08'in command(['scontrol','show','hostnames',found.group(1)]).splitlines()
  assert account()is None
  for row in rows():
   if re.fullmatch(r'910380[.]\d+',row[0])and row[2].split()[0].rstrip('+')not in TERMINAL:
    assert 'a14r2n08'not in command(['scontrol','show','hostnames',row[4]]).splitlines(),'foreign actual active step on assigned node'
  q['before']=probe('before',[]);q['original_start']=time.time();save()
  p,args=start(root,go['GPU_source_manifest_SHA256'],start_unix=q['original_start']);q['submissions'].append(dict(owned_srun_PID=p.pid,owned_original_PGID=p.pid,argv=args));save()
  while time.monotonic()<deadline-65:
   row=account();q['last_accounting']=row;q['srun_rc']=p.poll();save()
   terminal=bool(row and row[2].split()[0].rstrip('+')in TERMINAL)
   if p.poll()not in [None,0]or(terminal and row[2:4]!=['COMPLETED','0:0']):raise RuntimeError('original GPU step failed; no replay')
   if terminal and p.poll()==0:break
   time.sleep(3)
  else:raise TimeoutError('one original component budget exhausted')
  owner,pin=visible(root/'results'/('Kimi-split-V128-'+actual+'.json'),min(deadline-40,time.monotonic()+15))
  assert owner['actual_step']==actual and owner['source_manifest_SHA256']==go['GPU_source_manifest_SHA256']
  assert owner['passed']and owner['mechanical_passed']and owner['numeric_agreement']and owner['normal_Runtime_close']and owner['close']['rc']==0
  assert owner['visible_count']==1 and len(owner['outputs'])==20 and all(v['finite']and v['native_rc']==0 for v in owner['outputs'])and owner['observed_warm_net_capacity_sufficient']
  assert owner['physical_PCI'].lower()in[p.lower()for p in q['before']['physical_PCI']]
  q.update(owner_report=pin,owner_PID=owner['PID'],numeric_agreement=owner['numeric_agreement'],normal_Runtime_destroy=True)
  q['post']=probe('post',[owner['PID']]);q['passed']=True
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
