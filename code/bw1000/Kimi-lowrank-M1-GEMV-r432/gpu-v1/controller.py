"""One bounded single Runtime/one-GPU Kimi lowrank projection component."""
from pathlib import Path,PurePosixPath
import json,hashlib,subprocess,os,re,time,signal,traceback,datetime
NAME='Kimi-lowrank-M1-GEMV-r432/gpu-v1';LABEL='Kimi-r432-lowrank-M1-GEMV-v1';GPU_SECONDS=600
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

def clean_env():return{k:v for k,v in os.environ.items()if k not in ['BASH_ENV','ENV','PYTHONSTARTUP','PYTHONINSPECT','LD_PRELOAD'] and (not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_','INFINILM_','INFINICORE_','KIMI_','GLM_','GLOBAL_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER'])}
def gpu_command(root):
 root=Path(root);return['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','--ntasks-per-node=1','-c4','-w','a02r3n03','--gres=dcu:1','--mem=8192M','-t','00:10:00','--job-name='+LABEL,'--output='+str(root/'logs'/(LABEL+'-%J.log')),'--export=ALL','bash',str(root/'incoming'/NAME/'resource_wave_host_alias_outer.sh'),'probe']
def start(root,pin,popen=None,start_unix=None):
 popen=popen or subprocess.Popen;args=gpu_command(root);env=clean_env();env.update(WAVE_INSIDE_R2323=NAME+'/GPU_inside.sh',KIMI_R432_GPU_SOURCE_SHA=pin,KIMI_R432_GPU_START=str(start_unix if start_unix is not None else time.time()))
 with(Path(root)/'logs'/(LABEL+'-srun.log')).open('x')as f:p=popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 return p,args
def verify(root,go):
 assert go['execute_once']is True and re.fullmatch('[0-9a-f]{40}',go['publication_commit'])and go['publication_commit']!='0'*40
 d=Path(root)/'incoming'/NAME;assert Path(__file__).resolve().parent==d.resolve(),'Actual controller path identity'
 assert NAME=='Kimi-lowrank-M1-GEMV-r432/gpu-v1'and LABEL=='Kimi-r432-lowrank-M1-GEMV-v1'
 mf=d/'GPU_SOURCE_MANIFEST.json';assert sha(mf)==go['GPU_source_manifest_SHA256'];m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(d/n)==h,n
 host=lambda value:Path(root).joinpath(*PurePosixPath(value).relative_to('/refactor').parts)
 receipt=host(m['CPU_receipt']);assert sha(receipt)==m['CPU_receipt_SHA256'];artifact=json.loads(receipt.read_bytes())
 assert artifact['passed']and artifact['actual_CPU']=='910407.869'and artifact['HOST_production_TUs']==artifact['GPU_TUs']==1
 assert artifact['library_SHA256']==m['library_SHA256']and sha(host(artifact['library']))==m['library_SHA256']
 for unit in artifact['compiled']:
  assert sha(host(unit['object']))==unit['object_SHA256']and sha(host(unit['MD']))==unit['MD_SHA256']
  for p,h in unit['dependency_SHA256'].items():assert sha(host(p))==h,p
 assert sha(host(m['BLAS_policy']['path']))==m['BLAS_policy']['SHA256']
 assert sha(host(m['single_Runtime_affinity']['actual_source']))==m['single_Runtime_affinity']['source_SHA256']
 claim=m['resource_claim'];assert(claim['parent'],claim['node'],claim['cards'],claim['cores'])==(910371,'a02r3n03',1,4)
 assert m['modes']==[0,1,0,1,1,0]and m['timed_projection_calls']==12 and m['guarded_diagnostics']==2
 args=gpu_command(root);assert '-N1'in args and '-n1'in args and '-c4'in args and '--gres=dcu:1'in args and args[args.index('-w')+1]==claim['node']
 assert '--jobid='+str(claim['parent'])in args and str(d/'resource_wave_host_alias_outer.sh')in args
 for p in d.glob('*.sh'):assert b'\r'not in p.read_bytes()
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
 root=Path(root);m=verify(root,go);watch=root/'results/Kimi432-watch-v1.json';q=dict(completed=False,passed=False,GPU_cards=1,publication_commit=go['publication_commit'],GPU_source_manifest_SHA256=go['GPU_source_manifest_SHA256'],submissions=[],actual_step=None,original_start=time.time(),profile=0,no_retry=True)
 with watch.open('x')as f:json.dump(q,f)
 p=None;actual=None;env=clean_env();deadline=time.monotonic()+900
 def save():
  q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=watch.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(watch)
 def command(args):return subprocess.check_output(args,env=env,text=True,timeout=30)
 def rows():return[line.split('|')for line in command(['sacct','-S','2026-11-01','-j','910371','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,NodeList%100']).splitlines()if line.strip()]
 def account():
  nonlocal actual
  matches=[row for row in rows()if row[1]==LABEL];assert len(matches)<=1,'Duplicate label never authorizes another step'
  if not matches:return None
  row=matches[0];assert re.fullmatch(r'910371\.\d+',row[0])and command(['scontrol','show','hostnames',row[4]]).splitlines()==['a02r3n03'];assert actual in [None,row[0]];actual=row[0];q['actual_step']=actual;return row
 def probe(suffix,expected):
  args=['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a02r3n03','--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+suffix,'python3','-B',str(root/'incoming'/NAME/'resource_probe.py'),json.dumps(expected)]
  z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=150);records=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')];assert z.returncode==0 and len(records)==1 and records[0]['passed'],z.stderr[-1000:];return records[0]
 def known_pids():
  if not actual:return[]
  f=root/'results'/('Kimi432-'+actual+'.json')
  try:return[json.loads(f.read_bytes())['PID']]
  except(FileNotFoundError,json.JSONDecodeError,KeyError):return[]
 try:
  parent=command(['scontrol','show','job','910371','-o']);assert'UserId=aclsylqidf('in parent and'JobState=RUNNING'in parent;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',parent);assert nodes and'a02r3n03'in command(['scontrol','show','hostnames',nodes.group(1)]).splitlines()
  assert account()is None
  for row in rows():
   if re.fullmatch(r'910371\.\d+',row[0])and row[2].split()[0].rstrip('+')not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE']:
    assert'a02r3n03'not in command(['scontrol','show','hostnames',row[4]]).splitlines(),'Foreign active step on assigned node'
  q['before']=probe('before',[]);q['original_start']=time.time();save();p,args=start(root,go['GPU_source_manifest_SHA256'],start_unix=q['original_start']);q['submissions'].append(dict(owned_srun_PID=p.pid,owned_original_PGID=p.pid,argv=args));save();unresolved=None
  while time.monotonic()<deadline-230:
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
  until=min(deadline-150,time.monotonic()+90);f=root/'results'/('Kimi432-'+actual+'.json');owner=visible(f,until)
  assert owner['actual_step']==actual and owner['source_manifest_SHA256']==go['GPU_source_manifest_SHA256']and owner['passed']and owner['normal_destroy']and owner['destroy_rc']==0
  assert owner['visible_count']==1 and owner['actual_call_counts']==[3,4,3,4]and len(owner['records'])==12 and len(owner['diagnostics'])==2
  assert owner['wrong_owner_rejected']and owner['actual_weight_layout']==[3584,7168,7168,1,1,7168,3584,3584,1,1]
  for projection in range(2):
   records=[v for v in owner['records']if v['projection']==projection];assert[v['mode']for v in records]==[0,1,0,1,1,0]
   assert all(v['rc']==0 and v['output_valid_by_producer']and v['vs_CPU_reference']['finite']and v['vs_CPU_reference']['NRMSE']<=.02 and v['vs_original_OFF']['NRMSE']<=.02 for v in records)
  assert owner['entry_affinity']==owner['affinity_after_create']==owner['final_affinity']and len(owner['entry_affinity'])>=4
  q.update(owner_report=dict(path=str(f),SHA256=sha(f),PID=owner['PID'],physical_PCI=owner['physical_PCI']),normal1=True,srun_exit0=True,Slurm_COMPLETED0=True)
  q['post']=probe('post',[owner['PID']]);assert owner['physical_PCI'].lower()in[p.lower()for p in q['post']['physical_PCI']]
  q.update(passed=True,component_numerics_and_lifecycle_diagnostic=True,performance_audit_required=True,Model_quality_qualified=False)
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
