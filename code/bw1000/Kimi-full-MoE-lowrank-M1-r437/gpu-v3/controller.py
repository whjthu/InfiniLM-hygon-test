"""One bounded actual TP16/PP1 complete-MoE component, two eight-owner nodes."""
from pathlib import Path,PurePosixPath
import json,hashlib,subprocess,os,re,time,signal,traceback,datetime
NAME='Kimi-full-MoE-lowrank-M1-r437/gpu-v3';LABEL='Kimi-r437-full-MoE-lowrank-v3';GPU_SECONDS=600
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
 root=Path(root);return['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','-w','a02r3n03,a02r3n07','--gres=dcu:8','--mem=128000M','-t','00:10:00','--job-name='+LABEL,'--output='+str(root/'logs'/(LABEL+'-%J-%t.log')),'--export=ALL','bash',str(root/'incoming'/NAME/'resource_wave_host_alias_outer.sh'),'probe']
def start(root,pin,popen=None,start_unix=None,master_addr=None):
 popen=popen or subprocess.Popen;args=gpu_command(root);env=clean_env();env.update(WAVE_INSIDE_R2323=NAME+'/GPU_inside.sh',KIMI_R437_GPU_SOURCE_SHA=pin,KIMI_R437_GPU_START=str(start_unix if start_unix is not None else time.time()),KIMI_R437_MASTER_ADDR=master_addr or"recording-only-address")
 with(Path(root)/'logs'/(LABEL+'-srun.log')).open('x')as f:p=popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 return p,args
def verify(root,go):
 assert go['execute_once']is True and re.fullmatch('[0-9a-f]{40}',go['publication_commit'])and go['publication_commit']!='0'*40
 d=Path(root)/'incoming'/NAME;assert Path(__file__).resolve().parent==d.resolve(),'Actual controller path identity'
 assert NAME=='Kimi-full-MoE-lowrank-M1-r437/gpu-v3'and LABEL=='Kimi-r437-full-MoE-lowrank-v3'
 mf=d/'GPU_SOURCE_MANIFEST.json';assert sha(mf)==go['GPU_source_manifest_SHA256'];m=json.loads(mf.read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(d/n)==h,n
 host=lambda value:Path(root).joinpath(*PurePosixPath(value).relative_to('/refactor').parts)
 receipt=host(m['CPU_receipt']);assert sha(receipt)==m['CPU_receipt_SHA256'];artifact=json.loads(receipt.read_bytes())
 assert artifact['passed']and artifact['actual_CPU']=='910407.894'and artifact['new_production_HOST_TUs']==artifact['pure_sched_contract_HOST_TUs']==1 and artifact['new_GPU_TUs']==0
 assert artifact['library_SHA256']==m['library_SHA256']and sha(host(artifact['library']))==m['library_SHA256']
 new_source=host(m['owner_source_root']);assert sha(new_source/'SOURCE_MANIFEST.json')==m['owner_source_SHA256']
 for n,h in json.loads((new_source/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(new_source/n)==h,n
 unit=artifact['compiled'];assert sha(host(unit['object']))==unit['object_SHA256']and sha(host(unit['MD']))==unit['MD_SHA256']
 for p,h in unit['dependency_SHA256'].items():
  if PurePosixPath(p).is_relative_to('/refactor'):assert sha(host(p))==h,p
  else:assert p.startswith(('/usr/','/opt/'))
 parent=host(m['parent_source_root']);assert sha(parent/'SOURCE_MANIFEST.json')==m['parent_source_SHA256']
 for n,h in json.loads((parent/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(parent/n)==h,n
 old=host(m['compiled_CPU883_receipt']);assert sha(old)==m['compiled_CPU883_receipt_SHA256'];compiled=json.loads(old.read_bytes())
 for unit in compiled['compiled']:
  assert sha(host(unit['object']))==unit['object_SHA256']and sha(host(unit['MD']))==unit['MD_SHA256']
  for p,h in unit['dependency_SHA256'].items():
   if PurePosixPath(p).is_relative_to('/refactor'):assert sha(host(p))==h,p
   else:assert p.startswith(('/usr/','/opt/')),'Unexpected dependency namespace'
 for p,h in m['mapped_dependencies_SHA256'].items():assert sha(host(p))==h,p
 assert sha(host(m['BLAS_policy']['path']))==m['BLAS_policy']['SHA256']
 claim=m['resource_claim'];assert(claim['parent'],claim['nodes'],claim['cards'],claim['cores_per_node'])==(910371,['a02r3n03','a02r3n07'],16,64)
 assert claim['GPU_max_s']==600 and claim['keeper_max_s']==900 and claim['TP']==16 and claim['PP']==1
 assert m['modes']==[0,1,0,1,1,0]and m['calls_per_owner']==6 and m['total_forward_calls']==96
 args=gpu_command(root);assert '-N2'in args and '-n2'in args and '-c64'in args and '--gres=dcu:8'in args and args[args.index('-w')+1]==','.join(claim['nodes'])
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
 root=Path(root);m=verify(root,go);watch=root/'results/Kimi437-watch-v3.json';q=dict(completed=False,passed=False,GPU_cards=16,publication_commit=go['publication_commit'],GPU_source_manifest_SHA256=go['GPU_source_manifest_SHA256'],submissions=[],actual_step=None,original_start=time.time(),profile=0,no_retry=True)
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
  row=matches[0];assert re.fullmatch(r'910371\.\d+',row[0])and command(['scontrol','show','hostnames',row[4]]).splitlines()==['a02r3n03','a02r3n07'];assert actual in [None,row[0]];actual=row[0];q['actual_step']=actual;return row
 def probe(suffix,expected):
  args=['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','-w','a02r3n03,a02r3n07','--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+suffix,'python3','-B',str(root/'incoming'/NAME/'resource_probe.py'),json.dumps(expected)]
  probe_env=dict(env,KIMI_R437_POST_DEADLINE=str(min(time.time()+90,q['original_start']+880)))
  z=subprocess.run(args,env=probe_env,text=True,capture_output=True,timeout=min(150,max(1,deadline-time.monotonic())));records=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')];q.setdefault('CPU_probe_records',[]).append(dict(phase=suffix,rc=z.returncode,stdout=z.stdout,stderr=z.stderr));save();assert z.returncode==0 and len(records)==2 and all(v['passed']for v in records)and{v['node']for v in records}=={'a02r3n03','a02r3n07'},z.stderr[-1000:];assert len({(v['node'],p)for v in records for p in v['physical_PCI']})==16;return records
 def known_pids():
  found={}
  if actual:
   for node,host in enumerate(['a02r3n03','a02r3n07']):
    f=root/'results'/('Kimi437-'+actual+'-node%d.json'%node)
    try:found[host]=[json.loads(f.read_bytes())['PID']]
    except(FileNotFoundError,json.JSONDecodeError,KeyError):pass
  return found

 try:
  parent=command(['scontrol','show','job','910371','-o']);assert'UserId=aclsylqidf('in parent and'JobState=RUNNING'in parent;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',parent);assert nodes and{'a02r3n03','a02r3n07'}<=set(command(['scontrol','show','hostnames',nodes.group(1)]).splitlines())
  assert account()is None
  for row in rows():
   if re.fullmatch(r'910371\.\d+',row[0])and row[2].split()[0].rstrip('+')not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE']:
    assert not{'a02r3n03','a02r3n07'}&set(command(['scontrol','show','hostnames',row[4]]).splitlines()),'Foreign active step on assigned nodes'
  q['before']=probe('before',{});head=next(v for v in q['before']if v['node']=='a02r3n03');assert head['ib_ipv4'];master=next(iter(head['ib_ipv4'].values()));__import__('ipaddress').IPv4Address(master);q['master_addr']=master;q['original_start']=time.time();save();p,args=start(root,go['GPU_source_manifest_SHA256'],start_unix=q['original_start'],master_addr=master);q['submissions'].append(dict(owned_srun_PID=p.pid,owned_original_PGID=p.pid,argv=args));save();unresolved=None
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
  until=min(deadline-150,time.monotonic()+90);reports=[];pids={}
  for node,host in enumerate(['a02r3n03','a02r3n07']):
   f=root/'results'/('Kimi437-'+actual+'-node%d.json'%node);owner=visible(f,until)
   assert owner['actual_step']==actual and owner['node_rank']==node and owner['node']==host and owner['source_manifest_SHA256']==go['GPU_source_manifest_SHA256']and owner['passed']and owner['normal_close']
   assert owner['visible_count']==8 and len(set(owner['physical_PCI_by_visible_device']))==8 and len(owner['raw_outputs'])==48
   inventory=next(v for v in q['before']if v['node']==host);assert set(owner['physical_PCI_by_visible_device'])=={v.lower()for v in inventory['physical_PCI']}
   rows=owner['CAPI_result']['reports'];assert len(rows)==8 and owner['CAPI_result']['normal_CommunicationGroup_close']
   for rank,row in enumerate(rows):
    assert row['completed']and row['normal_Runtime_close']and row['both_modules_closed']and row['global_tp_rank']==node*8+rank
    assert row['affinity_restored']and row['restored_mask']==row['entry_mask']and row['pinned_mask']==row['owner_mask']==row['entry_mask'][rank*4:rank*4+4]and[v['mode']for v in row['arms']]==[0,1,0,1,1,0]
    assert all(v['down_up_selected_delta']==[v['mode'],v['mode']]and v['nonfinite']==0 and v['input_and_tail_canary_unchanged']for v in row['arms'])
   reports.append(dict(path=str(f),SHA256=sha(f),PID=owner['PID'],node=host));pids[host]=[owner['PID']]
  q.update(owner_reports=reports,normal16=True,srun_exit0=True,Slurm_COMPLETED0=True,all_96_forward_results_persisted=True)
  q['post']=probe('post',pids);q.update(passed=True,component_numerics_and_lifecycle_diagnostic=True,performance_audit_required=True,Model_quality_qualified=False)

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
