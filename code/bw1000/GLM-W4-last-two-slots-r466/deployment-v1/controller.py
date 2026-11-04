from pathlib import Path
import argparse,datetime,hashlib,json,os,re,signal,socket,subprocess,sys,time,traceback
from startup_gate import StartupGate, StartupTimeout, seal_startup_records, drain_after_stop
D=Path(__file__).resolve().parent
R=D.parents[2]
PACKAGE=R/'incoming/GLM444-verified-ready-socket-CPU-v1/provider'
CLIENT_PACKAGE=R/'incoming/GLM-W4-fewshot-quality-r444-client-v1'
SLOT=None;PARENT=None;RUN=None;LABEL='GLM-r466-last-two';NODES=None;CONFIG=None

def configure_slot(index):
 global SLOT,PARENT,RUN,NODES,CONFIG
 assert type(index)is int and index in (4,5), 'Only explicitly claimed last slots4/5 are admitted; prior slots must not replay'
 slots=json.loads((D/'SLOTS.json').read_bytes())['slots'];assert len(slots)==8
 SLOT=slots[index];assert SLOT['slot']==SLOT['shard_index']==index and SLOT['shards']==8
 PARENT=SLOT['parent'];RUN=SLOT['run'];NODES=SLOT['nodes'];CONFIG=D/('config-slot%d.json'%index)
 assert SLOT['HTTP']==SLOT['Native']==64 and SLOT['cards']==16
 return SLOT

def scope_contract(m):
 config=json.loads(CONFIG.read_bytes())
 assert NODES==config['nodes']==m['resource_claim']['slots'][SLOT['slot']]['nodes']
 assert SLOT==m['resource_claim']['slots'][SLOT['slot']]
 assert (config['TP'],config['PP'],config['rows'],config['context'])==(8,2,1,32768)
 assert (config['Native_seconds'],config['service_seconds'])==(28800,28200)
 assert CONFIG.parent==D and PACKAGE==R/'incoming/GLM444-verified-ready-socket-CPU-v1/provider'
 assert SLOT['label']==LABEL and config['ready_metadata_IPs']==SLOT['ready_metadata_IPs'] and config['ready_port']==43844
 assert R in (Path('/refactor'),Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014'))
 return dict(slot=SLOT['slot'],parent=PARENT,nodes=NODES,cards=16,HTTP=64,shards=8,shard_index=SLOT['slot'],individual_original_epoch=True)

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean_environment(base):
 return{k:v for k,v in base.items()if k not in['BASH_ENV','ENV','PYTHONSTARTUP','PYTHONINSPECT']and((not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_','GLM_')))or k in['SLURM_CONF','SLURM_CONF_SERVER'])}
def publish(path,q):
 with Path(path).open('x')as f:json.dump(q,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def verify(digest):
 assert sha(D/'SOURCE_MANIFEST.json')==digest;m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 assert sha(PACKAGE/'SOURCE_MANIFEST.json')==m['provider_source_SHA256']
 for n,h in json.loads((PACKAGE/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(PACKAGE/n)==h,n
 assert sha(CLIENT_PACKAGE/'client-v1/SOURCE_MANIFEST.json')==m['client_manifest_SHA256']
 for n,h in json.loads((CLIENT_PACKAGE/'client-v1/SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(CLIENT_PACKAGE/n)==h,n
 assert sha(CONFIG)==m['config_files_SHA256'][CONFIG.name]
 m['config_SHA256']=sha(CONFIG);scope_contract(m);return m

def verify_gpu(digest,m):
 assert digest
 paths=[p for p in [D/'GPU_MANIFEST-last-two.json']if p.is_file()and sha(p)==digest]
 assert len(paths)==1
 q=json.loads(paths[0].read_bytes())
 assert q['GPU_GO']and q['Model_GO']and q['deployment_source_SHA256']==sha(D/'SOURCE_MANIFEST.json')
 assert SLOT['slot']in q['admitted_slots'], 'This fixed slot has not been admitted; no intent or job'
 assert q['admitted_slots']==[4,5], 'Only the two concrete remaining replicas are admitted'
 assert q.get('maximum_admitted_slots')==2
 assert q['provider_source_SHA256']==m['provider_source_SHA256']and q['resource_claim']==m['resource_claim']
 assert q['client_manifest_SHA256']==m['client_manifest_SHA256']
 for p,h in q['CPU_receipts_SHA256'].items():
  relative=Path(p).relative_to('/refactor');raw=(R/relative).read_bytes();assert hashlib.sha256(raw).hexdigest()==h
  result=json.loads(raw);assert result['passed']and result['completed']
 assert len(q['CPU_receipts_SHA256'])==4
 path='/refactor/candidates/GLM444-startup-diagnostic-v4-CPU/CPU-result.json'
 assert q['CPU_receipts_SHA256'][path]=='0ee1db3bb566f297528574ed3c4d2d2c04800800071ac7223e14344c8b0370c2'
 result=json.loads((R/Path(path).relative_to('/refactor')).read_bytes())
 assert result['actual_CPU']=='910407.939'and result['source_manifest_SHA256']=='831609820dff256dbc64107d571bea0cb5568cbdf89320c8d099a447ac1d8fbe'
 assert result['GPU']==0 and not result['Native_dlopen']and not result['SDK_executed']
 extra=q['verified_ready_CPU_receipts'];assert set(extra)=={'883','884'}
 for step,items in extra.items():
  assert len(items)==2
  for role,item in enumerate(items):
   relative=Path(item['path']).relative_to('/refactor');raw=(R/relative).read_bytes();assert hashlib.sha256(raw).hexdigest()==item['SHA256']
   value=json.loads(raw);assert value['role']==role and value['actual_CPU']=='910376.'+step
   assert value['recording_metadata_only']and value['GPU']==0 and not value['Native_dlopen']and not value['SDK']
   if step=='883':
    assert value['actual_whole_source_CLI']
    if role==1:assert len(value['cases'])==18 and all(c['passed']for c in value['cases'])and not value['errors']
    else:assert 'multiple values for argument' in value['errors'][0]['error']
   else:
    assert value['completed']and value['passed']and not value['errors']
    assert not value['Native_SDK_maps']and not value['forbidden_modules_present']
    assert value['reused_CPU883_only']['local_cases']==18
    assert value['post']==[dict(IP=['12.1.2.74','12.1.2.75'][role],port=p,bindable=True)for p in(43843,43844)]
 receipt=q['remaining_scope_CPU_receipt']
 assert receipt['source_manifest_SHA256']==sha(D/'SOURCE_MANIFEST.json')
 raw=(R/Path(receipt['path']).relative_to('/refactor')).read_bytes()
 assert hashlib.sha256(raw).hexdigest()==receipt['SHA256']
 value=json.loads(raw)
 assert value['completed']and value['passed']and value['slots']==[4,5]
 assert value['source_manifest_SHA256']==sha(D/'SOURCE_MANIFEST.json')
 assert value['GPU']==0 and not value['Native_dlopen']and not value['SDK_execution']
 assert value['actual_CPU'].startswith('910407.')
 return True
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
    assert re.fullmatch(re.escape(str(PARENT))+r'[.]\d+',got[0]) and(step is None or step==got[0]),'Owned cleanup step identity mismatch'
    step=got[0];row=got
   result['observations'].append(dict(step=step,row=got,srun_rc=proc.poll(),group_live=group_live()))
  except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError,AssertionError,ValueError)as error:
   result['observations'].append(dict(step=step,observation_error=repr(error),srun_rc=proc.poll(),group_live=group_live()))
  terminal=bool(row and row[2].split()[0].rstrip('+') in TERMINAL)
  # If accounting never yielded a step, absence is UNKNOWN, even if the local srun exits.
  return terminal and proc.poll() is not None and not group_live()
 def signal_owned(kind):
  if step is not None:
   assert re.fullmatch(re.escape(str(PARENT))+r'[.]\d+',step),'Never signal an unknown or parent allocation'
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

def gpu_environment(base,epoch,m):
 env=clean_environment(base)
 env.update(WAVE_INSIDE_R2323='GLM-W4-last-two-slots-r466/deployment-v1/inside.sh',GLM_R444_SLOT=str(SLOT['slot']),
  GLM_R425_RUN=RUN,GLM_R425_GPUSTART=str(epoch),GLM_R425_CONFIG_SHA=m['config_SHA256'],
  GLM_R425_SOURCE_SHA=m['provider_source_SHA256'],GLM_SOURCE_MANIFEST_SHA=m['provider_source_SHA256'],GLM_RUN_R5040=RUN,
  GLM_PP_SIZE_R3603='2',GLM_CACHE_ROWS_R3627='1',GLM_CACHE_CAP_R3627='32768',GLM_NETWORK_TRIAL='shca',
  GLM_BOUNDED_ROPE_R4279='1',GLM_SHARED_CONTROL_R4279='1',GLM_PIPELINE_TAIL_NODE=NODES[1],
  GLM_EXACT_BANKS_R4188='1',GLM_CHANNELS_R4363='0',GLM_VARIANT_R4363='layer77-5783',
  INFINILM_OWNER_CPU_WIDTH_R2392='4',USE_WEIGHT_CACHE_R3604='0',GLM_SERVICE_HOST_R143='127.0.0.1',GLM_SERVICE_PORT_R143='43843')
 return env

def gpu_command():
 return ['srun','--jobid='+str(PARENT),'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1',
  '-N2','-n2','--ntasks-per-node=1','-c64','--gres=dcu:8','--mem=224000M','-w',','.join(NODES),'-t','08:10:00',
  '--job-name='+LABEL+'-'+RUN,'--export=ALL','--output='+str(R/'logs'/(LABEL+'-'+RUN+'-%J-%t.log')),
  'bash',str(D/'resource_wave_host_alias_outer.sh'),'probe']

def client_command(actual,m,epoch):
 return ['srun','--jobid='+str(PARENT),'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2',
  '--gres=none','--mem=8192M','-w',NODES[1],'-t','07:50:00','--job-name='+LABEL+'-'+RUN+'-client','--export=ALL',
  '--output='+str(R/'logs'/(LABEL+'-'+RUN+'-client-%J.log')),'bash',str(D/'resource_wave_host_alias_outer.sh'),'probe']

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--preflight-only',action='store_true');p.add_argument('--slot',type=int,required=True);p.add_argument('--gpu-manifest-sha');a=p.parse_args(argv);configure_slot(a.slot)
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit);m=verify(a.manifest_sha)
 from preflight import check
 proof=check(D,PACKAGE,m,SLOT,gpu_command())
 if a.preflight_only:print(json.dumps(dict(passed=True,source_only=True,GPU=0,Slurm=0,proof=proof)));return 0
 assert verify_gpu(a.gpu_manifest_sha,m)and os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]=='a14r2n20'
 cleanup_limit=time.monotonic()+32380;watch=R/'results'/(LABEL+'-'+RUN+'-watch.json');folder=R/'results/GLM-W4-fewshot-quality-r444'/RUN
 assert not folder.exists(),'Existing run directory: reconcile original; never replay'
 w=dict(completed=False,passed=False,publication_commit=a.publication_commit,deployment_source_SHA256=a.manifest_sha,
  provider_source_SHA256=m['provider_source_SHA256'],CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
  HTTP_planned=64,planned_global_HTTP=512,slot=SLOT['slot'],resource_claim=SLOT,warms=0,retry=False,reload=False,forced_cleanup=False,actual_GPU=None,actual_client=None)
 publish(watch,w);gpu=client=None;epoch=None;stop_sent=False;visibility={};expected_pids={}
 def save():
  w['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=watch.with_suffix('.tmp');t.write_text(json.dumps(w,indent=2,allow_nan=False)+'\n');t.replace(watch)
 def command(args,timeout=25):return subprocess.check_output(args,text=True,timeout=min(timeout,max(1,cleanup_limit-time.monotonic())))
 def rows(timeout=25):return [x.split('|')for x in command(['sacct','-S','2026-10-31','-j',str(PARENT),'-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100'],timeout=timeout).splitlines()if'|'in x]
 def hosts(value):return command(['scontrol','show','hostnames',value]).splitlines()
 def exact(label,timeout=25):
  values=[r for r in rows(timeout)if r[1]==label];assert len(values)<=1,'Duplicate owned label'
  return values[0]if values else None
 def visible(path):
  path=Path(path)
  try:q=json.loads(path.read_bytes())
  except(FileNotFoundError,json.JSONDecodeError)as error:
   since=visibility.setdefault(str(path),time.monotonic())
   if time.monotonic()-since>=90:raise TimeoutError('Bounded terminal artifact visibility:'+str(path))from error
   return None
  visibility.pop(str(path),None);return q
 def stop():
  nonlocal stop_sent
  if stop_sent or not w['actual_GPU']or not folder.exists():return
  # Pure protocol import; no Native/Torch/HTTP. Exact shared Run identity.
  sys.path.insert(0,str(PACKAGE));from public_glm_native.protocol import Run
  config=json.loads(CONFIG.read_bytes());config.update(runtime_directory=str(R/'results/GLM-W4-fewshot-quality-r444'),
   run=RUN,actual_step=w['actual_GPU'],source_manifest_SHA256=m['provider_source_SHA256'],provider_config_SHA256=m['config_SHA256'],gpu_start_epoch=epoch)
  Run(config).stop();stop_sent=True;w['normal_STOP_published']=True;save()
 def idle(phase):
  # Even a metadata/collector failure cannot discard already recorded child PIDs.
  for stage,node in enumerate(NODES):
   known=set(expected_pids.get(node,[]))
   for name,key in [('leader-%d.json'%stage,'PID'),('actors-%d.json'%stage,'owned_process_PIDs')]:
    try:
     value=json.loads((folder/name).read_bytes())[key];known.update(value if isinstance(value,list)else[value])
    except(FileNotFoundError,json.JSONDecodeError,KeyError):pass
   expected_pids[node]=sorted(known)
  env=dict(clean_environment(os.environ),GLM_R425_EXPECTED_PIDS=json.dumps(expected_pids),GLM_R425_POST_DEADLINE_UNIX=str(min(time.time()+90,epoch+32280 if epoch else time.time()+90)))
  cmd=['srun','--jobid='+str(PARENT),'--overlap','--exact','--immediate=30','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(NODES),'-t','00:02:00','--job-name='+LABEL+'-'+RUN+'-'+phase,'python3','-B',str(D/'idle.py')]
  z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(115,max(1,cleanup_limit-time.monotonic())))
  w.setdefault('CPU_idle_observations',[]).append(dict(phase=phase,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr));save()
  values=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')]
  assert z.returncode==0 and len(values)==2 and {v['node']for v in values}==set(NODES)
  assert all(v['CPU_only']and not v['GPU_initialized']and not v['kfd_pids']and not v['matching_component_processes']and not v['expected_PIDs_still_exist']and v['ports_bindable']=={'43843':True,'43844':True}for v in values)
  assert len({(v['node'],pci)for v in values for pci in v['physical_PCI']})==16
  return values
 try:
  parent=command(['scontrol','show','job',str(PARENT),'-o']);assert'JobState=RUNNING'in parent and set(NODES)<=set(hosts(re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1)))
  assert re.search(r'(?:^| )UserId='+re.escape(os.environ['USER'])+r'\(',parent),'Allocation user mismatch'
  w['parent_allocation']=parent
  for row in rows():
   assert not row[1].startswith(LABEL+'-'+RUN),'Existing label: reconcile original, no replay'
   if re.fullmatch(re.escape(str(PARENT))+r'[.]\d+',row[0])and row[2].split()[0].rstrip('+')not in TERMINAL:
    assert not set(hosts(row[4]))&set(NODES),'Live overlapping step'
  w['fresh_before']=idle('before');epoch=time.time();w.update(original_GPU_start_unix=epoch,service_deadline_unix=epoch+28200,Native_deadline_unix=epoch+28800,GPU_deadline_unix=epoch+29400,keeper_deadline_unix=epoch+32400);save()
  with(R/'logs'/(LABEL+'-'+RUN+'-srun.log')).open('x')as log:
   gpu=subprocess.Popen(gpu_command(),env=gpu_environment(os.environ,epoch,m),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_srun_PID=gpu.pid,GPU_owned_PGID=gpu.pid);save()
  client_output=R/'results'/(LABEL+'-'+RUN+'-client');identity=client_output.with_name(client_output.name+'-identity.json')
  done_since=None;startup_gate=None
  while time.time()<epoch+29400:
   startup_allowed=False
   if client is None:
    if time.time()>=epoch+2700:raise StartupTimeout('Original GPUepoch startup2700 expired before client; no clock reset')
    if w['actual_GPU']and folder.exists():
     if startup_gate is None:
      sys.path.insert(0,str(PACKAGE));from public_glm_native.protocol import Run
      startup_config=json.loads(CONFIG.read_bytes());startup_config.update(runtime_directory=str(R/'results/GLM-W4-fewshot-quality-r444'),run=RUN,actual_step=w['actual_GPU'],source_manifest_SHA256=m['provider_source_SHA256'],provider_config_SHA256=m['config_SHA256'],gpu_start_epoch=epoch)
      startup_gate=StartupGate(Run(startup_config))
     try:startup_allowed=startup_gate.observe(time.time())
     finally:w.update(startup_gate.state());save()
   try:
    gr=exact(LABEL+'-'+RUN);cr=exact(LABEL+'-'+RUN+'-client')if client is not None else None
   except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as error:
    if gpu.poll()not in(None,0)or(client is not None and client.poll()not in(None,0)):raise RuntimeError('Owned srun nonzero during observation loss')from error
    w.setdefault('transient_account_errors',[]).append(dict(error=repr(error),original_epoch=epoch));save();time.sleep(15);continue
   for key,row in [('actual_GPU',gr),('actual_client',cr)]:
    if row:
     assert w[key]in(None,row[0]);w[key]=row[0];w[key+'_Slurm']=row
     if row[2].split()[0].rstrip('+')in TERMINAL:assert row[2]=='COMPLETED'and row[3]=='0:0','Owned execution failed:'+repr(row)
   if gpu.poll()not in(None,0):raise RuntimeError('Owned GPU srun nonzero')
   if client is not None and client.poll()not in(None,0):raise RuntimeError('Owned public client srun nonzero')
   ready=folder/'MODEL_READY.json'
   if client is None and startup_allowed and ready.exists()and w['actual_GPU']:
    assert startup_gate.observe(time.time()), 'Recheck actual STOP/FAILED before client launch'
    value=visible(ready)
    if value is not None:
     expected=dict(run=RUN,actual_step=w['actual_GPU'],source_manifest_SHA256=m['provider_source_SHA256'],provider_config_SHA256=m['config_SHA256'],gpu_start_epoch=epoch,Native_seconds=28800)
     assert value['identity']==expected and value['Model_READY']and value['actual_owner_count']==16
     env=gpu_environment(os.environ,epoch,m);env.update(WAVE_INSIDE_R2323='GLM-W4-last-two-slots-r466/deployment-v1/client_inside.sh',GLM_R444_ACTUAL_GPU=w['actual_GPU'],GLM_R444_DEPLOY_MF=a.manifest_sha,CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
     with(R/'logs'/(LABEL+'-'+RUN+'-client-srun.log')).open('x')as log:client=subprocess.Popen(client_command(w['actual_GPU'],m,epoch),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     w.update(client_srun_PID=client.pid,client_owned_PGID=client.pid,client_result=str(client_output/'result.json'));save()
   if client is not None and(client_output/'result.json').exists():
    value=visible(client_output/'result.json')
    if value is not None and value['completed']:
     w['client_result_SHA256']=sha(client_output/'result.json');w['client_passed']=value['execution_passed'];stop()
     assert value['execution_passed']and value['actual_HTTP']==value['planned_HTTP']==len(value['rows'])==64 and not value['errors'],'Client infrastructure/state failure; wrong/cap/parse retained without early quality stop'
   if client is not None and client.poll()==0 and gpu.poll()==0:
    done_since=time.monotonic()if done_since is None else done_since
    if gr and cr and gr[2:4]==cr[2:4]==['COMPLETED','0:0']and stop_sent:break
    if time.monotonic()-done_since>90:raise TimeoutError('srun/Slurm exit propagation bounded90s')
   elif gpu.poll()==0:
    done_since=time.monotonic()if done_since is None else done_since
    if time.monotonic()-done_since>90:raise RuntimeError('GPU exited before complete public plan')
   save();time.sleep(min(15,max(0,epoch+29400-time.time())))
  else:raise TimeoutError('Original GPU29400 deadline; no retry/reload')
  from collect import collect
  summary=collect(R,RUN,w['actual_GPU'],m['provider_source_SHA256'],client_output/'result.json',min(time.time()+90,epoch+32280),NODES,SLOT['slot'],CLIENT_PACKAGE)
  expected_pids=summary['expected_node_PIDs'];w.update(summary)
  w['post']=idle('post');w.update(completed=True,passed=True,state='NORMAL_64_FEWSHOT_PAIRED_REQUESTS_COMPLETE_QUALITY_SCORES_RETAINED')
 except BaseException as error:
  w.update(completed=True,passed=False,error=repr(error),traceback=traceback.format_exc(),state='FAILED_OR_UNKNOWN_NO_RETRY')
  startup_failed=client is None and gpu is not None
  if startup_failed:
   w['client_launch_blocked']=True;save()
   if folder.exists():
    try:w['startup_original_snapshot']=seal_startup_records(folder,watch.with_name(watch.stem+'-startup-diagnostics.json'));save()
    except BaseException as capture:w['startup_snapshot_error']=repr(capture);save()
  try:stop()
  except BaseException as close:w['STOP_error']=repr(close)
  if startup_failed and stop_sent and w['actual_GPU']:
   try:w['startup_STOP_drain']=drain_after_stop(gpu,exact,LABEL+'-'+RUN,w['actual_GPU']);save()
   except BaseException as close:w['startup_STOP_drain_error']=repr(close)
  for label,proc,key in [(LABEL+'-'+RUN,gpu,'actual_GPU'),(LABEL+'-'+RUN+'-client',client,'actual_client')]:
   if proc is None:continue
   if key=='actual_GPU'and w.get('startup_STOP_drain',{}).get('normal_GPU_STOP_drain'):continue
   w['forced_cleanup']=True
   try:w.setdefault('owned_cleanup',[]).append(close_owned(label,proc,w[key],exact,cleanup_limit))
   except BaseException as close:w.setdefault('cleanup_errors',[]).append(repr(close))
  try:w['post_after_failure']=idle('failure-post')
  except BaseException as close:w['post_error']=repr(close)
 finally:save()
 print(json.dumps({k:w.get(k)for k in ['completed','passed','CPU_keeper','actual_GPU','actual_client','state','error']}));return 0 if w['passed']else 1

if __name__=='__main__':raise SystemExit(main())

