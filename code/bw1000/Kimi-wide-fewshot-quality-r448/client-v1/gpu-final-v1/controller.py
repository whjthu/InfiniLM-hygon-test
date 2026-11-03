from pathlib import Path
import argparse,datetime,hashlib,json,os,re,signal,socket,subprocess,sys,time,traceback
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/Kimi-wide-fewshot-quality-r448/client-v1'
FINAL=D/'gpu-final-v1'
sys.path.append(str(D))
PACKAGE=R/'incoming/Kimi-public-incremental-text-r434'
PARENT=R/'incoming/public-Kimi-native-provider-r428/runtime-v2'
RUN='public-kimi-wide-20261104-a'
LABEL='Kimi-r448-wide-quality'
NODES=['a14r2n08','a14r2n09','a14r2n10','a14r2n12']
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean_environment(base):
 return{k:v for k,v in base.items()if k not in['BASH_ENV','ENV','PYTHONSTARTUP','PYTHONINSPECT']and((not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_','GLM_','KIMI_','INFINILM_','GLOBAL_')))or k in['SLURM_CONF','SLURM_CONF_SERVER'])}
def publish(path,q):
 with Path(path).open('x')as f:json.dump(q,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def host(path):
 return R/Path(path).relative_to('/refactor')

def verify(digest):
 scope_contract()
 assert sha(D/'SOURCE_MANIFEST.json')==digest;m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 for name in ['inside.sh','resource_wave_host_alias_outer.sh']:assert b'\r'not in(D/name).read_bytes(),'Published Shell must use LF: '+name
 assert sha(PACKAGE/'SOURCE_MANIFEST.json')==m['provider_source_SHA256']
 for n,h in json.loads((PACKAGE/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(PACKAGE/n)==h,n
 assert sha(D/'config.json')==m['config_SHA256']
 config=json.loads((D/'config.json').read_bytes())
 assert Path(__file__).resolve().parent==FINAL.resolve(),'Final controller path mismatch'
 assert NODES==config['nodes']==m['resource_claim']['nodes'],'Controller/config/claim node identity mismatch'
 claim=m['resource_claim'];assert claim['parent']==910380 and claim['cards']==len(NODES)*8==32
 assert (claim['TP'],claim['PP'],claim['cores_per_node'],claim['host_mem_MiB_per_node'])==(16,2,64,128000)
 assert (claim['GPU_max_s'],claim['Native_max_s'],claim['service_s'],claim['CPU_keeper']['max_s'])==(29400,28800,28200,32400)
 assert (claim['HTTP'],claim['Native_requests'],claim['warm'])==(256,256,0)
 assert claim['original_epoch_only']and not claim['retry']and not claim['reload']
 assert PACKAGE==R/'incoming/Kimi-public-incremental-text-r434'
 assert sha(PARENT/'SOURCE_MANIFEST.json')==m['qualified_parent_SHA256']
 for n,h in json.loads((PARENT/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(PARENT/n)==h,n
 assert config['model_manifest'].startswith('/refactor/incoming/public-Kimi-native-provider-r428/runtime-v2/')
 assert config['Native_SHA256']=='8df87f0741b13c90f658174a793f0068efb4eb8097eddf1d5f44edb4d208e1a2'
 public=host(config['public_provider_root']);assert sha(public/'SOURCE_MANIFEST.json')=='122b69d7c1571f1523e15c672839e2d4479e91863e80018b4a8bb60c2ce93724'
 for n,h in json.loads((public/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(public/n)==h,n
 assert sha(host(config['model_manifest']))==config['model_manifest_SHA256'];model=json.loads(host(config['model_manifest']).read_bytes())
 native=host(model['native_receipt']);assert sha(native)==model['native_receipt_SHA256'];receipt=json.loads(native.read_bytes())
 assert receipt['completed']and receipt['CPU_only']and not receipt['GPU']and not receipt['Native_dlopen']
 assert receipt['native_SHA256']==config['Native_SHA256']and sha(host(receipt['native']))==config['Native_SHA256']
 for p,h in receipt['all_linked_object_SHA256'].items():assert sha(host(p))==h,p
 assert sha(host(config['Common_extension']))==config['Common_extension_SHA256']
 cpu=host(m['text_CPU_receipt']);assert sha(cpu)==m['text_CPU_receipt_SHA256'];q=json.loads(cpu.read_bytes())
 assert q['passed']and q['completed']and q['actual_CPU']=='910407.876'and len(q['cases'])==11 and q['GPU']==0 and not q['Native_dlopen']
 assert q['source_manifest_SHA256']==m['provider_source_SHA256']
 for field in ['initial_module_maps','final_module_maps']:
  assert not q[field]['project_Native_maps']
 assert config['qualified_parent_root']=='/refactor/incoming/public-Kimi-native-provider-r428/runtime-v2'
 assert (config['Native_seconds'],config['service_seconds'])==(28800,28200)
 enc=host(m['encoding_CPU_receipt']);assert sha(enc)==m['encoding_CPU_receipt_SHA256'];q=json.loads(enc.read_bytes())
 assert q['passed']and q['completed']and q['actual_CPU']=='910407.937'and len(q['cases'])==256
 assert q['actual_Kimi_codec']and q['actual_public_protocol']and q['actual_prepared_FIFO']and not q['GPU']and not q['Native_dlopen']
 pins=json.loads((D/'INPUT_PINS.json').read_bytes());assert pins['encoding_SHA256']==m['encoding_CPU_receipt_SHA256']
 assert sha(host(pins['requests_path']))==pins['requests_SHA256']==q['requests_file_SHA256']
 assert sha(host(pins['cases_path']))==pins['cases_SHA256']
 return m

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

def gpu_environment(base,epoch,m,nonce):
 env=clean_environment(base)
 env.update(WAVE_INSIDE_R2323='Kimi-wide-fewshot-quality-r448/client-v1/inside.sh',
  KIMI_R428_RUN=RUN,KIMI_R428_GPUSTART=str(epoch),KIMI_R428_CONFIG_SHA=m['config_SHA256'],
  KIMI_R428_SOURCE_SHA=m['provider_source_SHA256'],KIMI_R428_NONCE=nonce,
  KIMI_USE_RAMCACHE_R3679='0',KIMI_REDUCE_POLICY='0',
  INFINILM_DENSE_PREFILL_F32_R2475='1',INFINILM_MXFP4_TILE_TP_R2559='0',
  GLOBAL_OFFSET_R2354='0',GLOBAL_MODEL_MODE_R2354='shca',KIMI_CAPACITY_ONLY_R9145='0',KIMI_TP_NODES='2',KIMI_PP_SIZE='2')
 return env

def gpu_command():
 return ['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1',
  '-N4','-n4','--ntasks-per-node=1','-c64','--gres=dcu:8','--mem=128000M','-w',','.join(NODES),'-t','08:10:00',
  '--job-name='+LABEL+'-'+RUN,'--export=ALL','--output='+str(R/'logs'/(LABEL+'-'+RUN+'-%J-%t.log')),
  'bash',str(D/'resource_wave_host_alias_outer.sh'),'probe']

def client_command(actual,m,epoch):
 output=R/'results'/(LABEL+'-'+RUN+'-client');pins=json.loads((D/'INPUT_PINS.json').read_bytes())
 config=json.loads((D/'config.json').read_bytes())
 return ['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2',
  '--gres=none','--mem=8192M','-w',NODES[0],'-t','07:50:00','--job-name='+LABEL+'-'+RUN+'-client',
  '--output='+str(R/'logs'/(LABEL+'-'+RUN+'-client-%J.log')),'python3','-B','-u',str(D/'client_launch.py'),
  '--identity',str(output.with_name(output.name+'-identity.json')),'--client',str(D/'client.py'),'--client-sha',m['files_SHA256']['client.py'],
  '--base-url','http://127.0.0.1:43844/v1','--runtime-dir',str(R/'results/public-Kimi-text-provider-r434'),
  '--run',RUN,'--expected-step',actual,'--source-sha',m['provider_source_SHA256'],'--config-sha',m['config_SHA256'],
  '--model-sha',config['model_manifest_SHA256'],'--cases',str(host(pins['cases_path'])),
  '--requests',str(host(pins['requests_path'])),'--encoding-receipt',str(host(pins['encoding_path'])),
  '--output-dir',str(output),'--deadline-unix',str(epoch+28200),'--gpu-start-epoch',str(epoch)]

def scope_contract():
 assert D==R/'incoming/Kimi-wide-fewshot-quality-r448/client-v1'
 assert PACKAGE==R/'incoming/Kimi-public-incremental-text-r434'and len(set(NODES))==4
 assert RUN=='public-kimi-wide-20261104-a'and LABEL=='Kimi-r448-wide-quality'
 gpu=gpu_command();assert all(v in gpu for v in ['--jobid=910380','-N4','-n4','-c64','--gres=dcu:8','--mem=128000M','08:10:00'])
 assert gpu[gpu.index('-w')+1]==','.join(NODES)
 m=dict(files_SHA256={'client.py':'CPU-contract-client'},provider_source_SHA256='CPU-source',config_SHA256='CPU-config')
 argv=client_command('910380.0',m,100.)
 assert argv[argv.index('--deadline-unix')+1]=='28300.0'and argv[argv.index('--gpu-start-epoch')+1]=='100.0'
 assert argv[argv.index('--expected-step')+1]=='910380.0'and argv[argv.index('--client')+1]==str(D/'client.py')
 assert '--gres=none'in argv and argv[argv.index('-w')+1]==NODES[0]
 return dict(nodes=NODES,cards=32,TP=16,PP=2,C=1,external_requests=256,warm=0,
  GPU_s=29400,Native_s=28800,service_s=28200,keeper_s=32400,original_epoch=True,
  actual_GPU_argv=gpu,actual_client_argv=argv,Slurm_executed=False,Native=False)

def verify_execution(expected,m):
 assert re.fullmatch('[0-9a-f]{64}',expected)
 assert sha(FINAL/'GPU_MANIFEST.json')==expected;go=json.loads((FINAL/'GPU_MANIFEST.json').read_bytes())
 assert go['GPU_GO']and go['Model_GO']and not go['reload']and not go['retry']
 assert go['production_source_manifest_SHA256']==sha(D/'SOURCE_MANIFEST.json')
 assert not m['GPU_GO']and not m['Model_GO'],'Original CPU-only claim must remain false'
 assert sha(FINAL/'SOURCE_MANIFEST.json')==go['final_source_manifest_SHA256'];mf=json.loads((FINAL/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in mf['files_SHA256'].items():assert sha(FINAL/n)==h,n
 assert go['resource_claim']['cards']==32 and go['resource_claim']['nodes']==NODES and go['HTTP']==256
 assert (go['resource_claim']['Native_max_s'],go['resource_claim']['service_s'],go['resource_claim']['GPU_max_s'],go['resource_claim']['CPU_keeper']['max_s'])==(28800,28200,29400,32400)
 cpu={}
 for name,pin in go['actual_CPU_receipts'].items():
  path=host(pin['path']);assert sha(path)==pin['SHA256'];cpu[name]=json.loads(path.read_bytes())
 assert cpu['937']['passed']and cpu['937']['completed']and cpu['937']['actual_CPU']=='910407.937'and len(cpu['937']['cases'])==256
 assert not cpu['942']['passed']and cpu['942']['completed']and cpu['942']['error']=="KeyError('node')"and cpu['942']['phases']==['actual_controller_verify_scope_source937_and_parent_metadata_bytes','actual_final_Bash_exports_typedCLI4roles_noServe']
 assert cpu['943']['passed']and cpu['943']['completed']and cpu['943']['production_manifest_SHA256']==go['production_source_manifest_SHA256']
 assert [r['actual_local_HTTP']for r in cpu['943']['actual_changed_entry']['scenarios']]==[256,1,0,0]
 assert all(not q['GPU']and not q['Native_dlopen']for q in cpu.values())
 assert cpu['943']['inherited942_source_and_actualBash4role_proof']['receipt_SHA256']==go['actual_CPU_receipts']['942']['SHA256']
 return go

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--preflight-only',action='store_true');p.add_argument('--execution-manifest-sha',required=True);a=p.parse_args(argv)
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit);m=verify(a.manifest_sha);go=verify_execution(a.execution_manifest_sha,m)
 from preflight import check
 proof=check(D,PACKAGE,m)
 if a.preflight_only:print(json.dumps(dict(passed=True,source_only=True,GPU=0,Slurm=0,proof=proof)));return 0
 assert go['GPU_GO']and go['Model_GO']and os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]=='a14r2n20'
 cleanup_limit=time.monotonic()+32380;watch=R/'results'/(LABEL+'-'+RUN+'-watch.json');folder=R/'results/public-Kimi-text-provider-r434'
 assert not list(folder.glob('*'+RUN+'*')),'Existing run artifacts: follow original, never replay'
 w=dict(completed=False,passed=False,publication_commit=a.publication_commit,deployment_source_SHA256=a.manifest_sha,
  provider_source_SHA256=m['provider_source_SHA256'],CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
  execution_manifest_SHA256=a.execution_manifest_sha,final_source_manifest_SHA256=go['final_source_manifest_SHA256'],HTTP_planned=256,warms=0,retry=False,reload=False,forced_cleanup=False,actual_GPU=None,actual_client=None)
 publish(watch,w);gpu=client=None;epoch=None;stop_sent=False;visibility={};expected_pids={}
 def save():
  w['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=watch.with_suffix('.tmp');t.write_text(json.dumps(w,indent=2,allow_nan=False)+'\n');t.replace(watch)
 def command(args,timeout=25):return subprocess.check_output(args,text=True,timeout=min(timeout,max(1,cleanup_limit-time.monotonic())))
 def rows():return [x.split('|')for x in command(['sacct','-S','2026-10-31','-j','910380','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100']).splitlines()if'|'in x]
 def hosts(value):return command(['scontrol','show','hostnames',value]).splitlines()
 def exact(label):
  values=[r for r in rows()if r[1]==label];assert len(values)<=1,'Duplicate owned label'
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
  # Exact source/config/run/original-epoch STOP identity consumed by public watcher.
  path=folder/('STOP-'+RUN+'.json')
  identity=dict(run=RUN,source_manifest_SHA256=m['provider_source_SHA256'],config_SHA256=m['config_SHA256'],gpu_start_epoch=epoch)
  publish(path,dict(identity=identity,action='stop'));stop_sent=True;w['normal_STOP_published']=True;save()

 def idle(phase):
  client_id=R/'results'/(LABEL+'-'+RUN+'-client-identity.json')
  if phase!='before'and client_id.exists():
   v=json.loads(client_id.read_bytes());assert v['client_SHA256']==m['files_SHA256']['client.py']and v['client']==str(D/'client.py')and v['node']==NODES[0]
   if w['actual_client']:assert v['actual_CPU']==w['actual_client']
   assert type(v['PID'])is int and v['PID']>0
   expected_pids.setdefault(NODES[0],[]).append(v['PID']);w['actual_client_process_identity']=v
  for node_index,node in enumerate(NODES):
   known=set(expected_pids.get(node,[]));claim=folder/('claim-'+RUN+'-node%d.json'%node_index)
   try:known.add(json.loads(claim.read_bytes())['PID'])
   except(FileNotFoundError,json.JSONDecodeError,KeyError):pass
   expected_pids[node]=sorted(known)
  env=dict(clean_environment(os.environ),KIMI_R428_EXPECTED_PIDS=json.dumps(expected_pids),KIMI_R428_POST_DEADLINE_UNIX=str(min(time.time()+90,epoch+32280 if epoch else time.time()+90)))
  cmd=['srun','--jobid=910380','--overlap','--exact','--immediate=30','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(NODES),'-t','00:02:00','--job-name='+LABEL+'-'+RUN+'-'+phase,'python3','-B',str(D/'idle.py')]
  z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(115,max(1,cleanup_limit-time.monotonic())))
  w.setdefault('CPU_idle_observations',[]).append(dict(phase=phase,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr));save()
  values=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')]
  assert z.returncode==0 and len(values)==4 and{v['node']for v in values}==set(NODES)
  assert all(v['CPU_only']and not v['GPU_initialized']and not v['kfd_pids']and not v['matching_component_processes']and not v['expected_PIDs_still_exist']and all(v['ports_bindable'].values())for v in values)
  assert len({(v['node'],pci)for v in values for pci in v['physical_PCI']})==32
  return values

 try:
  parent=command(['scontrol','show','job','910380','-o']);assert'JobState=RUNNING'in parent and set(NODES)<=set(hosts(re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1)))
  assert re.search(r'(?:^| )UserId='+re.escape(os.environ['USER'])+r'\(',parent),'Allocation user mismatch'
  w['parent_allocation']=parent
  for row in rows():
   assert not row[1].startswith(LABEL+'-'+RUN),'Existing label: reconcile original, no replay'
   if re.fullmatch(r'910380[.]\d+',row[0])and row[2].split()[0].rstrip('+')not in TERMINAL:
    assert not set(hosts(row[4]))&set(NODES),'Live overlapping step'
  w['fresh_before']=idle('before');nonce=__import__('secrets').token_hex(24);w['nonce_SHA256']=hashlib.sha256(nonce.encode()).hexdigest();epoch=time.time();w.update(original_GPU_start_unix=epoch,service_deadline_unix=epoch+28200,Native_deadline_unix=epoch+28800,GPU_deadline_unix=epoch+29400,keeper_deadline_unix=epoch+32400);save()
  with(R/'logs'/(LABEL+'-'+RUN+'-srun.log')).open('x')as log:
   gpu=subprocess.Popen(gpu_command(),env=gpu_environment(os.environ,epoch,m,nonce),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_srun_PID=gpu.pid,GPU_owned_PGID=gpu.pid);save()
  client_output=R/'results'/(LABEL+'-'+RUN+'-client');identity=client_output.with_name(client_output.name+'-identity.json')
  done_since=None
  while time.time()<epoch+29400:
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
   if client is None and w['actual_GPU']:
    from handoff import ready_handoff
    config=json.loads((D/'config.json').read_bytes())
    ready=ready_handoff(folder,RUN,w['actual_GPU'],m['provider_source_SHA256'],m['config_SHA256'],config['model_manifest_SHA256'])
    if ready is not None:
     w['actual_ready_handoff']=ready;save()
     # Client proves PID-owned HTTP LISTEN; firstPOST is part of the fixed256.
     env=clean_environment(os.environ)
     with(R/'logs'/(LABEL+'-'+RUN+'-client-srun.log')).open('x')as log:client=subprocess.Popen(client_command(w['actual_GPU'],m,epoch),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     w.update(client_srun_PID=client.pid,client_owned_PGID=client.pid,client_result=str(client_output/'result.json'),all4_weights_ready=True);save()
   if client is not None and(client_output/'result.json').exists():
    value=visible(client_output/'result.json')
    if value is not None and value['completed']:
     w['client_result_SHA256']=sha(client_output/'result.json');w['client_passed']=value['execution_passed'];stop()
     assert value['execution_passed']and value['attempted_HTTP']==value['completed_HTTP']==256,'Public infrastructure failure retained, wrong/cap/parse outcomes do not stop quality plan'
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
  summary=collect(R,RUN,w['actual_GPU'],m['provider_source_SHA256'],client_output/'result.json',min(time.time()+90,epoch+32280))
  expected_pids=summary['expected_node_PIDs'];w.update(summary)
  w['post']=idle('post');w.update(completed=True,passed=True,state='KIMI_PUBLIC_SINGLE_BASELINE256_COMPLETE_NORMAL32_QUALITY_RETAINED')
 except BaseException as error:
  w.update(completed=True,passed=False,error=repr(error),traceback=traceback.format_exc(),state='FAILED_OR_UNKNOWN_NO_RETRY')
  try:stop()
  except BaseException as close:w['STOP_error']=repr(close)
  for label,proc,key in [(LABEL+'-'+RUN,gpu,'actual_GPU'),(LABEL+'-'+RUN+'-client',client,'actual_client')]:
   if proc is None:continue
   w['forced_cleanup']=True
   try:w.setdefault('owned_cleanup',[]).append(close_owned(label,proc,w[key],exact,cleanup_limit))
   except BaseException as close:w.setdefault('cleanup_errors',[]).append(repr(close))
  try:w['post_after_failure']=idle('failure-post')
  except BaseException as close:w['post_error']=repr(close)
 finally:save()
 print(json.dumps({k:w.get(k)for k in ['completed','passed','CPU_keeper','actual_GPU','actual_client','state','error']}));return 0 if w['passed']else 1

if __name__=='__main__':raise SystemExit(main())

