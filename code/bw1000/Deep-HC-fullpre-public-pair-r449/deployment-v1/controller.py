from pathlib import Path
import argparse,datetime,hashlib,json,os,re,signal,socket,subprocess,sys,time,traceback
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/Deep-HC-fullpre-public-pair-r449/deployment-v1';PACKAGE=R/'incoming/Deep-HC-fullpre-public-pair-r449/source-v1'
RUN='fullpre-pair-20261104-a';LABEL='Deep-r449-HC-fullpre-pair';NODES=['a02r4n04']
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean_environment(base):
 return{k:v for k,v in base.items()if k not in['BASH_ENV','ENV','PYTHONSTARTUP','PYTHONINSPECT']and((not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_','GLM_','DEEP_','INFINILM_DEEP_')))or k in['SLURM_CONF','SLURM_CONF_SERVER'])}
def publish(path,q):
 with Path(path).open('x')as f:json.dump(q,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def verify(digest):
 assert sha(D/'SOURCE_MANIFEST.json')==digest;m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
 assert sha(PACKAGE/'SOURCE_MANIFEST.json')==m['provider_source_SHA256']
 for n,h in json.loads((PACKAGE/'SOURCE_MANIFEST.json').read_bytes())['files_SHA256'].items():assert sha(PACKAGE/n)==h,n
 assert sha(D/'config.json')==m['config_SHA256'];return m
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
 env.update(WAVE_INSIDE_R2323='Deep-HC-fullpre-public-pair-r449/deployment-v1/inside.sh',DEEP_R449_MODEL_RUN=RUN,DEEP_R449_MODEL_GPUSTART=str(epoch),DEEP_R449_MODEL_CONFIG_SHA=m['config_SHA256'],DEEP_R449_MODEL_SOURCE_SHA=m['provider_source_SHA256'],INFINILM_OWNER_CPU_WIDTH_R2392='4')
 return env

def gpu_command():
 return ['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1',
  '-N1','-n1','--ntasks-per-node=1','-c32','--gres=dcu:8','--mem=128000M','-w',','.join(NODES),'-t','00:45:00',
  '--job-name='+LABEL+'-'+RUN,'--export=ALL','--output='+str(R/'logs'/(LABEL+'-'+RUN+'-%J-%t.log')),
  'bash',str(D/'resource_wave_host_alias_outer.sh'),'probe']

def client_command(actual,m,epoch):
 output=R/'results'/(LABEL+'-'+RUN+'-client')
 return ['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2',
  '--gres=none','--mem=8192M','-w',NODES[0],'-t','00:40:00','--job-name='+LABEL+'-'+RUN+'-client',
  '--output='+str(R/'logs'/(LABEL+'-'+RUN+'-client-%J.log')),'python3','-B','-u',str(D/'client_launch.py'),
  '--identity',str(output.with_name(output.name+'-identity.json')),'--client',str(D/'client.py'),'--client-sha',m['files_SHA256']['client.py'],
  '--base-url','http://127.0.0.1:43882/v1','--runtime-dir',str(R/'results/Deep-HC-fullpre-public-r449'),
  '--run',RUN,'--expected-step',actual,'--source-sha',m['provider_source_SHA256'],'--config-sha',m['config_SHA256'],
  '--fixtures',str(D/'FIXTURE_CARD.json'),'--output-dir',str(output),'--deadline-unix',str(epoch+2400)]

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv)
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit);m=verify(a.manifest_sha)
 from preflight import check
 proof=check(D,PACKAGE,m)
 if a.preflight_only:print(json.dumps(dict(passed=True,source_only=True,GPU=0,Slurm=0,proof=proof)));return 0
 assert m['GPU_GO']and m['Model_GO']and os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]=='a14r2n20'
 cleanup_limit=time.monotonic()+2980;watch=R/'results'/(LABEL+'-'+RUN+'-watch.json');folder=R/'results/Deep-HC-fullpre-public-r449'/RUN
 assert not folder.exists(),'Existing run directory: reconcile original; never replay'
 w=dict(completed=False,passed=False,publication_commit=a.publication_commit,deployment_source_SHA256=a.manifest_sha,
  provider_source_SHA256=m['provider_source_SHA256'],CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],
  HTTP_planned=6,warms=2,retry=False,reload=False,forced_cleanup=False,actual_GPU=None,actual_client=None)
 publish(watch,w);gpu=client=None;epoch=None;stop_sent=False;visibility={};expected_pids={}
 def save():
  w['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=watch.with_suffix('.tmp');t.write_text(json.dumps(w,indent=2,allow_nan=False)+'\n');t.replace(watch)
 def command(args,timeout=25):return subprocess.check_output(args,text=True,timeout=min(timeout,max(1,cleanup_limit-time.monotonic())))
 def rows():return [x.split('|')for x in command(['sacct','-S','2026-10-31','-j','910374','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100']).splitlines()if'|'in x]
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
  # Pure protocol import; no Native/Torch/HTTP. Exact shared Run identity.
  sys.path.insert(0,str(R/'incoming/public-Deep-native-provider-r427/source-v3'));from public_deep_native.protocol import Run
  config=json.loads((D/'config.json').read_bytes());config.update(runtime_directory=str(R/'results/Deep-HC-fullpre-public-r449'),
   run=RUN,actual_step=w['actual_GPU'],source_manifest_SHA256=m['provider_source_SHA256'],provider_config_SHA256=m['config_SHA256'],gpu_start_epoch=epoch)
  Run(config).stop();stop_sent=True;w['normal_STOP_published']=True;save()
 def idle(phase):
  # Even a metadata/collector failure cannot discard already recorded child PIDs.
  for node in NODES:
   known=set(expected_pids.get(node,[]))
   try:known.add(json.loads((folder/'FRONTEND.json').read_bytes())['PID'])
   except(FileNotFoundError,json.JSONDecodeError,KeyError):pass
   expected_pids[node]=sorted(known)
  env=dict(clean_environment(os.environ),DEEP_R449_MODEL_EXPECTED_PIDS=json.dumps(expected_pids),DEEP_R449_MODEL_POST_DEADLINE_UNIX=str(min(time.time()+90,epoch+2880 if epoch else time.time()+90)))
  cmd=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(NODES),'-t','00:02:00','--job-name='+LABEL+'-'+RUN+'-'+phase,'python3','-B',str(D/'idle.py')]
  z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(115,max(1,cleanup_limit-time.monotonic())))
  w.setdefault('CPU_idle_observations',[]).append(dict(phase=phase,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr));save()
  values=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')]
  assert z.returncode==0 and len(values)==1 and {v['node']for v in values}==set(NODES)
  assert all(v['CPU_only']and not v['GPU_initialized']and not v['kfd_pids']and not v['matching_component_processes']and not v['expected_PIDs_still_exist']and v['ports_bindable']=={'43882':True}for v in values)
  assert len({(v['node'],pci)for v in values for pci in v['physical_PCI']})==8
  return values
 try:
  parent=command(['scontrol','show','job','910374','-o']);assert'JobState=RUNNING'in parent and set(NODES)<=set(hosts(re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1)))
  assert re.search(r'(?:^| )UserId='+re.escape(os.environ['USER'])+r'\(',parent),'Allocation user mismatch'
  w['parent_allocation']=parent
  for row in rows():
   assert not row[1].startswith(LABEL+'-'+RUN),'Existing label: reconcile original, no replay'
   if re.fullmatch(r'910374[.]\d+',row[0])and row[2].split()[0].rstrip('+')not in TERMINAL:
    assert not set(hosts(row[4]))&set(NODES),'Live overlapping step'
  w['fresh_before']=idle('before');epoch=time.time();w.update(original_GPU_start_unix=epoch,service_deadline_unix=epoch+2400,Native_deadline_unix=epoch+2580,GPU_deadline_unix=epoch+2700,keeper_deadline_unix=epoch+3000);save()
  with(R/'logs'/(LABEL+'-'+RUN+'-srun.log')).open('x')as log:
   gpu=subprocess.Popen(gpu_command(),env=gpu_environment(os.environ,epoch,m),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_srun_PID=gpu.pid,GPU_owned_PGID=gpu.pid);save()
  client_output=R/'results'/(LABEL+'-'+RUN+'-client');identity=client_output.with_name(client_output.name+'-identity.json')
  done_since=None
  while time.time()<epoch+2700:
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
   if client is None and ready.exists()and w['actual_GPU']:
    value=visible(ready)
    if value is not None:
     expected=dict(run=RUN,actual_step=w['actual_GPU'],source_manifest_SHA256=m['provider_source_SHA256'],provider_config_SHA256=m['config_SHA256'],gpu_start_epoch=epoch,Native_seconds=2580)
     assert value['identity']==expected and value['Model_READY']and value['Native_internal_owner_count']==8
     env=dict(clean_environment(os.environ),CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
     with(R/'logs'/(LABEL+'-'+RUN+'-client-srun.log')).open('x')as log:client=subprocess.Popen(client_command(w['actual_GPU'],m,epoch),env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     w.update(client_srun_PID=client.pid,client_owned_PGID=client.pid,client_result=str(client_output/'result.json'));save()
   if client is not None and(client_output/'result.json').exists():
    value=visible(client_output/'result.json')
    if value is not None and value['completed']:
     w['client_result_SHA256']=sha(client_output/'result.json');w['client_passed']=value['passed'];stop()
     assert value['passed']and value['attempted_HTTP']==value['completed_HTTP']==6,'Public request/gold failure retained'
   if client is not None and client.poll()==0 and gpu.poll()==0:
    done_since=time.monotonic()if done_since is None else done_since
    if gr and cr and gr[2:4]==cr[2:4]==['COMPLETED','0:0']and stop_sent:break
    if time.monotonic()-done_since>90:raise TimeoutError('srun/Slurm exit propagation bounded90s')
   elif gpu.poll()==0:
    done_since=time.monotonic()if done_since is None else done_since
    if time.monotonic()-done_since>90:raise RuntimeError('GPU exited before complete public plan')
   save();time.sleep(min(15,max(0,epoch+2700-time.time())))
  else:raise TimeoutError('Original GPU2700 deadline; no retry/reload')
  from collect import collect
  summary=collect(R,RUN,w['actual_GPU'],m['provider_source_SHA256'],client_output/'result.json',min(time.time()+90,epoch+2880))
  expected_pids=summary['expected_node_PIDs'];w.update(summary)
  w['post']=idle('post');w.update(completed=True,passed=True,state='PUBLIC_HC0_FULLPRE_MODEL_SIX_REQUESTS_NORMAL')
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

