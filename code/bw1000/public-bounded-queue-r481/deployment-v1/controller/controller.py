"""One immutable C1 queue experiment per keeper; bounded original epoch."""
from pathlib import Path
import argparse,datetime,json,os,re,socket,subprocess,sys,time,traceback
from support import *
from qualified_cleanup import clean_environment,close_owned,TERMINAL

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--instance',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv)
 assert re.fullmatch('[0-9a-f]{40}',a.publication_commit);m,v,c,cp=verify(a.manifest_sha,a.instance)
 if a.preflight_only:print(json.dumps(dict(passed=True,source_only=True,GPU=0,Slurm=0,instance=a.instance,Native=False)));return 0
 assert m['GPU_GO']and m['Model_GO']and os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]=='a14r2n20'
 label=LABEL+'-'+v['run'];node=v['node'];base=R/'results'/label;runtime=folder(c,v);watch=base.with_name(base.name+'-watch.json');client_out=base.with_name(base.name+'-client')
 assert not watch.exists()and not runtime.exists()and not client_out.exists();exclusive(watch,{})
 entry=time.monotonic();limit=entry+120;gpu=client=None;expected={node:[]};epoch=None;visibility={};exit_since=None
 w=dict(completed=False,passed=False,instance=a.instance,node=node,publication_commit=a.publication_commit,deployment_source_SHA256=a.manifest_sha,provider_source_SHA256=PROVIDER_SHA,CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],HTTP_planned=20,warm=0,retry=False,reload=False,forced_cleanup=False,actual_GPU=None,actual_client=None)
 def save():w['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(watch,w)
 def remaining(seconds):
  left=limit-time.monotonic()
  if left<=0:raise TimeoutError('Original keeper cleanup deadline')
  return min(seconds,left)
 def command(args,seconds=25,env=None):return subprocess.check_output(args,env=env,text=True,timeout=remaining(seconds))
 def rows():return[x.split('|')for x in command(['sacct','-S','2026-11-05','-j','910374','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100']).splitlines()if'|'in x]
 def exact(name):
  q=[x for x in rows()if x[1]==name];assert len(q)<=1,'Duplicate owned label';return q[0]if q else None
 def hosts(x):return command(['scontrol','show','hostnames',x]).splitlines()
 def visible(path):
  try:q=read(path)
  except(FileNotFoundError,json.JSONDecodeError):
   start=visibility.setdefault(str(path),time.monotonic())
   if time.monotonic()-start>90:raise TimeoutError('Original90s terminal artifact propagation:'+str(path))
   return None
  visibility.pop(str(path),None);return q
 def stop(fallback=False):
  if not w['actual_GPU']or not runtime.exists():return
  sc=server_config(v,w['actual_GPU'],epoch,c,cp);sys.path.insert(0,str(PACKAGE));from public_deep_model.protocol import Run
  run=Run(sc)
  if(runtime/'STOP.json').exists():run.read('STOP.json');w.setdefault('STOP_owner','client_or_existing_exactidentity')
  elif fallback:run.stop();w['STOP_owner']='controller_failure_or_terminal_fallback'
  else:return # Normal STOP belongs to the client; wait original visibility bound.
  save()
 def idle(phase):
  try:expected[node]=sorted(set(expected[node]+[read(runtime/'FRONTEND.json')['PID']]))
  except(FileNotFoundError,json.JSONDecodeError,KeyError):pass
  env=dict(clean_environment(os.environ),DEEP_R481_EXPECTED_PIDS=json.dumps(expected),DEEP_R481_POST_DEADLINE_UNIX=str(min(time.time()+90,epoch+CLEANUP_SECONDS if epoch else time.time()+90)))
  cmd=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=512M','-w',node,'-t','00:02:00','--job-name='+label+'-'+phase,HOST_PYTHON,'-B',str(D/'idle.py')]
  z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=remaining(115));w.setdefault('CPU_idle_observations',[]).append(dict(phase=phase,rc=z.returncode,stdout=z.stdout,stderr=z.stderr));save();q=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')]
  assert z.returncode==0 and len(q)==1 and q[0]['node']==node;x=q[0];assert x['KFD_available']and not x['GPU_initialized']and not x['kfd_pids']and not x['matching_component_processes']and not x['expected_PIDs_still_exist']and x['ports_bindable']=={'43882':True}and len(set(x['physical_PCI']))==8
  return x
 try:
  parent=command(['scontrol','show','job','910374','-o']);assert'JobState=RUNNING'in parent and'UserId=aclsylqidf('in parent;assert node in hosts(re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1));w['parent_allocation']=parent
  for row in rows():
   assert not row[1].startswith(label),'Existinglabel requires original reconcile'
   if re.fullmatch(r'910374[.]\d+',row[0])and row[2].split()[0].rstrip('+')not in TERMINAL:assert node not in hosts(row[4]),'Fresh overlapping step'
  w['fresh_before']=idle('before');mono=time.monotonic();assert mono+CLEANUP_SECONDS<=entry+KEEPER_SECONDS;epoch=time.time();limit=mono+CLEANUP_SECONDS;w.update(original_GPU_start_unix=epoch,service_deadline_unix=epoch+SERVICE_SECONDS,Native_deadline_unix=epoch+NATIVE_SECONDS,GPU_deadline_unix=epoch+GPU_SECONDS,cleanup_deadline_unix=epoch+CLEANUP_SECONDS,keeper_deadline_unix=epoch+KEEPER_SECONDS);save()
  env=clean_environment(os.environ);env.update(WAVE_INSIDE_R2323='public-bounded-queue-r481/deployment-v1/controller/inside.sh',DEEP_R481_CONFIG=str(cp),DEEP_R481_CONFIG_SHA=SHA(cp),DEEP_R481_RUN=v['run'],DEEP_R481_GPUSTART=str(epoch),DEEP_R481_PROVIDER_SHA=PROVIDER_SHA,INFINILM_OWNER_CPU_WIDTH_R2392='4')
  args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','--ntasks-per-node=1','-c32','--gres=dcu:8','--mem=128000M','-w',node,'-t','06:00:00','--job-name='+label,'--export=ALL','--output='+str(R/'logs'/(label+'-%J-%t.log')),'bash',str(D/'resource_wave_host_alias_outer.sh'),'probe']
  with(R/'logs'/(label+'-srun.log')).open('x')as log:gpu=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_srun_PID=gpu.pid,GPU_owned_PGID=gpu.pid,GPU_srun_startticks=Path('/proc/'+str(gpu.pid)+'/stat').read_text().split()[21]);save()
  while time.time()<epoch+GPU_SECONDS:
   try:gr=exact(label);cr=exact(label+'-client')if client is not None else None
   except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as error:
    if gpu.poll()not in(None,0)or(client and client.poll()not in(None,0)):raise RuntimeError('Owned srun failed during observation loss')from error
    w.setdefault('transient_account_errors',[]).append(repr(error));save();time.sleep(15);continue
   for key,row in [('actual_GPU',gr),('actual_client',cr)]:
    if row:
     assert w[key]in(None,row[0]);w[key]=row[0];w[key+'_Slurm']=row
     if row[2].split()[0].rstrip('+')in TERMINAL:assert row[2:4]==['COMPLETED','0:0'],'Owned step failed'
   if gpu.poll()not in(None,0)or(client and client.poll()not in(None,0)):raise RuntimeError('Owned srun nonzero')
   if client is None and w['actual_GPU']and(runtime/'MODEL_READY.json').exists():
    ready=visible(runtime/'MODEL_READY.json')
    if ready is not None:
     assert ready['identity']==identity(v,w['actual_GPU'],epoch,cp)and ready['Model_READY']and ready['Native_internal_owner_count']==8 and ready['Native_SHA256']==NATIVE_SHA and ready['HC_mode']==0
     assert ready['fixed_profile']['fixed_mode']==ready['fixed_profile']['r366_fixed']==0 and ready['fixed_profile']['HC_groups']==86;assert ready['queue_policy']==dict(enabled=v['queue_policy'],physical_pending_capacity=2,queue_wait_seconds=5.)
     runtime_config=base.with_name(base.name+'-client-server.json');exclusive(runtime_config,server_config(v,w['actual_GPU'],epoch,c,cp));cc=base.with_name(base.name+'-client-config.json');exclusive(cc,dict(private_plan=str(hostpath(m['private_plan']['path'])),private_plan_SHA256=m['private_plan']['SHA256'],base_url='http://127.0.0.1:43882/v1',output=str(client_out),pair=v['pair'],queue_policy=v['queue_policy'],deadline_unix=epoch+SERVICE_SECONDS,server_config=str(runtime_config),server_config_SHA256=SHA(runtime_config)))
     ce=dict(clean_environment(os.environ),CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1',PYTHONPATH=str(DEPLOY)+':'+str(R/'incoming/public-token-clock-r480')+':'+str(PACKAGE));identity_file=base.with_name(base.name+'-client-identity.json');args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w',node,'-t','05:40:00','--job-name='+label+'-client','--output='+str(R/'logs'/(label+'-client-%J.log')),HOST_PYTHON,'-B','-u',str(D/'client_launch.py'),'--identity',str(identity_file),'--client',str(DEPLOY/'finite_client.py'),'--client-sha',m['files_SHA256']['finite_client.py'],'--config',str(cc)]
     with(R/'logs'/(label+'-client-srun.log')).open('x')as log:client=subprocess.Popen(args,env=ce,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     w.update(client_srun_PID=client.pid,client_owned_PGID=client.pid,client_result=str(client_out/'result.json'));save()
   if client and(client_out/'result.json').exists():
    result=visible(client_out/'result.json')
    if result is not None and result['completed']:
     assert len(result['rows'])==20 and{r['index']for r in result['rows']}==set(range(20));w.update(client_result_SHA256=SHA(client_out/'result.json'),client_quality_summary=result['quality_summary']);stop()
   if client and client.poll()==0 and gpu.poll()==0:
    exit_since=time.monotonic()if exit_since is None else exit_since
    if gr and cr and gr[2:4]==cr[2:4]==['COMPLETED','0:0']and(runtime/'STOP.json').exists():break
    if time.monotonic()-exit_since>90:raise TimeoutError('Original90s accounting exit propagation')
   elif gpu.poll()==0:
    exit_since=time.monotonic()if exit_since is None else exit_since
    if time.monotonic()-exit_since>90:raise RuntimeError('GPU ended without completed finite client')
   save();time.sleep(min(15,max(0,epoch+GPU_SECONDS-time.time())))
  else:raise TimeoutError('Original GPU21600 deadline, no reload/retry')
  try:
   raw=command(['sacct','-j',w['actual_GPU'],'-S','2026-11-05','-n','-P','-o','JobIDRaw,ElapsedRaw,State,ExitCode'],12);cost=[x.split('|')for x in raw.splitlines()if x.split('|')[0]==w['actual_GPU']];assert len(cost)==1 and cost[0][1].isdigit()and cost[0][2:4]==['COMPLETED','0:0'];w['GPU_allocation_cost']=dict(actual_step=w['actual_GPU'],ElapsedRaw_s=int(cost[0][1]),cards=8,allocated_cardseconds=8*int(cost[0][1]),includes_load_and_cleanup=True,not_GPU_busy_or_online_throughput=True)
  except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError,AssertionError,ValueError)as error:w['GPU_allocation_cost']=dict(state='UNKNOWN',error=repr(error),not_imputed_from_clientclock=True)
  from collect import collect
  summary=collect(runtime,identity(v,w['actual_GPU'],epoch,cp),client_out/'result.json',base.with_name(base.name+'-client-identity.json'),min(time.time()+90,epoch+CLEANUP_SECONDS),private_plan=hostpath(m['private_plan']['path']),private_plan_SHA256=m['private_plan']['SHA256']);expected[node]=summary['expected_PIDs'];w.update(summary);w['post']=idle('post');w.update(completed=True,passed=True,execution_passed=True,quality_passed=None,state='FINITE20_QUEUE_CAPABILITY_NORMAL_QUALITY_RECORDED')
 except BaseException as error:
  w.update(completed=True,passed=False,state='FAILED_OR_UNKNOWN_NO_RETRY',error=repr(error),traceback=traceback.format_exc())
  try:stop(fallback=True)
  except BaseException as e:w['STOP_error']=repr(e)
  for name,proc,key in [(label,gpu,'actual_GPU'),(label+'-client',client,'actual_client')]:
   if proc is not None:
    w['forced_cleanup']=True
    try:w.setdefault('owned_cleanup',[]).append(close_owned(name,proc,w[key],exact,limit))
    except BaseException as e:w.setdefault('cleanup_errors',[]).append(repr(e))
  try:w['post_after_failure']=idle('failure-post')
  except BaseException as e:w['post_error']=repr(e)
 finally:save()
 return 0 if w.get('execution_passed')else 1
if __name__=='__main__':raise SystemExit(main())
