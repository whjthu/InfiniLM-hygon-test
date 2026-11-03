"""Exactly two NoV16 replicas, 64HTTP each, one common epoch, no resubmission."""
from pathlib import Path
import argparse,datetime,hashlib,json,math,os,re,signal,socket,subprocess,time,traceback
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/GLM-NoV-W4-quality-r418';D=S/'execution-v1';RUN='quality-20261103-a';LABEL='GLM-r418-two-NoV16-quality';PAIRS=[['a02r4n12','a02r4n13'],['a02r4n14','a02r4n15']]
TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean_environment(base):
 return {k:v for k,v in base.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
def publish(path,q):
 with Path(path).open('x')as f:json.dump(q,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def environment(base,index,epoch,loader,profile,cancel):
 nodes=PAIRS[index];env=clean_environment(base)
 env.update(WAVE_INSIDE_R2323='GLM-NoV-W4-quality-r418/execution-v1/owner/GPU_inside_r290.sh',GLM_R290_GPUSTART_UNIX=str(epoch),GLM_SOURCE_MANIFEST_SHA=loader,GLM_RUN_R5040=RUN+'-r'+str(index),GLM_R404_CANCEL_FILE='/refactor/control/'+Path(cancel).name,GLM_R418_REPLICA_INDEX=str(index),GLM_R417_SOURCE_SHA256=profile,GLM_SERVICE_PROFILE_R417='glm-nov16-w4',GLM_R417_DEFAULT_PROFILE='glm-nov16-w4',
  GLM_R418_SERVICE_SECONDS='13500',GLM_R418_NATIVE_SECONDS='14100',GLM_R418_GPU_SECONDS='14400',GLM_R418_KEEPER_SECONDS='15000',GLM_PP_SIZE_R3603='2',GLM_CACHE_ROWS_R3627='1',GLM_CACHE_CAP_R3627='32768',GLM_NETWORK_TRIAL='shca',GLM_BOUNDED_ROPE_R4279='1',GLM_SHARED_CONTROL_R4279='1',GLM_PIPELINE_TAIL_NODE=nodes[1],GLM_EXACT_BANKS_R4188='1',GLM_CHANNELS_R4363='0',GLM_VARIANT_R4363='layer77-5783',INFINILM_OWNER_CPU_WIDTH_R2392='4',USE_WEIGHT_CACHE_R3604='0',GLM_SERVICE_HOST_R143='127.0.0.1',GLM_SERVICE_PORT_R143='43843',GLM_REGULAR_FIXTURE_RECEIPT_R180='/refactor/results/GLM-regular-fixture-r178a-910407.539/CPU-sizing.json',GLM_REGULAR_FIXTURE_RECEIPT_SHA256_R180='ed2de5e3a0035655ae89f38c65359acf567821b5d254279eebc60847d5f89cbf')
 return env
def quality_args(rec,manifest,epoch,root=R,source=S):
 R0=Path(root);S0=Path(source);ready=R0/'results'/('GLM-r417-service-ready-'+rec['run']+'.json');output=R0/'results'/(LABEL+'-'+rec['run']+'-quality-client')
 return ['--base-url','http://127.0.0.1:43843/v1','--ready',str(ready),'--service-report',str(R0/'results'/('GLM-latest-full-owner-r290-'+rec['run']+'-stage1-rank0.json')),'--fixtures',str(S0/'fixtures-r1'),'--encoding-receipt',str(R0/'candidates/GLM-NoV-W4-quality-encoding-r418-CPU-v2/encoding.json'),'--encoding-sha',manifest['encoding_receipt_SHA256'],'--encoding-source-sha',manifest['encoding_source_SHA256'],'--profile-manifest-sha',manifest['profile_manifest_SHA256'],'--loader-manifest-sha',manifest['loader_manifest_SHA256'],'--expected-step',rec['actual_GPU'],'--output-dir',str(output),'--shard-index',str(rec['index']),'--shards','2','--native-offset','0','--deadline-unix',str(epoch+13500)]
def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--publication-commit',required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args(argv);assert re.fullmatch('[0-9a-f]{40}',a.publication_commit)
 from verify_execution import verify
 m=verify(R,S,a.manifest_sha)
 from check_shell_layout import check as check_shell
 shell=check_shell(D,Path('/public/home/aclsylqidf/hygon-inference-20261013/models/ZhipuAI-GLM-5.3/config.json'),R/'incoming/GLM-NoV-W4-normal-profile-r417',m['loader_manifest_SHA256'],m['profile_manifest_SHA256'])
 if a.preflight_only:print(json.dumps(dict(preflight_passed=True,GPU=0,Slurm=0,intent_created=False,loader_manifest_SHA256=m['loader_manifest_SHA256'],profile_manifest_SHA256=m['profile_manifest_SHA256'],actual_four_role_shell_contract=shell)));return 0
 assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname().split('.')[0]=='a14r2n20'and m['GPU_GO']and m['Model_GO']
 cleanup_limit=time.monotonic()+14980;cancel=R/'control'/(LABEL+'-'+RUN+'-cancel.json');watchpath=R/'results'/(LABEL+'-'+RUN+'-watch.json')
 assert not watchpath.exists()and not cancel.exists()
 w=dict(completed=False,passed=False,publication_commit=a.publication_commit,GPU_manifest_SHA256=a.manifest_sha,CPU_keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],forced_cleanup=False,replicas=[],HTTP_planned=128,logical_Native_planned=128,warms=0,one_epoch=True,retry=False,reload=False);publish(watchpath,w)
 replicas=[];epoch=None;visibility_since={}
 def save():
  w['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();t=watchpath.with_suffix('.tmp');t.write_text(json.dumps(w,indent=2)+'\n');t.replace(watchpath)
 def command(args,timeout=25):return subprocess.check_output(args,text=True,timeout=min(timeout,max(1,cleanup_limit-time.monotonic())))
 def rows():return [line.split('|')for line in command(['sacct','-S','2026-10-31','-j','910376','-n','-P','-o','JobIDRaw,JobName%180,State,ExitCode,NodeList%100']).splitlines()if'|'in line]
 def exact(label,expected_nodes=None):
  found=[r for r in rows()if r[1]==label];assert len(found)<=1
  if not found:return None
  row=found[0]
  if expected_nodes is not None and row[4]not in('','None','Unknown','(null)'):assert set(hosts(row[4]))==set(expected_nodes)
  return row
 def visible(path):
  path=Path(path)
  try:value=json.loads(path.read_bytes())
  except(FileNotFoundError,json.JSONDecodeError)as error:
   first=visibility_since.setdefault(str(path),time.monotonic());w.setdefault('artifact_visibility_observations',[]).append(dict(path=str(path),error=repr(error),original_epoch=epoch));save()
   if time.monotonic()-first>90:raise TimeoutError('Original-epoch bounded artifact visibility:'+str(path))
   return None
  visibility_since.pop(str(path),None);return value
 def hosts(value):return command(['scontrol','show','hostnames',value]).splitlines()
 def idle(phase,pids=None):
  env=dict(clean_environment(os.environ),GLM_R418_EXPECTED_PIDS=json.dumps(pids or{}),GLM_R418_POST_DEADLINE_UNIX=str(min(time.time()+90,(epoch+14880)if epoch else time.time()+90)))
  cmd=['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(sum(PAIRS,[])),'-t','00:02:00','--job-name='+LABEL+'-'+RUN+'-'+phase,'python3','-B',str(D/'idle.py')]
  z=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=min(115,max(1,cleanup_limit-time.monotonic())));w.setdefault('CPU_idle_observations',[]).append(dict(phase=phase,returncode=z.returncode,stdout=z.stdout,stderr=z.stderr));save()
  values=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')];assert z.returncode==0 and len(values)==4 and {v['node']for v in values}==set(sum(PAIRS,[]))
  assert all(v['CPU_only']and not v['GPU_initialized']and not v['kfd_pids']and not v['matching_component_processes']and not v['expected_PIDs_still_exist']and v['ports_bindable']=={'43843':True}for v in values)
  assert len({(v['node'],p)for v in values for p in v['physical_PCI']})==32
  return values
 try:
  parent=command(['scontrol','show','job','910376','-o']);assert'JobState=RUNNING'in parent and set(sum(PAIRS,[]))<=set(hosts(re.search(r'(?:^| )NodeList=([^ ]+)',parent).group(1)))
  for r in rows():
   if r[2].split()[0].rstrip('+')not in TERMINAL and re.fullmatch(r'910376[.]\d+',r[0]):assert not set(hosts(r[4]))&set(sum(PAIRS,[])),'Existinglive same-range step;no launch'
   assert not r[1].startswith(LABEL+'-'+RUN),'Existinglabel must reconcile original'
  w['fresh_before']=idle('before');epoch=time.time();w.update(original_GPU_start_unix=epoch,service_deadline_unix=epoch+13500,Native_deadline_unix=epoch+14100,GPU_deadline_unix=epoch+14400,keeper_deadline_unix=epoch+15000);save()
  for i,nodes in enumerate(PAIRS):
   label=LABEL+'-'+RUN+'-r'+str(i);env=environment(os.environ,i,epoch,m['loader_manifest_SHA256'],m['profile_manifest_SHA256'],cancel)
   cmd=['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','--gres=dcu:8','--mem=224000M','-w',','.join(nodes),'-t','04:00:00','--job-name='+label,'--export=ALL','--output='+str(R/'logs'/(label+'-%J-%t.log')),'bash',str(D/'resource_wave_host_alias_outer.sh'),'probe']
   rec=dict(index=i,nodes=nodes,label=label,run=RUN+'-r'+str(i),client_submitted=False,stop_sent=False,actual_GPU=None,actual_client=None);w['replicas'].append(rec)
   with(R/'logs'/(label+'-srun-client.log')).open('x')as log:gpu=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   replicas.append(dict(record=rec,gpu=gpu,client=None));rec['srun_PID']=gpu.pid;save()
  while time.time()<epoch+14400:
   for rep in replicas:
    rec=rep['record'];i=rec['index'];label=rec['label']
    try:row=exact(label,rec['nodes'])
    except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as error:
     if rep['gpu'].poll()not in(None,0):raise RuntimeError('Owned GPU srun nonzero during observation loss')
     if rep['client']is not None and rep['client'].poll()not in(None,0):raise RuntimeError('Owned client srun nonzero during observation loss')
     w.setdefault('transient_observation_errors',[]).append(dict(label=label,error=repr(error),actual_GPU=rec['actual_GPU'],original_epoch=epoch));save();continue
    if row:
     assert rec['actual_GPU']in(None,row[0]);rec['actual_GPU']=row[0];rec['last_Slurm']=row
     if row[2].split()[0].rstrip('+')in TERMINAL:assert row[2]=='COMPLETED'and row[3]=='0:0','OriginalfailedGPU retained'
    if rep['gpu'].poll()not in(None,0):raise RuntimeError('Owned GPU srun failed:'+label)
    ready=R/'results'/('GLM-r417-service-ready-'+rec['run']+'.json')
    if ready.exists()and rec['actual_GPU']and not rec['client_submitted']:
     identity=visible(ready)
     if identity is None:continue
     assert identity['actual_step']==rec['actual_GPU']and identity['loader_parent_source_manifest_SHA256']==m['loader_manifest_SHA256']and identity['service_profile_manifest_SHA256']==m['profile_manifest_SHA256']
     output=R/'results'/(LABEL+'-'+rec['run']+'-quality-client');client_identity=R/'results'/(LABEL+'-'+rec['run']+'-client-identity.json')
     args=quality_args(rec,m,epoch)
     cmd=['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c2','--gres=none','--mem=8192M','-w',rec['nodes'][1],'-t','03:45:00','--job-name='+label+'-client','--output='+str(R/'logs'/(label+'-client-%J.log')),'python3','-B','-u',str(D/'client_launch.py'),'--identity',str(client_identity),'--client',str(S/'client-v1/quality_client.py'),'--client-sha',m['client_source_SHA256'],*args]
     client_env=dict(clean_environment(os.environ),CUDA_VISIBLE_DEVICES='-1',HIP_VISIBLE_DEVICES='-1',ROCR_VISIBLE_DEVICES='-1')
     with(R/'logs'/(label+'-client-srun.log')).open('x')as log:rep['client']=subprocess.Popen(cmd,env=client_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     rec.update(client_submitted=True,client_srun_PID=rep['client'].pid,client_result=str(output/'result.json'),client_identity=str(client_identity));save()
    if rec['client_submitted']:
     cp=Path(rec['client_identity'])
     if cp.exists():
      ci=visible(cp)
      if ci is not None:rec['actual_client']=ci['actual_CPU']
     result=Path(rec['client_result'])
     if result.exists():
      cr=visible(result)
      if cr is None:continue
      if cr['completed']:
       assert cr['execution_passed']and cr['actual_HTTP']==cr['planned_HTTP']==len(cr['rows'])==64 and not cr['errors'],'Clientinfra failure;wrongscores/caps alone do notfail'
       rec['client_summary']=cr['summary'];rec['client_result_SHA256']=sha(result)
       if not rec['stop_sent']:publish(R/'control'/('GLM-r417-service-stop-'+rec['run']+'.json'),dict(normal_stop=True,run=rec['run'],profile='glm-nov16-w4'));rec['stop_sent']=True;save()
     if rep['client'].poll()not in(None,0):raise RuntimeError('Owned client srun nonzero:'+label)
   if all(r['record']['stop_sent']and r['gpu'].poll()==0 and r['client']is not None and r['client'].poll()==0 for r in replicas):break
   save();time.sleep(min(15,max(0,epoch+14400-time.time())))
  else:raise TimeoutError('Original14400 GPU deadline;no retry/reload')
  from collect_quality import collect
  summary=collect(R,RUN,replicas,m,epoch);w.update(summary);w['post']=idle('post',summary['expected_node_PIDs']);w.update(completed=True,passed=True,state='NORMAL_TWO_REPLICAS128_QUALITY_REQUESTS_COMPLETE_SCORES_REPORTED_NOT_NOINFERIORITY')
 except BaseException as error:
  w.update(error=repr(error),traceback=traceback.format_exc(),completed=True,passed=False,state='FAILED_OR_UNKNOWN_NO_RETRY')
  if not cancel.exists():publish(cancel,dict(cancelled=True,run=RUN,reason=type(error).__name__))
  for rep in replicas:
   for label,proc in [(rep['record']['label'],rep['gpu']),(rep['record']['label']+'-client',rep['client'])]:
    try:
     row=exact(label)
     if row and row[2].split()[0].rstrip('+')not in TERMINAL:w['forced_cleanup']=True;subprocess.run(['scancel','--signal=TERM',row[0]],check=True,timeout=15)
     if proc is not None:
      try:proc.wait(timeout=20)
      except subprocess.TimeoutExpired:w['forced_cleanup']=True;os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=10)
    except BaseException as close:w.setdefault('cleanup_errors',[]).append(dict(label=label,error=repr(close)))
  try:w['post_after_failure']=idle('failure-post')
  except BaseException as close:w['post_error']=repr(close)
 finally:save()
 print(json.dumps({k:w.get(k)for k in ['completed','passed','CPU_keeper','state','error']}));return 0 if w['passed']else 1
if __name__=='__main__':raise SystemExit(main())
