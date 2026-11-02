"""One bounded actual HC default-OFF/T1-ON Model residency. No retry/reload or import-only GPU job."""
import argparse,pathlib,json,hashlib,subprocess,os,time,re,datetime,socket
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=pathlib.Path(__file__).resolve().parent;S=R/'incoming/Deep-paired-quality-r375/owner';REPLICA=int(os.environ['DEEP_R375_REPLICA']);assert REPLICA in range(4);PARENT,NODE=[(910374,'a02r4n01'),(910374,'a02r4n05'),(910374,'a02r4n07'),(910371,'a02r3n16')][REPLICA];LABEL='Deep-paired-quality-r375-r'+str(REPLICA);W=R/'results'/(LABEL+'-watch.json');TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'};sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
from dso_binding import verify_dso,CPU621_SHA
from namespace_binding import verify_namespace,WRAPPER_SHA as NAMESPACE_WRAPPER_SHA,CPU_SHA as NAMESPACE_CPU_SHA,terminal_failure
OWNER_SHA='1b93c3db22ab59c6768e401dc46f71b630a768934fe316135b53d780d1b3fafb'
CPU_SOURCE_SHA='7e7cfdf907a80fc2fbf01791cd6c70fbbda1640347c054241cd7c3c1fb9a128e'


def validate_scope(g):
 if g.get('GPU_authorized') is not True:raise PermissionError('Published exact four-replica delegated quality claim required')
 assert g['publication_branch']=='bw1000-experiments' and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])
 assert g['replica']==REPLICA and g['GPU_parent']==PARENT and g['GPU_node']==NODE
 assert g['owner_manifest_SHA256']==OWNER_SHA and re.fullmatch('[0-9a-f]{64}',g['controller_manifest_SHA256'])
 assert g['CPU_keeper_parent']==910407 and g['CPU_keeper_node']=='a14r2n20'
 for k in ['replica','GPU_parent','cards','TP','PP','context','ordinary_budget','thinking_budget','max_HTTP','max_Native','GPU_max_s','Native_max_s','CPU_keeper_max_s']:assert type(g[k])is int
 assert (g['cards'],g['TP'],g['PP'],g['context'],g['ordinary_budget'],g['thinking_budget'],g['max_HTTP'],g['max_Native'])==(8,8,1,32768,4096,8192,66,66)
 assert (g['GPU_max_s'],g['Native_max_s'],g['CPU_keeper_max_s'])==(7200,7140,7800)
 assert g['one_attempt'] is True and g['no_retry_reload'] is True and g['prefill_chunk']==1024
 assert g['CPU_source_manifest_SHA256']==CPU_SOURCE_SHA
 assert re.fullmatch('[0-9a-f]{64}',g['CPU_entry_receipt_SHA256']) and re.fullmatch(r'910407[.]\d+',g['CPU_actual_step'])
 assert g['wrapper_SHA256']==NAMESPACE_WRAPPER_SHA and g['DSO_CPU621_receipt_SHA256']==CPU621_SHA
 return g

def host_path(value):
 p=str(value);return R/p[len('/refactor/'):] if p.startswith('/refactor/') else pathlib.Path(p)

def verify_sources(g):
 validate_scope(g);assert sha(D/'SOURCE_MANIFEST.json')==g['controller_manifest_SHA256']
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
 assert sha(S/'SOURCE_MANIFEST.json')==OWNER_SHA;sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items())
 cp=R/'candidates/Deep-paired-quality-r375'/('CPU-'+g['CPU_actual_step'])/'CPU-result.json'
 assert sha(cp)==g['CPU_entry_receipt_SHA256'];c=json.loads(cp.read_bytes())
 assert c['completed'] and c['passed'] and c['CPU_only'] and c['GPU']==c['Model']==0 and not c['Native_dlopen'] and not c['weights_opened']
 assert c['actual_CPU']==g['CPU_actual_step'] and c['owner_manifest_SHA256']==OWNER_SHA and c['CPU_source_manifest_SHA256']==CPU_SOURCE_SHA
 assert c['inherited_encoding_749'] and c['no_reencoding_of_129']
 source=R/'incoming/Deep-paired-quality-r375/repair-v4/CPU_SOURCE_MANIFEST.json';assert sha(source)==CPU_SOURCE_SHA
 cm=json.loads(source.read_bytes());assert all(sha(source.parent/n)==h for n,h in cm['files_SHA256'].items())
 cases=c['whole_entry_cases'];seams=[x for x in cases if x['kind']=='actual_worker_report_to_actual_client_main66']
 assert {x['replica'] for x in seams}==set(range(4)) and all(x['completed_HTTP']==66 and x['passed'] for x in seams)
 assert any(x['kind']=='actual_client_main_recorded4096cap_continues66' and x['completed_HTTP']==66 for x in cases)
 assert {x['scenario'] for x in cases if x['kind']=='actual_worker_serve_cleanup' and x['passed']} >= {'success66','coldload','Native-failure','HC-ready-failure','HC-hit-failure','client-failure'}
 ep=host_path(c['encoding_receipt_path']);assert sha(ep)==c['encoding_receipt_SHA256']==g['encoding_receipt_SHA256']=='f17342d2ce78e73b9f8853720c6b477c947ccf8d71a7503dcd7e04e391e16447'
 assert g['encoding_receipt_container_path']==c['encoding_receipt_path']
 e=json.loads(ep.read_bytes());assert e['completed'] and e['passed'] and e['actual_prepared_ingress'] and e['retired_actual_requests']==129
 assert e['GPU']==e['Model']==0 and not e['weights_loaded'] and e['first_actual_encode_finish_real_stream_finally_retired']
 pin=json.loads((S/'NATIVE_BUILD_PIN.json').read_bytes());native=R/'candidates/Deep-HC-T1-Model-r366-cpu-v1/CPU-build.json';assert sha(native)==pin['CPU_receipt_SHA256'];build=json.loads(native.read_bytes())
 assert build['passed'] and build['library_SHA256']==g['Native_SHA256']==pin['library_SHA256']
 for n,h in build['all_objects_SHA256'].items():assert sha(host_path(n))==h
 assert sha(R/'candidates/Deep-HC-T1-Model-r366-cpu-v1/libDeep_HC_T1_Model_r366.so')==g['Native_SHA256']
 verify_namespace(R,D);verify_dso(R,R/'incoming/Deep-latest-full-owner-r300','Deep')
 return c

def terminal_report(path,visibility_s=90):
 end=time.monotonic()+visibility_s
 while True:
  try:
   value=json.loads(path.read_bytes())
   if value.get('completed')is True and (value.get('passed')is False or value.get('normal_Native_shutdown')is True):return value
  except (FileNotFoundError,json.JSONDecodeError):pass
  if time.monotonic()>=end:raise TimeoutError('Terminal report not complete within bounded90s visibility; no GPU replay')
  time.sleep(1)

def clean_env():return{k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
def parent(parent,node,env):
 z=subprocess.check_output(['scontrol','show','job',str(parent),'-o'],env=env,text=True,timeout=25);assert 'UserId=aclsylqidf('in z and 'JobState=RUNNING'in z;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',z).group(1);expanded=subprocess.check_output(['scontrol','show','hostnames',nodes],env=env,text=True,timeout=15).splitlines();assert node in expanded

def validate_account_row(row):
 assert re.fullmatch(rf'{PARENT}[.]\d+',row[0])and row[1]==LABEL and row[4]==NODE,'Accounting row is not the exact owned parent/label/node'

def main():
 p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();g=json.loads(pathlib.Path(a.execution_go).read_bytes());verify_sources(g);assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname()=='a14r2n20';assert not W.exists();env=clean_env();parent(910407,'a14r2n20',env);parent(PARENT,NODE,env)
 w={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'GPU_submitted':False,'actual_GPU':None,'CPU_keeper_actual':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'publication_commit':g['publication_commit'],'owner_manifest_SHA256':OWNER_SHA,'controller_manifest_SHA256':g['controller_manifest_SHA256'],'scope':'One of four independent8card Native976f HC OFFON/ONOFF quality replicas; custom0shot128 cases globally; full4096/32768/1024; scores/caps do not gate completion','no_retry_reload':True,'GPU_max_s':7200,'CPU_keeper_max_s':7800,'Model_qualified':False,'namespace_wrapper_SHA256':NAMESPACE_WRAPPER_SHA,'namespace_CPU618_SHA256':NAMESPACE_CPU_SHA,'DSO_CPU621_inside_SHA256':CPU621_SHA};proc=None;actual=None
 def save():
  w['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=W.with_suffix('.tmp');tmp.write_text(json.dumps(w,indent=2)+chr(10));tmp.replace(W)
 def account():
  rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-31','-j',str(PARENT),'-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%100'],env=env,text=True,timeout=25).splitlines()];rows=[v for v in rows if v[1]==LABEL];assert len(rows)<=1;
  for row in rows:validate_account_row(row)
  return rows[0]if rows else None
 def no_foreign_steps(ignore=()):
  rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-31','-j',str(PARENT),'-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%100'],env=env,text=True,timeout=25).splitlines()]
  for row in rows:
   if re.fullmatch(rf'{PARENT}[.]\d+',row[0])and row[0]not in ignore and row[2].split()[0].rstrip('+')not in TERMINAL:
    nodes=subprocess.check_output(['scontrol','show','hostnames',row[4]],env=env,text=True,timeout=15).splitlines();assert NODE not in nodes,'Active foreign step intersects target node'
  return True
 def idle(suffix,owner_tids=()):
  args=['srun','--jobid='+str(PARENT),'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',NODE,'--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+suffix,'python3',str(D/'idle_r375.py'),'--node',NODE,'--owner-tids',json.dumps(list(owner_tids))];z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];assert z.returncode==0 and len(rows)==1 and rows[0]['node']==NODE and not rows[0]['kfd_pids']and rows[0]['HTTP43882_bindable'] and rows[0]['owner_tids_all_gone'];return rows[0]
 with W.open('x') as first:json.dump(w,first);first.flush();os.fsync(first.fileno())
 save()
 try:
  assert account()is None,'Existing exact label must be followed, never replayed';no_foreign_steps();w['fresh_before']=idle('before');no_foreign_steps({w['fresh_before']['actual_step']});save();start=time.monotonic();epoch=time.time();w['GPU_start_UTC_unix']=epoch;save()
  export='ALL,WAVE_INSIDE_R2323=Deep-paired-quality-r375/owner/GPU_inside_r277.sh,DEEP_R277_GPUSTART_UNIX='+str(epoch)+',DEEP_SOURCE_MANIFEST_SHA='+OWNER_SHA+',DEEP_R375_REPLICA='+str(REPLICA)+',DEEP_R375_ENCODING_PATH='+g['encoding_receipt_container_path']+',DEEP_R375_ENCODING_SHA='+g['encoding_receipt_SHA256']
  gpu_env=dict(env)
  for item in export.split(',')[1:]:
   key,value=item.split('=',1);gpu_env[key]=value
  args=['srun','--jobid='+str(PARENT),'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c32','-w',NODE,'--gres=dcu:8','--mem=128000M','-t','02:00:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J.log')),'--export='+export,'bash',str(D/'resource_wave_host_alias_outer.sh'),'probe']
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(args,env=gpu_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_submitted=True,srun_client_PID=proc.pid,state='FOLLOW_SAME_SINGLE_OWNED_GPU_HANDLE');save();end=start+7260
  while time.monotonic()<end:
   try:row=account();w.pop('account_observation_error',None)
   except(subprocess.TimeoutExpired,subprocess.CalledProcessError,OSError)as e:w['account_observation_error']=repr(e);save();time.sleep(15);continue
   if row:
    assert actual is None or actual==row[0];actual=row[0];w.update(actual_GPU=actual,last_Slurm=row)
    if row[2].split()[0].rstrip('+')in TERMINAL:w['terminal']=row;save();break
   if proc.poll()is not None:
    w.setdefault('srun_exit_observed_monotonic',time.monotonic());w['srun_exit_code']=proc.poll()
    if time.monotonic()-w['srun_exit_observed_monotonic']>60:raise RuntimeError('srun exited; actual accounting still unresolved; follow exact label, no restart')
   save();time.sleep(15)
  else:
   row=account()
   if row and row[0]==actual and row[2].split()[0].rstrip('+')not in TERMINAL:
    w['owned_runtime_over_authorized_GPU_bound']=True;save();subprocess.run(['scancel','--signal=TERM',actual],env=env,check=True,timeout=25)
   raise TimeoutError('Original GPU runtime bound exceeded; follow same owned handle to terminal, no replay')
  exit_end=time.monotonic()+90
  while proc.poll() is None and time.monotonic()<exit_end:time.sleep(1)
  w['srun_exit_code']=proc.poll();assert w['srun_exit_code'] is not None,'srun exit unresolved after terminal; no replay'
  final_report=None;owner_tids=[];report_error=None
  if not terminal_failure(w['terminal']) and w['srun_exit_code']==0:
   try:
    final_report=terminal_report(R/'results'/('Deep-paired-quality-r375-'+actual+'.json'))
    owner_tids=[x['tid'] for x in final_report.get('owner_exit_observation',{}).get('owners',[])]
   except BaseException as exc:report_error=exc;w['terminal_report_error']=repr(exc)
  w['post']=idle('post',owner_tids);w['post_empty']=True;w['post_owner_TID_count']=len(owner_tids);w['post_owner_TID_evidence']='KNOWN' if owner_tids else 'UNKNOWN_NO_THREAD_PROOF';save()
  if report_error is not None:raise report_error
  if terminal_failure(w['terminal']) or w['srun_exit_code']!=0:
   w.update(completed=True,state='TERMINAL_GPU_FAILED_POST_CLEAN_NO_RETRY',Model_qualified=False,normal_Model_lifecycle_qualified=False);save();return
  result=R/'results'/('Deep-paired-quality-r375-'+actual+'.json');w['result_path']=str(result)
  if True:
   q=final_report if final_report is not None else terminal_report(result);w.update(result_SHA256=sha(result),result_summary={k:q.get(k)for k in ['completed','passed','normal_server_shutdown','normal_Native_shutdown','actual_HTTP','actual_Native','owner_exit_observation','Model_qualification_scope']});w['Model_qualified']=bool(w['terminal'][2:4]==['COMPLETED','0:0']and q.get('passed')and q.get('completed')and q.get('actual_HTTP')==66 and q.get('actual_Native')==66 and q.get('attempted_Native')==66 and q.get('completed_Native')==66 and q.get('normal_server_shutdown')and q.get('normal_Native_shutdown')and q.get('owner_exit_observation',{}).get('all8_unique_threads_exited'))
  w.update(completed=True,state='HC_MODEL_PAIR_COMPLETE_PENDING_INDEPENDENT_TIMING_AUDIT'if w['Model_qualified']else'TERMINAL_NO_NEW_MODEL_QUALIFICATION');save()
 except BaseException as e:
  w.update(error=repr(e),state='FAILED_BEFORE_GPU'if not w['GPU_submitted']else('TERMINAL_GPU_POST_OR_MODEL_AUDIT_FAILED_NO_RETRY'if 'terminal'in w else'FOLLOW_SAME_HANDLE_NO_RETRY'),completed='terminal'in w,Model_qualified=False);save()
  # An observation failure never means a job died. Slurm/inside absolute bounds
  # remain in force; root follows the recorded handle/unique label to terminal.
  raise
if __name__=='__main__':main()
