"""One delegated independent semantic68 original8card residency. No retry/reload or import-only GPU job."""
import argparse,pathlib,json,hashlib,subprocess,os,time,re,datetime,socket
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=pathlib.Path(__file__).resolve().parent;S=R/'incoming/Deep-prefill-chunk-pair-r348/owner';LABEL='Deep-prefill-chunk-pair-Model-r348';NODE='a02r4n01';W=R/'results'/(LABEL+'-watch.json');TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'};sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
from dso_binding import verify_dso,CPU621_SHA
from namespace_binding import verify_namespace,WRAPPER_SHA as NAMESPACE_WRAPPER_SHA,CPU_SHA as NAMESPACE_CPU_SHA,terminal_failure
OWNER_SHA='40f69be372f89c4ed5345a4d326bbd1936423d5330f40da3dcf435a9b2f80d4e';CPU_SHA='159879833e64dd8ab758e71c714ca1a5fa70063005691374bb0632b0dd470235';CPU_ENTRY_SHA='170e4b651207a1abdabe5cc44f63df85da3b1951815b5c041c56431ee14c14f9'

def validate_scope(g):
 if g.get('GPU_authorized')is not True:raise PermissionError('Published concrete delegated prefill pair execution claim required')
 assert g['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',g['publication_commit']);assert g['owner_manifest_SHA256']==OWNER_SHA and re.fullmatch('[0-9a-f]{64}',g['controller_manifest_SHA256'])
 assert g['GPU_parent']==910374 and g['GPU_node']==NODE and g['CPU_keeper_parent']==910407 and g['CPU_keeper_node']=='a14r2n20'
 for k in ['GPU_parent','CPU_keeper_parent','cards','TP','PP','context','ordinary_budget','thinking_budget','max_HTTP','max_Native','GPU_max_s','Native_max_s','CPU_keeper_max_s']:assert type(g[k])is int
 assert(g['cards'],g['TP'],g['PP'],g['context'],g['ordinary_budget'],g['thinking_budget'],g['max_HTTP'],g['max_Native'])==(8,8,1,32768,128,128,6,6)
 assert(g['GPU_max_s'],g['Native_max_s'],g['CPU_keeper_max_s'])==(2700,2670,3300)and g['one_attempt']is True and g['no_retry_reload']is True
 assert g['CPU658_receipt_SHA256']==CPU_SHA and g['wrapper_SHA256']==NAMESPACE_WRAPPER_SHA and g['DSO_CPU621_receipt_SHA256']==CPU621_SHA
 return g

def verify_sources(g):
 validate_scope(g);assert sha(D/'SOURCE_MANIFEST.json')==g['controller_manifest_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items());assert sha(S/'SOURCE_MANIFEST.json')==OWNER_SHA;sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items())
 cp=R/'candidates/Deep-prefill-chunk-pair-r348-CPU/CPU-entry.json';assert sha(cp)==CPU_SHA;c=json.loads(cp.read_bytes());assert c['completed']and c['passed']and c['CPU_only']and not c['GPU']and not c['Native_dlopen']and not c['weights_opened']and c['actual_step']=='910407.658'and len(c['cases'])==17
 assert sha(S.parent/'CPU_entry.py')==CPU_ENTRY_SHA
 verify_namespace(R,D);verify_dso(R,R/'incoming/Deep-latest-full-owner-r300','Deep')

def clean_env():return{k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
def parent(parent,node,env):
 z=subprocess.check_output(['scontrol','show','job',str(parent),'-o'],env=env,text=True,timeout=25);assert 'UserId=aclsylqidf('in z and 'JobState=RUNNING'in z;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',z).group(1);expanded=subprocess.check_output(['scontrol','show','hostnames',nodes],env=env,text=True,timeout=15).splitlines();assert node in expanded

def validate_account_row(row):
 assert re.fullmatch(r'910374[.]\d+',row[0])and row[1]==LABEL and row[4]==NODE,'Accounting row is not the exact owned parent/label/node'

def main():
 p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();g=json.loads(pathlib.Path(a.execution_go).read_bytes());verify_sources(g);assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname()=='a14r2n20';assert not W.exists();env=clean_env();parent(910407,'a14r2n20',env);parent(910374,NODE,env)
 w={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'GPU_submitted':False,'actual_GPU':None,'CPU_keeper_actual':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'publication_commit':g['publication_commit'],'owner_manifest_SHA256':OWNER_SHA,'controller_manifest_SHA256':g['controller_manifest_SHA256'],'scope':'One unchanged afc6 TP8PP1 residency; actual frozen8489token toolprompt sixHTTP warm1024/2048 thenABBA, output128; prefillpair only, bufferedHTTP not clientTTFT/SLO','no_retry_reload':True,'GPU_max_s':2700,'CPU_keeper_max_s':3300,'Model_qualified':False,'namespace_wrapper_SHA256':NAMESPACE_WRAPPER_SHA,'namespace_CPU618_SHA256':NAMESPACE_CPU_SHA,'DSO_CPU621_inside_SHA256':CPU621_SHA};proc=None;actual=None
 def save():
  w['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=W.with_suffix('.tmp');tmp.write_text(json.dumps(w,indent=2)+chr(10));tmp.replace(W)
 def account():
  rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-31','-j','910374','-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%100'],env=env,text=True,timeout=25).splitlines()];rows=[v for v in rows if v[1]==LABEL];assert len(rows)<=1;
  for row in rows:validate_account_row(row)
  return rows[0]if rows else None
 def no_foreign_steps(ignore=()):
  rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-31','-j','910374','-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%100'],env=env,text=True,timeout=25).splitlines()]
  for row in rows:
   if re.fullmatch(r'910374[.]\d+',row[0])and row[0]not in ignore and row[2].split()[0].rstrip('+')not in TERMINAL:
    nodes=subprocess.check_output(['scontrol','show','hostnames',row[4]],env=env,text=True,timeout=15).splitlines();assert NODE not in nodes,'Active foreign step intersects target node'
  return True
 def idle(suffix):
  args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',NODE,'--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+suffix,'python3',str(D/'idle_r283.py')];z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];assert z.returncode==0 and len(rows)==1 and rows[0]['node']==NODE and not rows[0]['kfd_pids']and rows[0]['HTTP43882_bindable'];return rows[0]
 save()
 try:
  assert account()is None,'Existing exact label must be followed, never replayed';no_foreign_steps();w['fresh_before']=idle('before');no_foreign_steps({w['fresh_before']['actual_step']});save();start=time.monotonic();epoch=time.time();w['GPU_start_UTC_unix']=epoch;save()
  export='ALL,WAVE_INSIDE_R2323=Deep-prefill-chunk-pair-r348/owner/GPU_inside_r277.sh,DEEP_R277_GPUSTART_UNIX='+str(epoch)+',DEEP_SOURCE_MANIFEST_SHA='+OWNER_SHA
  args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c32','-w',NODE,'--gres=dcu:8','--mem=128000M','-t','00:45:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J.log')),'--export='+export,'bash',str(D/'resource_wave_host_alias_outer.sh'),'probe']
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_submitted=True,srun_client_PID=proc.pid,state='FOLLOW_SAME_SINGLE_OWNED_GPU_HANDLE');save();end=start+2760
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
  w['post']=idle('post');w['post_empty']=True
  if terminal_failure(w['terminal']):
   w.update(completed=True,state='TERMINAL_GPU_FAILED_POST_CLEAN_NO_RETRY',Model_qualified=False,normal_Model_lifecycle_qualified=False);save();return
  result=R/'results'/('Deep-prefill-chunk-pair-r348-'+actual+'.json');w['result_path']=str(result)
  if result.exists():
   q=json.loads(result.read_bytes());w.update(result_SHA256=sha(result),result_summary={k:q.get(k)for k in ['completed','passed','normal_server_shutdown','normal_Native_shutdown','actual_HTTP','actual_Native','owner_exit_observation','Model_qualification_scope']});w['Model_qualified']=bool(w['terminal'][2:4]==['COMPLETED','0:0']and q.get('passed')and q.get('completed')and q.get('actual_HTTP')==6 and q.get('actual_Native')==6 and q.get('normal_server_shutdown')and q.get('normal_Native_shutdown')and q.get('owner_exit_observation',{}).get('all8_unique_threads_exited'))
  w.update(completed=True,state='PREFILL_PAIR_COMPLETE_PENDING_INDEPENDENT_TIMING_AUDIT'if w['Model_qualified']else'TERMINAL_NO_NEW_MODEL_QUALIFICATION');save()
 except BaseException as e:
  w.update(error=repr(e),state='FAILED_BEFORE_GPU'if not w['GPU_submitted']else('TERMINAL_GPU_POST_OR_MODEL_AUDIT_FAILED_NO_RETRY'if 'terminal'in w else'FOLLOW_SAME_HANDLE_NO_RETRY'),completed='terminal'in w,Model_qualified=False);save()
  # An observation failure never means a job died. Slurm/inside absolute bounds
  # remain in force; root follows the recorded handle/unique label to terminal.
  raise
if __name__=='__main__':main()
