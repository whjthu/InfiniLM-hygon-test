"""One separately authorized original8card residency. No retry/reload or import-only GPU job."""
import argparse,pathlib,json,hashlib,subprocess,os,time,re,datetime,socket
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=pathlib.Path(__file__).resolve().parent;S=R/'incoming/Deep-latest-full-owner-r287';LABEL='Deep-latest99-full-Model-r283';NODE='a02r4n01';W=R/'results'/(LABEL+'-watch.json');TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'};sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
OWNER_SHA='12a4160d5703a9364929c41d5b9b69772a600254f2084ec2774d585023b17df1';CPU_SHA='215635334f529316a8b804f828babd2cf5faa599be898feaa8f80651dab38913'
def validate_scope(g):
 if g.get('GPU_authorized')is not True:raise PermissionError('Explicit additional Deep GPU execution GO required')
 assert g['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',g['publication_commit']);assert g['owner_manifest_SHA256']==OWNER_SHA and re.fullmatch('[0-9a-f]{64}',g['controller_manifest_SHA256']);assert g['GPU_parent']==910374 and g['GPU_node']==NODE and g['CPU_keeper_parent']==910407 and g['CPU_keeper_node']=='a14r2n20'
 for k in ['GPU_parent','CPU_keeper_parent','cards','TP','PP','context','ordinary_budget','thinking_budget','max_HTTP','max_Native','GPU_max_s','Native_max_s','CPU_keeper_max_s']:assert type(g[k])is int
 assert(g['cards'],g['TP'],g['PP'],g['context'],g['ordinary_budget'],g['thinking_budget'],g['max_HTTP'],g['max_Native'])==(8,8,1,32768,4096,8192,103,102)
 assert g['GPU_max_s']==29400 and g['Native_max_s']==28800 and g['CPU_keeper_max_s']==32400 and g['one_attempt']is True and g['no_retry_reload']is True
 return g

def verify_sources(g):
 validate_scope(g);assert sha(D/'SOURCE_MANIFEST.json')==g['controller_manifest_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items());assert sha(S/'SOURCE_MANIFEST.json')==OWNER_SHA;sm=json.loads((S/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S/n)==h for n,h in sm['files_SHA256'].items());cp=R/'candidates/Deep-owner-cleanup-r289/CPU-prepared.json';assert sha(cp)==CPU_SHA;c=json.loads(cp.read_bytes());assert c['completed']and c['passed']and c['CPU_only']and c['GPU']==c['Model']==c['production_HTTP']==0 and c['owner_source_manifest_SHA256']==OWNER_SHA and c['guarded_preweight_import_PASS']and c['inherited_CPU610_client_contract']and len(c['cleanup_cases'])==2
 assert sha(R/'incoming/resource_wave_local_r3097_outer.sh')=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'

def clean_env():return{k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
def parent(parent,node,env):
 z=subprocess.check_output(['scontrol','show','job',str(parent),'-o'],env=env,text=True,timeout=25);assert 'UserId=aclsylqidf('in z and 'JobState=RUNNING'in z;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',z).group(1);expanded=subprocess.check_output(['scontrol','show','hostnames',nodes],env=env,text=True,timeout=15).splitlines();assert node in expanded

def validate_account_row(row):
 assert re.fullmatch(r'910374[.]\d+',row[0])and row[1]==LABEL and row[4]==NODE,'Accounting row is not the exact owned parent/label/node'

def main():
 p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();g=json.loads(pathlib.Path(a.execution_go).read_bytes());verify_sources(g);assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname()=='a14r2n20';assert not W.exists();env=clean_env();parent(910407,'a14r2n20',env);parent(910374,NODE,env)
 w={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'GPU_submitted':False,'actual_GPU':None,'CPU_keeper_actual':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'publication_commit':g['publication_commit'],'owner_manifest_SHA256':OWNER_SHA,'controller_manifest_SHA256':g['controller_manifest_SHA256'],'scope':'Latest99 complete normal+multicall+fixedGSM32+fixedMMLU32 same residency; bufferedHTTP no incrementalSLO claim','no_retry_reload':True,'GPU_max_s':29400,'CPU_keeper_max_s':32400,'Model_qualified':False};proc=None;actual=None
 def save():
  w['observed_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();tmp=W.with_suffix('.tmp');tmp.write_text(json.dumps(w,indent=2)+chr(10));tmp.replace(W)
 def account():
  rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-31','-j','910374','-n','-P','-o','JobID,JobName%120,State,ExitCode,NodeList%100'],env=env,text=True,timeout=25).splitlines()];rows=[v for v in rows if v[1]==LABEL];assert len(rows)<=1;
  for row in rows:validate_account_row(row)
  return rows[0]if rows else None
 def idle(suffix):
  args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',NODE,'--gres=none','--mem=512M','-t','00:02:00','--job-name='+LABEL+'-'+suffix,'python3',str(D/'idle_r283.py')];z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];assert z.returncode==0 and len(rows)==1 and rows[0]['node']==NODE and not rows[0]['kfd_pids']and rows[0]['HTTP43882_bindable'];return rows[0]
 save()
 try:
  assert account()is None,'Existing exact label must be followed, never replayed';w['fresh_before']=idle('before');save();start=time.monotonic();epoch=time.time();w['GPU_start_UTC_unix']=epoch;save()
  export='ALL,WAVE_INSIDE_R2323=Deep-latest-full-owner-r287/GPU_inside_r277.sh,DEEP_R277_GPUSTART_UNIX='+str(epoch)+',DEEP_SOURCE_MANIFEST_SHA='+OWNER_SHA
  args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c32','-w',NODE,'--gres=dcu:8','--mem=128000M','-t','08:10:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J.log')),'--export='+export,'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_submitted=True,srun_client_PID=proc.pid,state='FOLLOW_SAME_SINGLE_OWNED_GPU_HANDLE');save();end=start+30000
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
  w['post']=idle('post');w['post_empty']=True;result=R/'results'/('Deep-latest-full-owner-r287-'+actual+'.json');w['result_path']=str(result)
  if result.exists():
   q=json.loads(result.read_bytes());w.update(result_SHA256=sha(result),result_summary={k:q.get(k)for k in ['completed','passed','normal_server_shutdown','normal_Native_shutdown','actual_HTTP','actual_Native','owner_exit_observation','Model_qualification_scope']});w['Model_qualified']=bool(w['terminal'][2:4]==['COMPLETED','0:0']and q.get('passed')and q.get('completed')and q.get('actual_HTTP')==103 and q.get('actual_Native')==102 and q.get('normal_server_shutdown')and q.get('normal_Native_shutdown')and q.get('owner_exit_observation',{}).get('all8_unique_threads_exited'))
  w.update(completed=True,state='FINITE_NORMAL_MULTICALL_COARSE_PILOTS_PASS_PENDING_ROOT_AUDIT'if w['Model_qualified']else'TERMINAL_NO_NEW_MODEL_QUALIFICATION');save()
 except BaseException as e:
  w.update(error=repr(e),state='FAILED_BEFORE_GPU'if not w['GPU_submitted']else'FOLLOW_SAME_HANDLE_NO_RETRY');save()
  # An observation failure never means a job died. Slurm/inside absolute bounds
  # remain in force; root follows the recorded handle/unique label to terminal.
  raise
if __name__=='__main__':main()
