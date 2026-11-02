"""One current976f fixed-teacher numerical diagnosis; no HTTP or performance claim."""
import argparse,pathlib,json,hashlib,subprocess,os,time,re,datetime,socket
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=pathlib.Path(__file__).resolve().parent;S=R/'incoming/Deep-HC-teacher-forced-logits-r397/source-v3/owner';LABEL='Deep-HC-teacher-forced-logits-r397-numeric-v1';NODE='a02r4n01';W=R/'results'/(LABEL+'-watch.json');TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'};sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
from dso_binding import verify_dso,CPU621_SHA
from namespace_binding import verify_namespace,WRAPPER_SHA as NAMESPACE_WRAPPER_SHA,CPU_SHA as NAMESPACE_CPU_SHA,terminal_failure
OWNER_SHA='66e93d47f89d9f8c7645d41427e1ae264e59ddeb3a2cc400abd6aa68db8ccfa6'
SOURCE_SHA='d3493e9f6ff56072564f78c9fe5f3b06176f2eb2be0ce532b2180ccd19be04df'
CPU_WHOLE_SHA='fb1789e0cfb0a4f05d9a7160de9f02981585a9fc78328cc66fa4f3c059e1f682'
CPU_ENTRY_SHA='e4cac684955ae2d6ccffabe5a77e5723b556601e824c8d8753ce204751bce65b'
NATIVE_SHA='976f97b51e9f4b11dfb2439d2c97f2a1e7788e4de05ee8a8a0138f33e38b6ed1'
def host(p):return R/pathlib.Path(p).relative_to('/refactor')if str(p).startswith('/refactor/')else pathlib.Path(p)
def validate_scope(g):
 if g.get('GPU_authorized')is not True or g.get('root_consumed_final_GPU_scope')is not True:raise PermissionError('Published exact teacher diagnosis scope must be root consumed')
 assert g['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])
 assert g['owner_manifest_SHA256']==OWNER_SHA and g['source_manifest_SHA256']==SOURCE_SHA and re.fullmatch('[0-9a-f]{64}',g['controller_manifest_SHA256'])
 assert g['GPU_parent']==910374 and g['GPU_node']==NODE and g['CPU_keeper_parent']==910407 and g['CPU_keeper_node']=='a14r2n20'
 keys=['GPU_parent','CPU_keeper_parent','cards','TP','PP','context','max_HTTP','max_Native_sequences','max_forwards','teacher_tokens','output_decisions','prefill_forwards_each','decode_forwards_each','GPU_max_s','Native_max_s','worker_max_s','CPU_keeper_max_s']
 assert all(type(g[k])is int for k in keys)
 assert(g['cards'],g['TP'],g['PP'],g['context'],g['max_HTTP'],g['max_Native_sequences'],g['max_forwards'])==(8,8,1,32768,0,4,368)
 assert(g['teacher_tokens'],g['output_decisions'],g['prefill_forwards_each'],g['decode_forwards_each'])==(83,84,9,83)
 assert(g['GPU_max_s'],g['Native_max_s'],g['worker_max_s'],g['CPU_keeper_max_s'])==(1800,1770,1680,2100)and g['one_attempt']is True and g['no_retry_reload']is True
 assert g['Native_SHA256']==NATIVE_SHA and g['prefill_chunk']==1024 and g['prompt_tokens']==8489
 assert g['HC_modes']==[0,1,1,0]and all(type(x)is int for x in g['HC_modes'])and g['observer_enabled']is False
 assert g['teacher_payload_SHA256']=='30f8834b470540ab6da6a15838370a06f9e8401415a9718fc26aecbb51838def'and g['teacher_IDs_SHA256']=='596df3e1131e55184edc034f57f8447e025b91a38c7d6a20a69eaa8b61f39513'
 assert g['CPU_whole_receipt_SHA256']==CPU_WHOLE_SHA and g['CPU_entry_receipt_SHA256']==CPU_ENTRY_SHA
 assert g['wrapper_SHA256']==NAMESPACE_WRAPPER_SHA and g['DSO_CPU621_receipt_SHA256']==CPU621_SHA
 return g

def verify_sources(g):
 validate_scope(g);assert sha(D/'SOURCE_MANIFEST.json')==g['controller_manifest_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
 assert sha(S.parent/'SOURCE_MANIFEST.json')==SOURCE_SHA;sm=json.loads((S.parent/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(S.parent/n)==h for n,h in sm['files_SHA256'].items())
 assert sha(S/'SOURCE_MANIFEST.json')==OWNER_SHA
 cp=R/'candidates/Deep-HC-teacher-forced-logits-r397-CPU-v3/CPU-whole.json';assert sha(cp)==CPU_WHOLE_SHA;c=json.loads(cp.read_bytes())
 assert c['completed']and c['passed']and c['actual_CPU']=='910407.792'and c['owner_manifest_SHA256']==OWNER_SHA and c['source_manifest_SHA256']==SOURCE_SHA
 assert all(c[k]is False for k in ['GPU','Native_dlopen','weights_opened','new_compile'])
 ep=cp.with_name('CPU-entry.json');assert sha(ep)==CPU_ENTRY_SHA;e=json.loads(ep.read_bytes());assert e['completed']and e['passed']and len(e['cases'])==10 and all(x['passed']for x in e['cases'])
 assert e['actual_loader_AST']['Native_forward_AST_unchanged']and e['actual_loader_AST']['original_finally_destroy_retained']
 assert e['actual_CPU_prompt']['prompt_IDs_SHA256']=='681977ee292994eac340bae5c1aadf7a6b136890df725d02bb75fc8f31ecf0df'and e['actual_CPU_prompt']['prompt_count']==8489
 success=next(x for x in e['cases']if x['case']=='success');assert success['actual_forward_leaf_calls']==368 and success['actual_reset_leaf_calls']==4 and success['actual_bootstrap_to_module_callback']and success['eight_fake_Native_owner_threads_exited']
 assert {'nonzero-code','cleanup-failure','forward-and-cleanup-failure'}<={x['case']for x in e['cases']}
 private=R/'incoming/Deep-HC-teacher-forced-logits-r397/private/teacher-prefix.json';assert sha(private)==g['teacher_payload_SHA256']
 pin=json.loads((S/'NATIVE_BUILD_PIN.json').read_bytes());bp=host(pin['CPU_receipt']);assert sha(bp)==pin['CPU_receipt_SHA256'];b=json.loads(bp.read_bytes());assert b['passed']and b['library_SHA256']==NATIVE_SHA and sha(host(b['library']))==NATIVE_SHA
 assert len(b['all_objects_SHA256'])==49 and all(sha(host(n))==h for n,h in b['all_objects_SHA256'].items())
 verify_namespace(R,D);verify_dso(R,R/'incoming/Deep-latest-full-owner-r300','Deep')

def clean_env():return{k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
def parent(parent,node,env):
 z=subprocess.check_output(['scontrol','show','job',str(parent),'-o'],env=env,text=True,timeout=25);assert 'UserId=aclsylqidf('in z and 'JobState=RUNNING'in z;nodes=re.search(r'(?:^| )NodeList=([^ ]+)',z).group(1);expanded=subprocess.check_output(['scontrol','show','hostnames',nodes],env=env,text=True,timeout=15).splitlines();assert node in expanded

def validate_account_row(row):
 assert re.fullmatch(r'910374[.]\d+',row[0])and row[1]==LABEL and row[4]==NODE,'Accounting row is not the exact owned parent/label/node'

def main():
 p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();g=json.loads(pathlib.Path(a.execution_go).read_bytes());verify_sources(g);assert os.environ['SLURM_JOB_ID']=='910407'and socket.gethostname()=='a14r2n20';assert not W.exists();env=clean_env();parent(910407,'a14r2n20',env);parent(910374,NODE,env)
 w={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'completed':False,'GPU_submitted':False,'actual_GPU':None,'CPU_keeper_actual':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'publication_commit':g['publication_commit'],'owner_manifest_SHA256':OWNER_SHA,'controller_manifest_SHA256':g['controller_manifest_SHA256'],'scope':'One exact976f TP8PP1 load, frozen8489 teacher83, HC0/1/1/0 four reset sequences,9prefill+83decode each,368max,0HTTP. Numerical diagnosis only, no EOS/quality/performance qualification','no_retry_reload':True,'GPU_max_s':1800,'CPU_keeper_max_s':2100,'Model_qualified':False,'Diagnostic_qualified':False,'namespace_wrapper_SHA256':NAMESPACE_WRAPPER_SHA,'namespace_CPU618_SHA256':NAMESPACE_CPU_SHA,'DSO_CPU621_inside_SHA256':CPU621_SHA};proc=None;actual=None
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
  export='ALL,WAVE_INSIDE_R2323=Deep-HC-teacher-forced-logits-r397/controller-v1/GPU_inside_r397.sh,DEEP_R277_GPUSTART_UNIX='+str(epoch)+',DEEP_SOURCE_MANIFEST_SHA='+OWNER_SHA
  args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c32','-w',NODE,'--gres=dcu:8','--mem=128000M','-t','00:30:00','--job-name='+LABEL,'--output='+str(R/'logs'/(LABEL+'-%J.log')),'--export='+export,'bash',str(D/'resource_wave_host_alias_outer.sh'),'probe']
  gpu_env=dict(env,WAVE_INSIDE_R2323='Deep-HC-teacher-forced-logits-r397/controller-v1/GPU_inside_r397.sh',DEEP_R277_GPUSTART_UNIX=str(epoch),DEEP_SOURCE_MANIFEST_SHA=OWNER_SHA)
  with(R/'logs'/(LABEL+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(args,env=gpu_env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  w.update(GPU_submitted=True,srun_client_PID=proc.pid,state='FOLLOW_SAME_SINGLE_OWNED_GPU_HANDLE');save();end=start+1860
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
  w['srun_exit_code']=proc.wait(timeout=30);w['post']=idle('post');w['post_empty']=True
  if terminal_failure(w['terminal']):
   w.update(completed=True,state='TERMINAL_GPU_FAILED_POST_CLEAN_NO_RETRY',Model_qualified=False,normal_Model_lifecycle_qualified=False);save();return
  result=R/'results'/('Deep-HC-teacher-forced-logits-r397-'+actual+'.json');w['result_path']=str(result)
  if result.exists():
   q=json.loads(result.read_bytes());w.update(result_SHA256=sha(result),result_summary={k:q.get(k)for k in ['completed','passed','actual_step','legacy_tag','normal_Native_shutdown','actual_forwards','actual_forward_attempts','actual_resets','actual_reset_attempts','owner_exit_observation','error','primary_error','cleanup_errors']})
   seq=q.get('sequences',[]);valid_sequences=len(seq)==4 and [v.get('HC_mode')for v in seq]==[0,1,1,0]and all(v.get('completed')is True and v.get('reset_ok')is True and v.get('prefill_calls')==9 and v.get('decode_calls')==83 and [x.get('output_index')for x in v.get('decisions',[])]==list(range(84))and all(x.get('finite')is True for x in v.get('decisions',[]))for v in seq)
   w['Diagnostic_qualified']=bool(w['terminal'][2:4]==['COMPLETED','0:0']and w['srun_exit_code']==0 and q.get('actual_step')==actual and q.get('passed')is True and q.get('completed')is True and q.get('actual_forwards')==q.get('actual_forward_attempts')==368 and q.get('actual_resets')==q.get('actual_reset_attempts')==4 and q.get('HTTP_requests')==0 and valid_sequences and q.get('normal_Native_shutdown')is True and q.get('owner_exit_observation',{}).get('all8_unique_threads_exited')is True and q.get('HC_clear_rc')==q.get('HC_after_clear')==0 and q.get('observer_disarm_rc')==[0,0]and not q.get('cleanup_errors'))
  w.update(completed=True,state='TEACHER_DIAGNOSIS_COMPLETE_PENDING_OFFLINE_AUDIT'if w['Diagnostic_qualified']else'TERMINAL_NO_DIAGNOSTIC_QUALIFICATION');save()
 except BaseException as e:
  w.update(error=repr(e),state='FAILED_BEFORE_GPU'if not w['GPU_submitted']else('TERMINAL_GPU_POST_OR_MODEL_AUDIT_FAILED_NO_RETRY'if 'terminal'in w else'FOLLOW_SAME_HANDLE_NO_RETRY'),completed='terminal'in w,Model_qualified=False);save()
  # An observation failure never means a job died. Slurm/inside absolute bounds
  # remain in force; root follows the recorded handle/unique label to terminal.
  raise
if __name__=='__main__':main()
