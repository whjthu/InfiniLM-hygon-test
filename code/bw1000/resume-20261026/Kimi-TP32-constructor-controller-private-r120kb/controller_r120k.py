"""Exactly one owned four-node GPU component; no retry or model promotion."""
import argparse,json,pathlib,subprocess,os,time,datetime,socket,traceback,sys
from contract_r120k import ROOT,INC,SOURCE,LABEL,NODES,CPU,CPU_SHA,NATIVE_SHA,gate,sha,clean_env
a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--go-file',required=True);o=a.parse_args()
m,go,cpu,wrapper=gate(o.manifest_sha,o.go_file);env=clean_env();keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];assert keeper.startswith('910407.')and socket.gethostname()=='a14r2n20'
rp=ROOT/(LABEL+'-launch.json');wp=ROOT/(LABEL+'-watch.json');assert not rp.exists()and not wp.exists()
TERM={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
known=None;terminal=None;proc=None;post_attempted=False;w={'qualified':False,'GPU_submitted':False,'restarts':0,'CPU_receipt_SHA256':CPU_SHA,'Native_SHA256':NATIVE_SHA,'wrapper_SHA256':go['wrapper_SHA256'],'wrapper_CPU_receipt_SHA256':go['wrapper_CPU_receipt_SHA256'],'controller_manifest_SHA256':o.manifest_sha,'publication_commit':go['publication_commit'],'root_GO_SHA256':sha(o.go_file),'scope':'constructor only: one KDA+one MLA+head+all93cachebuffers+metadata allgather; no93parameterconstruct/no weights/no forward'}
def write():
 w.update(heartbeat_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),keeper_step=keeper,keeper_PID=os.getpid(),keeper_hostname=socket.gethostname())
 p=wp.with_suffix('.tmp');p.write_text(json.dumps(w,indent=2));p.replace(wp)
def norm(s):return s.split()[0].rstrip('+')
def acct():return [s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-24','-j','910375','-n','-P','-o','JobIDRaw,JobName%110,State%40,ExitCode,NodeList%120'],env=env,text=True,timeout=25).splitlines()if len(s.split('|'))>=5]
def idle(tag):
 args=['srun','--jobid=910375','--overlap','--exact','--immediate=30','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(NODES),'-t','00:02:00','--job-name='+LABEL+'-'+tag,'python3',str(INC/'idle_ib_port_r120k.py'),str(m['master_port']),tag]
 z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')]
 assert z.returncode==0 and len(rows)==4 and {q['node']for q in rows}==set(NODES),(z.returncode,rows,z.stderr)
 rows.sort(key=lambda q:NODES.index(q['node']));assert all(q['passed']and not q['kfd_pids']and not q['CUDA_initialized']for q in rows)
 if tag=='before':assert all(q['port_checked']and q['wildcard_port_bind_free']for q in rows)
 assert len({q['ib0_ipv4'][0]for q in rows})==4 and len({q['step']for q in rows})==1
 handle=rows[0]['step'];deadline=time.monotonic()+30
 while True:
  matches=[q for q in acct()if q[0]==handle and q[1]==LABEL+'-'+tag];assert len(matches)<=1
  if matches and norm(matches[0][2])in TERM:assert matches[0][2:4]==['COMPLETED','0:0'];break
  assert time.monotonic()<deadline;time.sleep(2)
 return {'rows':rows,'terminal':matches[0],'completed_monotonic':time.monotonic(),'args':args}
with rp.open('x')as f:json.dump({'manifest_SHA256':o.manifest_sha,'publication_commit':go['publication_commit'],'root_GO_SHA256':sha(o.go_file),'restarts':0,'one_GPU_step_only':True},f,indent=2)
write()
try:
 for p,h in m['required_existing_SHA256'].items():assert sha(ROOT/p)==h,p
 parent=subprocess.check_output(['scontrol','show','job','910375','-o'],env=env,text=True,timeout=25);assert 'UserId=aclsylqidf('in parent and 'JobState=RUNNING'in parent
 spec=next(t.split('=',1)[1]for t in parent.split()if t.startswith('NodeList='));owned=subprocess.check_output(['scontrol','show','hostnames',spec],env=env,text=True,timeout=25).splitlines();assert set(NODES)<=set(owned);w['owned_parent_scontrol']=parent
 rows=acct();assert not any(q[1]==LABEL for q in rows)
 for q in rows:
  if q[0].startswith('910375.')and q[0].split('.')[-1].isdigit()and norm(q[2])not in TERM:
   active_nodes=subprocess.check_output(['scontrol','show','hostnames',q[4]],env=env,text=True,timeout=20).splitlines();assert not(set(NODES)&set(active_nodes)),q
 w['fresh_before']=idle('before');runtime=json.loads((INC/'runtime-template-r120k.json').read_bytes());assert runtime['explicit_constructor_GO']is False
 fresh={'qualified':True,'UTC':w['fresh_before']['rows'][0]['utc'],'KFD_empty_all4':True,'nodes':NODES,'port_free_all4':True,'rows':w['fresh_before']['rows'],'terminal':w['fresh_before']['terminal']};runtime.update(explicit_constructor_GO=True,root_source_published=True,Native_CPU_receipt_SHA256=CPU_SHA,wrapper_CPU_receipt_SHA256=go['wrapper_CPU_receipt_SHA256'],wrapper_SHA256=go['wrapper_SHA256'],master_addr=w['fresh_before']['rows'][0]['ib0_ipv4'][0],fresh_owned32_before=fresh,root_GO_SHA256=sha(o.go_file),publication_commit=go['publication_commit'],controller_manifest_SHA256=o.manifest_sha)
 age=time.monotonic()-w['fresh_before']['completed_monotonic'];assert age<=30;runtime['fresh_before_age_at_submit_seconds']=age
 runtime_path=ROOT/'results'/(LABEL+'-runtime.json')
 with runtime_path.open('x')as f:json.dump(runtime,f,indent=2)
 w.update(runtime_manifest=str(runtime_path),runtime_manifest_SHA256=sha(runtime_path),ready_before_GPU=True);write()
 args=['srun','--jobid=910375','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c32','--gres=dcu:8','--mem=128000M','-w',','.join(NODES),'-t','00:20:00','--job-name='+LABEL,'--output='+str(ROOT/'logs'/(LABEL+'-%J-%t.log')),'env','WAVE_INSIDE_R2323=Kimi-TP32-constructor-controller-r120k/GPU_inside_r120k.sh','R120K_MANIFEST_SHA256='+o.manifest_sha,'R120K_GO_FILE='+str(pathlib.Path(o.go_file).resolve()).replace(str(ROOT),'/refactor',1),'R120K_RUNTIME_SHA256='+sha(runtime_path),'R120K_RUNTIME='+str(runtime_path).replace(str(ROOT),'/refactor',1),'bash',str(ROOT/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(ROOT/'logs'/(LABEL+'-client.log')).open('x')as log:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 w.update(GPU_submitted=True,args=args,GPU_client_PID=proc.pid,state='following_one_unique_handle');write();start=time.monotonic();exit_at=None
 while time.monotonic()-start<1280:
  matches=[q for q in acct()if q[1]==LABEL];assert len(matches)<=1
  if matches:
   row=matches[0];assert known is None or known==row[0];known=row[0];w.update(actual_handle=known,Slurm=row);write()
   if norm(row[2])in TERM:terminal=row;w['terminal']=row;break
  elif proc.poll()is not None:
   if exit_at is None:exit_at=time.monotonic()
   elif time.monotonic()-exit_at>60:raise RuntimeError('client_exited_accounting_unknown_no_retry')
  time.sleep(10)
 else:raise RuntimeError('bounded_watch_unknown_follow_recorded_handle_no_retry')
 assert known is not None;proc.wait(timeout=30);w['GPU_client_returncode']=proc.returncode;write()
 # Post probes are permitted only after the same actual GPU handle is terminal.
 post_attempted=True;w['post_KFD']=idle('post');w['post_KFD_empty']=True;write()
 owners=[];samples=[];w['node_results']=[]
 for node in range(4):
  p=ROOT/'results'/('Kimi-TP32-construct-r120k-'+known+'-node'+str(node)+'.json');q=json.loads(p.read_bytes());w['node_results'].append({'path':str(p),'SHA256':sha(p)})
  assert q['completed']and q['all8normal_before_communicators']and q['node_rank']==node and q['returncode']==0 and not q['full93_model_parameters_constructed']and q['forward_calls']==0
  assert q['CPU_receipt_SHA256']==CPU_SHA and q['Native_SHA256']==NATIVE_SHA and q['actual_private_provider_SHA256']==NATIVE_SHA and q['Common_SHA256']==go['Common_SHA256']and q['actual_Common_provider_SHA256']==go['Common_SHA256']and q['private_Native0f_loaded']and not q['rejected_selfcontained_DSO_loaded'] and q['wrapper_CPU_receipt_SHA256']==go['wrapper_CPU_receipt_SHA256']and q['wrapper_SHA256']==go['wrapper_SHA256']and q['runtime_manifest_SHA256']==w['runtime_manifest_SHA256']and q['controller_manifest_SHA256']==o.manifest_sha
  assert q['fullModel_budget']=='UNKNOWN_STOP'and not q['Model_promotion']and len(q['owners'])==8
  for owner in q['owners']:
   rank=owner['global_rank'];assert rank==node*8+owner['local_rank']and owner['normal_Runtime_close']and owner['real32_BF16_headwidth_allgather']and not owner['checkpoint_opened']and owner['model_forward_calls']==0
   assert owner['head']['logical_bytes']==73400320
   kd=owner['KDA']['tensors'];md=owner['MLA']['tensors']
   for k in ['q_proj.weight','k_proj.weight','v_proj.weight']:assert kd[k]['shape']==[384,7168]
   assert kd['o_proj.weight']['shape']==[7168,384]and md['q_b_proj.weight']['shape'][0]==576 and md['kv_b_proj.weight']['shape']==[768,512]and md['g_proj.weight']['shape']==[384,7168]and md['o_proj.weight']['shape']==[7168,384]
   ca=owner['cache'];assert ca['bytes']==1086011136 and ca['KDA_layers']==69 and ca['MLA_layers']==24 and sorted(z['layer']for z in ca['shapes'])==list(range(93))
   for z in ca['shapes']:
    if 'KV'in z:assert z['KV']==[2,68,256,3,192]
    else:assert z['conv']==[17,1152,3]and z['recurrent']==[17,3,128,128]
   phases=['before_components','after_components_and_cache','after_collective','after_components_closed'];total=set()
   for phase in phases:
    z=owner[phase];assert z['free']>4*1024**3 and z['total']>z['free']and z['used']==z['total']-z['free'];total.add(z['total']);samples.append({'global_rank':rank,'phase':phase,**z})
   assert len(total)==1 and owner['before_components']['free']>7*1024**3;owners.append(rank)
  log=ROOT/'logs'/(LABEL+'-'+known+'-'+str(node)+'.log');text=log.read_text(errors='replace');assert text.count('KIMI_TP32_CONSTRUCT_R120K_NORMAL')==1 and 'Using network IBext_v8'in text and 'Using network Socket'not in text;w['node_results'][-1]['log_SHA256']=sha(log)
 assert sorted(owners)==list(range(32))and len(samples)==128
 w['memory_phase_samples']=samples;w['memory_observation']={'scope':'128 CUDA phase samples over32owners; not a whole-run or driver/CCL peak','minimum_phase_sample_free_bytes':min(z['free']for z in samples),'maximum_phase_sample_used_bytes':max(z['used']for z in samples),'actual_sampled_total_bytes_values':sorted({z['total']for z in samples}),'graphPinned':'UNKNOWN_NOT_SAMPLED','whole_run_memory_peak':'UNKNOWN','fullModel_budget':'UNKNOWN_STOP'}
 w['head_metadata_rowcoverage']={'global_rank_ids':sorted(owners),'local_head_rows':5120,'global_row_range':[0,163840],'transport':'C++ checked all32 rank-tag BF16 blocks of5120; synthetic metadata only, no user activations'}
 w['qualified']=terminal[2:4]==['COMPLETED','0:0']and proc.returncode==0;w.update(completed=True,state='terminal_constructor_normal32_postchecked',normal_owners=32,network='IBext_v8/noSocketData',GDR='UNKNOWN',Model_promotion=False,fullModel_budget='UNKNOWN_STOP',checkpoint_opened=False,model_forward_calls=0,full93_parameter_constructor=False);write();assert w['qualified']

except BaseException as e:
 w.update(error=repr(e),traceback=traceback.format_exc(),qualified=False,state='failed_before_GPU'if proc is None else'terminal_failure'if terminal else'unknown_follow_same_handle_no_restart',actual_handle=known)
 if terminal and not post_attempted:
  try:post_attempted=True;w['error_post_KFD']=idle('errorpost')
  except BaseException as pe:w['post_error']=repr(pe)
 elif proc and not terminal:w['post_not_run']='Actual GPU handle not terminal; no restart/cancel of other jobs. Root must follow recorded same handle.'
 write();raise
