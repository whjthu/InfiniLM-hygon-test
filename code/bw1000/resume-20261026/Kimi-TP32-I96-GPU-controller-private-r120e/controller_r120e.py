"""Exactly one owned four-node GPU component; no retry or model promotion."""
import argparse,json,pathlib,subprocess,os,time,datetime,socket,traceback,sys
from contract_r120e import ROOT,INC,SOURCE,LABEL,NODES,CPU,CPU_SHA,DSO_SHA,gate,sha,clean_env
a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--go-file',required=True);o=a.parse_args()
m,go,cpu=gate(o.manifest_sha,o.go_file);env=clean_env();keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];assert keeper.startswith('910407.')and socket.gethostname()=='a14r2n20'
rp=ROOT/(LABEL+'-launch.json');wp=ROOT/(LABEL+'-watch.json');assert not rp.exists()and not wp.exists()
TERM={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}
known=None;terminal=None;proc=None;post_attempted=False;w={'qualified':False,'GPU_submitted':False,'restarts':0,'CPU_receipt_SHA256':CPU_SHA,'library_SHA256':DSO_SHA,'controller_manifest_SHA256':o.manifest_sha,'publication_commit':go['publication_commit'],'root_GO_SHA256':sha(o.go_file),'scope':'standalone layer1 MoE TP32/I96 original FP4 math0; no fullModel/head/KDA'}
def write():
 w.update(heartbeat_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),keeper_step=keeper,keeper_PID=os.getpid(),keeper_hostname=socket.gethostname())
 p=wp.with_suffix('.tmp');p.write_text(json.dumps(w,indent=2));p.replace(wp)
def norm(s):return s.split()[0].rstrip('+')
def acct():return [s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-24','-j','910375','-n','-P','-o','JobIDRaw,JobName%110,State%40,ExitCode,NodeList%120'],env=env,text=True,timeout=25).splitlines()if len(s.split('|'))>=5]
def idle(tag):
 args=['srun','--jobid=910375','--overlap','--exact','--immediate=30','--cpu-bind=none','-N4','-n4','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(NODES),'-t','00:02:00','--job-name='+LABEL+'-'+tag,'python3',str(INC/'idle_ib_port_r120e.py'),str(m['master_port']),tag]
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
 w['fresh_before']=idle('before');runtime=json.loads((INC/'runtime-template-r120e.json').read_bytes());assert runtime['explicit_component_GO']is False
 runtime.update(explicit_component_GO=True,master_addr=w['fresh_before']['rows'][0]['ib0_ipv4'][0],fresh_before=w['fresh_before'],root_GO_SHA256=sha(o.go_file),publication_commit=go['publication_commit'],controller_manifest_SHA256=o.manifest_sha)
 age=time.monotonic()-w['fresh_before']['completed_monotonic'];assert age<=30;runtime['fresh_before_age_at_submit_seconds']=age
 runtime_path=ROOT/'results'/(LABEL+'-runtime.json')
 with runtime_path.open('x')as f:json.dump(runtime,f,indent=2)
 w.update(runtime_manifest=str(runtime_path),runtime_manifest_SHA256=sha(runtime_path),ready_before_GPU=True);write()
 args=['srun','--jobid=910375','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N4','-n4','--ntasks-per-node=1','-c32','--gres=dcu:8','--mem=128000M','-w',','.join(NODES),'-t','00:20:00','--job-name='+LABEL,'--output='+str(ROOT/'logs'/(LABEL+'-%J-%t.log')),'env','WAVE_INSIDE_R2323=Kimi-TP32-I96-GPU-controller-r120e/GPU_inside_r120e.sh','R120_COMPONENT_GO=1','R120E_MANIFEST_SHA256='+o.manifest_sha,'R120E_GO_FILE='+str(pathlib.Path(o.go_file).resolve()).replace(str(ROOT),'/refactor',1),'R120E_RUNTIME_SHA256='+sha(runtime_path),'R120_RUNTIME_MANIFEST='+str(runtime_path).replace(str(ROOT),'/refactor',1),'bash',str(ROOT/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
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
 paths=[];owners=[];w['node_results']=[]
 for node in range(4):
  p=ROOT/'results'/('Kimi-TP32-I96-MoE-r120-'+known+'-node'+str(node)+'.json');q=json.loads(p.read_bytes());paths.append(str(p));w['node_results'].append({'path':str(p),'SHA256':sha(p)})
  assert q['completed']and q['normal_CUDA_cleanup']and q['node_rank']==node and q['native']['all8normal_before_communicators']
  assert q['CPU_receipt_SHA256']==CPU_SHA and q['library_SHA256']==DSO_SHA and q['manifest_SHA256']==w['runtime_manifest_SHA256']and q['controller_manifest_SHA256']==o.manifest_sha
  assert q['input']['base_binary_distinct_rows']==8 and len(q['native']['reports'])==8 and len(q['raw_output_proof'])==144
  for owner in q['native']['reports']:
   rank=owner['global_tp_rank'];assert rank==node*8+owner['local_rank']and owner['node_rank']==node and owner['TP']==32 and owner['PP']==1
   assert owner['normal_Runtime_close']and owner['module_closed']and owner['all32_loaded_barrier']and len(owner['arms'])==18
   assert {(x['T'],x['arm'])for x in owner['arms']}=={(t,a)for t in [8,16,64]for a in range(6)}
   for arm in owner['arms']:assert arm['nonfinite']==0 and arm['input_readonly_tail_canary']and arm['private_provider_counter_delta_zero']
   for stage in ['physical_before_load','physical_after_load']:
    z=owner[stage];assert z['required_keys']==5384 and z['expert_keys']==5376 and z['global_I_range']==[rank*96,(rank+1)*96]and z['global_W2_packed_column_range']==[rank*48,(rank+1)*48]and z['global_W2_scale_column_range']==[rank*3,(rank+1)*3]and z['required_tensor_logical_bytes']<2*1024**3
   assert owner['physical_before_load']==owner['physical_after_load'];owners.append(rank)
  log=ROOT/'logs'/(LABEL+'-'+known+'-'+str(node)+'.log');s=log.read_text(errors='replace');assert s.count('KIMI_TP32_I96_R120E_NORMAL')==1 and 'Using network IBext_v8'in s and 'Using network Socket'not in s;w['node_results'][-1]['log_SHA256']=sha(log)
 assert sorted(owners)==list(range(32));sys.path.insert(0,str(SOURCE));from summarize_r120 import summarize
 w['summary']=summarize(paths);w['qualified']=terminal[2:4]==['COMPLETED','0:0']and proc.returncode==0;w.update(completed=True,state='terminal_normal32_postchecked',normal_owners=32,network='IBext_v8/noSocketData',GDR='UNKNOWN',Model_promotion=False,fullModel_budget='UNQUALIFIED',speedup_claim=False);write();assert w['qualified']
except BaseException as e:
 w.update(error=repr(e),traceback=traceback.format_exc(),qualified=False,state='failed_before_GPU'if proc is None else'terminal_failure'if terminal else'unknown_follow_same_handle_no_restart',actual_handle=known)
 if terminal and not post_attempted:
  try:post_attempted=True;w['error_post_KFD']=idle('errorpost')
  except BaseException as pe:w['post_error']=repr(pe)
 elif proc and not terminal:w['post_not_run']='Actual GPU handle not terminal; no restart/cancel of other jobs. Root must follow recorded same handle.'
 write();raise
