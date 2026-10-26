import pathlib,json,hashlib,subprocess,os,sys,time,uuid,datetime,socket,traceback,re
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming/GLM-deferred-Model-r9226';label='GLM-deferred-Model-r9226';parent='910376';nodes=['a02r4n12','a02r4n13'];keeper=os.environ.get('SLURM_JOB_ID','')+'.'+os.environ.get('SLURM_STEP_ID','');assert keeper.startswith(parent+'.')and keeper.split('.')[-1].isdigit();host=socket.gethostname();env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
record=r/(label+'-launch.json');wp=r/(label+'-watch.json');term={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'};sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();norm=lambda s:s.split()[0].rstrip('+')
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,q):
 q.update(CPU_keeper_step=keeper,CPU_keeper_host=host,CPU_keeper_PID=os.getpid(),heartbeat_UTC=now());tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(p)
def accounting():return [s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobID,JobName%100,State%40,ExitCode,NodeList%120'],env=env,text=True,timeout=30).splitlines()]
def idle(suffix):
 a=['srun','--jobid='+parent,'--overlap','--exact','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','-w',','.join(nodes),'--gres=none','--mem=512M','-t','00:03:00','--job-name='+label+'-'+suffix,'python3',str(r/'incoming/verify_gpu_idle_r5738.py')];z=subprocess.run(a,env=env,capture_output=True,text=True,timeout=190);assert z.returncode==0,(z.stdout,z.stderr);q=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];assert sorted(v['node']for v in q)==nodes and all(v['kfd_pids']==[]for v in q);return q
assert not record.exists()and not wp.exists();d={'state':'intent','run':'GLM-deferred-r9226-'+uuid.uuid4().hex[:16],'label':label,'parent':parent,'nodes':nodes,'created_UTC':now(),'manifest_SHA256':sha(inc/'GPU-manifest.json'),'restarts':0};write(record,d);w={'state':'checking','run':d['run'],'actual_handle':None,'warmups':{},'qualified':False};write(wp,w);process=None
try:
 m=json.loads((inc/'GPU-manifest.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in m['existing_remote_SHA256'].items():assert sha(r/n)==h,n
 cp=json.loads((inc/'CPU-result.json').read_bytes());assert cp['completed']and cp['actual_nodes']==nodes
 rows=accounting();assert any(v[0]==parent and norm(v[2])=='RUNNING'for v in rows)and not any(v[1]==label for v in rows)
 for v in rows:
  if v[0]!=keeper and v[0].startswith(parent+'.')and v[0].split('.')[-1].isdigit()and norm(v[2])not in term:
   nn=subprocess.check_output(['scontrol','show','hostnames',v[4]],env=env,text=True,timeout=20).splitlines();assert not set(nodes)&set(nn),v
 d['fresh_before_KFD']=idle('before');write(record,d);w.update(state='watch_ready_before_GPU',ready=True);write(wp,w)
 args=['srun','--jobid='+parent,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','-w',','.join(nodes),'--gres=dcu:8','--mem=240000M','-t','01:30:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J-%t.log')),'env','GLM_SHARED_PREFIX_VALIDATE_R5814=0','GLM_PREFLIGHT_ONLY_R5194=0','GLM_DECODE_ROW_DEDUP_R5386=0','GLM_RUN_R5040='+d['run'],'GLM_BOUNDED_ROPE_R4279=1','GLM_SHARED_CONTROL_R4279=1','GLM_PP_SIZE_R3603=2','GLM_CACHE_ROWS_R3627=32','GLM_CACHE_CAP_R3627=640','GLM_NETWORK_TRIAL=shca','GLM_PIPELINE_TAIL_NODE=a02r4n13','GLM_EXACT_BANKS_R4188=1','GLM_CHANNELS_R4363=0','GLM_VARIANT_R4363=layer77-5783','INFINILM_OWNER_CPU_WIDTH_R2392=4','WAVE_INSIDE_R2323=GLM-deferred-Model-r9226/GPU_inside.sh','USE_WEIGHT_CACHE_R3604=0','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(r/'logs'/(label+'-srun-client.log')).open('x')as f:process=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d.update(state='unique_GPU_submitted_follow_exact_handle',client_PID=process.pid,args=args);write(record,d);started=time.monotonic();last_acct=0;known=None;cancelled=False
 while time.monotonic()-started<5650:
  if time.monotonic()-last_acct>10:
   matching=[v for v in accounting()if v[1]==label];assert len(matching)<=1
   if matching:
    v=matching[0];assert known is None or known==v[0];known=v[0];w.update(actual_handle=known,Slurm=v,state='running_or_following_same_handle');write(wp,w)
    if norm(v[2])in term:w['terminal']=v;break
   last_acct=time.monotonic()
  for stage in [0,1]:
   p=r/'results'/('GLM-warm-r9226-'+d['run']+'-s'+str(stage)+'-r0.json')
   if p.exists():
    try:q=json.loads(p.read_bytes())
    except (ValueError,FileNotFoundError):continue
    assert q['run']==d['run']and q['stage']==stage;w['warmups'][str(stage)]=q
    if q['warm_active']and time.time()-q['UTC_epoch']>q['stall_limit_seconds']+1 and known and not cancelled:
     rows=accounting();exact=[v for v in rows if v[0]==known and v[1]==label];assert len(exact)==1
     w.update(warm_stall_failure={'observed_UTC':now(),'stage':stage,'progress':q,'exact_owned_step':known},state='warm_stall_evidence_saved_before_precise_cancel');write(wp,w);subprocess.run(['scancel',known],env=env,check=True,timeout=30);cancelled=True
  write(wp,w);time.sleep(2)
 else:w['watch_bound_expired']=True;write(wp,w);raise RuntimeError('onehandle watch bound, no retry')
 d.update(state='terminal_observed',actual_handle=known,terminal=w['terminal']);write(record,d)
 try:
  z=subprocess.run([sys.executable,str(inc/'audit.py')],env=env,capture_output=True,text=True,timeout=180);w['audit_returncode']=z.returncode;w['audit_stderr']=z.stderr[-6000:]
  if z.returncode==0:
   q=json.loads(z.stdout);audit=r/'results'/(label+'-audit.json');audit.write_text(json.dumps(q,indent=2));w.update(audit_path=str(audit),audit_SHA256=sha(audit),audit_qualified=q['qualified'])
  sys.path.insert(0,str(r/'incoming'));from glm_crossrequest_environment_r9059 import network_log_gate
  w['logs']=[]
  for node in [0,1]:
   p=r/'logs'/(label+'-'+known+'-'+str(node)+'.log');text=p.read_text(errors='replace');w['logs'].append({'path':str(p),'SHA256':sha(p),'network_gate':network_log_gate(text),'normal_marker':'GLM_DEFERRED_NODE_R9226_NORMAL '+str(node)in text,'errors':[x for x in text.splitlines()if re.search(r'Traceback|AssertionError|RuntimeError|Segmentation fault|HIP error|CUDA error|rocblas_status',x)][-40:]})
 except BaseException as e:w.update(audit_error=repr(e),audit_traceback=traceback.format_exc())
 finally:
  try:w['post_KFD']=idle('post');w['post_passed']=True
  except BaseException as e:w.update(post_error=repr(e),post_passed=False)
 w.update(state='terminal_audited_postchecked',qualified=bool(w['terminal'][2:4]==['COMPLETED','0:0']and w.get('audit_qualified')and w.get('post_passed')and all(v['normal_marker']and not v['errors']and v['network_gate']['passed']for v in w.get('logs',[]))));write(wp,w);d.update(state='completed_keeper',qualified=w['qualified']);write(record,d)
except BaseException as e:
 d.update(error=repr(e),traceback=traceback.format_exc(),state='unknown_GPU_follow_same_handle'if process else'failed_before_GPU');write(record,d);w.update(error=repr(e));write(wp,w);raise
