import common,os,pathlib,json,subprocess,socket,time,datetime,hashlib,traceback
r=common.ROOT;D=common.SRC;env,_=common.clean_env();label='Kimi-prefill-pool-r133';hp=r/'results'/('Kimi-prefill-pool-r133-control.json');assert not hp.exists();m=json.loads((D/'GPU-manifest-source.json').read_bytes());g={'parent':'910379','nodes':['a14r1n11','a14r2n05']};sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();w={'CPU_actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'host':socket.gethostname(),'PID':os.getpid(),'GPU_submitted':False,'completed':False,'ready':False,'label':label,'source_manifest_SHA256':sha(D/'GPU-manifest-source.json')};actual=None;process=None
def save():w['heartbeatUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();common.atomic(hp,w)
def idle(suffix):
 name=label+'-'+suffix;args=['srun','--jobid=910379','--overlap','--exact','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(g['nodes']),'-t','00:02:00','--job-name='+name,'python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')];z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')];assert z.returncode==0 and len(rows)==2 and {x['node']for x in rows}==set(g['nodes'])and all(x['passed']and not x['kfd_pids']for x in rows),(z.returncode,rows,z.stderr);steps={x['step']for x in rows};assert len(steps)==1;terminal=common.terminal('910379',name,steps.pop(),env);assert terminal[2:4]==['COMPLETED','0:0'];return {'rows':rows,'terminal':terminal}
save()
try:
 common.verify_parent(g,env)
 sourcepins=json.loads((D/'controller-source-pins.json').read_bytes())
 for rel,h in sourcepins['files_SHA256'].items():assert sha(D/rel)==h,rel
 go=json.loads((D/'GPU-GO-after-source-publication.json').read_bytes());assert go['authorized_by']=='parent-root' and go['manifest_SHA256']==w['source_manifest_SHA256'] and len(go['Git_publication_SHA'])==40
 for p,h in m['source_SHA256'].items():assert sha(p.replace('/refactor',str(r),1))==h,p
 evidence=json.loads((D/'actual-CPU458-evidence.json').read_bytes());assert evidence['actual_terminal'][2:4]==['COMPLETED','0:0'];assert sha(evidence['receipt_path'].replace('/refactor',str(r),1))==evidence['receipt_SHA256'];assert evidence['native_SHA256']==m['native_SHA256']and evidence['library_SHA256']==m['library_SHA256']
 prelabel=label+'-imports';a=['srun','--jobid=910379','--overlap','--exact','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c32','--gres=none','--mem=8192M','-w',','.join(g['nodes']),'-t','00:04:00','--job-name='+prelabel,'--export=ALL,WAVE_INSIDE_R2323=Kimi-prefill-pool-controller-r133/preflight_inside_r133.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=280);rows=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')];assert z.returncode==0 and len(rows)==2 and all(x['passed']and x['CPU_only']and not x['GPU']and not x['dlopen']and x['source_manifest_SHA256']==w['source_manifest_SHA256']for x in rows),(z.returncode,rows,z.stderr[-2000:]);steps={x['actual_step']for x in rows};assert len(steps)==1;terminal=common.terminal('910379',prelabel,steps.pop(),env);assert terminal[2:4]==['COMPLETED','0:0'];w['CPU_import_closure']={'rows':rows,'terminal':terminal}
 w['fresh_before']=idle('before');m['master_addr']=next(q['ib0_ipv4'][0]for q in w['fresh_before']['rows']if q['node']=='a14r1n11');common.atomic(D/'GPU-manifest.json',m);w['runtime_manifest_SHA256']=sha(D/'GPU-manifest.json');w['ready']=True;w['watch_ready_before_GPU']=True;save()
 a=['srun','--jobid=910379','--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c32','--gres=dcu:8','--mem=128000M','-w',','.join(g['nodes']),'-t','00:20:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J-%t.log')),'--export=ALL,WAVE_INSIDE_R2323=Kimi-prefill-pool-controller-r133/GPU_inside_r133.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];fo=(r/'logs'/(label+'-client.log')).open('x');process=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=fo,stderr=subprocess.STDOUT);w.update(GPU_submitted=True,client_PID=process.pid,command_scope='owned same Slurmstep, no rawargv/environment persisted');save();deadline=time.monotonic()+1280
 while True:
  row=common.account('910379',label,env)
  if row:
   if actual is None:actual=row[0]
   assert actual==row[0];w['actual_handle']=actual;w['actual_row']=row
   if row[2].split()[0].rstrip('+')not in common.ACTIVE:w['terminal']=row;save();break
  assert time.monotonic()<deadline,'componentboundedwatchdeadline';save();time.sleep(10)
 w['post_KFD']=idle('post');w['results']=[];allowners=[]
 for node in range(2):
  p=r/'results'/(label+'-'+actual+'-node%d.json'%node);raw=p.read_bytes();q=json.loads(raw);w['results'].append({'node':node,'path':str(p),'SHA256':hashlib.sha256(raw).hexdigest(),'completed':q.get('completed'),'normal_cleanup':q.get('normal_CUDA_cleanup'),'all8owners_closed':q.get('native',{}).get('all8normal_before_communicators'),'error':q.get('error')});assert q['completed']and q['normal_CUDA_cleanup']and q['native']['all8normal_before_communicators'];assert len(q['native']['reports'])==8
  assert q['actual_helper_binding']['SHA256']==m['helper_expected_SHA256']
  for owner in q['native']['reports']:
   assert owner['normal_Runtime_close']and owner['module_closed']and len(owner['arms'])==10 and not owner['profile']and len(owner['current_branch'])==1;assert owner['actual_helper_path']==q['actual_helper_binding']['path'];allowners.append(owner)
   for arm in owner['arms']:assert arm['T']==576 and arm['finite']and arm['output_relative_RMS']<=.01 and arm['input_readonly_and_tail_canary']and arm['driver_samples']>0
   assert owner['current_branch'][0]['I192_calls']==owner['current_branch'][0]['W2pair_calls']==0
  log=(r/'logs'/(label+'-'+actual+'-'+str(node)+'.log')).read_text(errors='replace');assert log.count('KIMI_PREFILL_POOL_R132_NORMAL')==1 and'Using network IBext_v8'in log and'Using network Socket'not in log
 from interpret_memory_r133 import interpret
 w['memory_interpretation']=interpret(allowners);w['passed']=w['terminal'][2:4]==['COMPLETED','0:0'];w['audit']={'normal_owners':16,'GPU':True,'component_only':True,'not_Model_or_currentrequest_bottleneck':True,'Model_fit':'PARK','post_KFD_empty':True,'network':'IBext_v8/noSocketData','GDR':'UNKNOWN'};w['completed']=True;save()
except BaseException as e:
 w['error']=repr(e);w['traceback']=traceback.format_exc();save()
 if w.get('GPU_submitted')and not actual:
  for _ in range(12):
   row=common.account('910379',label,env)
   if row:actual=row[0];w['actual_handle']=actual;break
   time.sleep(5)
 if w.get('GPU_submitted')and actual:
  row=common.account('910379',label,env)
  if row and row[0]==actual and row[2].split()[0].rstrip('+')in common.ACTIVE:subprocess.run(['scancel','--signal=TERM',actual],env=env,text=True,capture_output=True,timeout=30);w['terminal']=common.terminal('910379',label,actual,env,seconds=150);save()
  try:w['error_post_KFD']=idle('errorpost')
  except BaseException as ce:w['post_error']=repr(ce)
  save()
 raise
