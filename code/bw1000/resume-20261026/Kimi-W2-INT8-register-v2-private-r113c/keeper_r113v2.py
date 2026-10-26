import common_r113v2 as common
import os,pathlib,json,subprocess,socket,time,datetime,hashlib,traceback
r=common.ROOT;D=common.SRC;env,_=common.clean_env();label='Kimi-W2-INT8-register-v2-r113c';hp=r/'results'/('Kimi-W2-INT8-register-v2-r113c-control.json');assert not hp.exists();m=json.loads((D/'GPU-manifest-source-r113v2b.json').read_bytes());g={'parent':'910375','nodes':['a02r4n08','a02r4n09']};sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();w={'CPU_actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'host':socket.gethostname(),'PID':os.getpid(),'GPU_submitted':False,'completed':False,'ready':False,'label':label,'source_manifest_SHA256':sha(D/'GPU-manifest-source-r113v2b.json')};actual=None;process=None
def save():w['heartbeatUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();common.atomic(hp,w)
def idle(suffix):
 name=label+'-'+suffix;args=['srun','--jobid=910375','--overlap','--exact','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(g['nodes']),'-t','00:02:00','--job-name='+name,'python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')];z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')];assert z.returncode==0 and len(rows)==2 and {x['node']for x in rows}==set(g['nodes'])and all(x['passed']and not x['kfd_pids']for x in rows),(z.returncode,rows,z.stderr);steps={x['step']for x in rows};assert len(steps)==1;terminal=common.terminal('910375',name,steps.pop(),env);assert terminal[2:4]==['COMPLETED','0:0'];return {'rows':rows,'terminal':terminal}
save()
try:
 common.verify_parent(g,env)
 go=json.loads((D/'GPU-GO-after-source-publication-r113v2b.json').read_bytes());assert go['authorized_by']=='parent-root' and go['manifest_SHA256']==w['source_manifest_SHA256'] and len(go['Git_publication_SHA'])==40
 for p,h in m['source_SHA256'].items():assert sha(p.replace('/refactor',str(r),1))==h,p
 w['CPUpreflight_reused']={'actual_handle':'910375.741','scope':'existing identicalRuntime/loader importenvironment; no newCPU gate'};w['CPU_engineering_closure']={'actual_handle':'910375.755','CPU_only':True,'GPU':False,'receipt_SHA256':'81d3dbcb6ddad2587294de16446b117c6514d69bc40e70c31330df33e8b9e1e0','I8_ISA_CPU':'910375.755 sixregisterscaleW2+oneunchangedwarmatom; noF64'};w['fresh_before']=idle('before');m['master_addr']=next(q['ib0_ipv4'][0]for q in w['fresh_before']['rows']if q['node']=='a02r4n08');common.atomic(D/'GPU-manifest-r113v2b.json',m);w['runtime_manifest_SHA256']=sha(D/'GPU-manifest-r113v2b.json');w['ready']=True;w['watch_ready_before_GPU']=True;save()
 a=['srun','--jobid=910375','--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c32','--gres=dcu:8','--mem=128000M','-w',','.join(g['nodes']),'-t','00:20:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J-%t.log')),'--export=ALL,WAVE_INSIDE_R2323=Kimi-W2-INT8-register-v2-r113c/GPU_inside_r113v2.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];fo=(r/'logs'/(label+'-client.log')).open('x');process=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=fo,stderr=subprocess.STDOUT);w.update(GPU_submitted=True,client_PID=process.pid,args=a);save();deadline=time.monotonic()+1280
 while True:
  row=common.account('910375',label,env)
  if row:
   if actual is None:actual=row[0]
   assert actual==row[0];w['actual_handle']=actual;w['actual_row']=row
   if row[2].split()[0].rstrip('+')not in common.ACTIVE:w['terminal']=row;save();break
  assert time.monotonic()<deadline,'componentboundedwatchdeadline';save();time.sleep(10)
 w['post_KFD']=idle('post');w['results']=[]
 for node in range(2):
  p=r/'results'/(label+'-'+actual+'-node%d.json'%node);raw=p.read_bytes();q=json.loads(raw);w['results'].append({'node':node,'path':str(p),'SHA256':hashlib.sha256(raw).hexdigest(),'completed':q.get('completed'),'normal_cleanup':q.get('normal_CUDA_cleanup'),'all8owners_closed':q.get('native',{}).get('all8normal_before_communicators'),'error':q.get('error')});assert q['completed']and q['normal_CUDA_cleanup']and q['native']['all8normal_before_communicators'];assert len(q['native']['reports'])==8
  for owner in q['native']['reports']:
   assert owner['normal_Runtime_close'] and owner['module_closed'] and len(owner['arms'])==24 and len(owner['profile'])==0 and len(owner['current_branch'])==3 and owner['INT8_atom_CPU_integer_oracle_all256_pass']
   for arm in owner['arms']:assert len(arm['samples'])==8 and arm['input_readonly_and_tail_canary'] and arm['profile_R9125_sync_OFF']
   for branch in owner['current_branch']:assert branch['I192_calls']==68 and branch['I192_tokens']==68*branch['T'] and branch['legacy_variant_calls']==0 and branch['W2_original_OFF_calls']==0 and branch['W2_pair_ON_calls']==34 and branch['I8_OFF_calls']==34 and branch['I8_ON_calls']==34
   assert [(b['T'],b['constructed_low_reuse']) for b in owner['current_branch']]==[(8,False),(64,False),(64,True)]
  log=(r/'logs'/(label+'-'+actual+'-'+str(node)+'.log')).read_text(errors='replace');assert log.count('KIMI_W2_INT8_R113_NORMAL')==1 and'Using network IBext_v8'in log and'Using network Socket'not in log
 w['passed']=w['terminal'][2:4]==['COMPLETED','0:0'];w['audit']={'normal_owners':16,'GPU':True,'component_only':True,'not_Model_or_currentrequest_bottleneck':True,'post_KFD_empty':True,'network':'IBext_v8/noSocketData','GDR':'UNKNOWN'};w['completed']=True;save()
except BaseException as e:
 w['error']=repr(e);w['traceback']=traceback.format_exc();save()
 if w.get('GPU_submitted')and actual:
  row=common.account('910375',label,env)
  if row and row[0]==actual and row[2].split()[0].rstrip('+')in common.ACTIVE:subprocess.run(['scancel','--signal=TERM',actual],env=env,text=True,capture_output=True,timeout=30);w['terminal']=common.terminal('910375',label,actual,env,seconds=150);save()
  try:w['error_post_KFD']=idle('errorpost')
  except BaseException as ce:w['post_error']=repr(ce)
  save()
 raise
