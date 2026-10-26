import common,os,pathlib,json,subprocess,socket,time,datetime,hashlib,traceback
r=common.ROOT;D=common.SRC;env,_=common.clean_env();label='Kimi-MoE-event-r9220e';hp=r/'results'/('Kimi-MoE-event-r9220e-control.json');assert not hp.exists();m=json.loads((D/'GPU-manifest-final-source.json').read_bytes());g={'parent':'910373','nodes':['a02r3n17','a02r3n18']};sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();w={'CPU_actual_step':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'host':socket.gethostname(),'PID':os.getpid(),'GPU_submitted':False,'completed':False,'ready':False,'label':label,'source_manifest_SHA256':sha(D/'GPU-manifest-final-source.json')};actual=None;process=None
def save():w['heartbeatUTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();common.atomic(hp,w)
def idle(suffix):
 name=label+'-'+suffix;args=['srun','--jobid=910373','--overlap','--exact','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c1','--gres=none','--mem=512M','-w',','.join(g['nodes']),'-t','00:02:00','--job-name='+name,'python3',str(r/'incoming/Kimi_Q0_idle_r8066.py')];z=subprocess.run(args,env=env,text=True,capture_output=True,timeout=160);rows=[json.loads(line)for line in z.stdout.splitlines()if line.startswith('{')];assert z.returncode==0 and len(rows)==2 and {x['node']for x in rows}==set(g['nodes'])and all(x['passed']and not x['kfd_pids']for x in rows),(z.returncode,rows,z.stderr);steps={x['step']for x in rows};assert len(steps)==1;terminal=common.terminal('910373',name,steps.pop(),env);assert terminal[2:4]==['COMPLETED','0:0'];return {'rows':rows,'terminal':terminal}
save()
try:
 common.verify_parent(g,env)
 for p,h in m['source_SHA256'].items():assert sha(p.replace('/refactor',str(r),1))==h,p
 name=label+'-CPUpreflight';a=['srun','--jobid=910373','--overlap','--exact','--cpu-bind=none','-N2','-n2','--ntasks-per-node=1','-c32','--gres=none','--mem=4096M','-w',','.join(g['nodes']),'-t','00:04:00','--job-name='+name,'--export=ALL,WAVE_INSIDE_R2323=Kimi-MoE-event-profile-r9220e/preflight_actual_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=260);packets=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];w['CPUpreflight']={'returncode':z.returncode,'packets':packets,'stderr':z.stderr};save();assert z.returncode==0 and len(packets)==2 and all(q['passed']and not q['GPU']and not q['dlopen']for q in packets),(z.returncode,z.stderr,packets);steps={q['actual_step']for q in packets};assert len(steps)==1;w['CPUpreflight']['terminal']=common.terminal('910373',name,steps.pop(),env);assert w['CPUpreflight']['terminal'][2:4]==['COMPLETED','0:0'];assert packets[0]['CUDAEvent_provider_SHA256']==packets[1]['CUDAEvent_provider_SHA256'];w['fresh_before']=idle('before');m['master_addr']=next(q['ib0_ipv4'][0]for q in w['fresh_before']['rows']if q['node']=='a02r3n17');m['pinned_libraries'].update(packets[0]['CUDAEvent_provider_SHA256']);common.atomic(D/'GPU-manifest.json',m);w['runtime_manifest_SHA256']=sha(D/'GPU-manifest.json');w['ready']=True;w['watch_ready_before_GPU']=True;save()
 a=['srun','--jobid=910373','--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c32','--gres=dcu:8','--mem=65536M','-w',','.join(g['nodes']),'-t','00:20:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J-%t.log')),'--export=ALL,WAVE_INSIDE_R2323=Kimi-MoE-event-profile-r9220e/GPU_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];fo=(r/'logs'/(label+'-client.log')).open('x');process=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=fo,stderr=subprocess.STDOUT);w.update(GPU_submitted=True,client_PID=process.pid,args=a);save();deadline=time.monotonic()+1280
 while True:
  row=common.account('910373',label,env)
  if row:
   if actual is None:actual=row[0]
   assert actual==row[0];w['actual_handle']=actual;w['actual_row']=row
   if row[2].split()[0].rstrip('+')not in common.ACTIVE:w['terminal']=row;save();break
  assert time.monotonic()<deadline,'componentboundedwatchdeadline';save();time.sleep(10)
 w['post_KFD']=idle('post');w['results']=[]
 for node in range(2):
  p=r/'results'/(label+'-'+actual+'-node%d.json'%node);raw=p.read_bytes();q=json.loads(raw);w['results'].append({'node':node,'path':str(p),'SHA256':hashlib.sha256(raw).hexdigest(),'completed':q.get('completed'),'normal_cleanup':q.get('normal_CUDA_cleanup'),'all8owners_closed':q.get('native',{}).get('all8normal_before_communicators'),'error':q.get('error')});assert q['completed']and q['normal_CUDA_cleanup']and q['native']['all8normal_before_communicators'];log=(r/'logs'/(label+'-'+actual+'-'+str(node)+'.log')).read_text(errors='replace');assert log.count('KIMI_MOE_EVENT_R9220C_NORMAL')==1 and'Using network IBext_v8'in log and'Using network Socket'not in log
 w['passed']=w['terminal'][2:4]==['COMPLETED','0:0'];w['audit']={'normal_owners':16,'GPU':True,'component_only':True,'not_Model_or_currentrequest_bottleneck':True,'post_KFD_empty':True,'network':'IBext_v8/noSocketData','GDR':'UNKNOWN'};w['completed']=True;save()
except BaseException as e:
 w['error']=repr(e);w['traceback']=traceback.format_exc();save()
 if w.get('GPU_submitted')and actual:
  row=common.account('910373',label,env)
  if row and row[0]==actual and row[2].split()[0].rstrip('+')in common.ACTIVE:subprocess.run(['scancel','--signal=TERM',actual],env=env,text=True,capture_output=True,timeout=30);w['terminal']=common.terminal('910373',label,actual,env,seconds=150);save()
  try:w['error_post_KFD']=idle('errorpost')
  except BaseException as ce:w['post_error']=repr(ce)
  save()
 raise
