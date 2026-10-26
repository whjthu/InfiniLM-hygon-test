import pathlib,json,os,subprocess,time,datetime,secrets,hashlib,socket,sys,traceback
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');S=R/'incoming/CCL-channel-abba-r9217';P='910373';N='a02r3n17,a02r3n18';LABEL='ccl-channel-r9217';H=R/'results/CCL-channel-control-r9217.json';assert not H.exists()
env={k:v for k,v in os.environ.items()if not(k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['SLURM_CONF','SLURM_CONF_SERVER'])and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
state={'CPU_actual':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'PID':os.getpid(),'host':socket.gethostname(),'GPU_submitted':False,'completed':False,'manifest_SHA256':sys.argv[1]};active={'RUNNING','PENDING','COMPLETING','CONFIGURING'}
def save():
 state['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();p=H.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(H)
def account(name):
 rows=[s.split('|')for s in subprocess.check_output(['sacct','-S','2026-10-26','-j',P,'-n','-P','-o','JobIDRaw,JobName%90,State%35,ExitCode,NodeList%80,Elapsed'],env=env,text=True,timeout=25).splitlines()if len(s.split('|'))>5 and s.split('|')[1]==name];assert len(rows)<=1;return rows[0]if rows else None
def final(name,handle):
 for i in range(50):
  v=account(name)
  if v and v[0]==handle and v[2].split()[0].rstrip('+')not in active:return v
  time.sleep(1)
 raise RuntimeError('accounting terminal timeout '+handle)
def run_cpu(label,mode):
 a=['srun','--jobid='+P,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c'+('32'if mode=='preflight'else'1'),'-w',N,'--gres=none','--mem=2048M','-t','00:02:00','--job-name='+label]
 if mode=='preflight':a+=['--export=ALL,WAVE_INSIDE_R2323=CCL-channel-abba-r9217/CPU_inside.sh','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 else:a+=['python3',str(R/'incoming/Kimi_Q0_idle_r8066.py')]
 z=subprocess.run(a,env=env,text=True,capture_output=True,timeout=145);rows=[json.loads(x)for x in z.stdout.splitlines()if x.startswith('{')];v={'returncode':z.returncode,'rows':rows,'stdout_tail':z.stdout[-7000:],'stderr_tail':z.stderr[-3500:]};state[label]=v;save();assert z.returncode==0 and len(rows)==2
 if mode!='preflight':assert all(x['passed']and not x['kfd_pids']for x in rows);handle=rows[0]['step']
 else:assert all(x['passed']and x['noCDLL_executed']for x in rows);handle=rows[0]['actual_step']
 v['terminal']=final(label,handle);assert v['terminal'][2:4]==['COMPLETED','0:0'];save()
save();gpu=None
try:
 m=S/'GPU-manifest-r9217.json';assert hashlib.sha256(m.read_bytes()).hexdigest()==sys.argv[1]
 for name,h in json.loads(m.read_bytes())['source_SHA256'].items():assert hashlib.sha256((S/name).read_bytes()).hexdigest()==h,name
 j=subprocess.check_output(['scontrol','show','job',P,'-o'],env=env,text=True,timeout=25);assert 'JobState=RUNNING'in j and 'UserId=aclsylqidf('in j;spec=next(x.split('=',1)[1]for x in j.split()if x.startswith('NodeList='));assert set(N.split(','))<=set(subprocess.check_output(['scontrol','show','hostnames',spec],env=env,text=True,timeout=25).splitlines());assert account(LABEL)is None,'never duplicate this label'
 for row in subprocess.check_output(['sacct','-j',P,'-n','-P','-o','JobIDRaw,State%30,NodeList%80'],env=env,text=True,timeout=25).splitlines():
  v=row.split('|')
  if len(v)>2 and'.'in v[0]and v[0].split('.')[-1].isdigit()and v[1].split()[0].rstrip('+')in active:assert not(set(N.split(','))&set(subprocess.check_output(['scontrol','show','hostnames',v[2]],env=env,text=True,timeout=25).splitlines())),('busy',v)
 run_cpu('ccl-channel-preflight-r9217','preflight');run_cpu('ccl-channel-before-r9217','idle');state['watch_ready_before_GPU']=True;save()
 state['arms']={}
 for arm in ['auto_a','c1','c4','auto_b']:
  label=LABEL+'-'+arm;assert account(label)is None
  gpuenv=dict(env,CCL_AR_ARM=arm,CCL_AR_CONTROL_PORT=str(39177+['auto_a','c1','c4','auto_b'].index(arm)),CCL_AR_NONCE=secrets.token_hex(32),GLOBAL_MODEL_MODE_R2354='shca')
  args=['srun','--jobid='+P,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c32','-w',N,'--gres=dcu:8','--mem=16384M','-t','00:03:00','--job-name='+label,'--output='+str(R/'logs'/('CCL-channel-r9217-'+arm+'-%J-%t.log')),'--export=ALL,WAVE_INSIDE_R2323=CCL-channel-abba-r9217/GPU_inside.sh','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
  f=(R/'logs'/('CCL-channel-r9217-'+arm+'-client.log')).open('x');gpu=subprocess.Popen(args,env=gpuenv,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL);del gpuenv;v={'GPU_client_PID':gpu.pid,'arm':arm};state['arms'][arm]=v;state.update(GPU_submitted=True,current_arm=arm);save();start=time.monotonic()
  while gpu.poll()is None:
   row=account(label)
   if row:v['actual']=row;v['handle']=row[0]
   assert time.monotonic()-start<210;save();time.sleep(3)
  f.close();row=account(label);assert row;v['handle']=row[0];v['terminal']=final(label,row[0]);v['client_exit']=gpu.returncode;v['results']=[]
  for node in [0,1]:
   p=R/'results'/('CCL-raw-AR-r9217-'+arm+'-node%d.json'%node)
   if p.exists():
    q=json.loads(p.read_bytes());v['results'].append({'node':node,'step':q['actual_step'],'normal':q['normal_owners'],'SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),'path':str(p)})
  run_cpu('ccl-channel-post-r9217-'+arm,'idle');v['qualified']=v['terminal'][2:4]==['COMPLETED','0:0']and len(v['results'])==2 and all(x['normal']==8 for x in v['results']);save();assert v['qualified'],('failedarm',arm)
 state['qualified']=len(state['arms'])==4 and all(v['qualified']for v in state['arms'].values());state['completed']=True;save()
except BaseException as e:
 state.update(error=repr(e),traceback=traceback.format_exc());save()
 if gpu is not None and gpu.poll()is not None:
  try:run_cpu('ccl-channel-exception-post-r9217','idle')
  except BaseException as x:state['post_error']=repr(x);save()
 raise
