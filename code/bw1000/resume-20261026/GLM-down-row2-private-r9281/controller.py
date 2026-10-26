import pathlib,json,subprocess,os,time,datetime,hashlib,socket,sys,traceback
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming/GLM-down-row2-r9281';label='GLM-down-row2-r9281';parent='910376';node='a02r4n12';keeper=os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'];assert keeper.startswith(parent+'.');env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']};rp=r/(label+'-launch.json');wp=r/(label+'-watch.json');term={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE'};sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,q):
 q.update(keeper_step=keeper,keeper_hostname=socket.gethostname(),keeper_PID=os.getpid(),heartbeat_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(q,indent=2));tmp.replace(p)
def acct():return [x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-24','-j',parent,'-n','-P','-o','JobID,JobName%100,State%40,ExitCode,NodeList%120'],env=env,text=True,timeout=30).splitlines()]
def idle(tag):
 z=subprocess.run(['srun','--jobid='+parent,'--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:03:00','--job-name='+label+'-'+tag,'python3',str(r/'incoming/verify_gpu_idle_r5738.py')],env=env,capture_output=True,text=True,timeout=190);assert z.returncode==0,(z.stdout,z.stderr);q=[json.loads(v)for v in z.stdout.splitlines()if v.startswith('{')];assert len(q)==1 and q[0]['node']==node and q[0]['kfd_pids']==[];return q[0]
assert not rp.exists()and not wp.exists();d={'state':'intent','restarts':0,'label':label,'manifest_SHA256':sha(inc/'GPU-manifest.json')};w={'ready':False,'qualified':False,'actual_handle':None};write(rp,d);write(wp,w);proc=None
try:
 m=json.loads((inc/'GPU-manifest.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
 for n,h in m['parent_SHA256'].items():assert sha(str(n).replace('/refactor',str(r),1))==h,n
 rows=acct();assert any(x[0]==parent and x[2]=='RUNNING'for x in rows);assert not any(x[1]==label for x in rows)
 for x in rows:
  if x[0]!=keeper and x[0].startswith(parent+'.')and x[0].split('.')[-1].isdigit()and x[2].split()[0].rstrip('+')not in term:assert node not in subprocess.check_output(['scontrol','show','hostnames',x[4]],env=env,text=True,timeout=20).splitlines(),x
 d['fresh_before_KFD']=idle('before');write(rp,d);w.update(ready=True,state='watchready_beforeGPU');write(wp,w)
 a=['srun','--jobid='+parent,'--overlap','--exact','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c8','-w',node,'--gres=dcu:1','--mem=32768M','-t','00:15:00','--job-name='+label,'--output='+str(r/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323=GLM-down-row2-r9281/GPU_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 with(r/'logs'/(label+'-client.log')).open('x')as f:proc=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 d.update(state='uniqueGPUsubmitted_follow_samehandle',GPUclientPID=proc.pid,args=a);write(rp,d);start=time.monotonic();last=0;known=None
 while time.monotonic()-start<990:
  if time.monotonic()-last>10:
   rr=[x for x in acct()if x[1]==label];assert len(rr)<=1
   if rr:
    v=rr[0];assert known is None or known==v[0];known=v[0];w.update(actual_handle=known,Slurm=v,state='following_samehandle');write(wp,w)
    if v[2].split()[0].rstrip('+')in term:w['terminal']=v;break
   last=time.monotonic()
  write(wp,w);time.sleep(2)
 else:raise RuntimeError('boundedwatch no retry')
 try:
  result=r/'results'/(label+'-'+known+'.json');w['result_exists']=result.exists()
  if result.exists():
   z=subprocess.run([sys.executable,str(inc/'verify.py'),str(result)],env=env,text=True,capture_output=True,timeout=60);w.update(audit_returncode=z.returncode,audit_stdout=z.stdout,audit_stderr=z.stderr,result_SHA256=sha(result));q=json.loads(result.read_bytes());w['performance']=q.get('performance');w['performance_admission_passed']=q.get('performance_admission_passed')
  log=r/'logs'/(label+'-'+known+'.log');txt=log.read_text(errors='replace');w.update(normal_marker='GLM_DOWN_ROW2_R9281_NORMAL'in txt,log_SHA256=sha(log),log_errors=[s for s in txt.splitlines()if any(t in s for t in ['Traceback','AssertionError','RuntimeError','Segmentation fault'])][-30:])
 finally:
  try:w.update(post_KFD=idle('post'),post_passed=True)
  except BaseException as e:w.update(post_passed=False,post_error=repr(e))
 w.update(state='terminal_audited_postchecked',qualified=bool(w['terminal'][2:4]==['COMPLETED','0:0']and w.get('audit_returncode')==0 and w.get('normal_marker')and not w.get('log_errors')and w.get('post_passed')));write(wp,w);d.update(state='completed_keeper',actual_handle=known,qualified=w['qualified']);write(rp,d)
except BaseException as e:d.update(error=repr(e),traceback=traceback.format_exc(),state='unknownGPU_follow_samehandle'if proc else'failed_beforeGPU');write(rp,d);w.update(error=repr(e));write(wp,w);raise
