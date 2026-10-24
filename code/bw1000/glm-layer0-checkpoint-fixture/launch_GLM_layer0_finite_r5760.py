"""NOT RUN. Unique finite single GPU; controller always post-KFD on terminal/failure."""
import pathlib,json,hashlib,subprocess,datetime,re,sys,time,os
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';inc=r/'incoming';state=r/'results/glm-layer0-controller-r5760.json';intent=r/'glm-layer0-intent-r5760.json';mp=inc/'GLM-layer0-GPU-manifest-r5760.json'
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(q):t=state.with_suffix('.tmp');t.write_text(json.dumps(q,indent=2));t.replace(state)
def idle(label):
 a=['srun','--jobid=910376','--overlap','--exact','-N1','-n1','-c1','-w','a02r4n12','--gres=none','--mem=1000M','-t','00:01:00','--job-name='+label,'env','CUDA_VISIBLE_DEVICES=','HIP_VISIBLE_DEVICES=','ROCR_VISIBLE_DEVICES=','python3',str(inc/'verify_gpu_idle_r5738.py')];p=subprocess.run(a,text=True,capture_output=True,timeout=90);return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
if '--controller'in sys.argv:
 q={'watching':True,'numeric_started':False,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'PID':os.getpid()};save(q);code=None
 try:
  args=json.loads(intent.read_text())['args']
  with(r/'logs/glm-layer0-r5760.log').open('x')as log:
   p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT);q.update(numeric_started=True,srunPID=p.pid);save(q);code=p.wait();q['srun_returncode']=code
 except BaseException as e:q['controller_error']=repr(e)
 finally:
  try:q['postKFD']=idle('glm-l0-post-r5760')
  except BaseException as e:q['postKFD_error']=repr(e)
  try:
   a=subprocess.check_output(['sacct','-S','2026-10-24','-j','910376','-n','-P','-o','JobID,JobName%60,State,ExitCode,Elapsed'],text=True);matches=[v for v in a.splitlines()if'|glm-l0-r5760|'in v];assert len(matches)==1,matches;q['actual_sacct']=matches[0];fields=matches[0].split('|');assert fields[2]=='COMPLETED'and fields[3]=='0:0',matches[0]
   pth=r/'results/glm-layer0-finite-r5760.json'
   if pth.exists():rep=json.loads(pth.read_text());q['report_SHA256']=sha(pth);q['component_completed']=rep.get('completed',False);q['normal_close']=rep.get('normal_close',False)
  except BaseException as e:q['audit_error']=repr(e)
  q['terminal_observed']=True;save(q)
 sys.exit(0 if code==0 and q.get('component_completed')and q.get('postKFD',{}).get('returncode')==0 and 'audit_error'not in q else 1)
else:
 m=json.loads(mp.read_text())
 for n,h in m['sources_SHA256'].items():assert sha(inc/n)==h,n
 for p,h in m['DSO_SHA256'].items():assert sha(pathlib.Path(p.replace('/refactor',str(r),1)))==h,p
 job=subprocess.check_output(['scontrol','show','job','910376','-o'],text=True);assert 'JobState=RUNNING'in job
 nodes=re.search(r'\bNodeList=(\S+)',job).group(1);assert 'a02r4n12'in subprocess.check_output(['scontrol','show','hostnames',nodes],text=True).split()
 live=subprocess.check_output(['scontrol','show','step','910376','-o'],text=True)
 for line in live.splitlines():
  if re.search(r'\bStepId=910376\.\d+\b',line):
   ns=re.search(r'\b(?:Nodes|NodeList)=(\S+)',line);assert ns,line
   assert 'a02r4n12'not in subprocess.check_output(['scontrol','show','hostnames',ns.group(1)],text=True).split(),line
 assert not state.exists();fresh=idle('glm-l0-pre-r5760');assert fresh['returncode']==0,fresh
 args=['srun','--jobid=910376','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c8','-w','a02r4n12','--gres=dcu:1','--mem=16000M','-t','00:10:00','--job-name=glm-l0-r5760','env','WAVE_INSIDE_R2323=GLM_layer0_finite_r5760_inside.sh','bash',str(inc/'resource_wave_local_r3097_outer.sh'),'probe']
 with intent.open('x')as f:json.dump({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'args':args,'manifest_SHA256':sha(mp),'fresh':fresh,'fresh_parent':job,'live':live},f)
 with(r/'logs/glm-layer0-controller-r5760.log').open('x')as log:p=subprocess.Popen([sys.executable,str(inc/pathlib.Path(__file__).name),'--controller'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps({'submitted_controller':True,'PID':p.pid,'numeric_scope':'OneGPU8CPU16GB layer0 fixture,5min inner/10minhard,notModel','intent':intent.name}))
