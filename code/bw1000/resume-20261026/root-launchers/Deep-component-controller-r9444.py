import pathlib,json,subprocess,os,hashlib,datetime,time,re
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');label='Deep-C128-cache-component-r152c';node='a02r4n05';parent='910374';D=R/'incoming'/label;out=R/'results/Deep-component-controller-r9444.json';assert not out.exists();env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_keeper':os.environ['SLURM_JOB_ID']+'.'+os.environ['SLURM_STEP_ID'],'state':'validating','GPU_submitted':False,'restarts':0};out.write_text(json.dumps(q,indent=2))
def save():q['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();out.write_text(json.dumps(q,indent=2))
def acct():return subprocess.check_output(['sacct','-S','2026-10-28','-j',parent,'-n','-P','-o','JobID,JobName%100,State%40,ExitCode,NodeList%120'],env=env,text=True,timeout=30).splitlines()
def idle(suffix):
 a=['srun','--jobid='+parent,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','1','--job-name='+label+'-'+suffix,'python3',str(R/'incoming/verify_gpu_idle_r5738.py')];z=subprocess.run(a,env=env,capture_output=True,text=True,timeout=70);assert z.returncode==0;rows=[json.loads(s)for s in z.stdout.splitlines()if s.startswith('{')];assert len(rows)==1 and rows[0]['node']==node and not rows[0]['kfd_pids'];return rows
try:
 cp=R/'candidates/Deep-C128-cache-component-r152/CPU-build.json';assert sha(cp)=='3cc5496a4c6e4a49de3bdad10e31b51f0ee3bb28d2ca634d2faa8e57e3dffa3a';c=json.loads(cp.read_bytes());assert c['completed']and c['actual_step']=='910407.492'and sha(pathlib.Path(str(c['library']).replace('/refactor',str(R),1)))==c['library_SHA256'];assert sha(D/'manifest-r152c.json')=='eff322ab5f24e173d10be1cca19c284ca35bd6a8ed6ff44b98abf10a1c1fac29'
 m=json.loads((D/'manifest-r152c.json').read_bytes())
 for n,h in m['source_SHA256'].items():assert sha(D/n)==h
 rows=acct();assert not any('|'+label+'|'in s for s in rows)
 for row in rows:
  a=row.split('|')
  if a[0].startswith(parent+'.')and a[0].split('.')[-1].isdigit()and a[2].split()[0].rstrip('+')not in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED']:
   hosts=subprocess.check_output(['scontrol','show','hostnames',a[4]],text=True,env=env,timeout=20).splitlines();assert node not in hosts
 q['before']=idle('before-r9444');save();args=['srun','--jobid='+parent,'--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N1','-n1','-c4','-w',node,'--gres=dcu:1','--mem=4096M','-t','4','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),'env','WAVE_INSIDE_R2323='+label+'/GPU_inside_r152c.sh','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];q.update(state='running_unique_GPU',GPU_submitted=True);save()
 with(R/'logs/Deep-component-r9444-client.log').open('x')as f:z=subprocess.run(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,timeout=300)
 q.update(GPU_client_returncode=z.returncode,state='GPU_client_terminal',actual_rows=[s for s in acct()if '|'+label+'|'in s]);save()
except BaseException as e:q.update(error=repr(e),state='failed_or_unknown_follow_exact_rows');save()
finally:
 if q.get('GPU_client_returncode')is not None:
  try:q['post']=idle('post-r9444');q['post_empty']=True
  except BaseException as e:q.update(post_empty=False,post_error=repr(e))
 q['state']='controller_finished';save()
