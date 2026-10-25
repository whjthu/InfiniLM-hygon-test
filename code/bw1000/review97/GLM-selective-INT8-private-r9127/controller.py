import pathlib,json,subprocess,time,os,hashlib,traceback
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');b=r/'candidates/GLM-selective-INT8-r9127';inc=r/'incoming/GLM-selective-INT8-r9127';out=r/'results/GLM-I8-Q1-controller-r9127.json';assert not out.exists();q={'completed':False,'controller_PID':os.getpid(),'GPU_submitted':False,'scope':'1DCU n12 operator only'};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def write():out.write_text(json.dumps(q,indent=2))
def idle(label):
 a=['srun','--jobid=910376','--overlap','--exact','-N1','-n1','-c1','-w','a02r4n12','--gres=none','--mem=1000M','-t','00:02:00','--job-name=GLM-I8-Q1-'+label+'-r9127','python3',str(r/'incoming/verify_gpu_idle_r5738.py')];z=subprocess.run(a,capture_output=True,text=True,timeout=100);assert z.returncode==0,(z.stdout,z.stderr);rows=[json.loads(s)for s in z.stdout.splitlines()if s.strip()];assert len(rows)==1 and rows[0]['node']=='a02r4n12';return rows[0]
try:
 cp=b/'CPU-closure-b.json';c=json.loads(cp.read_text());assert c['completed']and c['GPU_initialized']==False;assert sha(b/'libGLM_i8_r9127.so')==c['library_SHA256'];q['CPU_closure_SHA256']=sha(cp);q['CPU_producer']=c['producer'];q['source_SHA256']={n:sha(inc/n)for n in ['GPU_run.py','GPU_inside.sh','conversion.py']};assert b'\r'not in(inc/'GPU_inside.sh').read_bytes();subprocess.run(['bash','-n',str(inc/'GPU_inside.sh')],check=True)
 job=subprocess.check_output(['scontrol','show','job','910376','-o'],text=True);assert 'JobState=RUNNING'in job;q['fresh_sacct']=subprocess.check_output(['sacct','-S','2026-10-26','-j','910376','-n','-P','-o','JobID,JobName%100,State,NodeList%100'],text=True)
 for line in q['fresh_sacct'].splitlines():
  a=line.split('|')
  if '.'in a[0]and a[0].split('.')[-1].isdigit()and a[2]in ['RUNNING','PENDING','CONFIGURING','COMPLETING']:
   hosts=subprocess.check_output(['scontrol','show','hostnames',a[3]],text=True).splitlines();assert 'a02r4n12'not in hosts,a
 q['before']=idle('before');assert q['before']['kfd_pids']==[];write()
 args=['srun','--jobid=910376','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c8','-w','a02r4n12','--gres=dcu:1','--mem=32768M','-t','00:10:00','--job-name=GLM-I8-Q1-GPU-r9127','--output='+str(r/'logs/GLM-I8-Q1-GPU-r9127-%J.log'),'env','WAVE_INSIDE_R2323=GLM-selective-INT8-r9127/GPU_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];q['args']=args;q['GPU_submitted']=True;write()
 z=subprocess.run(args,capture_output=True,text=True,timeout=630);q['GPU_returncode']=z.returncode;q['controller_stdout']=z.stdout;q['controller_stderr']=z.stderr;q['after']=idle('post');q['post_empty']=q['after']['kfd_pids']==[];q['completed']=z.returncode==0 and q['post_empty'];q['producer_files']={str(p):{'SHA256':sha(p),'data':json.loads(p.read_text())}for p in(r/'results').glob('GLM-I8-Q1-r9127-910376.*.json')}
except BaseException as e:q['error']=repr(e);q['traceback']=traceback.format_exc()
write();print(json.dumps({'completed':q['completed'],'GPU_submitted':q['GPU_submitted'],'error':q.get('error')}),flush=True)
