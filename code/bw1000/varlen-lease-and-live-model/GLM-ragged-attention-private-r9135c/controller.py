import pathlib,json,subprocess,time,os,hashlib,traceback,sys
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');b=r/'candidates/GLM-ragged-attention-r9135c';inc=r/'incoming/GLM-ragged-attention-r9135c';out=r/'results/GLM-ragged-controller-r9135c.json';assert not out.exists();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert len(sys.argv)==2;manifest=inc/'GPU-manifest.json';assert sha(manifest)==sys.argv[1];m=json.loads(manifest.read_text());env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_'))and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']};q={'completed':False,'controller_PID':os.getpid(),'GPU_submitted':False,'watch_before_GPU':True,'manifest_SHA256':sha(manifest),'scope':'singleDCU n12 completeattentionhelper only, noModel/wholeLayer'}
def write():out.write_text(json.dumps(q,indent=2))
def cmd(a,**kw):return subprocess.run(a,env=env,**kw)
def acct():return subprocess.check_output(['sacct','-S','2026-10-25','-j','910376','-n','-P','-o','JobID,JobName%100,State,ExitCode,Elapsed,NodeList%100'],env=env,text=True)
def idle(label):
 a=['srun','--jobid=910376','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c1','-w','a02r4n12','--gres=none','--mem=1000M','-t','00:02:00','--job-name=GLM-ragged-'+label+'-r9135c','python3',str(r/'incoming/verify_gpu_idle_r5738.py')];z=cmd(a,capture_output=True,text=True,timeout=100);assert z.returncode==0,(z.stdout,z.stderr);rows=[json.loads(s)for s in z.stdout.splitlines()if s.strip()];assert len(rows)==1 and rows[0]['node']=='a02r4n12';return rows[0]
try:
 for name,h in m['source_SHA256'].items():assert sha(inc/name)==h,name
 cp=b/'CPU-build.json';c=json.loads(cp.read_text());assert c['completed']and not c['GPU_initialized'];assert sha(cp)==m['CPU_closure_SHA256']and c['library_SHA256']==m['library_SHA256']==sha(c['library']);assert all(sha(p)==h for p,h in c['dependencies'].items());assert b'\r'not in(inc/'GPU_inside.sh').read_bytes();cmd(['bash','-n',str(inc/'GPU_inside.sh')],check=True);q['CPU_producer']=c['producer'];q['fresh_sacct']=acct()
 for s in q['fresh_sacct'].splitlines():
  a=s.split('|')
  if a[0].startswith('910376.')and a[0].split('.')[-1].isdigit()and a[2]in ['RUNNING','PENDING','CONFIGURING','COMPLETING']:
   hosts=subprocess.check_output(['scontrol','show','hostnames',a[5]],env=env,text=True).splitlines();assert 'a02r4n12'not in hosts,a
 assert 'JobState=RUNNING'in subprocess.check_output(['scontrol','show','job','910376','-o'],env=env,text=True);q['before']=idle('before');assert not q['before']['kfd_pids'];write();args=['srun','--jobid=910376','--overlap','--exact','--kill-on-bad-exit=1','--cpu-bind=none','-N1','-n1','-c8','-w','a02r4n12','--gres=dcu:1','--mem=32768M','-t','00:06:00','--job-name=GLM-ragged-GPU-r9135c','--output='+str(r/'logs/GLM-ragged-GPU-r9135c-%J.log'),'env','WAVE_INSIDE_R2323=GLM-ragged-attention-r9135c/GPU_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe'];q.update(args=args,GPU_submitted=True);write();z=cmd(args,capture_output=True,text=True,timeout=390);q.update(GPU_returncode=z.returncode,controller_stdout=z.stdout,controller_stderr=z.stderr)
except BaseException as exc:q['error']=repr(exc);q['traceback']=traceback.format_exc()
finally:
 if q['GPU_submitted']:
  try:q['after']=idle('post');q['post_empty']=not q['after']['kfd_pids']
  except BaseException as exc:q['post_error']=repr(exc)
 q['Slurm']=[s for s in acct().splitlines()if'GLM-ragged-'in s];models=[s.split('|')for s in q['Slurm']if s.split('|')[1]=='GLM-ragged-GPU-r9135c'];q['actual_handle']=models[0][0]if len(models)==1 else None;q['producer_files']={str(p):{'SHA256':sha(p),'data':json.loads(p.read_text())}for p in(r/'results').glob('GLM-ragged-attention-r9135c-910376.*.json')};q['completed']=q.get('GPU_returncode')==0 and q.get('post_empty')is True and len(models)==1 and models[0][2:4]==['COMPLETED','0:0']and len(q['producer_files'])==1 and all(v['data']['completed']and v['data']['normal_close']and v['data']['normal_Runtime_expired']and v['data']['Torch_postclose']for v in q['producer_files'].values());write()
print(json.dumps({'completed':q['completed'],'actual_handle':q['actual_handle'],'GPU_submitted':q['GPU_submitted'],'error':q.get('error')}))
