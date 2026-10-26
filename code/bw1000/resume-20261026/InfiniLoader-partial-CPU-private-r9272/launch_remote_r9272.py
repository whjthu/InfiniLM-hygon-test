import pathlib,json,subprocess,os,hashlib,datetime
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';d=r/'incoming/InfiniLoader-partial-CPU-r9272'
pins=json.loads((d/'producer-source-r9272.json').read_text())
for rel,sha in pins['files'].items():assert hashlib.sha256((d/rel).read_bytes()).hexdigest()==sha,rel
job=subprocess.check_output(['scontrol','show','job','910380','-o'],text=True,timeout=20)
assert 'UserId=aclsylqidf(154459)' in job and 'JobState=RUNNING' in job and 'NodeList=a14r2n[08-10,12]' in job,job
rows=subprocess.check_output(['squeue','--steps','-j','910380','-h','-o','%i|%j|%N'],text=True,timeout=20).splitlines()
assert not [x for x in rows if x.split('|')[0].split('.')[-1].isdigit()],rows
assert not (r/'results/InfiniLoader-partial-CPU-r9272-launch.json').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_VISIBLE','HIP_VISIBLE','ROCR_VISIBLE'))}
env.update(CUDA_VISIBLE_DEVICES='',HIP_VISIBLE_DEVICES='',ROCR_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
a=['srun','--jobid=910380','--overlap','--exact','--cpu-bind=none','-N1','-n1','-c8','--gres=none','--mem=49152M','-w','a14r2n08','-t','00:30:00','--job-name=loader-partial-CPU-r9272','--output='+str(r/'logs/InfiniLoader-partial-CPU-r9272-%J.log'),'timeout','--kill-after=10s','1500','/usr/local/python3.12/bin/python3','-B',str(d/'partial_probe_r9272.py'),'run',str(r/'results/loader-cpu-r9272')]
f=(r/'logs/InfiniLoader-partial-CPU-r9272-client.log').open('x');p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
out={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'client_PID':p.pid,'args':a,'prior_steps':rows,'GPU':False,'source_producer_SHA256':hashlib.sha256((d/'producer-source-r9272.json').read_bytes()).hexdigest()};(r/'results/InfiniLoader-partial-CPU-r9272-launch.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
