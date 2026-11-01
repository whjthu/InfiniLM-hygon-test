import pathlib,json,subprocess,hashlib,os,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-operator-search-r339';nodes=['a02r3n03','a02r3n07'];commit=os.environ['SOURCE_COMMIT']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(sha(D/n)==v for n,v in m['files_SHA256'].items());assert sha(D/'search.so')=='8feb14507414b3f2b3afbecd413a700a7b258d795405bbef556faf65a561f502'
assert not(D/'GPU-intent.json').exists()
allocation=subprocess.check_output(['scontrol','show','job','910371','-o'],text=True);assert 'JobState=RUNNING'in allocation
allocated=subprocess.check_output(['scontrol','show','hostnames',next(a.split('=',1)[1]for a in allocation.split()if a.startswith('NodeList='))],text=True).splitlines();assert set(nodes)<=set(allocated)
records=[]
for node in nodes:
 active=subprocess.check_output(['squeue','--steps','-j','910371','-h','-o','%i|%j|%N'],text=True)
 for line in active.splitlines():
  fields=line.split('|')
  if not fields[0].strip().split('.')[-1].isdigit():continue
  hosts=subprocess.check_output(['scontrol','show','hostnames',fields[-1]],text=True).splitlines();assert node not in hosts,('busy',line)
 z=subprocess.run(['srun','--jobid=910371','--overlap','--exact','--immediate=30','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name=Kimi339-idle','python3',str(R/'incoming/verify_gpu_idle_r5819.py')],text=True,capture_output=True,timeout=140);assert z.returncode==0,(z.stdout,z.stderr);record=json.loads(z.stdout);assert record['node']==node and record['kfd_pids']==[];records.append(record)
claims=[]
for idx,node in enumerate(nodes):
 args=['srun','--jobid=910371','--overlap','--exact','--immediate=30','--kill-on-bad-exit=1','--cpu-bind=none','-N1','-n8','--ntasks-per-node=8','-c4','-w',node,'--gres=dcu:8','--mem=64000M','-t','00:12:00','--job-name=Kimi339-search','--output='+str(D/('GPU-node'+str(idx)+'-%J-%t.log')),'env','SOURCE_COMMIT='+commit,'SEED_OFFSET='+str(idx),'WAVE_INSIDE_R2323=Kimi-operator-search-r339/GPU_inside.sh','bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
 claims.append({'node':node,'args':args})
intent={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'idle':records,'manifest_SHA256':sha(D/'SOURCE_MANIFEST.json'),'claims':claims}
(D/'GPU-intent.json').write_text(json.dumps(intent,indent=2))
for idx,item in enumerate(claims):
 with(D/('GPU-submit'+str(idx)+'.log')).open('x')as f:p=subprocess.Popen(item['args'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 item['pid']=p.pid
print(json.dumps(intent))
