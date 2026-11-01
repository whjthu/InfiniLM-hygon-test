from pathlib import Path
import json,hashlib,subprocess,os,datetime
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-KDA-fused-Delta-r343';nodes=['a02r3n03','a02r3n07'];sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
commit=os.environ['SOURCE_COMMIT'];assert len(commit)==40
m=json.loads((D/'GPU_MANIFEST.json').read_bytes());mh=sha(D/'GPU_MANIFEST.json');assert mh==os.environ['GPU_MANIFEST_SHA'];assert all(sha(D/n)==h for n,h in m['files_SHA256'].items());assert not(D/'GPU-intent.json').exists()
cp=R/'candidates/Kimi-KDA-fused-Delta-r343/CPU-build.json';assert sha(cp)==m['CPU_receipt_SHA256'];c=json.loads(cp.read_bytes());assert c['completed']and c['passed']
for key in ['native','Main']:assert sha(Path(c[key].replace('/refactor/',str(R)+'/',1)))==c[key+'_SHA256']
allocation=subprocess.check_output(['scontrol','show','job','910371','-o'],text=True);assert 'JobState=RUNNING'in allocation
idle=[]
for node in nodes:
 for line in subprocess.check_output(['squeue','--steps','-j','910371','-h','-o','%i|%j|%N'],text=True).splitlines():
  f=line.split('|')
  if f[0].strip().split('.')[-1].isdigit():assert node not in subprocess.check_output(['scontrol','show','hostnames',f[-1]],text=True).splitlines(),line
 args=['srun','--jobid=910371','--overlap','--exact','--immediate=30','-N1','-n1','-c1','-w',node,'--gres=none','--mem=512M','-t','00:02:00','--job-name=Kimi343-idle','python3',str(D/'resource_probe.py')]
 z=subprocess.run(args,capture_output=True,text=True,timeout=140);assert z.returncode==0,(z.stdout,z.stderr);q=json.loads(z.stdout);assert q['KFD_empty'] and q['control30040_bindable'] and q['node']==node;idle.append(q)
head=idle[0]['ib0_ipv4'];nonce=os.urandom(16).hex();env='ALL,KIMI_R343_GPU_MANIFEST_SHA='+mh+',KIMI_ENGINE_HEAD='+head+',KIMI_ENGINE_NONCE='+nonce+',GLOBAL_PORT_R2354=23040,GLOBAL_MODEL_MODE_R2354=shca,GLOBAL_OFFSET_R2354=0,WAVE_INSIDE_R2323=Kimi-KDA-fused-Delta-r343/GPU_inside.sh'
args=['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=1','-N2','-n2','--ntasks-per-node=1','-c64','-w',','.join(nodes),'--gres=dcu:8','--mem=49152M','-t','00:10:00','--job-name=Kimi343-fused-Delta','--output='+str(D/'GPU-%J-%t.log'),'--export='+env,'bash',str(R/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
intent={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':commit,'GPU_manifest_SHA256':mh,'CPU_receipt_SHA256':sha(cp),'idle':idle,'args':args};(D/'GPU-intent.json').write_text(json.dumps(intent,indent=2))
with(D/'GPU-submit.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
intent['launcher_pid']=p.pid;print(json.dumps(intent))
