import datetime,hashlib,json,pathlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/GLM-prefill-window-CPU-r8701-launch.json';assert not out.exists()
m=r/'incoming/GLM-prefill-window-CPU-r8701-manifest.json';q=json.loads(m.read_text())
for n,h in q['sources_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
allocation=subprocess.check_output(['scontrol','show','job','910406','-o'],text=True,timeout=20);assert 'a14r2n19'in subprocess.check_output(['scontrol','show','hostnames',next(v.split('=',1)[1]for v in allocation.split()if v.startswith('NodeList='))],text=True,timeout=20).splitlines()
args=['srun','--jobid=910406','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c4','-w','a14r2n19','--gres=none','--mem=4096M','-t','00:05:00','--job-name=GLM-prefill-window-CPU-r8701','--output='+str(r/'logs/GLM-prefill-window-CPU-r8701-%J.log'),'env','WAVE_INSIDE_R2323=GLM_prefill_window_CPU_r8701_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
d={'CPU_only':True,'dlopen':False,'GPU':False,'source_manifest_SHA256':hashlib.sha256(m.read_bytes()).hexdigest(),'args':args,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_replay':True}
with out.open('x')as f:json.dump(d,f,indent=2)
with(r/'logs/GLM-prefill-window-CPU-r8701-controller.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d['pid']=p.pid;out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
