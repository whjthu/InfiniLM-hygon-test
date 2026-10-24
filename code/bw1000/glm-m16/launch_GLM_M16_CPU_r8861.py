import datetime,hashlib,json,pathlib,subprocess
r=pathlib.Path.home()/'hygon-inference-refactor-20261014';out=r/'results/GLM-M16-CPU-r8861-launch.json';assert not out.exists()
m=r/'incoming/GLM-M16-CPU-manifest-r8861.json';q=json.loads(m.read_text())
for n,h in q['sources_SHA256'].items():assert hashlib.sha256((r/'incoming'/n).read_bytes()).hexdigest()==h,n
allocation=subprocess.check_output(['scontrol','show','job','910407','-o'],text=True,timeout=20);hosts=subprocess.check_output(['scontrol','show','hostnames',next(v.split('=',1)[1]for v in allocation.split()if v.startswith('NodeList='))],text=True,timeout=20).splitlines();nodes=[v for v in hosts if v.endswith('n20')];assert len(nodes)==1;node=nodes[0]
busy=[v for v in subprocess.check_output(['sacct','-j','910407','-n','-P','-o','JobID,State,NodeList%80'],text=True).splitlines() if v.split('|')[0].startswith('910407.') and v.split('|')[0].split('.')[-1].isdigit() and v.split('|')[1].split()[0]=='RUNNING' and node in v.split('|')[2]];assert not busy,('CPU lease busy',busy)
args=['srun','--jobid=910407','--overlap','--exact','--kill-on-bad-exit=1','-N1','-n1','-c4','-w',node,'--gres=none','--mem=4096M','-t','00:05:00','--job-name=GLM-M16-CPU-r8861','--output='+str(r/'logs/GLM-M16-CPU-r8861-%J.log'),'env','WAVE_INSIDE_R2323=GLM_M16_CPU_r8861_inside.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
d={'CPU_only':True,'dlopen':False,'GPU':False,'source_manifest_SHA256':hashlib.sha256(m.read_bytes()).hexdigest(),'args':args,'actual_node':node,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'no_replay':True}
with out.open('x')as f:json.dump(d,f,indent=2)
with(r/'logs/GLM-M16-CPU-r8861-controller.log').open('x')as f:p=subprocess.Popen(args,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
d['pid']=p.pid;out.write_text(json.dumps(d,indent=2));print(json.dumps(d))
