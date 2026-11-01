from pathlib import Path
import base64,json
D=Path(__file__).resolve().parent;L=D.parents[3];S=D/'CPU-repair-r2'
files={p.name:base64.b64encode(p.read_bytes()).decode()for p in S.iterdir()if p.is_file()}
script="""from pathlib import Path
import json,base64,hashlib,subprocess,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-two-cohort-Model-r355/CPU-repair-r2';assert not D.exists();D.mkdir()
for n,b in FILES.items():
 with(D/n).open('xb')as f:f.write(base64.b64decode(b))
m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
old=subprocess.check_output(['sacct','-S','2026-11-01','-j','910407.687','-n','-P','-o','JobIDRaw,State,ExitCode'],text=True,timeout=25);assert'910407.687|FAILED|1:0'in old
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','--ntasks-per-node=1','-c4','--gres=none','--mem=8192M','-w','a14r2n20','-t','00:10:00','--job-name=GLM-r355-whole-CPU-r2','--output='+str(R/'logs/GLM-r355-whole-CPU-r2-%J.log'),'--export=ALL','bash',str(D.parent/'controller/resource_wave_host_alias_outer.sh'),'probe']
env=dict(os.environ,WAVE_INSIDE_R2323='GLM-two-cohort-Model-r355/CPU-repair-r2/CPU_inside.sh')
with(R/'logs/GLM-r355-whole-CPU-r2-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'GPU':0,'CPU_submitted':True,'PID':p.pid,'label':'GLM-r355-whole-CPU-r2','owner_source_unchanged':True,'original_failure':'910407.687'}))
""".replace('FILES',repr(files))
(L/'remote355-upload-CPU-r2.py').write_text(script,encoding='utf8')
print(json.dumps({'helper':'remote355-upload-CPU-r2.py','GPU':0}))
