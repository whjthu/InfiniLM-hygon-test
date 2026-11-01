from pathlib import Path
import json,subprocess,hashlib,datetime
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-long-draft-memory-floor-r362b';run='draft-pressure-20261102-r362-a'
watch=R/'results'/('Kimi-r362-watch-'+run+'.json');w=json.loads(watch.read_bytes());assert w['completed']and w['actual_GPU']=='910371.662'
files=[]
for name in ['Kimi-r362-watch-'+run+'.json','Kimi-r362-pressure-'+run+'.json','Kimi-r362-worker-'+run+'.json']:
 p=R/'results'/name;raw=p.read_bytes();files.append(dict(path=str(p),SHA256=hashlib.sha256(raw).hexdigest(),report=json.loads(raw)))
pid=files[1]['report'].get('owned_worker_PID')
code="""import runpy,io,contextlib,json,sys
from pathlib import Path
s=io.StringIO()
with contextlib.redirect_stdout(s):runpy.run_path(sys.argv[1],run_name='__main__')
q=json.loads(s.getvalue());q['expected_worker_PID']=int(sys.argv[2]);q['expected_worker_PID_still_exists']=Path('/proc',sys.argv[2]).exists();q['passed']=q['passed']and not q['expected_worker_PID_still_exists'];print(json.dumps(q))
"""
args=['srun','--jobid=910371','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=256M','-w','a02r3n15','-t','00:01:00','--job-name=Kimi-r362b-independent-post','python3','-B','-c',code,str(D/'idle.py'),str(pid)]
p=subprocess.run(args,text=True,capture_output=True,timeout=100);assert p.returncode==0,p.stderr
post=[json.loads(x)for x in p.stdout.splitlines()if x.startswith('{')];assert len(post)==1
account=subprocess.check_output(['sacct','-S','2026-11-01','-j','910371.662,910407.708','-n','-P','-o','JobIDRaw,State,ExitCode,Elapsed,NodeList%100'],text=True,timeout=25)
print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),run=run,files=files,independent_post=post[0],accounting=account,GPU_new=0)))
