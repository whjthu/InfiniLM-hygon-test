"""Root launches one bounded CPU keeper; all GPU execution belongs to it."""
from pathlib import Path
import json,hashlib,subprocess,os,datetime
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=Path(__file__).resolve().parent;label='Kimi-C1-gate-alias-GPU-r318'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());mh=sha(D/'SOURCE_MANIFEST.json')
assert mh==os.environ['KIMI_R318_MANIFEST_SHA'] and all(sha(D/n)==h for n,h in m['files_SHA256'].items())
cp=R/'candidates/Kimi-C1-gate-alias-CPU-r315/CPU-build.json';assert sha(cp)==m['CPU_receipt_SHA256']
c=json.loads(cp.read_bytes());assert c['completed'] and c['passed'] and c['CPU_only'] and not c['Native_dlopen']
for key in ['native','Main']:assert sha(str(c[key]).replace('/refactor/',str(R)+'/',1))==c[key+'_SHA256']
record=R/'control'/(label+'-keeper-launch.json');assert not record.exists() and not(R/'results'/(label+'-watch.json')).exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
for parent in ['910407','910380']:
 z=subprocess.check_output(['scontrol','show','job',parent,'-o'],env=env,text=True,timeout=25);assert 'UserId=aclsylqidf('in z and 'JobState=RUNNING'in z
rows=[x.split('|')for x in subprocess.check_output(['sacct','-S','2026-10-31','-j','910407,910380','-n','-P','-o','JobID,JobName%120,State,ExitCode'],env=env,text=True,timeout=25).splitlines()]
assert not any(len(x)>1 and x[1]in[label,label+'-keeper']for x in rows),'Existing label must be followed, never replayed'
with record.open('x')as f:json.dump({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_manifest_SHA256':mh,'GPU_s':600,'CPU_s':1200,'one_attempt':True,'publication_commit':os.environ['KIMI_R318_PUBLICATION_COMMIT']},f,indent=2)
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=2048M','-t','00:20:00','--job-name='+label+'-keeper','--output='+str(R/'logs'/(label+'-keeper-%J.log')),'env','KIMI_R318_MANIFEST_SHA='+mh,'python3','-B',str(D/'controller.py')]
with(R/'logs'/(label+'-keeper-client.log')).open('x')as log:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'keeper_submitted':True,'srun_client_PID':p.pid,'source_manifest_SHA256':mh,'GPU_only_after_new_idle_and_source_checks':True,'one_attempt':True}))
