"""Upload the published CPU capsule and submit exactly one existing-parent CPU step."""
from pathlib import Path
import json,hashlib,base64
D=Path(__file__).resolve().parent;mf=D/'CPU_SOURCE_MANIFEST.json';m=json.loads(mf.read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
files={n:base64.b64encode((D/n).read_bytes()).decode()for n in [*m['files_SHA256'],'CPU_SOURCE_MANIFEST.json']}
outer=D.parent/'GLM-prefill-window-interleave-r361/runtime-r1/controller/resource_wave_host_alias_outer.sh'
s='''from pathlib import Path
import json,hashlib,base64,subprocess,os,re,datetime
R=Path.home()/'hygon-inference-refactor-20261014';D=R/'incoming/GLM-smallM-FFN-feasibility-r367';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not D.exists();D.mkdir()
for n,b in FILES.items():(D/n).write_bytes(base64.b64decode(b))
assert sha(D/'CPU_SOURCE_MANIFEST.json')==PIN
m=json.loads((D/'CPU_SOURCE_MANIFEST.json').read_bytes())
for n,h in m['files_SHA256'].items():assert sha(D/n)==h
outer=R/'incoming/GLM-prefill-window-interleave-r361/runtime-r1/controller/resource_wave_host_alias_outer.sh';assert sha(outer)==OUTER_SHA
label='GLM-r367-M1-gate-CPU'
raw=subprocess.check_output(['sacct','-S','2026-11-01','-j','910407','-n','-P','-o','JobIDRaw,JobName%160,State,ExitCode,NodeList%100'],text=True,timeout=25)
assert not any(line.split('|')[1]==label for line in raw.splitlines()if line.strip()),'Existing label must be followed, never replayed'
parent=subprocess.check_output(['scontrol','show','job','910407','-o'],text=True,timeout=25);assert'UserId=aclsylqidf('in parent and'JobState=RUNNING'in parent
nodes=re.search(r'(?:^| )NodeList=([^ ]+)',parent);assert nodes
assert'a14r2n20'in subprocess.check_output(['scontrol','show','hostnames',nodes.group(1)],text=True,timeout=25).splitlines()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
env.update(WAVE_INSIDE_R2323='GLM-smallM-FFN-feasibility-r367/CPU_inside.sh',GLM367_CPU_SOURCE_SHA256=PIN)
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:15:00','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),'--export=ALL','bash',str(outer),'probe']
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit='de5ca6b666c9aa01abbb4b84a638f0c6fedc6a8d',CPU_SOURCE_MANIFEST_SHA256=PIN,wrapper_SHA256=OUTER_SHA,GPU=False,Native_dlopen=False,label=label,argv=args,original_GPU_submissions=0,parent=910407,node='a14r2n20',CPU=4,memory_MiB=8192,bound_seconds=900)
intent=R/'results/GLM-r367-CPU-intent.json'
with intent.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs'/(label+'-submit.log')).open('x')as log:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
q.update(owned_srun_PID=p.pid,owned_original_PGID=p.pid,srun_initial_exit=p.poll())
with(R/'results/GLM-r367-CPU-submitted.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps(q))
'''
for key,value in [('FILES',files),('PIN',sha(mf)),('OUTER_SHA',sha(outer))]:s=s.replace(key,repr(value))
compile(s,'remote367-CPU-submit.py','exec');p=D.parents[3]/'remote367-CPU-submit.py';p.write_bytes(s.encode());print(p)
