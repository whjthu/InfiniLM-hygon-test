from pathlib import Path
import hashlib,json,os,subprocess
R=Path.home()/'hygon-inference-refactor-20261014';D=R/'incoming/Kimi-C1-Delta-projection-fusion-r405';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mf=D/'SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI405_SOURCE_SHA256'];m=json.loads(mf.read_bytes());assert all(sha(D/n)==h for n,h in m['files_SHA256'].items())
assert not(R/'candidates/Kimi-C1-Delta-projection-fusion-r405').exists()
intent=R/'results/Kimi405-CPU-once-intent.json'
with intent.open('x')as f:json.dump(dict(source_manifest_SHA256=sha(mf),GPU=0,private_implementation_TUs=1),f);f.flush();os.fsync(f.fileno())
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']};env.update(KIMI405_SOURCE_SHA256=sha(mf),WAVE_INSIDE_R2323='Kimi-C1-Delta-projection-fusion-r405/CPU_inside.sh')
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:05:00','--job-name=Kimi405-I192-bounded-BF16-CPU','--export=ALL','bash',str(R/'incoming/Deep-C1-current-profile-r353/repair-r1/controller-v2/resource_wave_host_alias_outer.sh'),'probe']
log=R/'logs/Kimi405-I192-bounded-BF16-CPU.log'
with log.open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q=dict(actual_owned_CPU_PID=p.pid,owned_PGID=p.pid,argv=args,log=str(log),source_manifest_SHA256=sha(mf),GPU=0)
with(R/'results/Kimi405-CPU-submitted.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps(q))
