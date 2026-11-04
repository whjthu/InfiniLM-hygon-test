from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-vocab-parallel-embedding-r461/cpu-repair-v2'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mf=D/'CPU_SOURCE_MANIFEST.json';assert sha(mf)==os.environ['KIMI461_SOURCE_SHA256'];m=json.loads(mf.read_bytes())
for n,h in m['files_SHA256'].items():assert sha(D/n)==h,n
outer=R/'incoming/Deep-C1-current-profile-r353/repair-r1/controller-v2/resource_wave_host_alias_outer.sh'
assert sha(outer)==m['original_CPU_outer_SHA256']
assert not(R/'candidates/Kimi-vocab-parallel-embedding-r461-CPU-v2').exists()
label='Kimi461-vocab-parallel-embedding-CPU-v2'
prior=subprocess.check_output(['sacct','-j','910407','-P','-n','--format=JobIDRaw,JobName%120,State'],text=True,timeout=25)
assert not any(label in line for line in prior.splitlines()),'existing same CPU experiment'
allocation=subprocess.check_output(['scontrol','show','job','910407','-o'],text=True,timeout=25)
assert 'JobState=RUNNING'in allocation and 'a14r2n20'in allocation
intent=R/'results/Kimi461-CPU-v2-once-intent.json'
with intent.open('x')as f:
 json.dump(dict(source_manifest_SHA256=sha(mf),GPU=0,production_HOST_TUs=2,new_GPU_TUs=1,pure_contract_TUs=0,reused_CPU979_contracts=2,prior_accounting_checked=True,created_unix=time.time()),f);f.flush();os.fsync(f.fileno())
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
env.update(KIMI461_SOURCE_SHA256=sha(mf),WAVE_INSIDE_R2323='Kimi-vocab-parallel-embedding-r461/cpu-repair-v2/CPU_inside.sh')
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:05:00','--job-name='+label,'--export=ALL','bash',str(outer),'probe']
log=R/'logs/Kimi461-CPU-v2.log'
with log.open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q=dict(actual_owned_CPU_PID=p.pid,owned_PGID=p.pid,argv=args,log=str(log),source_manifest_SHA256=sha(mf),GPU=0)
with(R/'results/Kimi461-CPU-v2-submitted.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps(q))
