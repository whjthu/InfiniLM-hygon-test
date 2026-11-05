from pathlib import Path
import hashlib,json,os,subprocess,time
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-vocab-parallel-embedding-r461/split-v128-component-link-v2'
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());mf=H(D/'SOURCE_MANIFEST.json')
assert mf==os.environ['KIMI_SPLITV_COMPONENT_SHA256']
for n,h in m['files_SHA256'].items():assert H(D/n)==h,n
outer=R/'incoming/Deep-C1-current-profile-r353/repair-r1/controller-v2/resource_wave_host_alias_outer.sh'
assert H(outer)==m['original_CPU_outer_SHA256']
assert not(R/'candidates/Kimi-split-V128-component-link-v2').exists()
label='Kimi-split-V128-component-link-v2'
prior=subprocess.check_output(['sacct','-S','2026-11-05','-j','910407','-P','-n','--format=JobIDRaw,JobName%120,State'],universal_newlines=True,timeout=10)
assert not any(label in line for line in prior.splitlines())
allocation=subprocess.check_output(['scontrol','show','job','910407','-o'],universal_newlines=True,timeout=10)
assert 'JobState=RUNNING'in allocation and 'a14r2n20'in allocation
intent=R/'results/Kimi-split-V128-component-link-v2-once-intent.json'
with intent.open('x')as f:
 json.dump(dict(source_manifest_SHA256=mf,GPU=0,new_HOST_TUs=0,new_GPU_TUs=0,prior_accounting_checked=True,created_unix=time.time()),f);f.flush();os.fsync(f.fileno())
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
env.update(KIMI_SPLITV_COMPONENT_SHA256=mf,WAVE_INSIDE_R2323='Kimi-vocab-parallel-embedding-r461/split-v128-component-link-v2/CPU_inside.sh')
args=['srun','--jobid=910407','--overlap','--exact','--immediate=20','--cpu-bind=none','-N1','-n1','-c4','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:05:00','--job-name='+label,'--export=ALL','bash',str(outer),'probe']
log=R/'logs/Kimi-split-V128-component-link-v2.log'
with log.open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q=dict(actual_owned_CPU_PID=p.pid,owned_PGID=p.pid,argv=args,log=str(log),source_manifest_SHA256=mf,GPU=0,started_unix=time.time())
with(R/'results/Kimi-split-V128-component-link-v2-submitted.json').open('x')as f:json.dump(q,f,indent=2)
print(json.dumps(q))
