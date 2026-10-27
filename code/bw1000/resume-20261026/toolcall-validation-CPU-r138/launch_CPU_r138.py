"""Root stages separately. This invocation submits only one CPU preflight."""
import pathlib,json,hashlib,subprocess,os,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/toolcall-validation-CPU-r138';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((D/'CPU-source-manifest.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
record=R/'results/toolcall-Qwen06-r138-CPU-launch.json';assert not record.exists()and not(R/'candidates/toolcall-Qwen06-r138').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))}
outer=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(outer)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
args=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c8','-w','a02r4n01','--gres=none','--mem=16384M','-t','00:05:00','--job-name=toolcall-Qwen06-CPU-r138','--output='+str(R/'logs/toolcall-Qwen06-CPU-r138-%J.log'),'env','WAVE_INSIDE_R2323=toolcall-validation-CPU-r138/CPU_inside_r138.sh','bash',str(outer),'probe']
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU_requested':False,'manifest_SHA256':sha(D/'CPU-source-manifest.json'),'args':args}
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs/toolcall-Qwen06-CPU-r138-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q['client_PID']=p.pid;record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
