"""Root invokes only after necessary source staging; no GPU is requested."""
import pathlib,json,hashlib,subprocess,os,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-agent-service-r143';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert sha(D/'source-manifest-r143.json')=='623bd89fd65befb8ff020dc810e21fc1f3d8d1403792829f5c0fae0eadaacc3e';m=json.loads((D/'source-manifest-r143.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
record=R/'results/GLM-agent-service-r143-CPU-launch.json';assert not record.exists()and not(R/'candidates/GLM-agent-service-r143').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))};outer=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(outer)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
a=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=8192M','-t','00:10:00','--job-name=GLM-agent-service-CPU-r143','--output='+str(R/'logs/GLM-agent-service-CPU-r143-%J.log'),'env','WAVE_INSIDE_R2323=GLM-agent-service-r143/CPU_inside_r143.sh','bash',str(outer),'probe'];q={'CPU_only':True,'GPU_requested':False,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_manifest_SHA256':sha(D/'source-manifest-r143.json'),'args':a}
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs/GLM-agent-service-CPU-r143-client.log').open('x')as f:p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q['client_PID']=p.pid;record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
