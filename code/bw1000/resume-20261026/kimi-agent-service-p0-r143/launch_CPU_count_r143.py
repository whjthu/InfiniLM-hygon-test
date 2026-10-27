"""Root stages separately. Only one CPU tokenizer metadata job, no GPU."""
import pathlib,json,hashlib,subprocess,os,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/kimi-agent-service-p0-r143';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
m=json.loads((D/'tokenizer-probe-manifest-r143.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
record=R/'results/Kimi-agent-token-count-r143-launch.json';assert not record.exists()and not(R/'results/Kimi-agent-token-count-r143.json').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))}
outer=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(outer)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
args=['srun','--jobid=910379','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r1n11','--gres=none','--mem=4096M','-t','00:03:00','--job-name=Kimi-agent-token-count-r143','--output='+str(R/'logs/Kimi-agent-token-count-r143-%J.log'),'env','WAVE_INSIDE_R2323=kimi-agent-service-p0-r143/CPU_count_inside_r143.sh','bash',str(outer),'probe']
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU_requested':False,'manifest_SHA256':sha(D/'tokenizer-probe-manifest-r143.json'),'args':args}
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs/Kimi-agent-token-count-r143-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q['client_PID']=p.pid;record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
