import pathlib,json,hashlib,subprocess,os,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Deep-agent-P0-r143d';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();m=json.loads((D/'source-manifest-r143.json').read_bytes())
for name,h in m['source_SHA256'].items():assert sha(D/name)==h,name
record=R/'results/Deep-agent-P0-r143d-CPU-launch.json';assert not record.exists()and not(R/'candidates/Deep-agent-p0-r143d').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))};outer=R/'incoming/resource_wave_local_r3097_outer.sh';assert sha(outer)=='d3ce6c3196d5728a5f97d459d070fb7720df76d25cadc6be8af0dbb6eca75dce'
a=['srun','--jobid=910374','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c8','-w','a02r4n01','--gres=none','--mem=16384M','-t','00:05:00','--job-name=Deep-agent-P0-static-r143d','--output='+str(R/'logs/Deep-agent-P0-static-r143d-%J.log'),'env','WAVE_INSIDE_R2323=Deep-agent-P0-r143d/CPU_inside_r143.sh','bash',str(outer),'probe'];q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'static_only':True,'GPU':False,'source_manifest_SHA256':sha(D/'source-manifest-r143.json'),'args':a}
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs/Deep-agent-P0-static-r143d-client.log').open('x')as f:p=subprocess.Popen(a,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q['client_PID']=p.pid;record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
