"""Root stages and publishes source first. Submit one CPU-only keeper."""
import argparse,pathlib,json,hashlib,subprocess,os,re,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-agent-service-Model-r144d';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--go-file',required=True);a=p.parse_args()
assert sha(D/'service-source-manifest-r144d.json')==a.manifest_sha;m=json.loads((D/'service-source-manifest-r144d.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
go=json.loads(pathlib.Path(a.go_file).read_bytes());assert go['GPU_authorized']is True and go['source_published']is True and go['source_manifest_SHA256']==a.manifest_sha and go['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',go['publication_commit'])
record=R/'results/Kimi-agent-service-r144d-keeper-launch.json';assert not record.exists()and not(R/'results/Kimi-agent-service-r144d-watch.json').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=4096M','-t','02:15:00','--job-name=Kimi-agent-service-r144d-keeper','--output='+str(R/'logs/Kimi-agent-service-r144d-keeper-%J.log'),'python3',str(D/'launcher/service_keeper_r144d.py'),'--manifest-sha',a.manifest_sha,'--go-file',str(pathlib.Path(a.go_file).resolve())]
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_keeper_only':True,'source_manifest_SHA256':a.manifest_sha,'publication_commit':go['publication_commit'],'args':args,'restarts':0}
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs/Kimi-agent-service-r144d-keeper-client.log').open('x')as f:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q['CPU_client_PID']=proc.pid;record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
