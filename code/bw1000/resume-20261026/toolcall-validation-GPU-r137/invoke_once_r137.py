import pathlib,json,hashlib,subprocess,os,datetime,re
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/toolcall-validation-GPU-r137';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();g=json.loads((D/'root-GO-after-CPU.json').read_bytes());assert g['GPU_authorized']is True and g['source_published']is True and g['publication_branch']=='bw1000-experiments'and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])and g['GPU_manifest_SHA256']==sha(D/'GPU-source-manifest.json')
m=json.loads((D/'GPU-source-manifest.json').read_bytes())
for n,h in m['source_SHA256'].items():assert sha(D/n)==h,n
record=R/'results/toolcall-Qwen06-r137-keeper-launch.json';assert not record.exists()and not(R/'results/toolcall-Qwen06-r137-watch.json').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=2048M','-t','00:30:00','--job-name=toolcall-Qwen06-keeper-r137','--output='+str(R/'logs/toolcall-Qwen06-keeper-r137-%J.log'),'python3',str(D/'controller_r137.py')]
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_keeper_only':True,'publication_commit':g['publication_commit'],'CPU_receipt_SHA256':g['CPU_receipt_SHA256'],'GPU_source_manifest_SHA256':sha(D/'GPU-source-manifest.json'),'args':args}
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs/toolcall-Qwen06-keeper-r137-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q['client_PID']=p.pid;record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
