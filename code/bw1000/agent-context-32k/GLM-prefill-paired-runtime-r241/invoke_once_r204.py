"""Root-only one CPU keeper submission; controller owns one future GPU run."""
import argparse,datetime,hashlib,json,os,pathlib,re,subprocess
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/GLM-prefill-paired-r241';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--publication-commit',required=True);a.add_argument('--cpu-receipt-sha',required=True);a.add_argument('--execution-go',required=True);x=a.parse_args();assert sha(D/'source-manifest-r204.json')==x.manifest_sha and re.fullmatch('[0-9a-f]{40}',x.publication_commit)and re.fullmatch('[0-9a-f]{64}',x.cpu_receipt_sha)
mf=json.loads((D/'source-manifest-r204.json').read_bytes())
for p,h in mf['source_SHA256'].items():assert sha(D/p)==h
go=json.loads(pathlib.Path(x.execution_go).read_bytes());assert go['GPU_GO']is True and go['source_manifest_SHA256']==x.manifest_sha and go['CPU_receipt_SHA256']==x.cpu_receipt_sha and go['publication_commit']==x.publication_commit and go['maximum_HTTP_requests']==18 and go['GPU_Slurm_seconds']==22200 and go['single_attempt']is True
record=R/'GLM-prefill-paired-r241-keeper-launch.json';assert not record.exists()and not(R/'GLM-prefill-paired-r241-watch.json').exists()and not(R/'GLM-prefill-paired-r241-intent.json').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))}
state=subprocess.check_output(['scontrol','show','job','910407','-o'],env=env,text=True,timeout=20);assert 'JobState=RUNNING'in state and 'UserId=aclsylqidf('in state
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=2048M','-t','06:30:00','--job-name=GLM-prefill-paired-r241-keeper','--output='+str(R/'logs/GLM-prefill-paired-r241-keeper-%J.log'),'python3',str(D/'controller_r204.py'),'--manifest-sha',x.manifest_sha,'--publication-commit',x.publication_commit,'--cpu-receipt-sha',x.cpu_receipt_sha]
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_keeper_only':True,'source_manifest_SHA256':x.manifest_sha,'publication_commit':x.publication_commit,'args':args,'restarts':0}
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs/GLM-prefill-paired-r241-keeper-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q['client_PID']=p.pid;record.write_bytes((json.dumps(q,indent=2)+'\n').encode());print(json.dumps(q))
