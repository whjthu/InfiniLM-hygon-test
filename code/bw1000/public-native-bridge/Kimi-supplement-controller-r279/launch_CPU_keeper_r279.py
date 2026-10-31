"""Submit the one existing-source CPU keeper once. Never replay unknown submission."""
import argparse,json,pathlib,subprocess,os,datetime,hashlib
D=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();go=pathlib.Path(a.execution_go).resolve();q=json.loads(go.read_bytes());assert q['execution_authorized']and q['one_GPU_attempt']and q['GPU_max_s']==29400 and q['CPU_keeper_max_s']==32400
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==q['controller_manifest_SHA256'];m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in m['files_SHA256'].items())
record=R/'results/Kimi-latest99-supplement-Model-r279-keeper-launch.json';assert not record.exists()and not(R/'results/Kimi-latest99-supplement-Model-r279-watch.json').exists()
for parent in ['910407','910380']:
 z=subprocess.check_output(['scontrol','show','job',parent,'-o'],text=True,timeout=25);assert 'UserId=aclsylqidf('in z and 'JobState=RUNNING'in z
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=4096M','-t','09:00:00','--job-name=Kimi-latest99-supplement-Model-r279-keeper','--output='+str(R/'logs/Kimi-latest99-supplement-Model-r279-keeper-%J.log'),'python3',str(D/'service_keeper_r279.py'),'--execution-go',str(go)]
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
with record.open('x')as f:json.dump({'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'publication_commit':q['publication_commit'],'no_retries':True,'GPU_max_s':29400,'CPU_keeper_max_s':32400,'controller_manifest_SHA256':q['controller_manifest_SHA256']},f,indent=2)
with(R/'logs/Kimi-latest99-supplement-Model-r279-keeper-client.log').open('x')as f:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_keeper_client_PID':proc.pid,'actual_CPU_step':'PENDING_NOT_YET_OBSERVED','watch':str(R/'results/Kimi-latest99-supplement-Model-r279-watch.json'),'GPU_submitted_by_keeper_only_after_closure_and_idle':True,'no_retries':True}))
