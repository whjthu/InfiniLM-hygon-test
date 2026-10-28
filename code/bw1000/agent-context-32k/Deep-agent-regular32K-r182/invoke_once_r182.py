"""Start only an owned CPU keeper for the frozen one-run controller."""
import pathlib,json,hashlib,os,re,subprocess,datetime
R=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Deep-agent-regular32K-r182';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();g=json.loads((D/'root-GO-r182.json').read_bytes());assert g['GPU_authorized']is True and g['source_published']is True and g['source_manifest_SHA256']==sha(D/'manifest-r182.json')and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])and g['publication_branch']=='bw1000-experiments'
m=json.loads((D/'manifest-r182.json').read_bytes())
for p,h in m['source_SHA256'].items():assert sha(D/p)==h,p
record=R/'results/Deep-agent-regular32K-r182-keeper-launch.json';assert not record.exists()and not(R/'Deep-agent-regular32K-r182-watch.json').exists();job=str(g['CPU_keeper_parent']);node=g['CPU_keeper_node'];assert re.fullmatch('\d+',job)and re.fullmatch('a\d+r\d+n\d+',node)
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))};state=subprocess.check_output(['scontrol','show','job',job,'-o'],env=env,text=True,timeout=20);assert 'UserId=aclsylqidf('in state and 'JobState=RUNNING'in state and node in state
args=['srun','--jobid='+job,'--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w',node,'--gres=none','--mem=2048M','-t','04:30:00','--job-name=Deep-agent-regular32K-r182-keeper','--output='+str(R/'logs/Deep-agent-regular32K-r182-keeper-%J.log'),'python3',str(D/'controller_r182.py')]
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_keeper_only':True,'source_manifest_SHA256':sha(D/'manifest-r182.json'),'args':args,'restarts':0}
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs/Deep-agent-regular32K-r182-keeper-client.log').open('x')as f:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q['client_PID']=p.pid;record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
