"""No upload. Run only after root source-publication commit and explicit GPU GO.
This launcher allocates a CPU-only keeper; its pinned controller owns one GPU step.
"""
import argparse,pathlib,json,subprocess,hashlib,re,os,datetime
a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--go-file',required=True);opt=a.parse_args();r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');inc=r/'incoming/GLM-arrival-defer-r153';sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();mp=inc/'GPU-manifest.json';assert sha(mp)==opt.manifest_sha;m=json.loads(mp.read_bytes())
for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
gp=pathlib.Path(opt.go_file).resolve();assert gp.is_relative_to(r.resolve());g=json.loads(gp.read_bytes());assert g['GPU_authorized']is True and g['source_published']is True and g['controller_manifest_SHA256']==opt.manifest_sha and re.fullmatch('[0-9a-f]{40}',g['publication_commit'])and g['publication_branch']=='bw1000-experiments';assert g['CPU_receipt_SHA256']==m['CPU_receipt_SHA256']==sha(r/'candidates/GLM-critical-observer-r131/CPU-build.json')
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
runner_cpu=r/'candidates/GLM-arrival-defer-r153/CPU-prepare.json';runner=json.loads(runner_cpu.read_bytes());assert sha(runner_cpu)==g['runner_CPU_receipt_SHA256']and runner['completed']and runner['CPU_only']and not runner['GPU']and runner['manifest_SHA256']==opt.manifest_sha
record=r/'results/GLM-arrival-defer-r153-keeper-launch.json';assert not record.exists()and not(r/'GLM-arrival-defer-r153-launch.json').exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_'))and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=2048M','-t','02:15:00','--job-name=GLM-critical-Model-keeper-r153','--output='+str(r/'logs/GLM-critical-Model-keeper-r153-%J.log'),'python3',str(inc/'controller_r153.py'),'--manifest-sha',opt.manifest_sha,'--go-file',str(gp)]
q={'schema':1,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'keeper_CPU_only':True,'GPU_submitter':'Pinnedcontroller onlyafterrootGO+publishedcommit+freshidle','args':args,'controller_manifest_SHA256':sha(mp),'root_GO_SHA256':sha(gp),'publication_commit':g['publication_commit'],'CPU_receipt_SHA256':g['CPU_receipt_SHA256'],'restarts':0};record.write_text(json.dumps(q,indent=2))
with(r/'logs/GLM-arrival-defer-r153-keeper-client.log').open('x')as log:p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
q['CPUkeeper_client_PID']=p.pid;record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
