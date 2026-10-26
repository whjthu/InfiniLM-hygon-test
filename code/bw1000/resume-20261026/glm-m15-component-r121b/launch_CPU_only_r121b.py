"""Invoke only after root separately stages/verifies this exact package.
CPU-only Slurm step; never uploads and never submits GPU resources.
"""
import argparse,pathlib,json,hashlib,subprocess,os,datetime
a=argparse.ArgumentParser();a.add_argument('--node',required=True);a.add_argument('--parent',default='910407',choices=['910407']);a.add_argument('--manifest-sha',required=True);opt=a.parse_args()
assert opt.node.isalnum() and opt.node.startswith('a') and len(opt.node)<32
r=pathlib.Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');inc=r/'incoming/GLM-M15-component-r121b';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();mp=inc/'manifest-r121b.json'
assert len(opt.manifest_sha)==64 and sha(mp)==opt.manifest_sha
m=json.loads(mp.read_bytes())
for n,h in m['source_SHA256'].items():assert sha(inc/n)==h,n
assert not (r/'candidates/GLM-M15-component-r121b').exists()
receipt=r/'results/GLM-M15-CPU-launch-r121b.json';assert not receipt.exists()
env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_')) and k not in ['CUDA_VISIBLE_DEVICES','HIP_VISIBLE_DEVICES','ROCR_VISIBLE_DEVICES','GPU_DEVICE_ORDINAL']}
args=['srun','--jobid='+opt.parent,'--overlap','--exact','--kill-on-bad-exit=1','--cpu-bind=none','-N1','-n1','-c1','-w',opt.node,'--gres=none','--mem=2048M','-t','00:03:00','--job-name=GLM-M15-CPU-r121b','--output='+str(r/'logs/GLM-M15-CPU-r121b-%J.log'),'env','WAVE_INSIDE_R2323=GLM-M15-component-r121b/CPU_inside_r121b.sh','bash',str(r/'incoming/resource_wave_local_r3097_outer.sh'),'probe']
q={'schema':1,'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU_requested':False,'manifest_SHA256':sha(mp),'args':args};receipt.write_text(json.dumps(q,indent=2))
with (r/'logs/GLM-M15-CPU-r121b-controller.log').open('x') as log:p=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
q['client_PID']=p.pid;receipt.write_text(json.dumps(q,indent=2));print(json.dumps(q))
