"""One bounded, published two-node CPU-only host transfer. No implicit retry."""
from pathlib import Path
import os,time,json,hashlib,secrets,re,subprocess,argparse

R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
LABEL='Deep-PD-r381-hostTCP-v1';sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def execute(root,source,mf,commit,run,runner=subprocess.run):
 assert re.fullmatch('[0-9a-f]{40}',commit)and re.fullmatch('[a-z0-9-]+',run)
 root=Path(root);source=Path(source);assert sha(source/'SOURCE_MANIFEST.json')==mf
 m=json.loads((source/'SOURCE_MANIFEST.json').read_bytes());assert m['GPU_GO']is False and all(sha(source/n)==h for n,h in m['files_SHA256'].items())
 parent=runner(['scontrol','show','job','910376','-o'],capture_output=True,text=True,timeout=20)
 assert parent.returncode==0 and 'JobState=RUNNING'in parent.stdout and 'UserId=aclsylqidf('in parent.stdout
 assert re.search(r'(?:^| )NodeList=([^ ]+)',parent.stdout).group(1)=='a02r4n[12-15]'
 intent=root/'control'/('Deep381-hostTCP-'+run+'-intent.json');assert not intent.exists()
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 epoch=time.time();env.update(DEEP_R381_CPU_MANIFEST=mf,DEEP_R381_CPU_RUN=run,DEEP_R381_CPU_DEADLINE=str(epoch+90),DEEP_R381_CPU_NONCE=secrets.token_hex(32))
 with intent.open('x')as f:json.dump(dict(publication_commit=commit,manifest_SHA256=mf,CPU_only=True,GPU=0,original_epoch_unix=epoch,deadline_unix=epoch+90,no_retry=True),f,indent=2)
 args=['srun','--jobid=910376','--overlap','--exact','--immediate=30','--cpu-bind=none','--kill-on-bad-exit=0','-N2','-n2','--ntasks-per-node=1','-c1','-w','a02r4n12,a02r4n13','--gres=none','--mem=4096M','-t','00:02:00','--job-name='+LABEL,'python3','-B',str(source/'worker.py')]
 p=runner(args,env=env,capture_output=True,text=True,timeout=110)
 out=dict(schema=1,completed=False,passed=False,GPU=0,Native_dlopen=False,model_loaded=False,actual_KV_state=False,publication_commit=commit,source_manifest_SHA256=mf,run=run,srun_returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,args=args,reports=[])
 until=min(epoch+145,time.time()+30)
 for role in ['P','D']:
  path=root/'results'/('Deep381-hostTCP-'+run+'-'+role+'.json')
  while not path.exists():
   assert time.time()<until,'Bounded report visibility expired';time.sleep(.2)
  q=json.loads(path.read_bytes());out['reports'].append(dict(path=str(path),SHA256=sha(path),data=q))
 out['completed']=True
 out['passed']=p.returncode==0 and all(q['data']['completed']and q['data']['passed']and q['data']['normal_socket_close']for q in out['reports'])
 assert len({q['data']['actual_step']for q in out['reports']})==1
 out['actual_CPU']=out['reports'][0]['data']['actual_step']
 dest=root/'results'/('Deep381-hostTCP-'+run+'-CPU.json');dest.write_text(json.dumps(out,indent=2)+'\n')
 assert out['passed'],dict(report=str(dest),source_manifest_SHA256=mf)
 return out
def main():
 a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--publication-commit',required=True);a.add_argument('--run',required=True);a=a.parse_args()
 q=execute(R,Path(__file__).resolve().parent,a.manifest_sha,a.publication_commit,a.run)
 print(json.dumps(dict(completed=True,passed=q['passed'],actual_CPU=q['actual_CPU'],GPU=0,report=str(R/'results'/('Deep381-hostTCP-'+a.run+'-CPU.json')))))
if __name__=='__main__':main()
