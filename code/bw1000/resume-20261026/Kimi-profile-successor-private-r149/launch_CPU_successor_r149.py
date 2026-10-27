"""Submit one ordinary CPU waiter; root publishes/exports claim and GO first."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
D=R/'incoming/Kimi-profile-successor-r149'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--go-file',required=True);p.add_argument('--source-manifest-sha',required=True);a=p.parse_args()
    manifest=D/'source-manifest-r149.json';assert sha(manifest)==a.source_manifest_sha
    m=json.loads(manifest.read_bytes())
    for rel,h in m['files_SHA256'].items():assert sha(D/rel)==h
    go=json.loads(Path(a.go_file).read_bytes());assert go['source_manifest_SHA256']==a.source_manifest_sha
    from successor_after_P0_r149 import validate_claim
    claim=Path(go['claim_path']);assert claim.resolve().is_relative_to(R/'incoming') and sha(claim)==go['claim_SHA256'];validate_claim(go,json.loads(claim.read_bytes()))
    env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','MV2_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
    rows=subprocess.check_output(['squeue','--steps','-j','910407','-h','-o','%i|%j|%N'],env=env,text=True,timeout=25).splitlines();assert len([x for x in rows if x.split('|')[0].split('.')[-1].isdigit()])<4
    label='Kimi-profile-after-P0-r149';prior=subprocess.check_output(['sacct','-S','2026-10-27','-j','910407','-n','-P','-o','JobIDRaw,JobName%110'],env=env,text=True,timeout=25);assert not any(x.split('|')[1]==label for x in prior.splitlines()if '|'in x)
    record=R/'results/Kimi-profile-after-P0-r149-launch.json';assert not record.exists()
    args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=1024M','-w','a14r2n20','-t','04:05:00','--job-name='+label,'--output='+str(R/'logs'/(label+'-%J.log')),'python3','-B',str(D/'successor_after_P0_r149.py'),'--go-file',str(Path(a.go_file).resolve()),'--source-manifest-sha',a.source_manifest_sha]
    q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,'GPU_submitted_by_launcher':False,'retries':0,'source_manifest_SHA256':a.source_manifest_sha,'publication_commit':go['publication_commit']}
    with record.open('x')as f:json.dump(q,f)
    with (R/'logs'/(label+'-client.log')).open('x')as f:z=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    q['CPU_client_PID']=z.pid;record.write_text(json.dumps(q,indent=2));print(json.dumps(q))


if __name__=='__main__':main()
