"""The only GPU-intent entry. False/missing manifest or incompleteCPU cannot submit."""
from pathlib import Path
import os,json,time,secrets,subprocess,re,argparse
from keeper import verify,R

def launch(directory,manifest_SHA,commit,run):
 directory=Path(directory);verify(directory,manifest_SHA,R)
 assert re.fullmatch('[0-9a-f]{40}',commit)and re.fullmatch('[a-z0-9-]+',run)
 label='Deep-PD-r445-'+run;intent=R/'results'/(label+'-intent.json')
 assert not (R/'results'/(label+'-watch.json')).exists()
 assert all(not(R/'control'/('Deep382-'+run+'-'+role+'-listening.json')).exists()for role in ['P','D'])
 with intent.open('x')as f:json.dump(dict(run=run,publication_commit=commit,GPU_manifest_SHA256=manifest_SHA,CPU_keeper_only=True,GPU_submitted_by_invoker=False,no_retry=True),f,indent=2)
 env={k:v for k,v in os.environ.items()if not k.startswith(('SLURM_','PMI_','PMIX_','OMPI_','CUDA_','HIP_','ROCR_','GPU_'))or k in ['SLURM_CONF','SLURM_CONF_SERVER']}
 env.update(DEEP_PD_SESSION=secrets.token_hex(32),DEEP_PD_CPU_EPOCH=str(time.time()))
 args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=4096M','-t','01:35:00','--job-name='+label+'-keeper','--output='+str(R/'logs'/(label+'-keeper-%J.log')),'timeout','--signal=TERM','--kill-after=15s','5670','python3','-B',str(directory/'keeper.py'),'--manifest-sha',manifest_SHA,'--publication-commit',commit,'--run',run]
 with(R/'logs'/(label+'-keeper-client.log')).open('x')as log:p=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 return dict(CPU_keeper_submitted=True,GPU_submitted_by_invoker=False,PID=p.pid,run=run,publication_commit=commit,GPU_manifest_SHA256=manifest_SHA)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--publication-commit',required=True);a.add_argument('--run',required=True);a=a.parse_args();print(json.dumps(launch(Path(__file__).resolve().parent,a.manifest_sha,a.publication_commit,a.run)))
