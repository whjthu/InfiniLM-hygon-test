"""347 single CPU keeper intent; no direct GPU submission by this invoker."""
from pathlib import Path
import argparse,json,os,subprocess,uuid,time
from contract_r410 import CPU_SECONDS,publish_once,sha
from controller import R,D,verify_sources
from slurm_backend import SlurmBackend
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);a=p.parse_args();plan=json.loads(a.plan.read_bytes());verify_sources(R,D,plan,True)
 b=SlurmBackend(R,D,plan);b.validate_resources();prefix='Kimi-ordinary-r410-'+plan['run'];intent=R/'results'/(prefix+'-intent.json');watch=R/'results'/(prefix+'-watch.json')
 assert not watch.exists();publish_once(intent,dict(run=plan['run'],plan_SHA256=sha(a.plan),controller_source_manifest_SHA256=plan['controller_source_manifest_SHA256'],target_GPU_submitted=False,CPU_keeper_only=True,retry=False,reload=False,publication_commit=plan['publication_commit']))
 env=dict(b.env,KIMI410_CONTROLLER_NONCE=uuid.uuid4().hex+uuid.uuid4().hex,KIMI410_KEEPER_START_UNIX=str(time.time()))
 cmd=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=4096M','-t','01:40:00','--job-name='+b.label('keeper'),'--export=ALL','--output='+str(R/'logs'/(b.label('keeper')+'-%J.log')),'python3','-B',str(D/'controller.py'),'--plan',str(a.plan.resolve())]
 with(R/'logs'/(b.label('keeper')+'-srun-client.log')).open('x')as log:proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 q=dict(run=plan['run'],publication_commit=plan['publication_commit'],CPU_keeper_client_PID=proc.pid,CPU_keeper_original_PGID=proc.pid,watch=str(watch),argv=cmd,target_GPU_submitted_by_invoker=False)
 publish_once(R/'results'/(prefix+'-CPU-submission.json'),q);print(json.dumps(q))
if __name__=='__main__':main()
