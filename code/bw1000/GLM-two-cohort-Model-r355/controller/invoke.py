"""Publish one CPU keeper intent; keeper owns the four original stage processes."""
import argparse,json,subprocess,os
from pathlib import Path
from controller import R,D,verify
from backend import Backend,clean_env


def main():
    p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();go=Path(a.execution_go).resolve();g=json.loads(go.read_bytes());verify(R,D,g)
    b=Backend(R,D,g['run']);b.parents();assert all(b.account(s)is None for s in range(4));b.no_foreign()
    paths=[R/'results'/('GLM-r355-'+kind+'-'+g['run']+'.json')for kind in ['intent','watch']]
    assert not any(p.exists()for p in paths)
    with paths[0].open('x')as f:json.dump(dict(run=g['run'],CPU_keeper_only=True,publication_commit=g['publication_commit'],GPU_submitted=False),f)
    args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=4096M','-t','02:10:00','--job-name=GLM-r355-'+g['run']+'-keeper','--output='+str(R/'logs'/('GLM-r355-'+g['run']+'-keeper-%J.log')),'python3','-B',str(D/'controller.py'),'--execution-go',str(go)]
    with(R/'logs'/('GLM-r355-'+g['run']+'-keeper-client.log')).open('x')as log:
        proc=subprocess.Popen(args,env=clean_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(CPU_keeper_client_PID=proc.pid,run=g['run'],actual_GPU='NOT_SUBMITTED_BY_INVOKER')))


if __name__=='__main__':main()
