"""Submit only the one owned CPU keeper after separately supplied explicit GO."""
import argparse,pathlib,json,hashlib,subprocess,datetime,os
from controller_r283 import R,D,LABEL,W,verify_sources,clean_env,parent
p=argparse.ArgumentParser();p.add_argument('--execution-go',required=True);a=p.parse_args();go=pathlib.Path(a.execution_go).resolve();g=json.loads(go.read_bytes());verify_sources(g);env=clean_env();parent(910407,'a14r2n20',env);parent(910374,'a02r4n01',env);intent=R/'results'/(LABEL+'-intent.json');record=R/'results'/(LABEL+'-keeper-launch.json');assert not intent.exists()and not record.exists()and not W.exists()
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=4096M','-t','09:00:00','--job-name='+LABEL+'-keeper','--output='+str(R/'logs'/(LABEL+'-keeper-%J.log')),'python3',str(D/'controller_r283.py'),'--execution-go',str(go)]
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_keeper_only':True,'publication_commit':g['publication_commit'],'owner_manifest_SHA256':g['owner_manifest_SHA256'],'controller_manifest_SHA256':g['controller_manifest_SHA256'],'one_attempt_no_retry_reload':True,'actual_GPU':'NOT_SUBMITTED_BY_THIS_LAUNCHER'}
with intent.open('x')as f:json.dump(q,f,indent=2)
with record.open('x')as f:json.dump(q,f,indent=2)
with(R/'logs'/(LABEL+'-keeper-client.log')).open('x')as log:proc=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(q,CPU_keeper_client_PID=proc.pid,actual_CPU_step='PENDING_NOT_YET_OBSERVED',watch=str(W))))
