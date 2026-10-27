"""Root invocation only after staging/publication/GO. No upload or local run."""
import argparse,pathlib,json,subprocess,datetime
from contract_r120k import ROOT,INC,LABEL,gate,clean_env,sha
a=argparse.ArgumentParser();a.add_argument('--manifest-sha',required=True);a.add_argument('--go-file',required=True);o=a.parse_args();m,g,c,w=gate(o.manifest_sha,o.go_file)
record=ROOT/'results'/(LABEL+'-keeper-launch.json');assert not record.exists()and not(ROOT/(LABEL+'-launch.json')).exists()
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','-w','a14r2n20','--gres=none','--mem=2048M','-t','00:30:00','--job-name='+LABEL+'-keeper','--output='+str(ROOT/'logs'/(LABEL+'-keeper-%J.log')),'python3',str(INC/'controller_r120k.py'),'--manifest-sha',o.manifest_sha,'--go-file',str(pathlib.Path(o.go_file).resolve())]
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_keeper_only':True,'GPU_submitter':'single pinned controller after rootGO and fresh idle','args':args,'manifest_SHA256':o.manifest_sha,'GO_SHA256':sha(o.go_file),'publication_commit':g['publication_commit'],'restarts':0}
with record.open('x')as f:json.dump(q,f,indent=2)
with(ROOT/'logs'/(LABEL+'-keeper-client.log')).open('x')as log:p=subprocess.Popen(args,env=clean_env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
q['CPUkeeper_client_PID']=p.pid;record.write_text(json.dumps(q,indent=2));print(json.dumps(q))
