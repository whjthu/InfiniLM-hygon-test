"""Root invocation after publication-bound GO. This CPU keeper may submit ONE
GPU diagnostic; this is not a CPU-only preflight entry."""
import argparse,json,subprocess,datetime,pathlib
from contract_r183 import ROOT,INC,LABEL,gate,sha,clean_env
p=argparse.ArgumentParser();p.add_argument('--manifest-sha',required=True);p.add_argument('--go-file',required=True);a=p.parse_args();m,g,pins=gate(a.manifest_sha,a.go_file)
record=ROOT/'results'/(LABEL+'-keeper-launch.json');assert not record.exists()and not(ROOT/'results'/(LABEL+'-watch.json')).exists()
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=4096M','-w','a14r2n20','-t','00:25:00','--job-name='+LABEL+'-keeper','--output='+str(ROOT/'logs'/(LABEL+'-keeper-%J.log')),'python3','-B',str(INC/'controller_r183.py'),'--manifest-sha',a.manifest_sha,'--go-file',str(pathlib.Path(a.go_file).resolve())]
q={'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_keeper_only':True,'GPU_submitter':'single controller after fresh KFD/IB/ports and root published GO','manifest_SHA256':a.manifest_sha,'GO_SHA256':sha(a.go_file),'publication_commit':g['publication_commit'],'retries':0}
with record.open('x')as f:json.dump(q,f,indent=2)
with(ROOT/'logs'/(LABEL+'-keeper-client.log')).open('x')as f:proc=subprocess.Popen(args,env=clean_env(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
q['owned_CPUkeeper_client_PID']=proc.pid;record.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps(q))
