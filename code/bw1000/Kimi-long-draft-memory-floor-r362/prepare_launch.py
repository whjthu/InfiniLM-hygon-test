"""Materialize a controller helper only after the published final one-card claim."""
from pathlib import Path
import argparse,hashlib,json

p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--CPU-SHA',required=True);a=p.parse_args()
D=Path(__file__).resolve().parent;L=D.parents[3]
g=json.loads((D/'execution-GO.template.json').read_bytes());g.update(execution_selected=True,
    publication_commit=a.commit,CPU_receipt_SHA256=a.CPU_SHA,
    source_manifest_SHA256=hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest())
script="""from pathlib import Path
import json,hashlib,subprocess,os
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');D=R/'incoming/Kimi-long-draft-memory-floor-r362'
g=__EXECUTION_PLAN__;mf=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
assert hashlib.sha256((D/'SOURCE_MANIFEST.json').read_bytes()).hexdigest()==g['source_manifest_SHA256']
assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==h for n,h in mf['files_SHA256'].items())
assert hashlib.sha256((R/'candidates/Kimi-long-draft-memory-floor-r362/CPU-result.json').read_bytes()).hexdigest()==g['CPU_receipt_SHA256']
control=R/'control/Kimi-r362-execution-GO.json'
with control.open('x')as f:json.dump(g,f,indent=2)
args=['srun','--jobid=910407','--overlap','--exact','--immediate=30','--cpu-bind=none','-N1','-n1','-c1','--gres=none','--mem=2048M','-w','a14r2n20','-t','00:15:00','--job-name=Kimi-r362-keeper','--output='+str(R/'logs/Kimi-r362-keeper-%J.log'),'python3','-B',str(D/'controller.py'),'--execution-go',str(control)]
with(R/'logs/Kimi-r362-keeper-client.log').open('x')as f:p=subprocess.Popen(args,env=os.environ.copy(),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps({'CPU_keeper_submitted':True,'PID':p.pid,'label':'Kimi-r362-keeper','publication_commit':g['publication_commit'],'GPU_not_yet_submitted_by_this_helper':True}))
""".replace('__EXECUTION_PLAN__',repr(g))
(L/'remote362-launch-pressure.py').write_bytes(script.encode())
print(json.dumps(g))
