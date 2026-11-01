"""Produce the once-only invoker only after the new source and CPU/claim push."""
from pathlib import Path
import argparse,ast,json,re,hashlib

parser=argparse.ArgumentParser();parser.add_argument('--commit',required=True)
parser.add_argument('--target-CPU-SHA',required=True);parser.add_argument('--controller-CPU-SHA',required=True)
args=parser.parse_args();C=Path(__file__).resolve().parent;T=C.parent/'Kimi-long-target-bootstrap-r364';L=C.parents[3]
assert re.fullmatch('[0-9a-f]{40}',args.commit)
assert all(re.fullmatch('[0-9a-f]{64}',s)for s in [args.target_CPU_SHA,args.controller_CPU_SHA])
plan=json.loads((C/'execution-plan.template.json').read_bytes())
plan.update(execute_once=True,publication_commit=args.commit,
    controller_manifest_SHA256=hashlib.sha256((C/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
    target_manifest_SHA256=hashlib.sha256((T/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
    target_CPU_receipt_SHA256=args.target_CPU_SHA,CPU_receipt_SHA256=args.controller_CPU_SHA)
script="""from pathlib import Path
import json,subprocess
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');C=R/'incoming/Kimi-long-pair-controller-r365';plan=__PLAN__
p=R/'control/Kimi-long-run-d-r365-execution-plan.json'
with p.open('x')as f:json.dump(plan,f,indent=2);f.write('\\n')
process=subprocess.run(['python3','-B',str(C/'invoke_once.py'),'--plan',str(p)],capture_output=True,text=True,timeout=180)
assert process.returncode==0,process.stderr
rows=[json.loads(line)for line in process.stdout.splitlines()if line.startswith('{')]
assert len(rows)==1
print(json.dumps({'plan':str(p),'publication_commit':plan['publication_commit'],'invoker':rows[0],'target_GPU_submitted_by_this_helper':False}))
""".replace('__PLAN__',repr(plan))
ast.parse(script);(L/'remote364365-launch-run-d.py').write_bytes(script.encode())
print(json.dumps(dict(run=plan['run'],publication_commit=plan['publication_commit'],helper_AST_PASS=True,execution_helper_only=True)))
