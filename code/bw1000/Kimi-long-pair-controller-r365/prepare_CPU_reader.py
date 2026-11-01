from pathlib import Path
import ast

D=Path(__file__).resolve().parent;L=D.parents[3]
script="""from pathlib import Path
import json,hashlib,subprocess,sys
R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');C=R/'incoming/Kimi-long-pair-controller-r365';rows=[]
for directory in ['Kimi-long-target-bootstrap-r364','Kimi-long-pair-controller-r365']:
 p=R/'candidates'/directory/'CPU-result.json';q={'directory':directory,'receipt_exists':p.exists()}
 if p.exists():q.update(receipt=json.loads(p.read_bytes()),receipt_path=str(p),receipt_SHA256=hashlib.sha256(p.read_bytes()).hexdigest())
 log=p.parent/'checks.log'
 if log.exists():q.update(log_SHA256=hashlib.sha256(log.read_bytes()).hexdigest(),log_tail=log.read_text()[-12000:])
 rows.append(q)
account=subprocess.check_output(['sacct','-S','2026-11-01','-j','910407','-n','-P','-o','JobIDRaw,JobName%120,State,ExitCode,Elapsed'],text=True,timeout=25)
result={'GPU':0,'CPU_results':rows,'accounting':'\\n'.join(x for x in account.splitlines()if'Kimi-run-d-'in x)}
if all(q.get('receipt',{}).get('completed')and q['receipt']['passed']for q in rows):
 sys.path.insert(0,str(C));import controller,contract
 plan=json.loads((C/'execution-plan.template.json').read_bytes());plan.update(execute_once=True,
  controller_manifest_SHA256=hashlib.sha256((C/'SOURCE_MANIFEST.json').read_bytes()).hexdigest(),
  target_manifest_SHA256=rows[0]['receipt']['source_manifest_SHA256'],
  target_CPU_receipt_SHA256=rows[0]['receipt_SHA256'],CPU_receipt_SHA256=rows[1]['receipt_SHA256'])
 controller.verify_sources(R,C,plan)
 result['actual_source_verifier_and_both_CPU_receipts_PASS']=True
 result['actual_GPU_or_Native_initialized_by_reader']=False
print(json.dumps(result))
"""
ast.parse(script);(L/'remote364365-read-CPU.py').write_bytes(script.encode())
print('Full postCPU source/receipts verifier reader AST PASS; not executed')
