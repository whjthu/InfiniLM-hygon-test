"""Seal only read-only readers; never touch the frozen Model runtime."""
from pathlib import Path
import base64,hashlib
D=Path(__file__).resolve().parent
audit=(D/'audit_completed_r1.py').read_bytes();cost=(D/'audit_cost_r361.py').read_bytes()
for name,data in [('audit',audit),('cost',cost)]:compile(data,name,'exec')
script='''from pathlib import Path
import json,hashlib,base64,subprocess,io,contextlib
R=Path.home()/'hygon-inference-refactor-20261014';run='GLM-C2M1-r361-a'
directory=R/'incoming/GLM-r361-terminal-audit-r1';directory.mkdir(exist_ok=False)
audit=base64.b64decode(AUDIT_DATA);cost=base64.b64decode(COST_DATA)
assert hashlib.sha256(audit).hexdigest()==AUDIT_SHA and hashlib.sha256(cost).hexdigest()==COST_SHA
(directory/'audit_completed_r1.py').write_bytes(audit);(directory/'audit_cost_r361.py').write_bytes(cost)
capture=io.StringIO()
with contextlib.redirect_stdout(capture):exec(compile(audit,str(directory/'audit_completed_r1.py'),'exec'),{'__name__':'__main__'})
q=json.loads(capture.getvalue());q['reader_SHA256']={'audit_completed_r1.py':AUDIT_SHA,'audit_cost_r361.py':COST_SHA}
if q['passed']:
 output=R/'results/GLM-r361-independent-cost-r1.json';assert not output.exists()
 paths=[r['path']for r in q['owners']]
 p=subprocess.run(['python3','-B',str(directory/'audit_cost_r361.py'),'--owner-jsons',*paths,'--output',str(output)],text=True,capture_output=True,timeout=90)
 assert p.returncode==0,p.stderr
 c=json.loads(output.read_bytes())
 assert c['measured_OFF_s']==q['complete_pair_OFF_mean_s'] and c['measured_ON_s']==q['complete_pair_ON_mean_s']
 for row in ['0','1']:
  for field in q['primary_full_request_metrics'][row]:assert c['primary_full_request_metrics'][row][field]==q['primary_full_request_metrics'][row][field]
 q['independent_cost_reader_equal']=True;q['cost_path']=str(output);q['cost_SHA256']=hashlib.sha256(output.read_bytes()).hexdigest();q['cost']=c
sealed=R/'results/GLM-r361-independent-terminal-audit-r1.json'
with sealed.open('x')as f:json.dump(q,f,indent=2)
q['remote_terminal_audit_path']=str(sealed);q['remote_terminal_audit_SHA256']=hashlib.sha256(sealed.read_bytes()).hexdigest()
print(json.dumps(q))
'''
for key,value in [('AUDIT_DATA',base64.b64encode(audit).decode()),('COST_DATA',base64.b64encode(cost).decode()),('AUDIT_SHA',hashlib.sha256(audit).hexdigest()),('COST_SHA',hashlib.sha256(cost).hexdigest())]:script=script.replace(key,repr(value))
compile(script,'remote361-terminal-audit-r1.py','exec')
out=D.parents[3]/'remote361-terminal-audit-r1.py';out.write_bytes(script.encode())
print('Prepared read-only terminal audit',out)
