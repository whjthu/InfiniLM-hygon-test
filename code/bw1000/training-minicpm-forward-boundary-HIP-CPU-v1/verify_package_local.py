"""Verify new complete-source installation block and actually shipped manifest."""
from pathlib import Path
import ast,base64,hashlib,json
D=Path(__file__).resolve().parent;B=D.parent;S=D/'publication';prefix='code/bw1000/'+D.name;source=S/prefix
plan=json.loads((D/'publication-plan.json').read_bytes());task_path='coordination/tasks/'+D.name+'.json';task=json.loads((S/task_path).read_bytes())
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(n,v):
 raw=(json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode();(S/n).write_bytes(raw)
 plan['files'][n]=dict(bytes=len(raw),SHA256=sha(raw))
task['remote_files']=list(task['source_SHA256'])
mf=json.loads((source/'SOURCE_FILES.json').read_bytes())
assert set(mf['files'])<=set(task['remote_files'])
for name,pin in mf['files'].items():
 raw=(source/name).read_bytes();assert len(raw)==pin['bytes'] and sha(raw)==pin['SHA256']
 if name.endswith('.py'):ast.parse(raw)
receiver=(source/'receive_once.py').read_text(encoding='utf8');old=(B/'training-minicpm-first-tail-stop-HIP-CPU-v1/package/receive_once.py').read_text(encoding='utf8').replace('training-minicpm-first-tail-stop-HIP-CPU-v1',D.name)
a=receiver.index('for name,row in FILES.items():');b=receiver.index("save('intent.json'",a)
assert receiver[:a]==old[:old.index('for name,row in FILES.items():')]
assert receiver[b:]==old[old.index("save('intent.json'"):]
block=receiver[a:b];assert 'subprocess' not in block and 'import torch' not in block
fixture=D/'local-source-install';fixture.mkdir(exist_ok=False);(fixture/'source').mkdir();(fixture/'framework').exists() and (_ for _ in ()).throw(AssertionError())
files={n:dict(b64=base64.b64encode((source/n).read_bytes()).decode(),bytes=(source/n).stat().st_size,SHA256=task['source_SHA256'][n]) for n in task['remote_files']}
exec(compile(block,'actual_new_complete_source_install_block','exec'),dict(W=fixture,FILES=files,Path=Path,base64=base64,hashlib=hashlib,json=json))
fm=json.loads((fixture/'source/FRAMEWORK_FILES.json').read_bytes());assert len(fm['files'])==353
for n,pin in fm['files'].items():assert (fixture/'framework'/n).stat().st_size==pin['bytes'] and sha((fixture/'framework'/n).read_bytes())==pin['SHA256']
assert sha((fixture/'source/slurm_accounting.py').read_bytes())=='15a42839fd887a52e415104dcef2a4a548480be5da380f8cbd6980ddb55d3c0b'
assert 'from slurm_accounting import wait_terminal,is_terminal' in receiver
assert 'epoch=time.time();anchor=time.monotonic();deadline=epoch+300' in receiver
write(task_path,task)
v=dict(passed=True,actual_installation_block_exercised=True,complete_framework_files_verified=353,shipped_manifest_complete=True,unchanged_admission_deadline_srun_and_terminal_accounting_code=True,accounting_helper_unchanged_and_integrated=True,targetHOST_qualified=False,Slurm_steps=0,GPU_cards=0)
with (D/'local-package-checks.json').open('x') as f:json.dump(v,f,indent=2)
(D/'publication-plan.json').write_bytes((json.dumps(plan,indent=2)+'\n').encode());print(json.dumps(v))
