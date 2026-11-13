"""Verify the original HunyuanVideo import step, installed probe and result in place."""
from pathlib import Path
import ast,hashlib,json,subprocess
D=Path(__file__).resolve().parent
B=D.parent
launch=json.loads((D/'launch-intent.json').read_bytes())
tree=ast.parse((B/'read_extra40_approval_original_handles_once.py').read_bytes())
i=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ssh')
env={'__file__':__file__}
exec(compile(ast.Module(body=tree.body[:i+1],type_ignores=[]),'known-readonly-transport','exec'),env)
remote=r'''
from pathlib import Path
import hashlib,json,subprocess
w=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-hunyuanvideo-sft-native-import-CPU-v1')
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
run=lambda args:subprocess.check_output(args,text=True,timeout=12).splitlines()
i=read(w/'intent.json')
o=read(w/'outcome.json')
assert i['request_id']==o['request_id']==REQUEST and i['claim']==o['claim']==CLAIM
assert o['accounting_columns']==['JobIDRaw','JobName','State','ExitCode','ElapsedRaw','NodeList']
rows=[s.split('|') for s in run(['sacct','-n','-P','-j','1103606','--format=JobIDRaw,JobName%100,State,ExitCode,ElapsedRaw,NodeList']) if '|'+w.name+'|' in s]
assert len(rows)==1 and rows==o['accounting'] and rows[0][-1]==i['node']=='f13r3n13'
assert o['terminal']['terminal_visible'] and o['owned_srun_reaped']
assert not any(s.split('|')[1]==w.name for s in run(['squeue','--steps','-h','-j','1103606','-o','%i|%j']))
assert sha(w/'CPU.log')==o['log_SHA256']
assert all(sha(w/'src'/n)==p for n,p in i['source_files'].items())
p=w/'evidence/CPU-result.json'
value=None
if p.is_file():
    doc=o['CPU-result.json']
    assert sha(p)==doc['SHA256'] and read(p)==doc['value']
    value=doc['value']
    assert value['owner']['step']==rows[0][0] and value['owner']['node']==rows[0][-1]
    assert not value['model_constructed'] and not value['weights_read'] and not value['cache_read'] and value['GPU_cards']==0
if o['passed']:
    assert rows[0][2:4]==['COMPLETED','0:0'] and value['passed'] and value['native_files_verified']==38
    assert len(value['modules'])==4
print(json.dumps(dict(verified=True,passed=o['passed'],accounting=rows,CPU_result=value,
    outcome_SHA256=sha(w/'outcome.json'),CPU_log_SHA256=o['log_SHA256'],GPU_cards=0)))
'''
args=f"REQUEST={launch['request_id']!r}\nCLAIM={launch['claim']!r}\n"
r=subprocess.run(env['ssh'],input=(args+remote).encode(),capture_output=True,timeout=40)
if r.returncode==0:v=json.loads(r.stdout)
else:v=dict(rc=r.returncode,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),failure_types=[n for n in ('AssertionError','FileNotFoundError','Connection closed','Permission denied') if n.encode() in r.stderr])
with (D/'independent-terminal.json').open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
print(json.dumps(v))
