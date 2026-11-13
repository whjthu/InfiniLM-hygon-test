"""Independently confirm the installed source and original stage receipts."""
from pathlib import Path
import ast,hashlib,json,subprocess
B=Path(__file__).resolve().parent
D=B/'hunyuan-sft-source-stage-publication-v1'
binding=json.loads((D/'publication/code/bw1000/training-hunyuanvideo-original-source-v1/BINDING.json').read_bytes())
i=json.loads((D/'stage-intent.json').read_bytes())
o=json.loads((D/'stage-outcome.json').read_bytes())['value']
tree=ast.parse((B/'read_extra40_approval_original_handles_once.py').read_bytes())
n=next(k for k,v in enumerate(tree.body) if isinstance(v,ast.Assign) and isinstance(v.targets[0],ast.Name) and v.targets[0].id=='ssh')
env={'__file__':__file__}
exec(compile(ast.Module(body=tree.body[:n+1],type_ignores=[]),'known-readonly-transport','exec'),env)
remote=r'''
from pathlib import Path
import hashlib,json,stat
base=Path(ROOT)
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
intent=read(base/'intent.json')
outcome=read(base/'outcome.json')
assert intent['claim']==CLAIM and intent['request_id']==REQUEST
assert outcome==OUTCOME and outcome['passed'] is True
root=base/'source'
assert root.is_dir() and not root.is_symlink() and not (base/'source.incomplete').exists()
assert set(PINS)=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
for name,pin in PINS.items():
    p=root/name
    st=p.lstat()
    assert stat.S_ISREG(st.st_mode) and st.st_size==pin['bytes'] and stat.S_IMODE(st.st_mode)==pin['mode']
    assert sha(p)==pin['SHA256']
    if p.suffix=='.py':compile(p.read_bytes(),name,'exec')
print(json.dumps(dict(verified=True,claim=CLAIM,request_id=REQUEST,source=str(root),
    intent_SHA256=sha(base/'intent.json'),outcome_SHA256=sha(base/'outcome.json'),
    files=len(PINS),bytes=sum(x['bytes'] for x in PINS.values()),GPU_calls=0,
    model_imported=False,cache_bytes_read=0,weight_bytes_read=0)))
'''
args=f"ROOT={binding['target']!r}\nCLAIM={i['claim']!r}\nREQUEST={i['request_id']!r}\nOUTCOME={o!r}\nPINS={binding['files']!r}\n"
r=subprocess.run(env['ssh'],input=(args+remote).encode(),capture_output=True,timeout=40)
if r.returncode==0:v=json.loads(r.stdout)
else:v=dict(rc=r.returncode,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),failure_types=[x for x in ('AssertionError','Permission denied','Connection closed') if x.encode() in r.stderr])
with (D/'stage-independent.json').open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
print(json.dumps(v))
