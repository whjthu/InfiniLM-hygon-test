"""Stage the published exact HunyuanVideo source package once."""
from pathlib import Path
import ast,base64,hashlib,json,subprocess,uuid
B=Path(__file__).resolve().parents[0]
G=Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
D=B/'hunyuan-sft-source-stage-publication-v1'
label='training-hunyuanvideo-original-source-v1'
receipt=json.loads((D/'source-publication-receipt.json').read_bytes())
assert receipt['pushed'] and receipt['remote_head']==receipt['commit']
claim=receipt['commit']
git=['git','-c','safe.directory='+G.as_posix(),'-C',str(G)]
def blob(name):return subprocess.check_output(git+['show',claim+':'+name])
def sha(raw):return hashlib.sha256(raw).hexdigest()
task=json.loads(blob('coordination/tasks/'+label+'.json'))
assert task['stage_GO'] and not task['Model_GO']
def published(name):
    raw=blob(task['source']+'/'+name)
    assert sha(raw)==task['source_SHA256'][name]
    return raw
assert Path(__file__).read_text(encoding='utf-8-sig').rstrip().encode()+b'\n'==published('stage_hunyuan_source_once.py')
binding=json.loads(published('BINDING.json'))
root=B/'hunyuanvideo-1.5-exact-a100-package-v1'
files={}
for name,pin in binding['files'].items():
    raw=(root/name).read_bytes()
    assert len(raw)==pin['bytes'] and sha(raw)==pin['SHA256']
    files[name]=base64.b64encode(raw).decode('ascii')
assert len(files)==38
tree=ast.parse((B/'read_extra40_approval_original_handles_once.py').read_bytes())
i=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.Assign)
       and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='ssh')
env={'__file__':__file__}
exec(compile(ast.Module(body=tree.body[:i+1],type_ignores=[]),'known-readonly-transport','exec'),env)
remote=r'''
from pathlib import Path,PurePosixPath
import base64,hashlib,json,os,signal,stat
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('source stage deadline')))
signal.alarm(30)
w=Path(TARGET)
assert w.name=='training-hunyuanvideo-original-source-v1'
assert set(PINS)==set(FILES) and len(FILES)==38
raws={}
for name,pin in PINS.items():
    path=PurePosixPath(name)
    assert not path.is_absolute() and '..' not in path.parts
    raw=base64.b64decode(FILES[name],validate=True)
    assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    if path.suffix=='.py':compile(raw,name,'exec')
    raws[name]=raw
w.mkdir(mode=0o700)
def save(name,obj):
    with (w/name).open('x') as f:
        json.dump(obj,f,indent=2)
        f.flush()
        os.fsync(f.fileno())
save('intent.json',dict(claim=CLAIM,request_id=REQUEST,files=len(FILES)))
pending=w/'source.incomplete'
pending.mkdir(mode=0o700)
for name,raw in raws.items():
    path=pending/name
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:
        f.write(raw)
        os.fchmod(f.fileno(),PINS[name]['mode'])
        f.flush()
        os.fsync(f.fileno())
    assert hashlib.sha256(path.read_bytes()).hexdigest()==PINS[name]['SHA256']
    assert stat.S_IMODE(path.stat().st_mode)==PINS[name]['mode']
assert set(PINS)=={p.relative_to(pending).as_posix() for p in pending.rglob('*') if p.is_file()}
pending.rename(w/'source')
result=dict(passed=True,claim=CLAIM,request_id=REQUEST,source=str(w/'source'),
    files=len(FILES),bytes=sum(len(raw) for raw in raws.values()),python_files=sum(n.endswith('.py') for n in FILES),
    weights_bytes=0,cache_bytes=0,GPU_calls=0)
save('outcome.json',result)
print(json.dumps(result))
'''
request=uuid.uuid4().hex
payload=(f"TARGET={binding['target']!r}\nPINS={binding['files']!r}\nFILES={files!r}\nCLAIM={claim!r}\nREQUEST={request!r}\n"+remote).encode()
assert len(payload)<650000
with (D/'stage-intent.json').open('x',encoding='utf8') as f:
    json.dump(dict(claim=claim,request_id=request,payload_SHA256=sha(payload)),f)
r=subprocess.run(env['ssh'],input=payload,capture_output=True,timeout=50)
v=dict(rc=r.returncode,stderr_SHA256=sha(r.stderr))
if r.returncode==0:v['value']=json.loads(r.stdout)
else:v['failure_types']=[n for n in ('AssertionError','Permission denied','Connection closed','FileExistsError') if n.encode() in r.stderr]
with (D/'stage-outcome.json').open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
print(json.dumps(v))
