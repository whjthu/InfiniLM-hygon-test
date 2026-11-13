"""Reconstruct the observed A100 HunyuanVideo package from upstream and six exact overrides."""
from pathlib import Path, PurePosixPath
import ast
import base64
import hashlib
import json
import os
import subprocess

BASE = Path(__file__).resolve().parent
REPO = Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
A100 = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
COMPARE = BASE / 'hunyuan-upstream-comparison-v1.json'
comparison = json.loads(COMPARE.read_bytes())
record = REPO/'results/bw1000/2026-11-13/training-hunyuanvideo-sft-A100-native-package-file-identity.json'
def sha(raw):
    return hashlib.sha256(raw).hexdigest()
assert sha(record.read_bytes()) == comparison['package_record_SHA256']
original = json.loads(record.read_bytes())['package']
files = original['files']
assert len(files) == 38 and set(files) == set(comparison['files'])
overrides = {n: files[n] for n,v in comparison['files'].items() if v['classification']=='bytes_differ'}
assert len(overrides)==6 and all(n.startswith('hyvideo/') and n.endswith('.py') for n in overrides)
assert sum(x['bytes'] for x in overrides.values()) < 256*1024

remote = r'''
import base64,hashlib,json,os,stat
from pathlib import Path,PurePosixPath
root=Path(ROOT)
rows={}
def identity(st):
    return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns
for name,pin in FILES.items():
    rel=PurePosixPath(name)
    assert rel.parts[0]=='hyvideo' and '..' not in rel.parts and rel.suffix=='.py'
    path=root.joinpath(*rel.parts)
    assert path.resolve().is_relative_to(root.resolve())
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        before=os.fstat(f.fileno())
        assert stat.S_ISREG(before.st_mode) and before.st_size==pin['bytes'] and stat.S_IMODE(before.st_mode)==pin['mode']
        raw=f.read(pin['bytes']+1)
        assert identity(before)==identity(os.fstat(f.fileno()))==identity(path.lstat())
    assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    rows[name]=base64.b64encode(raw).decode('ascii')
print(json.dumps(rows))
'''
wire=f"ROOT={original['source_root']!r}\nFILES={overrides!r}\n"+remote
argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A100/'ssh_config'),
      'a100-26','timeout 30 python3 -B -u -']
run=subprocess.run(argv,input=wire.encode(),capture_output=True,timeout=40)
if run.returncode:
    detail=[x for x in ('AssertionError','Permission denied','Connection closed','AttributeError') if x.encode() in run.stderr]
    print(json.dumps(dict(rc=run.returncode,failure_types=detail,stderr_SHA256=sha(run.stderr))))
    raise SystemExit(1)
assert len(run.stdout) < 400*1024
values=json.loads(run.stdout)
assert set(values)==set(overrides)
upstream=BASE/'hunyuan-upstream-60783e7.git'
git=['git','--git-dir',str(upstream)]
revision=comparison['upstream_commit']
assert subprocess.check_output(git+['rev-parse',revision+'^{tree}'],text=True).strip()==comparison['upstream_tree']
raws={}
for name,pin in files.items():
    if name in overrides:
        raw=base64.b64decode(values[name],validate=True)
    else:
        assert comparison['files'][name]['classification']=='bytes_match'
        raw=subprocess.check_output(git+['show',revision+':'+name])
    assert len(raw)==pin['bytes'] and sha(raw)==pin['SHA256']
    if name.endswith('.py'):
        ast.parse(raw,filename=name)
    raws[name]=raw
destination=BASE/'hunyuanvideo-1.5-exact-a100-package-v1'
destination.mkdir()
for name,raw in raws.items():
    p=destination/name
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('xb') as f:f.write(raw)
assert {p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}==set(files)
assert all(sha((destination/n).read_bytes())==pin['SHA256'] for n,pin in files.items())
receipt=dict(upstream_commit=revision,upstream_tree=comparison['upstream_tree'],
    comparison_SHA256=sha(COMPARE.read_bytes()),A100_package_SHA256=sha(record.read_bytes()),
    upstream_files_reused=len(files)-len(overrides),A100_source_files_read=len(overrides),
    A100_source_bytes_transferred=sum(v['bytes'] for v in overrides.values()),files=files,
    parsed_python_files=sum(n.endswith('.py') for n in files),
    exact_bytes_verified=True,target_deployed=False,training_cache_transferred_bytes=0,
    model_weight_transferred_bytes=0,GPU_calls=0)
with (BASE/'hunyuan-exact-source-reconstruction-v1.json').open('x',encoding='utf8') as f:
    json.dump(receipt,f,indent=2)
print(json.dumps({k:v for k,v in receipt.items() if k!='files'}))
