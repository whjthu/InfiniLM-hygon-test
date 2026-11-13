"""Freeze original HunyuanVideo source identities without copying source or assets."""
import hashlib
import json
from pathlib import Path
import subprocess

base = Path(__file__).resolve().parent
repo = Path('C:/Users/WangHaojie/Documents/ChatGPT/海光集群/InfiniLM-hygon-test')
a100 = Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
evidence = repo / 'results/bw1000/2026-11-13/training-hunyuanvideo-sft-original-model-cache-and-target-availability.json'
evidence_raw = evidence.read_bytes()
origin = json.loads(evidence_raw)['source_assets']
lock = json.loads((base/'hygon-training-integration/generation/sources.lock.json').read_bytes())
locked_revision = lock['sources']['hunyuanvideo-1.5']['revision']
program = r'''
import datetime, hashlib, json, os, stat, subprocess
from pathlib import Path, PurePosixPath

root = Path(SOURCE)
def git(*args):
    return subprocess.check_output(['git','-C',str(root),*args],timeout=12,stderr=subprocess.PIPE)
git_metadata=(root/'.git').exists()
try:
    head=git('rev-parse','--verify','HEAD').decode().strip()
except subprocess.CalledProcessError as exc:
    git_reason=('repository_metadata_unavailable' if b'not a git repository' in (exc.stderr or b'') else
                'git_metadata_query_failed')
    head=None
else:
    git_reason=None

def package_inventory():
    selected=[]
    for path in (root/'hyvideo').rglob('*'):
        if path.is_symlink():
            raise ValueError('symlink under native import root')
        if '__pycache__' in path.parts or path.suffix in ('.pyc','.pyo'):
            continue
        if path.is_file():
            selected.append(path.relative_to(root).as_posix())
    selected += [p.name for p in root.glob('requirements*.txt') if p.is_file()]
    selected += [n for n in ('pyproject.toml','setup.py','setup.cfg') if (root/n).is_file()]
    return sorted(selected)

inventory=package_inventory()
assert 3 <= len(inventory) <= 1000 and len(set(inventory))==len(inventory)
files={}
total=0
for name in inventory:
    rel=PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts
    path=root.joinpath(*rel.parts)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        st=os.fstat(f.fileno())
        assert stat.S_ISREG(st.st_mode) and st.st_size<=4*1024**2
        raw=f.read(st.st_size+1)
        def identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        assert identity(st)==identity(os.fstat(f.fileno()))==identity(path.lstat()) and len(raw)==st.st_size
    total+=len(raw)
    assert total<=48*1024**2
    files[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest(),mode=stat.S_IMODE(st.st_mode))
for name,pin in ENTRY.items():
    assert files[name]['SHA256']==pin['SHA256']
assert inventory==package_inventory()
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_root=str(root),head=head,locked_revision=LOCK,revision_matches=head==LOCK if head else None,
    git_metadata=git_metadata,git_reason=git_reason,files=files,bytes_hashed=total,
    scope='hyvideo non-bytecode files and root packaging files only',
    weights_bytes_read=0,cache_bytes_read=0,GPU_calls=0)))
'''
wire = f"SOURCE={origin['source']!r}\nENTRY={origin['entry_source_files']!r}\nLOCK={locked_revision!r}\n" + program
args = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(a100/'ssh_config'),
        'a100-26','timeout 45 python3 -B -u -']
run = subprocess.run(args,input=wire.encode(),capture_output=True,timeout=60)
receipt = dict(rc=run.returncode,asset_evidence_SHA256=hashlib.sha256(evidence_raw).hexdigest(),
               stderr_SHA256=hashlib.sha256(run.stderr).hexdigest())
if run.returncode == 0:
    receipt['value'] = json.loads(run.stdout)
else:
    receipt['failure_types'] = [n for n in ('AssertionError','Connection closed','Permission denied','CalledProcessError','FileNotFoundError') if n.encode() in run.stderr]
with (base/'hunyuan-sft-original-source-package-v2.json').open('x',encoding='utf8') as out:
    json.dump(receipt,out,indent=2)
v=receipt.get('value',{})
print(json.dumps(dict(rc=run.returncode, files=len(v.get('files',{})), **{k:v.get(k) for k in (
    'head','revision_matches','git_metadata','git_reason','bytes_hashed')},
    failure_types=receipt.get('failure_types'))))
