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
    return subprocess.check_output(['git','-C',str(root),*args],timeout=12)
def snapshot():
    entries = git('ls-files','-z','--stage','--','hyvideo','requirements*.txt','pyproject.toml','setup.py','setup.cfg').split(b'\0')
    return [entry for entry in entries if entry]
head = git('rev-parse','HEAD').decode().strip()
tree = git('rev-parse','HEAD^{tree}').decode().strip()
assert git('rev-parse','--show-object-format').strip()==b'sha1'
index=snapshot()
assert 3 <= len(index) <= 1000
files={}
total=0
for item in index:
    spec, name = item.split(b'\t',1)
    mode, blob, stage = spec.decode().split()
    name = name.decode()
    rel = PurePosixPath(name)
    assert not rel.is_absolute() and '..' not in rel.parts and stage=='0' and mode in ('100644','100755')
    path = root.joinpath(*rel.parts)
    with os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
        st=os.fstat(f.fileno())
        assert stat.S_ISREG(st.st_mode) and st.st_size<=4*1024**2
        raw=f.read(st.st_size+1)
        after=os.fstat(f.fileno())
        current=path.lstat()
        def identity(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        assert identity(st)==identity(after)==identity(current) and len(raw)==st.st_size
    total+=len(raw)
    assert total<=48*1024**2 and name not in files
    sha=hashlib.sha256(raw).hexdigest()
    git_blob=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
    files[name]=dict(bytes=len(raw),SHA256=sha,index_blob=blob,index_matches_worktree=git_blob==blob,mode=mode)
for name,pin in ENTRY.items():
    assert files[name]['SHA256']==pin['SHA256']
status_raw=git('status','--porcelain=v1','-z','--untracked-files=all','--','hyvideo','requirements*.txt','pyproject.toml','setup.py','setup.cfg')
status_rows=[s for s in status_raw.split(b'\0') if s]
# Untracked Python files under the import root can change module resolution.
untracked_py=[s[3:].decode() for s in status_rows if s.startswith(b'?? hyvideo/') and s.endswith(b'.py')]
assert index==snapshot() and head==git('rev-parse','HEAD').decode().strip()
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_root=str(root),head=head,tree=tree,locked_revision=LOCK,revision_matches=head==LOCK,
    index_matches_worktree=all(f['index_matches_worktree'] for f in files.values()),scope_status_records=len(status_rows),
    scoped_status_SHA256=hashlib.sha256(status_raw).hexdigest(),untracked_python=untracked_py,
    files=files,bytes_hashed=total,weights_bytes_read=0,cache_bytes_read=0,GPU_calls=0)))
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
with (base/'hunyuan-sft-original-source-closure-v1.json').open('x',encoding='utf8') as out:
    json.dump(receipt,out,indent=2)
v=receipt.get('value',{})
print(json.dumps(dict(rc=run.returncode, files=len(v.get('files',{})), **{k:v.get(k) for k in (
    'head','revision_matches','index_matches_worktree','scope_status_records','untracked_python','bytes_hashed')},
    failure_types=receipt.get('failure_types'))))
