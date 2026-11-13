"""Independently stream-hash the exact original HunyuanVideo SFT transformer in place."""
from pathlib import Path
import hashlib,json,subprocess
B=Path(__file__).resolve().parent
A=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline')
first=B/'hunyuan-sft-original-transformer-header-v1.json'
header=json.loads(first.read_bytes())['value']
assert header['stat']['bytes']==33306632192 and header['companion_SHA256']
program=r'''
from pathlib import Path
import datetime,hashlib,json,os,stat,time
p=Path(HEADER['path'])
def identity(s):
    assert stat.S_ISREG(s.st_mode)
    return dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
start=time.monotonic()
h=hashlib.sha256()
with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb') as f:
    before=identity(os.fstat(f.fileno()))
    assert before==HEADER['stat']==identity(p.lstat())
    prefix=f.read(8)
    n=int.from_bytes(prefix,'little')
    assert len(prefix)==8 and n==HEADER['header_bytes']
    raw=f.read(n)
    assert len(raw)==n and hashlib.sha256(raw).hexdigest()==HEADER['header_SHA256']
    h.update(prefix);h.update(raw)
    copied=8+n
    while chunk:=f.read(8*1024**2):
        copied+=len(chunk)
        assert copied<=before['bytes']
        h.update(chunk)
    assert identity(os.fstat(f.fileno()))==identity(p.lstat())==before and copied==before['bytes']
actual=h.hexdigest()
print(json.dumps(dict(observed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    SHA256=actual,companion_matches=actual==HEADER['companion_SHA256'],stat=before,
    complete_bytes_read=copied,wall_seconds=time.monotonic()-start,GPU_calls=0,values_decoded=False)))
'''
args=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-F',str(A/'ssh_config'),
      'a100-26','timeout 200 python3 -B -u -']
r=subprocess.run(args,input=(f'HEADER={header!r}\n'+program).encode(),capture_output=True,timeout=215)
v=dict(rc=r.returncode,stderr_SHA256=hashlib.sha256(r.stderr).hexdigest(),header_receipt_SHA256=hashlib.sha256(first.read_bytes()).hexdigest())
if r.returncode==0:v['value']=json.loads(r.stdout)
else:v['failure_types']=[n for n in ('AssertionError','Connection closed','Permission denied','OSError') if n.encode() in r.stderr]
with (B/'hunyuan-sft-original-transformer-wholefile-v1.json').open('x',encoding='utf8') as f:json.dump(v,f,indent=2)
print(json.dumps(v))
