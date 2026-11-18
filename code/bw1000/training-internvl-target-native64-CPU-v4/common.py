"""One new ordinary CPU qualification; historical unknown operations stay untouched."""
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import time

D=Path(__file__).resolve().parent
P=json.loads((D/'PLAN.json').read_bytes())
RUN=Path(P['RUN'])
COLS='JobIDRaw,JobName%100,State%50,ExitCode,ElapsedRaw,NodeList%500,AllocTRES%200'

def need(ok,why):
    if not ok:raise ValueError(why)
require=need

def sha(raw):return hashlib.sha256(raw).hexdigest()

def read(path):
    path=Path(path);need(path.is_file() and not path.is_symlink() and path.stat().st_size<=2*1024**2,'bounded JSON source/receipt')
    return json.loads(path.read_bytes())

def pin(path):
    path=Path(path);before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_size<=32*1024**2,'bounded regular source/metadata/input')
    raw=path.read_bytes();after=path.lstat()
    need(all(getattr(before,k)==getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),'file changed while pinning')
    return dict(bytes=len(raw),SHA256=sha(raw))

def canonical(value):return json.dumps(value,sort_keys=True,allow_nan=False).encode()+b'\n'

def save(path,value,replace=False):
    path=Path(path);raw=canonical(value);need(len(raw)<=2*1024**2,'bounded private result')
    target=path.with_suffix('.tmp') if replace else path
    with target.open('wb' if replace else 'xb') as stream:stream.write(raw)
    if replace:target.replace(path)

def admission():
    c=read(D/'CLAIM.json')
    need(c['root_GO'] is c['CPU_GO'] is True,'new ordinary CPU scope not adopted')
    need(c['GPU_GO'] is c['Model_GO'] is c['training_GO'] is False,'CPU qualification only')
    return c

def verify():
    for name,expected in read(D/'SOURCE_FILES.json')['files'].items():
        part=PurePosixPath(name);need(not part.is_absolute() and '..' not in part.parts and part.as_posix()==name,'safe source member')
        need(pin(D/name)==expected,'published source changed: '+name)
    return pin(D/'SOURCE_FILES.json')

def clock(epoch):
    now=time.time();need(math.isfinite(epoch) and 0<epoch<=now<epoch+210,'original 300 epoch worker launch exhausted')
    return dict(epoch=epoch,deadline=epoch+300,work_deadline=epoch+210)

def remaining(deadline,mono,reserve=0):
    left=min(deadline-time.time(),mono-time.monotonic())-reserve
    need(left>0,'original epoch exhausted');return left
