import hashlib,json,math,os,socket,time
from pathlib import Path
D=Path(__file__).resolve().parent
P=json.loads((D/'PLAN.json').read_bytes());RUN=Path(P['RUN'])
def need(ok,why):
    if not ok:raise ValueError(why)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def save(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode();need(len(raw)<=262144,'RECEIPT_BOUND')
    tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def load(path,cap=262144):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'SMALL_REGULAR_JSON')
    return json.loads(path.read_bytes())
def own():
    raw=Path('/proc/self/stat').read_text();v=raw.rsplit(') ',1)[1].split()
    return dict(PID=os.getpid(),PPID=os.getppid(),PGID=os.getpgrp(),startticks=int(v[19]),UID=os.getuid(),host=socket.gethostname(),boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
def budget(epoch,offset):
    now=time.time();need(type(epoch) in (int,float) and math.isfinite(epoch) and 0<epoch<=now<epoch+offset,'ORIGINAL_EPOCH_EXHAUSTED')
    mono=time.monotonic()+epoch+offset-now
    def left(reserve=0):
        t=min(epoch+offset-time.time(),mono-time.monotonic())-reserve;need(t>0,'ORIGINAL_EPOCH_EXHAUSTED');return t
    return left
