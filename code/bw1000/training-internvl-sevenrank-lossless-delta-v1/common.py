"""Small source/receipt primitives; payload is never serialized through JSON."""
import hashlib,json,math,os,socket,stat,time
from pathlib import Path
D=Path(__file__).resolve().parent
P=json.loads((D/'PLAN.json').read_bytes())

def need(ok,why):
    if not ok:raise ValueError(why)

def sha(raw):return hashlib.sha256(raw).hexdigest()

def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()

def stable(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def no_links(p):
    p=Path(p);need(p.is_absolute(),'ABSOLUTE_REGISTERED_PATH')
    for q in [*reversed(p.parents),p]:need(not q.is_symlink(),'REGISTERED_PATH_SYMLINK')
    return p

def save(path,value,replace=False):
    raw=canonical(value);need(len(raw)<=262144,'SMALL_RECEIPT_BOUND')
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid())) if replace else path
    with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if replace:os.replace(tmp,path)

def load(path,cap=262144):
    need(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'SMALL_REGULAR_METADATA')
    return json.loads(path.read_bytes())

def own():
    fields=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    return dict(PID=os.getpid(),PPID=os.getppid(),PGID=os.getpgrp(),startticks=int(fields[19]),UID=os.getuid(),host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

def budget(epoch,offset):
    need(type(epoch)in(int,float) and math.isfinite(epoch) and 0<epoch<=time.time()<epoch+offset,'ORIGINAL900_EPOCH')
    end=time.monotonic()+epoch+offset-time.time()
    def left(reserve=0):
        v=min(epoch+offset-time.time(),end-time.monotonic())-reserve;need(v>0,'ORIGINAL_DEADLINE_EXPIRED');return v
    return left

def control(value):
    import sys
    raw=canonical(value);need(len(raw)<=65536,'CONTROL_BOUND');sys.stdout.buffer.write(raw);sys.stdout.buffer.flush()
