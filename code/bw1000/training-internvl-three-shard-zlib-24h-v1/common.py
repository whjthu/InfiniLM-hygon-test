"""Small metadata/identity primitives, all deadlines tied to one original86400s epoch."""
import hashlib,json,math,os,socket,stat,time
from pathlib import Path
D=Path(__file__).resolve().parent
P=json.loads((D/'PLAN.json').read_bytes())
CONTEXT_KEYS=('publication','request_id','epoch','deadline','source_manifest_SHA256','namespace_nonce')

def need(v,w):
    if not v:raise ValueError(w)

def sha(b):return hashlib.sha256(b).hexdigest()
def canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)+'\n').encode()
def stable(s):return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def base_context(c):return {k:c[k] for k in CONTEXT_KEYS}
def member(c):
    need(c['deadline']==c['epoch']+P['whole_seconds']==c['epoch']+86400 and type(c['file_id'])is int,'ONE_ORIGINAL86400_CONTEXT')
    rows=[r for r in P['files'] if r['file_id']==c['file_id']];need(len(rows)==1,'FROZEN_FILE_ID');return rows[0]

def no_links(p):
    p=Path(p);need(p.is_absolute(),'ABSOLUTE_REGISTERED_PATH')
    for q in [*reversed(p.parents),p]:need(not q.is_symlink(),'REGISTERED_SYMLINK')
    return p

def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def save(path,v,replace=False):
    raw=canonical(v);need(len(raw)<=262144,'SMALL_METADATA_BOUND')
    tmp=path.with_name(path.name+'.tmp-'+str(os.getpid())) if replace else path
    with tmp.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    if replace:os.replace(tmp,path)

def load(path,cap=262144):
    path=no_links(path);a=path.stat();need(stat.S_ISREG(a.st_mode) and a.st_size<=cap,'SMALL_REGULAR_METADATA');raw=path.read_bytes();need(stable(a)==stable(path.stat()),'SMALL_METADATA_STABLE');return json.loads(raw)

def own():
    f=Path('/proc/self/stat').read_text().rsplit(') ',1)[1].split()
    return dict(PID=os.getpid(),PPID=os.getppid(),PGID=os.getpgrp(),startticks=int(f[19]),UID=os.getuid(),host=socket.gethostname(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())

def budget(epoch,offset):
    need(type(epoch)in(int,float) and math.isfinite(epoch) and 0<epoch<=time.time()<epoch+offset,'ORIGINAL86400_DEADLINE')
    end=time.monotonic()+epoch+offset-time.time()
    def left(reserve=0):
        v=min(epoch+offset-time.time(),end-time.monotonic())-reserve;need(v>0,'ORIGINAL_DEADLINE_EXPIRED');return v
    return left

def exact_fd(fd,n,read=None):
    need(type(n)is int and 0<n<=P['block_bytes']+P['compressed_slack_bytes'],'BOUNDED_EXACT_READ');out=bytearray()
    while len(out)<n:
        b=(read or os.read)(fd,n-len(out));need(b,'SHORT_EXACT_INPUT');out.extend(b)
    return bytes(out)

def line_fd(fd,cap=65536):
    b=bytearray()
    while len(b)<cap:
        x=os.read(fd,1)
        if not x:return bytes(b)
        b.extend(x)
        if x==b'\n':return bytes(b)
    raise ValueError('CONTROL_LINE_BOUND')

def control(v):
    raw=canonical(v);need(len(raw)<=65536,'CONTROL_BOUND');used=0
    while used<len(raw):
        n=os.write(1,raw[used:]);need(n>0,'CONTROL_SHORT_WRITE');used+=n

def marker_value(c):return dict(schema=P['schema'],label=P['label'],context=base_context(c),first_receiver_file_id=2,common_root=P['common_root'],modelroot=P['modelroot'])
def verify_marker(c):
    path=no_links(Path(P['common_root'])/'CONTEXT.json');path_before=path.stat();fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_size<=16384 and before.st_uid==154459 and stable(before)==stable(path_before),'COMMON_MARKER_OWNER_REGULAR_BOUND')
        raw=bytearray()
        while len(raw)<before.st_size:
            b=os.read(fd,before.st_size-len(raw));need(b,'COMMON_MARKER_SHORT_EOF');raw.extend(b)
        need(not os.read(fd,1),'COMMON_MARKER_TRAILING');after=os.fstat(fd);path_after=path.stat()
        need(stable(before)==stable(after)==stable(path_after) and raw==canonical(marker_value(c)),'EXACT_CONTEXT_MARKER_OBSERVER_FD_PATH_STABLE')
        return dict(bytes=len(raw),SHA256=sha(raw),owner_UID=before.st_uid,stat=stable(before),stat_after=stable(after),observer=own(),stable_FD_path=True,value=marker_value(c),boundary='This observer saw the exact unpredictable context marker; no cross-observer st_dev/inode equality or all3 payload visibility claim')
    finally:os.close(fd)
