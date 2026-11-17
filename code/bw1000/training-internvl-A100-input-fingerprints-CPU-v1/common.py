"""Bounded stdlib utilities; single client-origin epoch, no SDK or model imports."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

D=Path(__file__).resolve().parent
P=json.loads((D/'PLAN.json').read_bytes())


def require(ok,why):
    if not ok:raise ValueError(why)


def sha(raw):return hashlib.sha256(raw).hexdigest()


def stamp(st):return [st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns]


def safe(path):
    path=Path(path)
    require(path.is_absolute() and '..' not in path.parts,'PATH_FORMAT')
    require(not any(x.is_symlink() for x in (path,*path.parents)),'LINKED_PATH')
    return path


def small(path,limit,pin=None):
    path=safe(path);before=path.lstat()
    require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=limit,'SMALL_FILE_BOUNDS')
    with path.open('rb') as stream:
        require(stamp(os.fstat(stream.fileno()))==stamp(before),'OPEN_IDENTITY')
        raw=stream.read(limit+1)
        require(stamp(os.fstat(stream.fileno()))==stamp(before),'READ_IDENTITY')
    require(stamp(path.lstat())==stamp(before) and len(raw)==before.st_size,'FINAL_IDENTITY')
    if pin:require(len(raw)==pin['bytes'] and sha(raw)==pin['SHA256'],'SMALL_FILE_PIN')
    return raw,stamp(before)


def unique(raw):
    def pairs(items):
        out={}
        for key,value in items:
            require(key not in out,'DUPLICATE_JSON_KEY');out[key]=value
        return out
    return json.loads(raw,object_pairs_hook=pairs)


def save(path,value,limit=128*1024):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    require(len(raw)<=limit,'OUTPUT_BOUNDS')
    with Path(path).open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return dict(bytes=len(raw),SHA256=sha(raw))


class Budget:
    def __init__(self,epoch):
        require(type(epoch) in (int,float) and math.isfinite(epoch) and epoch>0,'EPOCH')
        self.epoch=epoch;self.anchor=time.monotonic();now=time.time()
        self.remaining0=epoch+P['whole_seconds']-now
        self.work_remaining0=epoch+P['work_offset_seconds']-now
        require(0<self.remaining0<=P['whole_seconds'],'EPOCH_EXPIRED_OR_FUTURE')
    def remaining(self,work=False,reserve=0):
        end=self.epoch+(P['work_offset_seconds'] if work else P['whole_seconds'])
        monotonic_cap=self.work_remaining0 if work else self.remaining0
        value=min(end-time.time(),monotonic_cap-(time.monotonic()-self.anchor))-reserve
        require(value>0,'DEADLINE');return value


def owner(pid):
    raw=Path('/proc',str(pid),'stat').read_bytes()
    require(len(raw)<=4096,'PROC_STAT_BOUNDS');fields=raw.decode().rsplit(') ',1)[1].split()
    return dict(PID=pid,PPID=int(fields[1]),PGID=int(fields[2]),startticks=int(fields[19]),UID=os.stat('/proc/'+str(pid)).st_uid)
