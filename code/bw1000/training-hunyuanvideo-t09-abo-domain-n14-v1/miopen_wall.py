"""Gemma GPU child gate for an exact read-only MIOpen ELF and isolated cold state."""
import errno
from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import sys
import time


R=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014')
ORIGINAL=R/'incoming/t09-miopen-single-bind-c6025678606b/work-once/provider/libMIOpen.so.1.0'
LIB=Path('/opt/dtk/lib/libMIOpen.so.1.0')
HOME=Path('/miopen-private-home-v4')
REST=('.config','.config/miopen','.config/miopen/user-db','.config/miopen/miopen-lockfiles','.cache','.cache/miopen')
PIN={'bytes':18705784,'SHA256':'1b0c36a8528fccec6abbe9db907526b75713da7fc4346c9eea9d1264cfc06cba'}


def need(flag,why):
    if not flag:
        raise ValueError(why)


def key(st):
    return [st.st_dev,st.st_ino]


def stamp(st):
    return key(st)+[st.st_size,st.st_mode,st.st_uid,st.st_mtime_ns,st.st_ctime_ns]


def digest(fd,deadline,mirror=None):
    before=os.fstat(fd)
    need(stat.S_ISREG(before.st_mode) and before.st_size==PIN['bytes'],'EXACT_QUALIFIED_MIOPEN_EXTENT')
    read=0
    body=sha256()
    while True:
        need(time.time()<deadline,'NATIVE_MIOPEN_SHA_ORIGINAL_EPOCH')
        raw=os.read(fd,1<<20)
        if not raw:
            break
        read+=len(raw)
        need(read<=PIN['bytes'],'BOUNDED_LOCAL_MIOPEN_FD')
        body.update(raw)
        if mirror is not None:
            need(mirror.write(raw)==len(raw),'SHORT_PRIVATE_MIOPEN_COPY')
    need(dict(bytes=read,SHA256=body.hexdigest())==PIN and stamp(before)==stamp(os.fstat(fd)),
         'EXACT_STABLE_MIOPEN_ELF_SHA')
    return before


def open_exact(path):
    need(path.resolve(strict=True)==path,'EXACT_CANONICAL_MIOPEN_BIND_PATH')
    return os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK|os.O_CLOEXEC)


def fsync_dir(path):
    fd=os.open(path,os.O_DIRECTORY|os.O_NOFOLLOW|os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def stage(work,deadline):
    work.mkdir(mode=0o700)
    home=work/'private-home'
    home.mkdir(mode=0o700)
    for relative in REST:
        (home/relative).mkdir(mode=0o700)
    parent=work/'provider'
    parent.mkdir(mode=0o700)
    provider=parent/LIB.name
    src=open_exact(ORIGINAL)
    try:
        with provider.open('xb') as stream:
            before=digest(src,deadline,stream)
            need(stamp(ORIGINAL.stat())==stamp(before) and before.st_uid==os.getuid()
                 and stat.S_IMODE(before.st_mode)==0o400,'ORIGINAL_3600_SEALED_OWNER_COPY')
            stream.flush()
            os.fchmod(stream.fileno(),0o400)
            os.fsync(stream.fileno())
    finally:os.close(src)
    fsync_dir(parent)
    fsync_dir(work)
    fd=open_exact(provider)
    try:
        actual=digest(fd,deadline)
        need(actual.st_uid==os.getuid() and actual.st_nlink==1 and stat.S_IMODE(actual.st_mode)==0o400
             and stamp(provider.stat())==stamp(actual),'NEW_N12_SEALED_PRIVATE_ELF_RECHECK')
    finally:os.close(fd)
    return dict(provider=PIN,provider_dev_inode=key(actual),home_dev_inode=key(home.stat()),current_uid=os.getuid())


def maps_pin():
    from maps_probe import MAX_MAPS, inspect_maps

    with Path('/proc/self/maps').open('rb') as stream:
        raw=stream.read(MAX_MAPS+1)
    groups,malformed,related=inspect_maps(raw)
    rows=[{'identity':group['identity'],'executable_segments':group['executable_segments'],
           'segments':group['segments'],'deleted_segments':sum(item['deleted_segments'] for item in group['paths'].values())}
          for group in groups]
    return dict(groups=rows,malformed=len(malformed),related_segments=related,SHA256=sha256(raw).hexdigest())


def verify_mapping(before,after,identity):
    execs=[item for item in after['groups'] if item['executable_segments']>0]
    need(not before['groups'] and not before['malformed'] and not after['malformed'],
         'NO_PRIOR_OR_MALFORMED_MIOPEN_MAPPING')
    need(len(execs)==1 and execs[0]['identity']==identity and execs[0]['deleted_segments']==0,
         'ONE_ACTUAL_EXECUTABLE_MIOPEN_MAP_MATCHES_EXACT_BOUND_FD')


def actual_process_import(host,deadline):
    from mount_observation import explicit_readonly, observe

    need('torch' not in sys.modules and os.environ.get('HOME')==str(HOME)
         and os.environ.get('MIOPEN_USER_DB_PATH')==str(HOME/'.config/miopen/user-db')
         and os.environ.get('MIOPEN_CUSTOM_CACHE_DIR')==str(HOME/'.cache/miopen')
         and not any(os.environ.get(name) for name in
         ('LD_PRELOAD','LD_AUDIT','LD_PROFILE','XDG_CONFIG_HOME','XDG_CACHE_HOME','XDG_DATA_HOME',
          'MIOPEN_SYSTEM_DB_PATH','MIOPEN_LOCKFILE_DIR','MIOPEN_LOCKFILE_PATH')),
         'NORMAL_PRODUCTION_TORCH_SEARCH_AND_PRIVATE_HOME')
    home,lib=[observe(path) for path in (HOME,LIB)]
    need(home['at_exact_point'] is True and home['same_before_open_after_dev_inode'] is True
         and home['permissions']==0o700 and home['owner']==host['current_uid']==os.getuid()
         and home['three_reads']==['RW','RW','RW'] and home['dev_inode']==host['home_dev_inode']
         and all((HOME/name).resolve(strict=True)==HOME/name
                 and (HOME/name).stat().st_uid==os.getuid() and stat.S_IMODE((HOME/name).stat().st_mode)==0o700
                 for name in REST) and all(not list((HOME/name).iterdir()) for name in (REST[2],REST[3],REST[-1])),
         'EXACT_N12_COLD_PRIVATE_0700_HOME')
    need(explicit_readonly(lib) and lib['object_type']=='REGULAR' and lib['dev_inode']==host['provider_dev_inode']
         and lib['permissions']==0o400 and lib['owner']==os.getuid() and host['provider']==PIN,
         'N12_EFFECTIVE_CANONICAL_SINGLE_FILE_BIND_READONLY')
    fd=open_exact(LIB)
    try:
        start=digest(fd,deadline)
        need(key(start)==lib['dev_inode'],'N12_BOUND_FILE_IS_EXACT_HOST_PRIVATE_OBJECT')
        before=maps_pin()
        import torch

        need(torch.version.hip and not torch.version.cuda,'ACTUAL_GEMMA_HIP_TORCH_REQUIRED')
        after=maps_pin()
        os.lseek(fd,0,os.SEEK_SET)
        end=digest(fd,deadline)
        need(stamp(start)==stamp(end)==stamp(LIB.stat()),'N12_STABLE_SAME_FD_BEFORE_AFTER_TORCH_IMPORT')
        mapping=[os.major(start.st_dev),os.minor(start.st_dev),start.st_ino]
        verify_mapping(before,after,mapping)
    finally:os.close(fd)
    receipt=dict(provider=PIN,provider_dev_inode=key(start),bound_map_identity=mapping,readonly=lib['three_reads'],
        home_dev_inode=home['dev_inode'],private_home_cold_at_start=True,maps_before=before,maps_after=after,
        mapped_exact_same_FD_verified=True,native_DB_lock_cache_access_observed=False,process_pid=os.getpid())
    return torch,receipt


def readonly_failure(error):
    text=str(error).lower()
    return isinstance(error,OSError) and error.errno==errno.EROFS or 'read-only file system' in text or 'erofs' in text


class NativeReadonlyMIOpenError(RuntimeError):
    def __init__(self):
        super().__init__('native MIOpen EROFS verified in own stderr')


def stderr_cursor(path):
    before=path.lstat()
    need(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and before.st_size<=16*1024**2,
         'EXACT_HELD_NATIVE_STDERR_ONLY')
    return key(before),before.st_size


def stderr_readonly(path,cursor):
    ident,start=cursor
    fd=open_exact(path)
    try:
        before=os.fstat(fd)
        need(key(before)==ident and start<=before.st_size<=16*1024**2
             and before.st_size-start<=512*1024,'BOUNDED_CURRENT_SLOT_NATIVE_STDERR')
        os.lseek(fd,start,os.SEEK_SET)
        data=bytearray()
        remaining=before.st_size-start
        while remaining:
            chunk=os.read(fd,min(remaining,64*1024))
            need(bool(chunk),'NATIVE_STDERR_SHORT_READ')
            data.extend(chunk)
            remaining-=len(chunk)
        need(key(os.fstat(fd))==key(path.lstat())==ident,'NATIVE_STDERR_OWNER_IDENTITY_CHANGED')
    finally:os.close(fd)
    return private_readonly_record(data)


def private_readonly_record(raw):
    return any(b'miopen' in line and (b'read-only file system' in line or b'erofs' in line)
               for line in raw.lower().splitlines())


def continue_after_error(arm,ordinal,error):
    return (arm,ordinal)!=('base',0) and not isinstance(error,TimeoutError) and not readonly_failure(error)


def bind_first_success(output,record,pin):
    path=output/'base-return-00.json'
    payload=path.read_bytes()
    need(pin==dict(bytes=len(payload),SHA256=sha256(payload).hexdigest())
         and json.loads(payload)==record and record['status']=='COMPLETED'
         and record['arm']=='base' and record['ordinal']==0,'EXACT_PERSISTED_FIRST_BLIND_SLOT')
    fsync_dir(output)
    return pin
