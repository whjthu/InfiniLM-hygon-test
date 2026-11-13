"""Open and close each enumerated verbs context without allocating queues or contacting peers."""
import ctypes as c
import errno
import hashlib,json,os,stat
from pathlib import Path

def observe():
    nodes={}
    for path in sorted(Path('/dev/infiniband').glob('uverbs*')):
        meta=path.stat()
        if not stat.S_ISCHR(meta.st_mode):
            continue
        try:
            fd=os.open(path,os.O_RDWR|os.O_CLOEXEC)
            os.close(fd)
            error=None
        except OSError as exc:
            error=exc.errno
        nodes[path.name]=dict(major=os.major(meta.st_rdev),minor=os.minor(meta.st_rdev),open_errno=error)
    sysfs={}
    for device in sorted(Path('/sys/class/infiniband').glob('*')):
        sysfs[device.name]={p.name:(p/'state').read_text().strip() for p in sorted((device/'ports').glob('*'))}
    verbs=dict(library=None,devices=None)
    try:
        lib=c.CDLL('libibverbs.so.1',use_errno=True)
        verbs['library']='libibverbs.so.1'
        lib.ibv_get_device_list.argtypes=[c.POINTER(c.c_int)]
        lib.ibv_get_device_list.restype=c.POINTER(c.c_void_p)
        lib.ibv_get_device_name.argtypes=[c.c_void_p]
        lib.ibv_get_device_name.restype=c.c_char_p
        lib.ibv_free_device_list.argtypes=[c.POINTER(c.c_void_p)]
        lib.ibv_open_device.argtypes=[c.c_void_p]
        lib.ibv_open_device.restype=c.c_void_p
        lib.ibv_close_device.argtypes=[c.c_void_p]
        lib.ibv_close_device.restype=c.c_int
        count=c.c_int()
        c.set_errno(0)
        devices=lib.ibv_get_device_list(c.byref(count))
        if not devices:
            verbs['enumeration_errno']=c.get_errno()
        else:
            try:
                assert 0<=count.value<=64
                verbs['devices']=[lib.ibv_get_device_name(devices[i]).decode() for i in range(count.value)]
                contexts={}
                for device,name in zip(devices[:count.value],verbs['devices'],strict=True):
                    c.set_errno(0)
                    context=lib.ibv_open_device(device)
                    open_errno=c.get_errno() if not context else None
                    close_rc=None
                    close_errno=None
                    if context:
                        c.set_errno(0)
                        close_rc=lib.ibv_close_device(context)
                        close_errno=c.get_errno() if close_rc else None
                    contexts[name]=dict(opened=bool(context),open_errno=open_errno,
                        open_error_name=errno.errorcode.get(open_errno),close_rc=close_rc,close_errno=close_errno)
                verbs['contexts']=contexts
            finally:
                lib.ibv_free_device_list(devices)
    except OSError:
        verbs['error']='LIBIBVERBS_LOAD_FAILED'
    except AttributeError:
        verbs['error']='LIBIBVERBS_API_UNAVAILABLE'
    providers={}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6 or not fields[5].startswith('/'):
            continue
        path=Path(fields[5])
        if path.name.startswith(('libibverbs','libshca','libnl-')) and str(path) not in providers:
            assert 0<path.stat().st_size<=32*1024**2
            raw=path.read_bytes()
            providers[str(path)]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
    return dict(uverbs=nodes,sysfs=sysfs,verbs=verbs,mapped_provider_files=providers)

if __name__=='__main__':
    print(json.dumps(observe()),flush=True)
