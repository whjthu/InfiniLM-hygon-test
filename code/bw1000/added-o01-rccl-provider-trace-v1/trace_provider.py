"""Record the actual rank's library mappings and inspect existing verbs handles."""
import ctypes as c
import errno
import hashlib
import os
from pathlib import Path

PREFIXES=('libibverbs','libshca','libnl-','librccl','libnccl','libmlx5')

def providers():
    paths=set()
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6 or not fields[5].startswith('/'):
            continue
        path=Path(fields[5])
        if path.name.startswith(PREFIXES):
            paths.add(path)
    return sorted(paths)

def capture():
    found={}
    for path in providers():
        try:
            info=path.stat()
            if not 0<info.st_size<=256*1024**2:
                found[str(path)]={'bytes':info.st_size,'identity':'outside byte cap'}
                continue
            digest=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda: f.read(4*1024**2),b''):
                    digest.update(chunk)
            digest=digest.hexdigest()
            found[str(path)]={'bytes':info.st_size,'SHA256':digest}
        except OSError as exc:
            found[str(path)]={'errno':exc.errno}
    return found

def open_from_mapped_verbs():
    result={}
    for path in providers():
        if not path.name.startswith('libibverbs'):
            continue
        try:
            lib=c.CDLL(str(path),mode=os.RTLD_NOLOAD|os.RTLD_NOW,use_errno=True)
            for name,argtypes,restype in (
                ('ibv_get_device_list',[c.POINTER(c.c_int)],c.POINTER(c.c_void_p)),
                ('ibv_get_device_name',[c.c_void_p],c.c_char_p),
                ('ibv_open_device',[c.c_void_p],c.c_void_p),
                ('ibv_close_device',[c.c_void_p],c.c_int),
                ('ibv_free_device_list',[c.POINTER(c.c_void_p)],None),
            ):
                fn=getattr(lib,name)
                fn.argtypes=argtypes
                fn.restype=restype
            count=c.c_int()
            c.set_errno(0)
            devices=lib.ibv_get_device_list(c.byref(count))
            if not devices:
                result[str(path)]={'enumeration_errno':c.get_errno()}
                continue
            contexts={}
            try:
                assert 0<=count.value<=64
                for index in range(count.value):
                    device=devices[index]
                    name=lib.ibv_get_device_name(device).decode()
                    c.set_errno(0)
                    context=lib.ibv_open_device(device)
                    error=None if context else c.get_errno()
                    closed=lib.ibv_close_device(context) if context else None
                    contexts[name]=dict(opened=bool(context),errno=error,error=errno.errorcode.get(error),close_rc=closed)
            finally:
                lib.ibv_free_device_list(devices)
            result[str(path)]={'contexts':contexts}
        except (OSError,AttributeError) as exc:
            result[str(path)]={'error_type':type(exc).__name__,'errno':getattr(exc,'errno',None)}
    return result
