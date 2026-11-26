"""Parse and qualify all candidate MIOpen map groups without loading any library."""
from hashlib import sha256
import os
from pathlib import Path,PurePosixPath
import re
import stat
import time


EXPECTED=Path('/opt/dtk/lib/libMIOpen.so.1.0')
EXPECTED_PIN=dict(bytes=18705784,SHA256='1b0c36a8528fccec6abbe9db907526b75713da7fc4346c9eea9d1264cfc06cba')
PREFIXES=('/opt/dtk/lib/','/opt/dtk/lib64/','/opt/rocm/lib/','/opt/rocm/lib64/',
          '/usr/local/lib/','/training-runtime/site/','/.singularity.d/libs/')
FAMILY=re.compile(r'libMIOpen\.so(?:\.[0-9]{1,8}){0,4}')
ROW=re.compile(r'^([0-9a-f]+)-([0-9a-f]+)\s+([r-][w-][x-][ps])\s+([0-9a-f]+)\s+'
               r'([0-9a-f]+):([0-9a-f]+)\s+([0-9]+)\s+(/.*)$')
MAX_MAPS=4<<20
MAX_GROUPS=24
MAX_SEGMENTS=192
MAX_OBJECT=128<<20


def sha(raw):return sha256(raw).hexdigest()


def allowed(path):
    posix=PurePosixPath(path)
    return (posix.is_absolute() and '..' not in posix.parts and posix.as_posix()==path
            and any(path.startswith(prefix) for prefix in PREFIXES))


def metadata(name):
    basename=PurePosixPath(name).name
    return dict(path_SHA256=sha(name.encode('utf8','surrogateescape')),basename=basename if FAMILY.fullmatch(basename) else None,
                raw_path_is_expected=name==EXPECTED.as_posix(),trusted_read_root=allowed(name))


def inspect_maps(raw):
    """Return internal canonical candidate groups; raw path is never serialized."""
    if not 0<len(raw)<=MAX_MAPS:
        raise ValueError('BOUNDED_CURRENT_PROCESS_MAPS_EXTENT')
    found={}
    invalid=[]
    related=0
    segments=0
    for line in raw.decode('utf8','surrogateescape').splitlines():
        name=line.split(maxsplit=5)[-1]
        deleted=name.endswith(' (deleted)')
        path=name[:-10] if deleted else name
        basename=PurePosixPath(path).name
        if basename=='libMIOpen-recommend.so':
            related+=1
        if not FAMILY.fullmatch(basename):
            continue
        match=ROW.fullmatch(line)
        segments+=1
        if segments>MAX_SEGMENTS:
            raise ValueError('BOUNDED_PROVIDER_SEGMENTS')
        if not match or not path.startswith('/') or int(match[7])==0 or int(match[1],16)>=int(match[2],16):
            invalid.append(dict(segment_SHA256=sha(line.encode('utf8','surrogateescape')),**metadata(path)))
            continue
        key=(int(match[5],16),int(match[6],16),int(match[7]))
        current=found.setdefault(key,dict(identity=list(key),paths={},segments=0,executable_segments=0))
        if len(found)>MAX_GROUPS:
            raise ValueError('BOUNDED_PROVIDER_GROUP_COUNT')
        evidence=current['paths'].setdefault(path,dict(path=path,deleted_segments=0,segments=0,executable=0))
        evidence['deleted_segments']+=int(deleted)
        evidence['segments']+=1
        evidence['executable']+=int('x' in match[3])
        current['segments']+=1
        current['executable_segments']+=int('x' in match[3])
    return list(found.values()),invalid,related


def group_identity(info):
    return [os.major(info.st_dev),os.minor(info.st_dev),info.st_ino]


def stamp(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
