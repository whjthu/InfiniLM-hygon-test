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


def read_candidate(entry,group,deadline,*,authorized_aliases):
    name=entry['path']
    projected=dict(metadata(name),status='NOT_READ')
    if entry['deleted_segments']:
        return dict(projected,status='DELETED_MAPPING_PATH_NOT_OPENED')
    if name not in authorized_aliases or not allowed(name):
        return dict(projected,status='MAPPED_PATH_NOT_EXACTLY_AUTHORIZED')
    try:
        named=Path(name)
        resolved=named.resolve(strict=True)
        projected.update(resolved_path_SHA256=sha(str(resolved).encode('utf8','surrogateescape')),
                         resolves_to_expected=resolved==EXPECTED)
        if resolved!=EXPECTED:
            return dict(projected,status='RESOLVED_PATH_NOT_EXACT_FROZEN_PROVIDER')
        before=resolved.lstat()
        if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=MAX_OBJECT:
            return dict(projected,status='UNACCEPTABLE_LIBRARY_TYPE_OR_EXTENT')
        projected['path_dev_inode']=group_identity(before)
        if group_identity(before)!=group:
            return dict(projected,status='PATH_STAT_DOES_NOT_IDENTIFY_MAPPED_GROUP')
        fd=os.open(resolved,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK)
        with os.fdopen(fd,'rb') as stream:
            opened=os.fstat(stream.fileno())
            if stamp(opened)!=stamp(before) or group_identity(opened)!=group:
                return dict(projected,status='OPENED_FD_DOES_NOT_IDENTIFY_MAPPED_GROUP')
            content=sha256()
            size=0
            for block in iter(lambda:stream.read(1<<20),b''):
                if time.monotonic()>=deadline:
                    return dict(projected,status='ORIGINAL_HASH_BUDGET_EXHAUSTED')
                size+=len(block)
                if size>before.st_size:
                    return dict(projected,status='EXACT_FILE_EXTENT_CHANGED')
                content.update(block)
            if stamp(os.fstat(stream.fileno()))!=stamp(before):
                return dict(projected,status='FD_LIFETIME_CHANGED_DURING_HASH')
        if size!=before.st_size or stamp(resolved.lstat())!=stamp(before) or named.resolve(strict=True)!=resolved:
            return dict(projected,status='PATH_LIFETIME_CHANGED_DURING_HASH')
        return dict(projected,status='EXACT_MAPPED_INODE_FD_HASHED',pin=dict(bytes=size,SHA256=content.hexdigest()))
    except (OSError,ValueError,RuntimeError) as error:
        return dict(projected,status='MAPPED_OBJECT_OPEN_OR_READ_ERROR',error_type=type(error).__name__,
                    errno=getattr(error,'errno',None))


def select(groups,malformed_count):
    """Select only a unique executable exact group without competing executable objects."""
    if len({tuple(group['identity']) for group in groups})!=len(groups):
        return dict(state='DUPLICATE_MAP_GROUP_INPUT',selected=None)
    executable=[group for group in groups if group['executable_segments']>0]
    exact=[]
    unverified=[]
    other=[]
    for group in executable:
        observations=[item['pin'] for item in group['candidates'] if item['status']=='EXACT_MAPPED_INODE_FD_HASHED']
        if not observations:
            unverified.append(group)
        elif not all(example==observations[0] for example in observations):
            return dict(state='MAPPED_INODE_ALIAS_HASH_CONFLICT',selected=None)
        elif observations[0]==EXPECTED_PIN:
            exact.append(group)
        else:
            other.append(group)
    if malformed_count:
        return dict(state='MALFORMED_PROVIDER_MAPPING_ROWS',selected=None)
    if not groups:
        return dict(state='NO_MIOPEN_MAPPING',selected=None)
    if not executable:
        return dict(state='NO_EXECUTABLE_PROVIDER_MAPPING',selected=None)
    if len(exact)>1:
        return dict(state='MULTIPLE_EXECUTABLE_EXACT_GROUPS',selected=None)
    if exact and other:
        return dict(state='COMPETING_OTHER_EXECUTABLE_GROUP',selected=None)
    if unverified:
        return dict(state='EXECUTABLE_PROVIDER_GROUP_UNVERIFIED',selected=None)
    if exact:
        return dict(state='ONE_EXECUTABLE_EXACT_MIOPEN_GROUP',selected=exact[0]['identity'])
    return dict(state='NO_EXECUTABLE_EXACT_PROVIDER_BYTES',selected=None)


def capture(raw,deadline,*,reader):
    grouped,malformed,related=inspect_maps(raw)
    result=[]
    for group in grouped:
        identities=group['identity']
        candidates=[]
        for item in group['paths'].values():
            observable=reader(item,identities,deadline)
            candidates.append(dict(observable,mapped_segments=item['segments'],
                executable_segments=item['executable'],deleted_segments=item['deleted_segments']))
        result.append(dict(identity=identities,segments=group['segments'],
            executable_segments=group['executable_segments'],candidates=candidates))
    return dict(groups=result,malformed=malformed,related_non_provider_segments=related,maps_sha256=sha(raw),
                selection=select(result,len(malformed)))
