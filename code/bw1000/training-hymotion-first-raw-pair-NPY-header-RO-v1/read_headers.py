"""Explicit NPY magic/header reads only; never touch the array body."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import stat
import time


def main(binding,context):
    assert binding['root_GO'] is True and binding['RO_GO'] is True
    assert binding['GPU_GO'] is False and binding['array_body_GO'] is False
    deadline=context['epoch']+30;assert 0<=time.time()-context['epoch']<25
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    allowed=sorted(os.sched_getaffinity(0)-{0,1});assert len(allowed)>=2;os.sched_setaffinity(0,allowed[:2])
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    H=lambda b:hashlib.sha256(b).hexdigest();identity=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
    result=dict(context=context,sample_ID_SHA256=binding['sample_ID_SHA256'],planned_files=2,records={},
                explicit_bytes=0,explicit_array_body_bytes=0,np_load_Torch_or_GPU=False,
                whole_source_SHA256_verified=False,cache128_or_training_READY=False,
                physical_IO='UNKNOWN_OS_CACHE_AND_READAHEAD',header_pair_valid=False,source_min_frames=None,qualifies_true128=False)
    for role,pin in binding['files'].items():
        row=dict(path=pin['path'],manifest_whole_SHA256_reference=pin['manifest_whole_SHA256_reference'],
                 actual_whole_SHA256_verified=False,state='UNKNOWN',explicit_bytes=0)
        result['records'][role]=row;parts=[]
        try:
            path=Path(pin['path']);s=path.lstat();assert stat.S_ISREG(s.st_mode) and not path.is_symlink()
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
            with os.fdopen(fd,'rb',buffering=0) as stream:
                before=os.fstat(stream.fileno())
                def take(n):
                    assert 0<n<=4096-row['explicit_bytes']
                    data=b''
                    while len(data)<n:
                        block=stream.read(n-len(data))
                        assert block,'TRUNCATED_HEADER'
                        data+=block;row['explicit_bytes']+=len(block);result['explicit_bytes']+=len(block)
                    parts.append(data);return data
                assert take(6)==b'\x93NUMPY','NOT_NPY'
                major,minor=take(2);assert (major,minor) in ((1,0),(2,0),(3,0)),'NPY_VERSION'
                size=2 if major==1 else 4
                header_len=int.from_bytes(take(size),'little')
                assert 0<header_len<=4096-row['explicit_bytes'],'HEADER_CAP'
                header=take(header_len);assert header.endswith(b'\n'),'HEADER_NEWLINE'
                after=os.fstat(stream.fileno())
            assert identity(s)==identity(before)==identity(after)==identity(path.stat()),'SOURCE_CHANGED'
            parsed=ast.literal_eval(header.decode('utf-8' if major==3 else 'latin1').strip())
            assert type(parsed)is dict and set(parsed)=={'descr','fortran_order','shape'},'HEADER_SCHEMA'
            shape=parsed['shape'];dtype=parsed['descr']
            assert type(shape)is tuple and len(shape)==len(pin['shape_tail'])+1 and all(type(x)is int and x>0 for x in shape),'HEADER_SHAPE'
            assert list(shape[1:])==pin['shape_tail'] and shape[0]<=10000000,'SOURCE_FEATURE_SHAPE'
            assert type(dtype)is str and re.fullmatch(r'[<>=]f[248]',dtype),'NONFLOAT_OR_OBJECT_DTYPE'
            assert type(parsed['fortran_order'])is bool,'ORDER_TYPE'
            data_bytes=math.prod(shape)*int(dtype[-1])
            assert s.st_size==row['explicit_bytes']+data_bytes,'TRUNCATED_OR_TRAILING_FILE_LAYOUT'
            row.update(state='HEADER_VALID',NPY_version=[major,minor],shape=list(shape),dtype=dtype,
                       fortran_order=parsed['fortran_order'],header_bytes=row['explicit_bytes'],header_SHA256=H(b''.join(parts)),
                       file_stat_identity=identity(after),declared_array_bytes_NOT_READ=data_bytes,frames=shape[0])
        except FileNotFoundError:row.update(state='MISSING_EXACT_PATH_NO_FALLBACK')
        except Exception as exc:row.update(state='HEADER_READ_FAILED',error_type=type(exc).__name__,errno=getattr(exc,'errno',None),reason_SHA256=H(str(exc).encode()))
    valid=all(r['state']=='HEADER_VALID' for r in result['records'].values())
    result['header_pair_valid']=valid
    if valid:
        result['source_min_frames']=min(r['frames'] for r in result['records'].values())
        result['qualifies_true128']=result['source_min_frames']>=128
    result['state']='ONE_PAIR_HEADER_SHAPES_OBSERVED' if valid else 'UNKNOWN_SOURCE_FRAMES_MISSING_OR_INVALID_HEADER'
    result['limitation']='Only this exact pair header/shape/dtype; no wholefile SHA/value/finite check, no other127rows/cache128/A100training qualification'
    raw=json.dumps(result,separators=(',',':'),allow_nan=False).encode();assert len(raw)<=16384
    os.write(1,raw+b'\n')
    return 0 if valid else 2


if __name__=='__main__':raise SystemExit(main(BINDING,CONTEXT))
