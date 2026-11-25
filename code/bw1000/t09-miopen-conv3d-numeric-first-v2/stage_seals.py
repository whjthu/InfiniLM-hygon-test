"""Exclusive bounded evidence files with cross-phase linkage for the one smoke."""
from hashlib import sha256
import json
import os
from pathlib import Path
import stat

from scope import REQUEST, require

GUEST=Path('/t09-smoke-results')
NAMES=('01-numeric.json','02-aten.json','03-provider.json')
CAP=128<<10


def pin(raw):return dict(bytes=len(raw),SHA256=sha256(raw).hexdigest())


def life(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def seal(root,custody,idx,context,previous,value):
    require(0<=idx<len(NAMES),'EXACT_EVIDENCE_PHASE_NUMBER')
    right=GUEST if root is None else root
    directory=os.open(right,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        actual=os.fstat(directory)
        require([actual.st_dev,actual.st_ino]==custody and actual.st_uid==os.getuid()
                and stat.S_IMODE(actual.st_mode)==0o700,'ORIGINAL_WITNESSED_EVIDENCE_DIRECTORY')
        require(set(os.listdir(directory))==set(NAMES[:idx]),'EXACT_PRIOR_SEAL_ORDER')
        before=None
        if idx:
            raw,_=read_at(directory,NAMES[idx-1])
            before=pin(raw)
        require(before==previous,'EXACT_PREVIOUS_PHASE_SEAL_PIN')
        data=dict(request=REQUEST,step=context['step'],authority_SHA256=context['authority_SHA256'],
                  epoch=context['epoch'],sequence=idx+1,previous=previous,result=value)
        body=(json.dumps(data,sort_keys=True,allow_nan=False)+'\n').encode()
        require(len(body)<=CAP,'BOUNDED_PRIVATE_SEALED_RECEIPT')
        fd=os.open(NAMES[idx],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600,dir_fd=directory)
        with os.fdopen(fd,'wb') as stream:
            stream.write(body);stream.flush();os.fsync(stream.fileno())
        os.fsync(directory)
        return pin(body)
    finally:
        os.close(directory)


def read_at(directory,name):
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC|os.O_NONBLOCK,dir_fd=directory)
    with os.fdopen(fd,'rb') as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and
                stat.S_IMODE(before.st_mode)==0o600 and before.st_nlink==1 and 0<before.st_size<=CAP,
                'EXACT_PRIVATE_RECEIPT_REGULAR_FILE')
        buf=stream.read(CAP+1)
        require(len(buf)==before.st_size and life(os.fstat(stream.fileno()))==life(before),'RECEIPT_OPEN_FD_LIFETIME')
    require(life(os.stat(name,dir_fd=directory,follow_symlinks=False))==life(before),'RECEIPT_PATH_FD_LIFETIME')
    return buf,before


def consume(root,expected,require_inode=True):
    require(root.resolve(strict=True)==root,'EXACT_CANONICAL_RESULT_DIRECTORY')
    directory=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    previous=None
    result={}
    try:
        info=os.fstat(directory)
        require(info.st_uid==os.getuid() and stat.S_IMODE(info.st_mode)==0o700
                and (not require_inode or info.st_ino==expected['host_results_identity'][1]),
                'SAME_RESULT_DIRECTORY_OWNER_MODE_AND_INODE')
        names=set(os.listdir(directory))
        require(any(names==set(NAMES[:n]) for n in range(4)),'ONLY_A_PREFIX_OF_EXPECTED_SEALS')
        for idx,name in enumerate(NAMES):
            if name not in names:break
            raw,_=read_at(directory,name)
            data=json.loads(raw)
            require(data['request']==REQUEST and data['step']==expected['step'] and data['epoch']==expected['epoch']
                    and data['authority_SHA256']==expected['authority_SHA256']
                    and data['sequence']==idx+1 and data['previous']==previous,'ORIGINAL_NUMERIC_TRACE_PROVIDER_CHAIN')
            previous=pin(raw)
            result[name]=dict(pin=previous,result=data['result'])
    finally:os.close(directory)
    return result
