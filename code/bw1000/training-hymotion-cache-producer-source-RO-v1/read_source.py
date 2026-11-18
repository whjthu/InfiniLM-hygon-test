"""Read one pinned producer source; no import/execute of that source or dataset."""
import base64
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import time


def main(binding,context):
    assert binding['root_GO'] is True and binding['RO_GO'] is True
    assert binding['GPU_GO'] is False and binding['dataset_body_GO'] is False
    deadline=context['epoch']+30
    assert 0<=time.time()-context['epoch']<25
    signal.signal(signal.SIGALRM,lambda *_:os._exit(124))
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    affinity=sorted(os.sched_getaffinity(0)-{0,1});assert len(affinity)>=2;os.sched_setaffinity(0,affinity[:2])
    resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2))
    p=Path(binding['source']['path']);s=p.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_size==binding['source']['bytes']==16055
    identity=lambda x:[x.st_dev,x.st_ino,x.st_size,x.st_mtime_ns,x.st_ctime_ns]
    with os.fdopen(os.open(p,os.O_RDONLY|os.O_NOFOLLOW),'rb',buffering=0) as f:
        before=os.fstat(f.fileno());raw=f.read(s.st_size);after=os.fstat(f.fileno())
    assert len(raw)==s.st_size and identity(s)==identity(before)==identity(after)==identity(p.stat())
    digest=hashlib.sha256(raw).hexdigest();assert digest==binding['source']['SHA256']
    result=dict(context=context,host=socket.gethostname(),uid=os.getuid(),source=dict(binding['source'],file_identity=identity(after)),
                source_b64=base64.b64encode(raw).decode(),source_imported=False,source_executed=False,
                dataset_files_read=0,cache_tensor_bytes_read=0,weights_read=0,GPU_or_SDK=False,Slurm_steps=0)
    out=json.dumps(result,separators=(',',':')).encode();assert len(out)<=32768
    os.write(1,out+b'\n')


if __name__=='__main__':main(BINDING,CONTEXT)
