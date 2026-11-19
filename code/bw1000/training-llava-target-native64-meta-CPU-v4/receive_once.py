"""Future exclusive install; no pending-prerequisite discovery or fallback."""
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import sys
import time

RUN=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training-llava-target-native64-meta-CPU-v4')


def main():
    assert CONTEXT['deadline']==CONTEXT['epoch']+300 and 0<CONTEXT['epoch']<=time.time()<CONTEXT['deadline']
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original install epoch')))
    signal.setitimer(signal.ITIMER_REAL,CONTEXT['deadline']-time.time()-5)
    os.umask(0o077);assert not RUN.exists() and not RUN.is_symlink()
    decoded={};total=0
    for name,encoded in FILES.items():
        path=PurePosixPath(name);assert not path.is_absolute() and '..' not in path.parts and path.as_posix()==name
        raw=base64.b64decode(encoded,validate=True);total+=len(raw);assert total<=2*1024**2;decoded[name]=raw
    mfraw=decoded['SOURCE_FILES.json'];assert hashlib.sha256(mfraw).hexdigest()==CONTEXT['MF_SHA256']
    mf=json.loads(mfraw);assert set(decoded)==set(mf['files'])|{'SOURCE_FILES.json'}
    for name,pin in mf['files'].items():assert len(decoded[name])==pin['bytes'] and hashlib.sha256(decoded[name]).hexdigest()==pin['SHA256']
    # Independent endpoint re-check cannot be bypassed by passing an arbitrary publication.
    claim=json.loads(decoded['CLAIM.json'])
    assert claim['root_GO'] is claim['CPU_GO'] is True and claim['GPU_GO'] is claim['Model_GO'] is claim['training_GO'] is False
    RUN.mkdir(mode=0o700);source=RUN/'source';source.mkdir()
    for name,raw in decoded.items():
        path=source/name;path.parent.mkdir(exist_ok=True,parents=True)
        with path.open('xb') as stream:stream.write(raw)
    sys.path.insert(0,str(source))
    from verify_package import verify as verify_new_package
    verify_new_package(source)
    from common import admission,save
    admission();save(RUN/'install-intent.json',dict(CONTEXT,source_files=len(mf['files']),source_bytes=total,GPU=0))
    from host_launch import main as launch
    return launch(CONTEXT['publication'],CONTEXT['request_id'],CONTEXT['epoch'])


if __name__=='__main__':
    raise SystemExit(main())
