"""Exclusive source install and bounded one-shot CPU execution; injected pinned envelope."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path,PurePosixPath
import signal
import sys
import time

W=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-internvl-A100-input-fingerprints-v1')

def require(ok,why):
    if not ok:raise ValueError(why)


def main():
    require(type(EPOCH) in (int,float) and 0<time.time()-EPOCH<120,'CLIENT_ORIGIN_EPOCH')
    def expired(*unused):raise TimeoutError('INSTALL_ORIGIN_DEADLINE')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
    signal.setitimer(signal.ITIMER_REAL,EPOCH+870-time.time())
    require(len(REQUEST)==32 and not set(REQUEST)-set('0123456789abcdef'),'REQUEST')
    require(len(PUBLICATION)==40 and not set(PUBLICATION)-set('0123456789abcdef'),'PUBLICATION')
    require(not W.exists() and not W.is_symlink(),'EXISTING_NAMESPACE_NO_REPLAY')
    require(not any(p.is_symlink() for p in W.parents),'NAMESPACE_SYMLINK')
    require(type(FILES) is dict and 1<len(FILES)<=16,'SOURCE_COUNT')
    decoded={};total=0
    for name,pin in FILES.items():
        part=PurePosixPath(name);require(not part.is_absolute() and '..' not in part.parts and str(part)==name,'SOURCE_PATH')
        data=base64.b64decode(pin['b64'],validate=True)
        require(len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['SHA256'],'SOURCE_PIN')
        total+=len(data);require(total<=256*1024,'SOURCE_TOTAL');decoded[name]=data
    require(hashlib.sha256(decoded['SOURCE_FILES.json']).hexdigest()==MANIFEST_SHA256,'MANIFEST_SHA')
    mf=json.loads(decoded['SOURCE_FILES.json']);require(set(decoded)==set(mf['files'])|{'SOURCE_FILES.json'},'SOURCE_INVENTORY')
    for name,pin in mf['files'].items():require(len(decoded[name])==pin['bytes'] and hashlib.sha256(decoded[name]).hexdigest()==pin['SHA256'],'MF_MEMBER')
    claim=json.loads(decoded['CLAIM.json']);require(claim['CPU_GO'] is True and claim['GPU_GO'] is False and claim['case']=='T03.sft','CLAIM')
    W.mkdir(mode=0o700);source=W/'source';source.mkdir(mode=0o700);(W/'output').mkdir(mode=0o700)
    for name,data in decoded.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
        path.chmod(0o600)
    sys.path.insert(0,str(source))
    from common import Budget,save,owner
    budget=Budget(EPOCH)
    save(W/'install-intent.json',dict(request_id=REQUEST,publication=PUBLICATION,epoch=EPOCH,deadline=EPOCH+900,
        owner=owner(os.getpid()),source_manifest_SHA256=MANIFEST_SHA256,source_files=len(mf['files']),source_bytes=total,
        replay_allowed=False,device_compute_calls=0,driver_query_calls=0))
    print(json.dumps(dict(stage='REMOTE_SOURCE_INSTALLED',request_id=REQUEST,publication=PUBLICATION,
                         source_manifest_SHA256=MANIFEST_SHA256)),flush=True)
    budget.remaining(work=True,reserve=30)
    from run_once import main as run
    return run(PUBLICATION,REQUEST,EPOCH,MANIFEST_SHA256)


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        print(json.dumps(dict(stage='RECEIVER_FAILURE',request_id=REQUEST,error_type=type(error).__name__,device_compute_calls=0,driver_query_calls='UNKNOWN_IF_SUPERVISOR_STARTED')),flush=True)
        raise SystemExit(1)
