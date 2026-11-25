"""One explicitly injected ROOT transport; no SSH/helper/default route exists."""
import base64
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

from contract import D,ROOT,LABEL,REQUEST,DISPATCH_REQUEST,WHOLE,Clock,need,pin,verify
from held_child import cleanup_held
from authority import verify_authority

HOST_COMMAND='/usr/local/python3.12/bin/python3 -I -S -B -u -'

def transport_state(rc,held):
    return ('UNKNOWN_NO_REPLAY' if rc is None or rc==255 or held.get('reaped') is not True
            else 'RETURNED_REQUIRES_ORIGINAL_TERMINAL_CONSUMPTION')


def make_body(publication,epoch,manifest,authority):
    remote_root=ROOT.as_posix()
    need('\\' not in remote_root and PurePosixPath(remote_root).is_absolute()
         and remote_root=='/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/'+LABEL
         and PurePosixPath(remote_root).name==LABEL,
         'canonical absolute POSIX receiver root required')
    files=json.loads((D/'SOURCE_FILES.json').read_bytes())['files']
    payload={name:base64.b64encode((D/name).read_bytes()).decode() for name in files}
    payload['SOURCE_FILES.json']=base64.b64encode((D/'SOURCE_FILES.json').read_bytes()).decode()
    need(sum(len(v) for v in payload.values())<=2*1024**2,'source-only upload cap')
    intent=dict(request_id=REQUEST,dispatch_request_id=DISPATCH_REQUEST,publication=publication,epoch=epoch,deadline=epoch+WHOLE,
                source_manifest=manifest,label=LABEL,inference_only=True,planned_total=108,root_authority=authority)
    return ('import os,sys,json,base64,hashlib\nfrom pathlib import Path\n'
        +'remote_root='+repr(remote_root)+'\n'
        +'assert remote_root.startswith("/public/home/") and "\\\\" not in remote_root\n'
        +'root=Path(remote_root);assert root.is_absolute() and {p.name for p in root.iterdir()}=={"inference"}\n'
        +'src=root/"src";src.mkdir(mode=0o700)\n'
        +'payload='+repr(payload)+'\n'
        +'for name,value in payload.items():\n'
        +' p=src/name;assert not Path(name).is_absolute() and ".." not in Path(name).parts and "\\\\" not in name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value,validate=True))\n'
        +'with (root/"intent.json").open("x") as f:json.dump('+repr(intent)+',f,allow_nan=False,sort_keys=True)\n'
        +'entry="import sys,runpy;sys.path.insert(0,"+repr(str(src))+");runpy.run_path("+repr(str(src/"controller.py"))+",run_name=\'__main__\')"\n'
        +'os.execv(sys.executable,[sys.executable,"-I","-S","-B","-c",entry])\n').encode()


def launch(*,transport_start,publication,local_output,authority):
    epoch=time.time() # all preparation/upload/transport included; never reset after source check
    clock=Clock(epoch)
    need(callable(transport_start),'explicit root-coordinated transport callback required')
    need(isinstance(publication,str) and len(publication)==40
         and set(publication)<=set('0123456789abcdef'),'exact published source revision required')
    manifest=verify('GPU')
    verify_authority(authority,manifest,epoch)
    body=make_body(publication,epoch,manifest,authority)
    local_output=Path(local_output);local_output.mkdir(mode=0o700)
    intent=dict(request_id=DISPATCH_REQUEST,generation_request_id=REQUEST,publication=publication,epoch=epoch,deadline=epoch+WHOLE,
        source_manifest=manifest,remote_command=HOST_COMMAND,stdin_bytes=len(body),
        stdin_SHA256=hashlib.sha256(body).hexdigest(),transport_attempted=False)
    def record(name,obj):
        with (local_output/name).open('x',encoding='utf8') as f:json.dump(obj,f,sort_keys=True,allow_nan=False)
    record('intent.json',intent)
    child=None;anchor={};failure=None;rc=None;held={};transport_called=False
    try:
        clock.left(WHOLE,20)
        record('transport-attempt.json',dict(request_id=DISPATCH_REQUEST,generation_request_id=REQUEST,epoch=epoch,command=HOST_COMMAND,
            body_SHA256=intent['stdin_SHA256'],source_manifest=manifest))
        context={k:intent[k] for k in ('request_id','generation_request_id','publication','epoch','deadline','source_manifest','remote_command')}
        transport_called=True
        child=transport_start(context)
        need(isinstance(child,subprocess.Popen) and child.stdin is not None
             and child.stdout is not None and child.stderr is not None,
             'transport must return the actual held Popen with binary stdio pipes')
        anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True)
        # ROOT callback must Popen(start_new_session=True); this is recorded in its scope contract.
        out,err=child.communicate(input=body,timeout=clock.left(WHOLE,20))
        need(len(out)<=2*1024**2 and len(err)<=2*1024**2,'bounded transport logs exceeded')
        (local_output/'stdout.private').write_bytes(out);(local_output/'stderr.private').write_bytes(err)
        rc=child.returncode
        if rc==255:failure=dict(type='TransportDisconnected',execution_state='UNKNOWN_NO_REPLAY')
    except BaseException as e:
        failure=dict(type=type(e).__name__,reason=str(e)[:512],execution_state='UNKNOWN_NO_REPLAY' if transport_called else 'PRE_TRANSPORT_FAILED')
    finally:
        try:
            held=cleanup_held(child,anchor,epoch+WHOLE-2)
        except BaseException as e:
            held=dict(reaped=False,error_type=type(e).__name__)
        receipt=dict(request_id=DISPATCH_REQUEST,generation_request_id=REQUEST,publication=publication,epoch=epoch,deadline=epoch+WHOLE,
            returncode=rc,failure=failure,held=held,source_manifest=manifest,
            execution_state=transport_state(rc,held),transport_callback_called=transport_called,actual_Popen_returned=child is not None,
            effect_qualified=False)
        record('receipt.json',receipt)
    return receipt
