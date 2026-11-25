"""Dispatch one already installed and independently bound generation scope."""
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import time

from authority import verify_authority
from contract import D, DISPATCH_REQUEST, LABEL, REQUEST, WHOLE, Clock, need, verify
from held_child import cleanup_held

HOST_COMMAND='/usr/local/python3.12/bin/python3 -I -S -B -u -'
TARGET_PROJECT='/public/home/aclsylqidf/hygon-inference-refactor-20261014'


def stored(path, value):
    with path.open('x', encoding='utf8') as file:
        json.dump(value,file,allow_nan=False,sort_keys=True)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())


def make_body(publication,epoch,manifest,authority):
    source=TARGET_PROJECT+'/incoming/'+LABEL+'-source'
    results=TARGET_PROJECT+'/results/'+LABEL
    intent=dict(request_id=REQUEST,dispatch_request_id=DISPATCH_REQUEST,publication=publication,epoch=epoch,
                deadline=epoch+WHOLE,source_manifest=manifest,label=LABEL,inference_only=True,
                planned_total=108,root_authority=authority)
    entry=('import sys,runpy;sys.path.insert(0,'+repr(source)+');runpy.run_path('
           +repr(source+'/controller.py')+',run_name="__main__")')
    statements=[
        'import hashlib,json,os,sys',
        'from pathlib import Path',
        f'src=Path({source!r}); root=Path({results!r}); intent={intent!r}',
        'def require(ok):\n if not ok: raise ValueError("UNQUALIFIED_PRODUCTION_SOURCE_OR_ROOT")',
        'require(src.is_absolute() and src.resolve(strict=True)==src and root.parent.resolve(strict=True)==root.parent)',
        'def digest(raw): return {"bytes":len(raw),"SHA256":hashlib.sha256(raw).hexdigest()}',
        'manifest=(src/"SOURCE_FILES.json").read_bytes();require(digest(manifest)==intent["source_manifest"])',
        'members=json.loads(manifest)["files"];require(len(members)==37)',
        'for name,want in members.items():\n p=Path(name); q=src/p\n require(not p.is_absolute() and ".." not in p.parts and q.resolve(strict=True)==q and q.is_file() and q.stat().st_size==want["bytes"])\n require(digest(q.read_bytes())==want)',
        'root.mkdir(mode=0o700)',  # Atomic attempt identity before private owner files.
        '(root/"inference").mkdir(mode=0o700)',
        'with (root/"intent.json").open("x",encoding="utf8") as f:\n json.dump(intent,f,allow_nan=False,sort_keys=True);f.flush();os.fsync(f.fileno())',
        f'os.execv(sys.executable,[sys.executable,"-I","-S","-B","-c",{entry!r}])',
    ]
    body=('\n'.join(statements)+'\n').encode()
    need(len(body)<=16000,'bounded source-less owner payload')
    compile(body,'gemma-owner-entry','exec')
    return body


def launch(*,transport_start,publication,local_output,authority):
    epoch=time.time()
    clock=Clock(epoch)
    need(callable(transport_start) and isinstance(publication,str) and len(publication)==40
         and set(publication)<=set('0123456789abcdef'),'exact publication and one injected ROOT transport required')
    manifest=verify('GPU')
    verify_authority(authority,manifest,epoch,publication)
    body=make_body(publication,epoch,manifest,authority)
    local_output=Path(local_output)
    local_output.mkdir(mode=0o700)
    context=dict(request_id=DISPATCH_REQUEST,generation_request_id=REQUEST,publication=publication,epoch=epoch,
                 deadline=epoch+WHOLE,source_manifest=manifest,remote_command=HOST_COMMAND,
                 body_SHA256=sha256(body).hexdigest(),body_bytes=len(body),replay_permitted=False)
    stored(local_output/'intent.json',context)
    child=None
    rc=None
    error=None
    anchor={}
    attempted=False
    try:
        clock.left(WHOLE,25)
        stored(local_output/'transport-attempt.json',context)
        attempted=True
        child=transport_start(context)
        need(isinstance(child,subprocess.Popen) and child.stdin is not None and child.stdout is not None
             and child.stderr is not None,'actual held Popen with binary stdio pipes required')
        anchor=dict(PID=child.pid,PGID=child.pid,start_new_session=True)
        out,err=child.communicate(input=body,timeout=clock.left(WHOLE,20))
        need(len(out)<=2<<20 and len(err)<=2<<20,'bounded private transport logs exceeded')
        for name,raw in [('stdout.private',out),('stderr.private',err)]:
            with (local_output/name).open('xb') as stream:
                stream.write(raw)
        rc=child.returncode
    except BaseException as exc:
        error=type(exc).__name__
    finally:
        try:
            held=cleanup_held(child,anchor,epoch+WHOLE-2)
        except BaseException as exc:
            held=dict(reaped=False,error_type=type(exc).__name__)
        status='UNKNOWN_NO_REPLAY' if attempted else 'PRE_TRANSPORT_FAILED'
        if attempted and rc not in (None,255) and held.get('reaped') is True:
            status='RETURNED_REQUIRES_INDEPENDENT_TERMINAL'
        summary=dict(context,execution_state=status,returncode=rc,error_type=error,held=held,
                     transport_callback_called=attempted,actual_Popen_returned=child is not None)
        stored(local_output/'receipt.json',summary)
    return summary


def main(publication, transport):
    repo=D.parents[2]
    need(D==repo/'code'/'bw1000'/LABEL,'canonical independently published package required')
    local=repo/'local-only'
    authority_path=local/(LABEL+'-root-authority.json')
    need(authority_path.resolve(strict=True)==authority_path,'canonical private root authority required')
    with authority_path.open('rb') as stream:
        raw=stream.read(16385)
    need(0<len(raw)<=16384,'bounded root authority required')
    return launch(transport_start=transport,publication=publication,
                  local_output=local/(LABEL+'-once'),authority=json.loads(raw))
