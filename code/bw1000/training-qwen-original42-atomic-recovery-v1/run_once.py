"""Explicit-exception-only recovery, same W/labels, original receiver arbitration."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import threading
import time
from verify_recovery import H,need,require_permission,verify

D=Path(__file__).resolve().parent;G=D.parents[2]
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'


def save(root,name,value):
    with (root/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def main():
    origin=time.time();mono=time.monotonic();deadline=origin+180
    p=argparse.ArgumentParser();p.add_argument('--publication');p.add_argument('--request-id');p.add_argument('--verify-only',action='store_true');a=p.parse_args()
    mf,scope,claim,auth=verify(D)
    if a.verify_only:
        print(json.dumps(dict(source_verified=True,permission_received=auth['explicit_user_exception_received'],SSH=False)))
        return 0
    need(re.fullmatch('[0-9a-f]{40}',a.publication or '') and re.fullmatch('[0-9a-f]{32}',a.request_id or ''),'NEW_RECOVERY_CONTEXT')
    def left(reserve=0):
        n=min(deadline-time.time(),180-(time.monotonic()-mono))-reserve;need(n>0,'CLIENT_TRANSPORT_DEADLINE');return n
    def git(*args):
        r=subprocess.run(['git','-c','safe.directory='+str(G),*args],cwd=G,capture_output=True,timeout=min(8,left(10)),check=True)
        return r.stdout
    rel=D.relative_to(G).as_posix();base_root=scope['original_source_root']
    need(not git('diff','--cached','--name-only').strip(),'GIT_INDEX_NOT_EMPTY')
    need(not git('status','--porcelain','--',rel,base_root).strip(),'SOURCE_DIRTY')
    need(git('show',a.publication+':'+rel+'/SOURCE_FILES.json')==(D/'SOURCE_FILES.json').read_bytes(),'PUBLISHED_RECOVERY_SOURCE_MF')
    # Nothing below (including helper/config construction) is reachable without the real user exception.
    permission=require_permission(scope,claim,auth,a.request_id,lambda path:git('show',a.publication+':'+path))
    original=(D/'ORIGINAL_SOURCE_FILES.json').read_bytes()
    need(git('show',a.publication+':'+base_root+'/SOURCE_FILES.json')==original,'PUBLISHED_ORIGINAL17_MF')
    original_pins=json.loads(original)['files'];target_raw=(D/'TARGET_SOURCE_FILES.json').read_bytes();target=json.loads(target_raw)['files']
    files={}
    for name,pin in original_pins.items():
        raw=git('show',a.publication+':'+base_root+'/'+name)
        need(len(raw)==pin['bytes'] and H(raw)==pin['SHA256'],'PUBLISHED_ORIGINAL_MEMBER')
        if name=='verify_package.py':
            fixed=(D/'verify_package_fixed.py').read_bytes()
            need(fixed.replace(b'p.relative_to(FRAMEWORK_BASE).as_posix()',b'p.relative_to(FRAMEWORK).as_posix()')==raw,'SINGLE_REPAIR_REVERSE_BYTES')
            raw=fixed
        need(len(raw)==target[name]['bytes'] and H(raw)==target[name]['SHA256'] and len(raw)<200000,'TARGET_EXACT_BYTES')
        files[name]=dict(SHA256=H(raw),b64=base64.b64encode(raw).decode())
    files['SOURCE_FILES.json']=dict(SHA256=H(target_raw),b64=base64.b64encode(target_raw).decode())
    need(len(files)==18,'17_MEMBERS_PLUS_TARGET_MF')
    receiver=base64.b64decode(files['receive_once.py']['b64'],validate=True)
    # Execute exact old receiver; never create, remove, rename or repair W in this wrapper.
    body=('FILES='+repr(files)+'\nCLAIM='+repr(a.publication)+'\nREQUEST='+repr(a.request_id)+'\n').encode()+receiver
    need(len(body)<=512*1024,'SOURCE_ONLY_STDIN_CAP')
    need(H(HELPER.read_bytes())==HELPER_SHA,'QUALIFIED_SSH_HELPER')
    ssh=runpy.run_path(str(HELPER))['ssh']
    root=G/'local-only/training-qwen-original42-atomic-recovery-v1-original';root.mkdir(parents=True,exist_ok=False)
    context=dict(request_id=a.request_id,publication=a.publication,original_request=scope['original_request'],original42_state='UNKNOWN_PRESERVED',
                 client_origin=origin,client_deadline=deadline,server_origin='fresh receiver time.time(); same original7800s budget',
                 namespace=scope['namespace'],labels=scope['labels'],target_source_MF_SHA256=H(target_raw),permission=permission,
                 source_body_bytes=len(body),source_body_SHA256=H(body))
    save(root,'intent.json',context)
    (root/'body-source.private').write_bytes(body)
    save(root,'attempt.json',dict(context,exec_request_attempted=True))
    child=None;forced=threading.Event();cleanup=dict(reaped=False,kill_requested=False,errors=[])
    def interrupted(_s,_f):forced.set();raise InterruptedError('CLIENT_INTERRUPTED')
    signals=[getattr(signal,n) for n in ('SIGTERM','SIGINT','SIGHUP') if hasattr(signal,n)]
    for sig in signals:signal.signal(sig,interrupted)
    def expire():
        forced.set()
        try:
            if child is not None and child.poll() is None:child.kill()
        except OSError:pass
    timer=threading.Timer(left(3),expire);timer.daemon=True;timer.start()
    started=time.monotonic();failure=None
    with (root/'stdout.private').open('xb') as out,(root/'stderr.private').open('xb') as err:
        try:
            child=subprocess.Popen([*ssh,'/usr/local/python3.12/bin/python3 -I -S -B -'],cwd=G,stdin=subprocess.PIPE,stdout=out,stderr=err)
            need(not forced.is_set(),'EXPIRED_DURING_HELD_POPEN');left(3)
            save(root,'owned-ssh.json',dict(PID=child.pid,anchor='held_Popen',request_id=a.request_id))
            child.communicate(body,timeout=left(3))
        except BaseException as exc:failure=type(exc).__name__;forced.set()
        finally:
            for sig in signals:signal.signal(sig,signal.SIG_IGN)
            if child is not None:
                if child.poll() is None:
                    forced.set();cleanup['kill_requested']=True
                    try:child.kill()
                    except OSError as exc:cleanup['errors'].append(type(exc).__name__)
                try:child.wait(timeout=max(0,min(2,deadline-time.time())));cleanup['reaped']=True
                except (OSError,subprocess.TimeoutExpired) as exc:cleanup['errors'].append(type(exc).__name__)
            timer.cancel()
    rc=child.returncode if child else None
    result=dict(context,remote_outcome='UNKNOWN' if forced.is_set() or rc in (None,255) else 'RETURNED',returncode=rc,
                forced_local_SSH_kill=forced.is_set(),cleanup=cleanup,client_error_type=failure,
                client_monotonic_s=time.monotonic()-started,model_execution_state='UNKNOWN_UNTIL_ORIGINAL_CONTROLLER_EVIDENCE',
                original42_UNKNOWN_preserved=True,replay_allowed=False,Model_qualified=False)
    for name in ('stdout.private','stderr.private'):
        raw=(root/name).read_bytes();result[name]=dict(bytes=len(raw),SHA256=H(raw))
    if rc==0 and not forced.is_set():
        try:
            raw=(root/'stdout.private').read_bytes();need(len(raw)<=32768,'ACK_BOUND');ack=json.loads(raw)
            need(ack['claim_commit']==a.publication and ack['output']==scope['namespace'],'EXACT_RECEIVER_ACK')
            result['receiver_ack']={k:ack.get(k) for k in ('owned_controller','controller_ready','controller_returncode','GPU_execution_status')}
        except Exception as exc:
            result['receiver_ack_error_type']=type(exc).__name__
            result['remote_outcome']='UNKNOWN'
    save(root,'outcome.json',result)
    print(json.dumps(dict(remote_outcome=result['remote_outcome'],returncode=rc,receipt=str(root/'outcome.json'),
                         model_execution_state=result['model_execution_state'],original42_UNKNOWN_preserved=True)))
    return 0 if result['remote_outcome']=='RETURNED' and rc==0 else 1


if __name__=='__main__':raise SystemExit(main())
