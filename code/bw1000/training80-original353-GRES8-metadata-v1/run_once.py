"""Future published once entry; root alone may adopt then invoke this program."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import signal
import subprocess
import sys
import threading
import time

from verify_package import verify

D = Path(__file__).resolve().parent
G = D.parents[2]
REL = D.relative_to(G).as_posix()
HELPER = Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA = '3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'
REMOTE = '/public/home/aclsylqidf/hygon-inference-refactor-20261014/results/training80-original353-GRES8-metadata-v1'


def need(test, reason):
    if not test:
        raise ValueError(reason)


def save(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def reap_held(child, deadline):
    """Only the Popen this invocation created; no PID scan, no throwing left()."""
    observation = dict(kill_requested=False, reaped=False, errors=[])
    if child is None:
        return observation
    if child.poll() is None:
        observation['kill_requested'] = True
        try:
            child.kill()
        except OSError as exc:
            observation['errors'].append(type(exc).__name__)
    try:
        child.wait(timeout=max(0, min(2, deadline-time.time())))
        observation['reaped'] = True
    except (OSError, subprocess.TimeoutExpired) as exc:
        observation['errors'].append(type(exc).__name__)
    observation['returncode'] = child.poll()
    return observation


def main():
    epoch, origin_mono = time.time(), time.monotonic()
    p = argparse.ArgumentParser()
    p.add_argument('--publication', required=True)
    p.add_argument('--request-id', required=True)
    a = p.parse_args()
    need(re.fullmatch('[0-9a-f]{40}', a.publication) and re.fullmatch('[0-9a-f]{32}', a.request_id), 'CONTEXT')
    deadline = epoch+90

    def left(reserve=0):
        value = min(deadline-time.time(), 90-(time.monotonic()-origin_mono))-reserve
        need(value > 0, 'SINGLE_ORIGIN_DEADLINE')
        return value

    mf = verify(D, adopted=True)
    def git(*args):
        result = subprocess.run(['git','-c','safe.directory='+str(G),*args], cwd=G, capture_output=True,
                                timeout=min(8,left(25)), check=True)
        return result.stdout
    need(not git('diff','--cached','--name-only').strip(), 'GIT_INDEX_NOT_EMPTY')
    need(not git('status','--porcelain','--',REL).strip(), 'OWN_PUBLISHED_SOURCE_DIRTY')
    need(git('show',a.publication+':'+REL+'/SOURCE_FILES.json') == (D/'SOURCE_FILES.json').read_bytes(), 'PUBLISHED_MF_DIFFERS')
    prior=git('show',a.publication+':results/bw1000/2026-11-18/training80-current640-window-original80-permission-failure.json')
    need(len(prior)==12819 and hashlib.sha256(prior).hexdigest()=='9430f015224003a7d084e6325a3c433bb9d6af05cfc93ace7b1399c746dd185b',
         'PUBLISHED_ORIGINAL_KNOWN_FAILURE_REQUIRED')
    baseline=git('show',a.publication+':results/bw1000/2026-11-18/training80-original353-single-node-KFD-gpu-id-read-EPERM.json')
    need(len(baseline)==3777 and hashlib.sha256(baseline).hexdigest()=='cad4776cfb3835127a2ca0bfe0f43342c879a3134248b2c8f1bcc9f388036965','PUBLISHED_353_BASELINE_REQUIRED')
    need(hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA, 'QUALIFIED_HELPER')
    ssh = runpy.run_path(str(HELPER))['ssh']
    local = G/'local-only'/'training80-original353-GRES8-metadata-v1-original'
    local.mkdir(parents=True, exist_ok=False)
    context = dict(request_id=a.request_id,publication=a.publication,epoch=epoch,deadline=deadline,
                   original_failed_sampler='57a19407a6f34ef2af158e317cf3c1d8',exec_request_attempted=False)
    save(local/'intent.json', context)
    files = {name:base64.b64encode((D/name).read_bytes()).decode() for name in [*mf['files'],'SOURCE_FILES.json']}
    # This contains small project source only. Only source/ledger enter; node receipts stay R and a bounded fleet summary returns.
    payload = dict(files=files,epoch=epoch,deadline=deadline,request=a.request_id,publication=a.publication)
    bootstrap = '''import base64,hashlib,json,os,pathlib,runpy,signal,sys,time
p=json.loads(sys.stdin.buffer.read(262145));assert len(p['files'])<=16
deadline=p['deadline'];assert deadline==p['epoch']+90 and time.time()<deadline-20
signal.signal(signal.SIGALRM,lambda *_:os._exit(124));signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
root=pathlib.Path(REMOTE);assert not root.exists();root.mkdir(mode=0o700)
src=root/'src';src.mkdir(mode=0o700)
mf=json.loads(base64.b64decode(p['files']['SOURCE_FILES.json'],validate=True));assert set(p['files'])==set(mf['files'])|{'SOURCE_FILES.json'}
for name,data in p['files'].items():
 assert pathlib.Path(name).name==name and time.time()<deadline-20
 raw=base64.b64decode(data,validate=True)
 if name!='SOURCE_FILES.json':
  pin=mf['files'][name];assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
 with (src/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
sys.path.insert(0,str(src));sys.argv=[str(src/'controller.py'),'--epoch',str(p['epoch']),'--request-id',p['request'],'--publication',p['publication']]
runpy.run_path(str(src/'controller.py'),run_name='__main__')
'''.replace('REMOTE',repr(REMOTE))
    body = (bootstrap + '\n').encode()
    # Small -c; avoid Win32 command-line length overflow; body arrives on stdin.
    tiny = "import sys;exec(compile(sys.stdin.buffer.read(int(sys.argv[1])), '<training80-gres8-bootstrap>', 'exec'))"
    import shlex
    command = ' '.join(shlex.quote(x) for x in ['/usr/local/python3.12/bin/python3','-I','-S','-B','-c',tiny,str(len(body))])
    argv = [*ssh, command]
    need(len(subprocess.list2cmdline(argv)) < 30000, 'SHORT_COMMANDLINE')
    stdin = body+json.dumps(payload,separators=(',',':')).encode()
    need(len(stdin) <= 262144, 'SMALL_SOURCE_PAYLOAD')
    save(local/'attempt.json', dict(context,exec_request_attempted=True,source_body_SHA256=hashlib.sha256(stdin).hexdigest(),source_body_bytes=len(stdin)))
    child = None
    forced = threading.Event()
    timer = None
    cleanup = None
    catchable = [getattr(signal, name) for name in ('SIGTERM','SIGINT','SIGHUP') if hasattr(signal,name)]
    previous = {s:signal.getsignal(s) for s in catchable}
    def interrupted(_s, _f):
        forced.set()
        raise InterruptedError('OWNED_SSH_CLIENT_INTERRUPTED')
    for s in catchable:
        signal.signal(s,interrupted)
    started = time.monotonic()
    with (local/'stdout.private').open('xb') as out, (local/'stderr.private').open('xb') as err:
        try:
            def expire():
                forced.set()
                try:
                    if child is not None and child.poll() is None:
                        child.kill()
                except OSError:
                    pass
            timer = threading.Timer(left(3),expire)
            timer.daemon=True
            timer.start()
            child = subprocess.Popen(argv,cwd=G,stdin=subprocess.PIPE,stdout=out,stderr=err)
            # Held identity exists even if persisting the optional metadata fails.
            need(not forced.is_set(), 'CLIENT_EXPIRED_DURING_POPEN')
            left(3)
            save(local/'owned-ssh.json',dict(PID=child.pid,request_id=a.request_id,anchor='held_Popen',epoch=epoch))
            left()
            child.communicate(stdin,timeout=left(3))
        except BaseException as exc:
            forced.set()
            save(local/'client-error.json',dict(type=type(exc).__name__,message_SHA256=hashlib.sha256(str(exc).encode()).hexdigest()))
        finally:
            for s in catchable:
                signal.signal(s,signal.SIG_IGN)
            if child is not None and child.poll() is None:
                forced.set()
            cleanup = reap_held(child,deadline)
            if timer:
                timer.cancel()
    rc = child.returncode if child else None
    state = 'UNKNOWN' if forced.is_set() or rc in (None,255) else 'RETURNED'
    outcome = dict(context,exec_request_attempted=True,remote_outcome=state,returncode=rc,
                   forced_local_SSH_kill=forced.is_set(),held_cleanup=cleanup,
                   client_wall_s=time.monotonic()-started,finished=time.time())
    for name in ('stdout.private','stderr.private'):
        raw=(local/name).read_bytes()
        outcome[name]=dict(bytes=len(raw),SHA256=hashlib.sha256(raw).hexdigest())
    save(local/'outcome.json',outcome)
    for s,handler in previous.items():
        signal.signal(s,handler)
    print(json.dumps(dict(remote_outcome=state,returncode=rc,receipt=str(local/'outcome.json'))))
    return 0 if state=='RETURNED' and rc==0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
