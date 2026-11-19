"""New ordinary CPU once; no old query/transfer replay. Epoch includes local preparation."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import runpy
import re
import subprocess
import time
import uuid
from common import D,P,read,pin,save,admission,verify

G=D.parents[2]
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'


def main(publication,mf_sha):
    epoch=time.time();anchor=time.monotonic()
    admission()
    assert re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{64}',mf_sha)
    def left():
        value=min(epoch+300-time.time(),300-(time.monotonic()-anchor));assert value>0;return value
    verify();runpy.run_path(str(D/'verify_package.py'),run_name='new_cpu_package_verification')['verify'](D)
    mfraw=(D/'SOURCE_FILES.json').read_bytes();assert hashlib.sha256(mfraw).hexdigest()==mf_sha
    raw=subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),'cat-file','blob',publication+':'+D.relative_to(G).as_posix()+'/SOURCE_FILES.json'],timeout=min(12,left()))
    assert raw==mfraw
    assert not subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),'status','--porcelain=v1','--untracked-files=all','--',D.relative_to(G).as_posix()],timeout=min(12,left()))
    blobs={name:base64.b64encode((D/name).read_bytes()).decode() for name in json.loads(mfraw)['files']}
    blobs['SOURCE_FILES.json']=base64.b64encode(mfraw).decode();request=uuid.uuid4().hex
    local=G/'local-only'/(P['label']+'-original-dispatch');local.mkdir(mode=0o700)
    prefix='FILES='+repr(blobs)+'\nCONTEXT='+repr(dict(publication=publication,request_id=request,epoch=epoch,deadline=epoch+300,MF_SHA256=mf_sha))+'\n'
    body=prefix.encode()+(D/'receive_once.py').read_bytes();assert len(body)<=2*1024**2
    save(local/'intent.json',dict(publication=publication,request_id=request,epoch=epoch,deadline=epoch+300,payload_SHA256=hashlib.sha256(body).hexdigest(),MF_SHA256=mf_sha,GPU=0))
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==HELPER_SHA
    argv=runpy.run_path(str(HELPER))['ssh'];argv=[argv[0],'-o','StrictHostKeyChecking=yes',*argv[1:],'/usr/local/python3.12/bin/python3 -I -S -B -']
    save(local/'attempt.json',dict(request_id=request,publication=publication,epoch=epoch,deadline=epoch+300,
        scope=P['RUN'],SSH_attempt_about_to_start=True,exec_outcome='UNKNOWN_UNTIL_ORIGINAL_RECEIPT',replay_allowed=False))
    child=None;timed_out=False;reaped=False;failure=None;stdout=b'';stderr=b'';rc=None;cleanup_errors={}
    try:
        assert left()>5
        child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        stdout,stderr=child.communicate(input=body,timeout=max(.001,min(left()-5,epoch+295-time.time())))
    except subprocess.TimeoutExpired as error:
        timed_out=True;failure=type(error).__name__;stdout=error.output or b'';stderr=error.stderr or b''
    except BaseException as error:
        failure=type(error).__name__
    finally:
        # This final held-Popen cleanup never calls the rejecting admission clock.
        # The original monotonic deadline independently caps drain/reap at epoch+298.
        if child is not None:
            if child.poll() is None:
                try:child.kill()
                except ProcessLookupError:pass
                except BaseException as error:cleanup_errors['kill']=type(error).__name__
                budget=min(3.0,anchor+298-time.monotonic())
                if budget>0:
                    try:stdout,stderr=child.communicate(timeout=budget)
                    except subprocess.TimeoutExpired as error:
                        stdout=error.output or stdout;stderr=error.stderr or stderr
                        cleanup_errors['reap']='TimeoutExpired'
                    except BaseException as error:cleanup_errors['reap']=type(error).__name__
                else:cleanup_errors['reap']='ORIGINAL_CLEANUP_DEADLINE_EXHAUSTED'
            rc=child.poll();reaped=rc is not None
    for name,raw in (('stdout.private',stdout),('stderr.private',stderr)):
        with (local/name).open('xb') as stream:stream.write(raw)
    record=dict(request_id=request,publication=publication,transport_rc=rc,timed_out=timed_out,held_SSH_started=child is not None,
                held_SSH_reaped=reaped,local_error_type=failure,cleanup_errors=cleanup_errors,client_wall_seconds=time.monotonic()-anchor,
                remote_outcome='UNKNOWN' if timed_out or failure or cleanup_errors or not reaped or rc in (None,255) else 'RETURNED_REQUIRES_ORIGINAL_RECEIPT_AUDIT',
                stdout_bytes=len(stdout),stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_bytes=len(stderr),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),replay_allowed=False,GPU=0)
    save(local/'transport.json',record);print(json.dumps(record));return 0 if rc==0 and not timed_out and failure is None and not cleanup_errors and reaped else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publication',required=True);parser.add_argument('--source-manifest-sha',required=True)
    a=parser.parse_args();raise SystemExit(main(a.publication,a.source_manifest_sha))
