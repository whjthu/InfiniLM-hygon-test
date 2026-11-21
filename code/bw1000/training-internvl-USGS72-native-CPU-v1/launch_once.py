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
from common import D,P,read,pin,save,admission,verify

G=D.parents[2]
TARGET_COMMAND=('/usr/local/python3.12/bin/python3','-I','-S','-B','-')


def require_transport_start(callback):
    if not callable(callback):
        raise ValueError('ROOT_COORDINATOR_TRANSPORT_CALLBACK_REQUIRED_NO_FALLBACK')
    return callback


def check_transport_child(child):
    if not isinstance(child,subprocess.Popen) or any(getattr(child,k,None) is None for k in ('stdin','stdout','stderr')) or child.text_mode:
        raise ValueError('ROOT_TRANSPORT_MUST_RETURN_BINARY_PIPE_POPEN')
    return child


def main(publication,mf_sha,*,transport_start=None):
    epoch=time.time();anchor=time.monotonic()
    require_transport_start(transport_start)
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
    blobs['SOURCE_FILES.json']=base64.b64encode(mfraw).decode();request=P['request_id']
    assert request=='8110b01cd24846f8b6fcd08f0abda5a0'
    contract=read(D/'DATA_CONTRACT.json');data_blobs={}
    asset_publication=P['input_asset_publication']
    old=subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),'-C',str(G),'cat-file','blob',asset_publication+':code/bw1000/training-internvl-USGS72-HIP-input-asset-v1/ASSET_FILES.json'],timeout=min(12,left()))
    assert old==(D/'DATA_CONTRACT.json').read_bytes()
    for name,wanted in contract['files'].items():
        assert left()>5
        path=G/P['local_asset_root']/name;assert pin(path)==wanted
        data_blobs[name]=base64.b64encode(path.read_bytes()).decode()

    local=G/'local-only'/(P['label']+'-original-dispatch');local.mkdir(mode=0o700)
    prefix='FILES='+repr(blobs)+'\nDATA_FILES='+repr(data_blobs)+'\nCONTEXT='+repr(dict(publication=publication,request_id=request,epoch=epoch,deadline=epoch+300,MF_SHA256=mf_sha))+'\n'
    body=prefix.encode()+(D/'receive_once.py').read_bytes();assert len(body)<=2*1024**2
    intent=dict(publication=publication,request_id=request,epoch=epoch,deadline=epoch+300,payload_SHA256=hashlib.sha256(body).hexdigest(),MF_SHA256=mf_sha,GPU=0,
                transport_kind='ROOT_COORDINATOR_INJECTED_CALLBACK_NO_DEFAULT',target_command=TARGET_COMMAND)
    save(local/'intent.json',intent)
    assert read(local/'intent.json')==dict(intent,target_command=list(TARGET_COMMAND))
    transport_context=dict(request_id=request,epoch=epoch,deadline=epoch+300,
        transport_stop_deadline=epoch+295,transport_reap_deadline=epoch+298,
        publication=publication,source_manifest_SHA256=mf_sha,payload_SHA256=intent['payload_SHA256'],
        scope=P['RUN'],target_command=TARGET_COMMAND)
    save(local/'attempt.json',dict(request_id=request,publication=publication,epoch=epoch,deadline=epoch+300,
        scope=P['RUN'],root_transport_start_about_to_be_called=True,exec_outcome='UNKNOWN_UNTIL_ORIGINAL_RECEIPT',replay_allowed=False))
    child=None;start_attempted=False;timed_out=False;reaped=False;failure=None;stdout=b'';stderr=b'';rc=None;cleanup_errors={}
    try:
        assert left()>5
        start_attempted=True
        candidate=transport_start(transport_context)
        if isinstance(candidate,subprocess.Popen):
            child=candidate  # Retain ownership even if its pipe contract is invalid.
        check_transport_child(candidate)
        assert left()>5
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
    stdout=stdout.encode('utf-8',errors='replace') if isinstance(stdout,str) else stdout or b''
    stderr=stderr.encode('utf-8',errors='replace') if isinstance(stderr,str) else stderr or b''
    for name,raw in (('stdout.private',stdout),('stderr.private',stderr)):
        with (local/name).open('xb') as stream:stream.write(raw)
    record=dict(request_id=request,publication=publication,transport_rc=rc,timed_out=timed_out,transport_start_attempted=start_attempted,
                held_transport_started=True if child is not None else None if start_attempted else False,
                held_transport_reaped=reaped,transport_kind='ROOT_COORDINATOR_INJECTED_CALLBACK_NO_DEFAULT',
                local_error_type=failure,cleanup_errors=cleanup_errors,client_wall_seconds=time.monotonic()-anchor,
                remote_outcome='UNKNOWN' if timed_out or failure or cleanup_errors or not reaped or rc in (None,255) else 'RETURNED_REQUIRES_ORIGINAL_RECEIPT_AUDIT',
                stdout_bytes=len(stdout),stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_bytes=len(stderr),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),replay_allowed=False,GPU=0)
    save(local/'transport.json',record);print(json.dumps(record));return 0 if rc==0 and not timed_out and failure is None and not cleanup_errors and reaped else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publication',required=True);parser.add_argument('--source-manifest-sha',required=True)
    a=parser.parse_args();raise SystemExit(main(a.publication,a.source_manifest_sha))
