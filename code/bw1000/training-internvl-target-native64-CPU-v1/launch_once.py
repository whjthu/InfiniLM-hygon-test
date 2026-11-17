"""Source-only by default: root GO/pending checks precede all SSH and installation."""
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
from scope import D,P,read,pin,save,admission,verify

G=D.parents[2]
HELPER=Path('C:/Users/WangHaojie/.codex/visualizations/2026/10/12/01a130bd-3b97-73c2-8ecf-eda5151f1ba7/glm-paired-readonly-connection.py')
HELPER_SHA='3f00e0eac9b40eb3fee7d788776389bc22c3e87d20de0441312f04a2566178d1'


def main(publication,mf_sha):
    # As delivered, this fails before helper/config, namespace writes, or imports.
    admission(read(D/'CLAIM.json'),read(D/'PREREQUISITES.json'))
    assert re.fullmatch('[0-9a-f]{40}',publication) and re.fullmatch('[0-9a-f]{64}',mf_sha)
    epoch=time.time();anchor=time.monotonic()
    def left():
        value=min(epoch+900-time.time(),900-(time.monotonic()-anchor));assert value>0;return value
    verify();mfraw=(D/'SOURCE_FILES.json').read_bytes();assert hashlib.sha256(mfraw).hexdigest()==mf_sha
    raw=subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'show',publication+':'+D.relative_to(G).as_posix()+'/SOURCE_FILES.json'],timeout=min(12,left()))
    assert raw==mfraw
    assert not subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),'status','--porcelain=v1','--untracked-files=all','--',D.relative_to(G).as_posix()],timeout=min(12,left()))
    blobs={name:base64.b64encode((D/name).read_bytes()).decode() for name in json.loads(mfraw)['files']}
    blobs['SOURCE_FILES.json']=base64.b64encode(mfraw).decode();request=uuid.uuid4().hex
    local=G/'local-only'/P['label'];local.mkdir(mode=0o700)
    prefix='FILES='+repr(blobs)+'\nCONTEXT='+repr(dict(publication=publication,request_id=request,epoch=epoch,deadline=epoch+900,MF_SHA256=mf_sha))+'\n'
    body=prefix.encode()+(D/'receive_once.py').read_bytes();assert len(body)<=2*1024**2
    save(local/'intent.json',dict(publication=publication,request_id=request,epoch=epoch,deadline=epoch+900,payload_SHA256=hashlib.sha256(body).hexdigest(),MF_SHA256=mf_sha,GPU=0))
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest()==HELPER_SHA
    argv=runpy.run_path(str(HELPER))['ssh'];argv=[argv[0],'-o','StrictHostKeyChecking=yes',*argv[1:],'/usr/local/python3.12/bin/python3 -I -S -B -']
    try:
        result=subprocess.run(argv,input=body,capture_output=True,timeout=left())
        rc=result.returncode;stdout=result.stdout;stderr=result.stderr
    except subprocess.TimeoutExpired as error:rc=None;stdout=error.stdout or b'';stderr=error.stderr or b''
    for name,raw in (('stdout.private',stdout),('stderr.private',stderr)):
        with (local/name).open('xb') as stream:stream.write(raw)
    record=dict(request_id=request,publication=publication,transport_rc=rc,remote_outcome='UNKNOWN' if rc in (None,255) else 'RETURNED_REQUIRES_ORIGINAL_RECEIPT_AUDIT',
                stdout_bytes=len(stdout),stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_bytes=len(stderr),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),replay_allowed=False,GPU=0)
    save(local/'transport.json',record);print(json.dumps(record));return 0 if rc==0 else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publication',required=True);parser.add_argument('--source-manifest-sha',required=True)
    a=parser.parse_args();raise SystemExit(main(a.publication,a.source_manifest_sha))
