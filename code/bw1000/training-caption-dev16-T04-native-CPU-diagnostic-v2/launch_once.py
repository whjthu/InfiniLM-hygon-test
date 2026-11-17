"""One published fixed16 CPU dispatch; original client epoch includes all upload time."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import uuid

D = Path(__file__).resolve().parent
G = D.parents[2]
LOCAL = G/'local-only/training-caption-dev16-T04-native-CPU-diagnostic-v2-original-dispatch'
CONFIG = 'C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config'


def need(ok, why):
    if not ok:
        raise ValueError(why)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def pinned(path, pin, cap):
    before = path.lstat()
    need(path.is_file() and not path.is_symlink() and before.st_size <= cap, 'Unsafe bounded input')
    raw = path.read_bytes()
    after = path.lstat()
    need(all(getattr(before,k) == getattr(after,k) for k in ('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns')),
         'Input changed during read')
    need(len(raw) == pin['bytes'] and sha(raw) == pin['SHA256'], 'Frozen input SHA differs')
    return raw


def main(a):
    epoch, anchor = time.time(), time.monotonic()
    deadline = epoch+300
    need(re.fullmatch('[0-9a-f]{40}',a.publication) and
         all(re.fullmatch('[0-9a-f]{64}',x) for x in (a.claim_sha256,a.source_mf_sha256)), 'Exact publication pins required')
    need(not LOCAL.exists(), 'Original fixed scope exists: reconcile, never replay')
    LOCAL.mkdir(mode=0o700)
    request = uuid.uuid4().hex
    intent = dict(request_id=request, publication=a.publication, claim_SHA256=a.claim_sha256,
                  source_MF_SHA256=a.source_mf_sha256, epoch=epoch, deadline=deadline,
                  exec_request_attempted=False, planned_rows=16, replay_allowed=False)
    save(LOCAL/'intent.json', intent)
    result = dict(intent, passed=False, transport_rc=None, remote_outcome='NOT_ATTEMPTED', GPU_GO=False)
    stdout, stderr = b'', b''

    def remaining():
        value = min(deadline-time.time(),300-(time.monotonic()-anchor))
        need(value>0, 'Original client300s exhausted')
        return value

    def show(name):
        return subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),
            '-C',str(G),'show',a.publication+':'+name], timeout=min(12,remaining()), stderr=subprocess.DEVNULL)

    try:
        relative = D.relative_to(G).as_posix()
        mfraw = (D/'MF.json').read_bytes()
        need(sha(mfraw)==a.source_mf_sha256 and show(relative+'/MF.json')==mfraw, 'Published source MF differs')
        mf = json.loads(mfraw)
        need({p.name for p in D.iterdir() if p.is_file()} == set(mf)|{'MF.json'}, 'Source inventory differs')
        blobs = {name:pinned(D/name,pin,1024**2) for name,pin in mf.items()}
        claim = json.loads(blobs['CLAIM.json'])
        need(sha(blobs['CLAIM.json'])==a.claim_sha256 and show(relative+'/CLAIM.json')==blobs['CLAIM.json'], 'Published claim differs')
        need(claim['CPU_GO_future_once_after_publication_and_fresh_gate'] is True and claim['max_CPU_invocations']==1
             and claim['whole_epoch_seconds']==300 and claim['inside_deadline_seconds']==260
             and claim['supervisor_work_seconds']==270 and claim['CPU_affinity_excludes']==[0,1]
             and all(claim[k] is False for k in ('GPU_GO','training_GO','Dataset_GO')), 'CPU-only original scope differs')
        dirty = subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),
            '-C',str(G),'status','--porcelain','--untracked-files=all','--',relative],timeout=min(12,remaining()),stderr=subprocess.DEVNULL)
        need(not dirty, 'Own source is not clean')
        need(claim['private_DATA_MANIFEST_SHA256']=='5b758107194df4f2599cad3d2598fb4f012eb220c3a75b1052db4d64eac3fcd1' and claim['diagnostic_scope'] is True and claim['uploaded_JPEG_bytes']==0 and claim['uploaded_model_metadata_bytes']==0, 'Diagnostic-only reuse boundary differs')
        files = {}
        for name in claim['runtime_src_files']:
            need(Path(name).name==name and name in blobs, 'Unsafe runtime source')
            raw = blobs[name]
            files['src/'+name] = dict(bytes=len(raw),SHA256=sha(raw),b64=base64.b64encode(raw).decode())
        need(len(files)==5,'Exact diagnostic runtime sources required')
        prefix = ('CLAIM='+repr(a.publication)+'\nCLAIM_SHA256='+repr(a.claim_sha256)+'\nREQUEST='+repr(request)
                  +'\nEPOCH='+repr(epoch)+'\nDATA_MANIFEST_SHA256='+repr(claim['private_DATA_MANIFEST_SHA256'])+'\nFILES='+repr(files)+'\n').encode()
        payload = prefix+blobs['receive_once.py']
        need(len(payload)<=262144, 'Upload exceeds original bounded data envelope')
        save(LOCAL/'exec-intent.json',dict(intent,exec_request_attempted=True,payload_bytes=len(payload),
             payload_SHA256=sha(payload),runtime_src_files=5,data_files=0,uploaded_JPEG_bytes=0,uploaded_model_metadata_bytes=0,diagnostic_scope=True,data_manifest_SHA256=claim['private_DATA_MANIFEST_SHA256']))
        result['exec_request_attempted'] = True
        argv = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','StrictHostKeyChecking=yes',
                '-F',CONFIG,'a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -u -']
        child = subprocess.run(argv,input=payload,capture_output=True,timeout=remaining())
        stdout, stderr = child.stdout, child.stderr
        need(len(stdout)<=262144 and len(stderr)<=65536, 'Bounded transport logs exceeded')
        docs = []
        for line in stdout.splitlines():
            try:
                value=json.loads(line)
            except (ValueError,UnicodeError):
                continue
            if isinstance(value,dict) and value.get('request_id')==request:
                docs.append(value)
        terminal = [v for v in docs if 'passed' in v and 'owned_group_reaped' in v]
        result.update(transport_rc=child.returncode,source_acknowledged=any(v.get('stage')=='REMOTE_SOURCE_AND_FIXED16_INSTALLED' for v in docs),
                      remote_outcome='UNKNOWN' if child.returncode==255 or len(terminal)!=1 else 'ORIGINAL_RETURNED',
                      terminal=terminal[0] if len(terminal)==1 else None)
        result['passed']=bool(child.returncode==0 and result['source_acknowledged'] and len(terminal)==1 and terminal[0]['passed'])
    except subprocess.TimeoutExpired as error:
        stdout,stderr=error.stdout or b'',error.stderr or b''
        result.update(remote_outcome='UNKNOWN',error_type='TimeoutExpired')
    except BaseException as error:
        result.update(error_type=type(error).__name__)
        if result['exec_request_attempted']:
            result['remote_outcome']='UNKNOWN'
    (LOCAL/'stdout.private.jsonl').write_bytes(stdout)
    (LOCAL/'stderr.private').write_bytes(stderr)
    result.update(stdout_SHA256=sha(stdout),stderr_SHA256=sha(stderr),stdout_bytes=len(stdout),stderr_bytes=len(stderr),
                  client_elapsed_seconds=time.monotonic()-anchor)
    save(LOCAL/'outcome.json',result)
    print(json.dumps({k:result.get(k) for k in ('request_id','epoch','deadline','passed','transport_rc','remote_outcome',
          'source_acknowledged','exec_request_attempted','error_type','stdout_bytes','stderr_bytes','GPU_GO')}))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--publication',required=True)
    parser.add_argument('--claim-sha256',required=True)
    parser.add_argument('--source-mf-sha256',required=True)
    raise SystemExit(main(parser.parse_args()))
