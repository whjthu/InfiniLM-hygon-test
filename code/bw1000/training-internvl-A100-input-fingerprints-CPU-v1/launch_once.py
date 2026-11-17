"""Future published-source dispatch; never run during source preparation."""
import argparse
import re
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

D=Path(__file__).resolve().parent
G=D.parents[2]
LOCAL=G/'local-only'/'training-internvl-A100-input-fingerprints-CPU-v1'
CONNECTION=Path('C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config')


def save(path,value):
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def cli(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--publication',required=True)
    parser.add_argument('--source-manifest-sha',required=True)
    parser.add_argument('--claim-sha256',required=True)
    args=parser.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{40}',args.publication):raise ValueError('exact publication required')
    if not all(re.fullmatch('[0-9a-f]{64}',v) for v in (args.source_manifest_sha,args.claim_sha256)):
        raise ValueError('exact MF and claim pins required')
    return args


def check_claim(claim):
    if not (claim['CPU_GO'] is True and claim['GPU_GO'] is False and claim['target_GO'] is False
            and claim['Model_GO'] is False and claim['case']=='T03.sft'
            and type(claim['maximum_CPU_invocations']) is int and claim['maximum_CPU_invocations']==1
            and claim['files']==64 and claim['LLaVA_included'] is False and claim['replay_allowed'] is False
            and claim['whole_seconds']==900 and claim['work_offset_seconds']==780):
        raise ValueError('published claim scope differs')


def main():
    args=cli();epoch,anchor=time.time(),time.monotonic()
    def remaining():
        value=min(epoch+900-time.time(),900-(time.monotonic()-anchor))
        if value<=0:raise TimeoutError('original client epoch exhausted')
        return value
    def git(*argv):
        return subprocess.check_output(['git','-c','safe.directory='+G.as_posix(),'-C',str(G),*argv],
            timeout=min(12,remaining()),stderr=subprocess.PIPE)
    if LOCAL.exists() or LOCAL.is_symlink():raise ValueError('fixed original local namespace already exists; no replay')
    relative=D.relative_to(G).as_posix()
    mfraw=(D/'SOURCE_FILES.json').read_bytes();manifest_sha=hashlib.sha256(mfraw).hexdigest();mf=json.loads(mfraw)
    if manifest_sha!=args.source_manifest_sha:raise ValueError('source manifest pin differs')
    if git('show',args.publication+':'+relative+'/SOURCE_FILES.json')!=mfraw:raise ValueError('published manifest differs')
    claimraw=(D/'CLAIM.json').read_bytes()
    if hashlib.sha256(claimraw).hexdigest()!=args.claim_sha256:raise ValueError('claim pin differs')
    if git('show',args.publication+':'+relative+'/CLAIM.json')!=claimraw:raise ValueError('published claim differs')
    check_claim(json.loads(claimraw))
    if git('status','--porcelain=v1','--untracked-files=all','--',relative):raise ValueError('own source is not clean')
    source={}
    for name,pin in mf['files'].items():
        part=Path(name)
        if part.is_absolute() or '..' in part.parts:raise ValueError('source member path differs')
        path=D/part
        if any(p.is_symlink() for p in (path,*path.parents)):raise ValueError('linked source path')
        raw=path.read_bytes()
        if len(raw)!=pin['bytes'] or hashlib.sha256(raw).hexdigest()!=pin['SHA256']:raise ValueError('source bytes differ')
        source[name]=raw
    if set(p.relative_to(D).as_posix() for p in D.rglob('*') if p.is_file())!=set(source)|{'SOURCE_FILES.json'}:
        raise ValueError('non-manifest file in source directory')
    source['SOURCE_FILES.json']=mfraw;commit=args.publication;request=uuid.uuid4().hex
    if any(p.is_symlink() for p in LOCAL.parents):raise ValueError('linked local namespace')
    LOCAL.mkdir(mode=0o700)  # fixed exclusive one-attempt scope; no renamed retry
    envelope={n:dict(bytes=len(v),SHA256=hashlib.sha256(v).hexdigest(),b64=base64.b64encode(v).decode()) for n,v in source.items()}
    payload=('PUBLICATION='+repr(commit)+'\nREQUEST='+repr(request)+'\nEPOCH='+repr(epoch)+'\nMANIFEST_SHA256='+repr(manifest_sha)+'\nFILES='+repr(envelope)+'\n').encode()+source['receive_once.py']
    save(LOCAL/'launch-intent.json',dict(request_id=request,publication=commit,epoch=epoch,deadline=epoch+900,
        source_manifest_SHA256=manifest_sha,claim_SHA256=args.claim_sha256,payload_SHA256=hashlib.sha256(payload).hexdigest(),
        exec_request_attempted=True,CPU_GO=True,GPU_GO=False,case='T03.sft',file_denominator=64,replay_allowed=False))
    # Existing connection and hostkey config only; never print/read credentials.
    remote='/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -u -'
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','StrictHostKeyChecking=yes','-F',str(CONNECTION),'a100-27',remote]
    rc=None;unknown=False
    try:
        result=subprocess.run(command,input=payload,capture_output=True,timeout=remaining())
        rc=result.returncode;stdout=result.stdout;stderr=result.stderr;unknown=rc==255
    except subprocess.TimeoutExpired as error:
        stdout=error.stdout or b'';stderr=error.stderr or b'';unknown=True
    except OSError as error:
        stdout=b'';stderr=type(error).__name__.encode();unknown=True
    for name,raw in [('original-stdout.private.jsonl',stdout),('original-stderr.private.txt',stderr)]:
        with (LOCAL/name).open('xb') as stream:stream.write(raw)
    docs=[]
    for line in stdout.splitlines():
        try:docs.append(json.loads(line))
        except (ValueError,UnicodeError):pass
    matching=[x for x in docs if type(x) is dict and x.get('request_id')==request]
    acks=[x for x in matching if x.get('stage')=='REMOTE_SOURCE_INSTALLED']
    terminal=[x for x in matching if x.get('stage')=='CPU_DERIVATION_TERMINAL']
    record=dict(request_id=request,publication=commit,epoch=epoch,transport_rc=rc,
        remote_outcome='UNKNOWN' if unknown or len(terminal)!=1 else 'ORIGINAL_RETURNED',
        source_acknowledged=len(acks)==1,terminal=terminal[0] if len(terminal)==1 else None,
        stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),
        stdout_bytes=len(stdout),stderr_bytes=len(stderr),replay_allowed=False,GPU_GO=False)
    save(LOCAL/'transport-outcome.json',record)
    print(json.dumps(record))
    return 0 if rc==0 and record['source_acknowledged'] and record['terminal'] and record['terminal']['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
