"""Future one-shot readonly audit of this new T04 producer; no payload read."""
import hashlib
import json
from pathlib import Path
import subprocess
import time
from launch_once import D,G,LOCAL,CONNECTION,save


def main():
    intent=json.loads((LOCAL/'launch-intent.json').read_bytes())
    transport=json.loads((LOCAL/'transport-outcome.json').read_bytes())
    assert transport['request_id']==intent['request_id'] and transport['transport_rc']==0
    assert transport['remote_outcome']=='ORIGINAL_RETURNED' and transport['source_acknowledged']
    terminal=transport['terminal'];assert terminal['passed'] is True and terminal['own_worker_reaped']
    mfraw=(D/'SOURCE_FILES.json').read_bytes();mf=json.loads(mfraw)
    assert hashlib.sha256(mfraw).hexdigest()==intent['source_manifest_SHA256']
    for name,pin in mf['files'].items():
        raw=(D/name).read_bytes();assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    scope=dict(request_id=intent['request_id'],publication=intent['publication'],epoch=intent['epoch'],deadline=intent['deadline'],
               source_manifest=dict(bytes=len(mfraw),SHA256=hashlib.sha256(mfraw).hexdigest()),source_files=mf['files'],
               producer_plan=json.loads((D/'PLAN.json').read_bytes()),original_outcome=terminal['outcome'],
               read_limit_bytes=1024**2,reader_seconds=60,known_observed_peak_RSS_bytes=terminal['observed_worker_peak_RSS_bytes'])
    assert scope['deadline']==scope['epoch']+900
    out=LOCAL/'independent-original-reader-once';out.mkdir(mode=0o700)
    body=('S='+repr(scope)+'\n').encode()+(D/'metadata_reader.py').read_bytes()
    command=['ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=12',
             '-F',str(CONNECTION),'a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -u -']
    save(out/'intent.json',dict(request_id=scope['request_id'],read_only=True,epoch=time.time(),whole_seconds=65,
                              original64_payloads_read=0,replay_allowed=False))
    try:
        result=subprocess.run(command,input=body,capture_output=True,timeout=65)
        stdout,stderr=result.stdout,result.stderr;rc=result.returncode
        assert len(stdout)<=131072 and len(stderr)<=65536
        value=json.loads(stdout) if rc==0 else None
    except subprocess.TimeoutExpired as error:
        stdout,stderr=error.stdout or b'',error.stderr or b'';rc=None;value=None
    for name,raw in (('stdout.private.json',stdout),('stderr.private',stderr)):
        with (out/name).open('xb') as stream:stream.write(raw)
    receipt=dict(returncode=rc,remote_outcome='RETURNED' if rc not in (None,255) else 'UNKNOWN',
                 stdout_bytes=len(stdout),stdout_SHA256=hashlib.sha256(stdout).hexdigest(),
                 stderr_bytes=len(stderr),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),
                 passed=bool(value and value.get('passed')),value=value,replay_allowed=False,original64_payloads_read=0,GPU_GO=False)
    save(out/'receipt.json',receipt)
    print(json.dumps({key:value for key,value in receipt.items() if key!='value'}))
    return 0 if receipt['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
