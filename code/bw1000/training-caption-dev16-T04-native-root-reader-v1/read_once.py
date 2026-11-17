"""One metadata-only audit of the actual original CPU request, with published source pins."""
import hashlib
import json
from pathlib import Path
import subprocess

D = Path(__file__).resolve().parent
G = D.parents[2]
SOURCE = G/'code/bw1000/training-caption-dev16-T04-native-CPU-v1'
LOCAL = G/'local-only/training-caption-dev16-T04-native-CPU-v1-original-dispatch'
REMOTE = '/home/whj/nvidia-baseline/alignment-assets/training-caption-dev16-T04-native-CPU-v1'


def save(path, value):
    with path.open('x',encoding='utf8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)


def main():
    intent=json.loads((LOCAL/'exec-intent.json').read_bytes())
    transport=json.loads((LOCAL/'outcome.json').read_bytes())
    assert intent['request_id']==transport['request_id'] and type(transport['transport_rc']) is int
    assert transport['transport_rc']!=255 and transport.get('terminal') is not None
    assert transport['remote_outcome']=='ORIGINAL_RETURNED'
    source_mf=(SOURCE/'MF.json').read_bytes()
    assert hashlib.sha256(source_mf).hexdigest()==intent['source_MF_SHA256']
    for name,pin in json.loads(source_mf).items():
        raw=(SOURCE/name).read_bytes()
        assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    reader_mf=(D/'MF.json').read_bytes()
    published=subprocess.check_output(['git','--no-optional-locks','-c','safe.directory='+G.as_posix(),
        '-C',str(G),'show',intent['publication']+':'+D.relative_to(G).as_posix()+'/MF.json'],
        timeout=12,stderr=subprocess.DEVNULL)
    assert published==reader_mf
    for name,pin in json.loads(reader_mf).items():
        raw=(D/name).read_bytes()
        assert len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['SHA256']
    source_pin=hashlib.sha256((SOURCE/'SOURCE_PINS.json').read_bytes()).hexdigest()
    out=LOCAL/'independent-reader-once'
    out.mkdir(mode=0o700)
    remote_result=REMOTE+'/independent-original-reader-result.json'
    args=['published-metadata-reader','--run',REMOTE,'--publication',intent['publication'],
          '--request',intent['request_id'],'--epoch',str(intent['epoch']),
          '--source-pins-sha256',source_pin,'--output',remote_result]
    code=('import signal,sys,json\nfrom pathlib import Path\nsignal.alarm(60)\nsys.argv='+repr(args)+'\n'
          +'exec(compile('+repr((D/'read_original.py').read_bytes())+",'published-metadata-reader','exec'))\n"
          +'print(json.dumps(json.loads(Path('+repr(remote_result)+').read_bytes()),allow_nan=False))\n').encode()
    assert len(code)<=131072
    save(out/'intent.json',dict(request_id=intent['request_id'],publication=intent['publication'],
         original_epoch=intent['epoch'],source_pins_SHA256=source_pin,reader_seconds=60,
         read_only_inputs=True,actual_new_processor_calls=0,replay_allowed=False))
    argv=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','StrictHostKeyChecking=yes',
          '-F','C:/Users/WangHaojie/.codex/worktrees/b9ba/训练/nvidia_baseline/ssh_config',
          'a100-27','/home/whj/nvidia-baseline/env-vlm/bin/python -I -S -B -u -']
    stdout,stderr,rc=b'',b'',None
    try:
        child=subprocess.run(argv,input=code,capture_output=True,timeout=70)
        stdout,stderr,rc=child.stdout,child.stderr,child.returncode
    except subprocess.TimeoutExpired as error:
        stdout,stderr=error.stdout or b'',error.stderr or b''
    assert len(stdout)<=262144 and len(stderr)<=65536
    (out/'stdout.private.jsonl').write_bytes(stdout)
    (out/'stderr.private').write_bytes(stderr)
    docs=[]
    for line in stdout.splitlines():
        try:
            value=json.loads(line)
        except (ValueError,UnicodeError):
            continue
        if isinstance(value,dict) and value.get('original_request')==intent['request_id'] and 'pins' in value:
            docs.append(value)
    value=docs[0] if len(docs)==1 and rc==0 else None
    result=dict(returncode=rc,remote_outcome='UNKNOWN' if rc in (None,255) else 'RETURNED',
        stdout_SHA256=hashlib.sha256(stdout).hexdigest(),stderr_SHA256=hashlib.sha256(stderr).hexdigest(),
        passed=bool(value and value.get('passed')),value=value,new_processor_calls=0,GPU_GO=False,replay_allowed=False)
    save(out/'receipt.json',result)
    print(json.dumps(dict(returncode=rc,passed=result['passed'],producer_passed=value.get('producer_passed') if value else None,
         actual_native_inference_rows=value.get('actual_native_inference_rows') if value else None,
         missing_rows=value.get('missing_rows') if value else None,new_processor_calls=0,GPU_GO=False)))
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
