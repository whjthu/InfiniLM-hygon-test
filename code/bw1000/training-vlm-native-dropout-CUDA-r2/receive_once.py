"""Install the exact published test without changing the original A100 environment."""
import base64,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-native-dropout-CUDA-r2');ROOT.mkdir(mode=0o700)
src=ROOT/'src';src.mkdir(mode=0o700);pins={}
for name,item in FILES.items():
    assert Path(name).name==name
    raw=base64.b64decode(item['b64'],validate=True);assert hashlib.sha256(raw).hexdigest()==item['SHA256']
    with (src/name).open('xb') as f:f.write(raw)
    pins[name]=item['SHA256']
with (ROOT/'installed.json').open('x') as f:json.dump({'claim_commit':CLAIM,'files':pins},f)
with (ROOT/'controller.log').open('xb') as log:
    r=subprocess.run(['python3','-B','-u',str(src/'run_once.py')],stdout=log,stderr=subprocess.STDOUT,timeout=210)
value={'controller_rc':r.returncode,'claim_commit':CLAIM}
for name in ('prelaunch.json','outcome.json'):
    if (ROOT/name).is_file():
        raw=(ROOT/name).read_bytes();v=json.loads(raw)
        if name=='outcome.json':v={k:x for k,x in v.items() if k not in ('result.json','progress.json')}
        value[name]={'SHA256':hashlib.sha256(raw).hexdigest(),'value':v}
with (ROOT/'outer-result.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
