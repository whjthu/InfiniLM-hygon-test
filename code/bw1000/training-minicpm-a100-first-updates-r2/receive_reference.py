"""Stage one immutable source package, then invoke its one bounded supervisor."""
import base64,hashlib,json,subprocess,os
from pathlib import Path
PREVIOUS=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-a100-first-updates-v1')
assert hashlib.sha256((PREVIOUS/'prelaunch-result.json').read_bytes()).hexdigest()=='a29ae77c19056c5516fed76df409e1c40943b551afda465e8264422571ef7048'
assert not (PREVIOUS/'gpu-intent.json').exists()
ROOT=PREVIOUS.with_name('hygon-minicpm-a100-first-updates-r2')
assert not ROOT.exists(),'Existing namespace must be reconciled, never replayed'
ROOT.mkdir(mode=0o700);src=ROOT/'src';src.mkdir(mode=0o700)
pins={}
for name,item in FILES.items():
    assert Path(name).name==name
    raw=base64.b64decode(item['b64'],validate=True)
    assert hashlib.sha256(raw).hexdigest()==item['SHA256']
    with (src/name).open('xb') as f:f.write(raw)
    pins[name]=item['SHA256']
with (ROOT/'installed.json').open('x') as f:json.dump({'claim_commit':CLAIM,'files':pins},f)
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
try:
    with (ROOT/'supervisor.log').open('xb') as log:
        r=subprocess.run(['/home/whj/nvidia-baseline/env-vlm/bin/python','-B','-u',str(src/'run_reference_once.py')],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1020)
    value={'supervisor_rc':r.returncode,'claim_commit':CLAIM}
except subprocess.TimeoutExpired:
    value={'supervisor_rc':None,'claim_commit':CLAIM,'state':'SUPERVISOR_OBSERVATION_TIMEOUT_OUTCOME_UNKNOWN'}
for name in ('prelaunch-result.json','outcome.json'):
    if (ROOT/name).is_file():
        raw=(ROOT/name).read_bytes();value[name]={'SHA256':hashlib.sha256(raw).hexdigest(),'value':json.loads(raw)}
with (ROOT/'outer-result.json').open('x') as f:json.dump(value,f,indent=2)
print(json.dumps(value))
