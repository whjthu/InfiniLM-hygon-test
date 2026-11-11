"""Stage one immutable source package, then invoke its one bounded supervisor."""
import base64,hashlib,io,json,subprocess,os,tarfile
from pathlib import Path
ROOT=Path('/home/whj/nvidia-baseline/alignment-assets/hygon-minicpm-native-masks-reference-v1')
assert not ROOT.exists(),'Existing namespace must be reconciled, never replayed'
ROOT.mkdir(mode=0o700);src=ROOT/'src';src.mkdir(mode=0o700)
pins={}
for name,item in FILES.items():
    assert Path(name).name==name
    raw=base64.b64decode(item['b64'],validate=True)
    assert hashlib.sha256(raw).hexdigest()==item['SHA256']
    with (src/name).open('xb') as f:f.write(raw)
    pins[name]=item['SHA256']
manifest=json.loads((src/'FRAMEWORK_FILES.json').read_bytes())
assert manifest['framework_commit']=='88592dfb1a7ac70b217cd54eaac90b7dea2a179b' and len(manifest['files'])==275
raw=(src/'FRAMEWORK.tar.gz').read_bytes();assert hashlib.sha256(raw).hexdigest()==manifest['archive_SHA256']
seen=set()
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 for item in archive:
  path=Path(item.name);assert not path.is_absolute() and '..' not in path.parts and path.parts[0]=='framework'
  if item.isdir():continue
  assert item.isfile() and item.name in manifest['files'] and item.name not in seen
  data=archive.extractfile(item).read();row=manifest['files'][item.name]
  assert len(data)==row['bytes'] and hashlib.sha256(data).hexdigest()==row['SHA256']
  p=ROOT/item.name;p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(data)
  p.chmod(row['mode']);seen.add(item.name)
assert seen==set(manifest['files'])
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
