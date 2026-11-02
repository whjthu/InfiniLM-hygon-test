"""Read exact current dispatch source and existing whole-layer bridge contract."""
from pathlib import Path
import json,hashlib,re,subprocess,datetime
R=Path.home()/'hygon-inference-refactor-20261014';host=lambda p:Path(str(p).replace('/refactor',str(R),1));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,GPU=False,dlopen=False,weights_opened=False)
p=R/'candidates/GLM-down-row2-Model-r9284/CPU-build.json';a=json.loads(p.read_bytes());q['dispatch_receipt']=dict(path=str(p),SHA256=sha(p));q['dispatch_sources']=[]
for c in a.get('compiled',[]):
 if 'dispatch'in c.get('object',''):
  s=host(c['source']);q['dispatch_sources'].append(dict(path=str(s),SHA256=sha(s),text=s.read_text()))
if not q['dispatch_sources']:
 for s in (R/'incoming/GLM-down-row2-Model-r9284').glob('*.cpp'):q['dispatch_sources'].append(dict(path=str(s),SHA256=sha(s),text=s.read_text()))
q['r9284_keys']=list(a);q['M1_previous']={}
for p in [R/'candidates/GLM-down-row2-r9281/CPU-build.json',R/'candidates/glm-vector-model-r8542/CPU-build.json']:
 a=json.loads(p.read_bytes());q['M1_previous'][str(p)]=dict(SHA256=sha(p),keys=list(a),compiled=a.get('compiled'))
md=R/'candidates/glm-transient-layer-r5681/private-0.d'
if md.exists():
 q['bridge_MD_SHA256']=sha(md)
 for name in re.findall(r'[^\s\\]+\.cpp',md.read_text()):
  p=host(name)
  if p.exists()and'bridge'in str(p):q['bridge']=dict(path=str(p),SHA256=sha(p),text=p.read_text())
sdk=R/'results/latest-sdk-r3672-GLM-build.json';a=json.loads(sdk.read_bytes());q['SDK_receipt']=dict(path=str(sdk),SHA256=sha(sdk),library_sha256=a['library_sha256'])
print(json.dumps(q))
