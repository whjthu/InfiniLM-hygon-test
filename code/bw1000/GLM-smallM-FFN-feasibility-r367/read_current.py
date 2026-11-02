"""Read-only current038d FFN provenance; no dlopen, weights or GPU."""
from pathlib import Path
import json,hashlib,subprocess,datetime,re
R=Path.home()/'hygon-inference-refactor-20261014'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
host=lambda p:Path(str(p).replace('/refactor',str(R),1))
paths=[R/'candidates/GLM-PP4-registry-repair-r184b/CPU-build.json',R/'candidates/GLM-critical-observer-r131/CPU-build.json']
q=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,GPU=False,dlopen=False,weights_opened=False,receipts=[],current_sources=[])
for p in paths:
 a=json.loads(p.read_bytes());q['receipts'].append(dict(path=str(p),SHA256=sha(p),library=a['library'],library_SHA256=a['library_SHA256'],reused_objects_SHA256=a['reused_objects_SHA256'],compiled=a['compiled'],dependency_SHA256=a['dependency_SHA256']))
 for obj,h in a['reused_objects_SHA256'].items():
  if any(s in obj.lower()for s in ['vector','ffn','routed']):
   o=host(obj);entry=dict(object=str(o),SHA256=sha(o),expected_SHA256=h);assert entry['SHA256']==h
   md=o.with_suffix('.d')
   if md.exists():
    text=md.read_text();sources=[host(x)for x in re.findall(r'[^\s\\]+\.(?:cpp|cu|hpp|h)',text)]
    entry['MD_SHA256']=sha(md);entry['source_files']=[dict(path=str(s),SHA256=sha(s))for s in sources if s.exists()]
   q['current_sources'].append(entry)
a=json.loads(paths[0].read_bytes());lib=host(a['library']);assert sha(lib)==a['library_SHA256']=='038d4ff2e7aee2e6db80f691dd885222f3e3160e25ae9d0e269296daf9596c00'
q['dynamic']=subprocess.check_output(['readelf','-d',str(lib)],text=True,timeout=25)
nm=subprocess.check_output(['nm','-D','--defined-only',str(lib)],text=True,timeout=25)
q['FFN_exports']=[x for x in nm.splitlines()if any(s in x for s in ['FeedForwardR19','Fp8MlpR19','glm_vec_gate','glm_grouped','vector_policy','vector_counts','glm_layer_forward'])]
print(json.dumps(q))
