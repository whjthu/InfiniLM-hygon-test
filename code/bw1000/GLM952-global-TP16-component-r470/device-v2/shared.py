"""Fixed topology/scalars; HOST/container roots are explicit, never fuzzy."""
from pathlib import Path
import hashlib,json,os,time
D=Path(__file__).resolve().parent;HOST=Path('/public/home/aclsylqidf/hygon-inference-refactor-20261014');R=Path('/refactor')if str(D).startswith('/refactor/')else HOST
LABEL='GLM-global-TP16-comm-r470-device-v2';NODES=['a02r3n15','a02r3n16'];OUT=R/'results'/LABEL
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def actual(p):return R/str(p)[len('/refactor/'):]if str(p).startswith('/refactor/')else Path(p)
def save(p,q):
 p=Path(p);t=p.with_name(p.name+'.tmp-'+str(os.getpid()));t.write_text(json.dumps(q,indent=2)+'\n');os.replace(t,p)
def wait(p,epoch):
 while time.time()<epoch+150:
  try:return json.loads(Path(p).read_bytes())
  except(FileNotFoundError,json.JSONDecodeError):time.sleep(.1)
 raise TimeoutError('Original150s component worker epoch')
def verify(pin):
 assert H(D/'GPU_MANIFEST.json')==pin;m=json.loads((D/'GPU_MANIFEST.json').read_bytes());assert H(D/'SOURCE_MANIFEST.json')==m['source_manifest_SHA256'];s=json.loads((D/'SOURCE_MANIFEST.json').read_bytes());assert all(H(D/n)==h for n,h in s['files_SHA256'].items());assert H(D/'GPU_RESOURCE_CLAIM.json')==m['claim_SHA256'];b=json.loads((D/'CPU_BINDING.json').read_bytes());assert H(actual(b['receipt']))==b['receipt_SHA256'];cpu=json.loads(actual(b['receipt']).read_bytes());assert cpu['passed']and cpu['actual_CPU']==b['actual_CPU']=='910407.1067'and cpu['source_manifest_SHA256']==b['phase_sourceMF']and cpu['new_HOST']==cpu['new_CABI']==1 and cpu['fresh_M_equals_MD']and not cpu['Runtime']and not cpu['GPU'];assert H(actual(cpu['library']))==b['library_SHA256']==cpu['library_SHA256']
 for p,h in cpu['verified_libraries'].items():assert H(actual(p))==h,p
 assert H(D.parent/'binding.py')==b['binding_SHA256']and H(D/'binding_trace.py')==b['binding_trace_SHA256'];return m,cpu
