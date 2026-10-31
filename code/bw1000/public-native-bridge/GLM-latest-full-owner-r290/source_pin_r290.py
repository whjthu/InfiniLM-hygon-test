"""One externally pinned source manifest; no hash-as-its-own-expected fallback."""
import pathlib,hashlib,json,os,importlib.util
D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def verify(expected=None):
 expected=expected if expected is not None else os.environ['GLM_SOURCE_MANIFEST_SHA']
 if len(expected)!=64 or sha(D/'SOURCE_MANIFEST.json')!=expected:raise ValueError('Pinned GLM source manifest mismatch')
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():
  p=D/n
  if not p.resolve().is_relative_to(D.resolve())or sha(p)!=h:raise ValueError('GLM source member mismatch: '+n)
 return m

def load_wire(expected):
 m=verify(expected);p=D/'wire_client_r290.py'
 if sha(p)!=m['files_SHA256']['wire_client_r290.py']:raise ValueError('Pinned HTTP transport mismatch')
 spec=importlib.util.spec_from_file_location('glm290_real_wire',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def profiles(expected):
 verify(expected);b=json.loads((D/'INPUT_BINDING.json').read_bytes());p=pathlib.Path(b['profile_manifest_path']);cpu=pathlib.Path(b['CPU_sizing_path'])
 if sha(cpu)!=b['CPU_sizing_SHA256']or sha(p)!=b['profile_manifest_SHA256']:raise ValueError('Original CPU/profile binding mismatch')
 rows=json.loads(p.read_bytes())
 if rows!=b['profiles']:raise ValueError('Frozen profile members differ')
 for r in rows:
  if sha(r['input_file'])!=r['input_SHA256']:raise ValueError('Actual profile input SHA mismatch')
 return rows
