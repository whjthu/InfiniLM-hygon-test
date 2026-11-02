"""One externally pinned source manifest; no hash-as-its-own-expected fallback."""
import pathlib,hashlib,json,os,importlib.util
D=pathlib.Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def verify(expected=None):
 expected=expected if expected is not None else os.environ['DEEP_SOURCE_MANIFEST_SHA']
 if len(expected)!=64 or sha(D/'SOURCE_MANIFEST.json')!=expected:raise ValueError('Pinned Deep source manifest mismatch')
 m=json.loads((D/'SOURCE_MANIFEST.json').read_bytes())
 for n,h in m['files_SHA256'].items():
  p=D/n
  if not p.resolve().is_relative_to(D.resolve())or sha(p)!=h:raise ValueError('Deep source member mismatch: '+n)
 return m

def load_wire(expected):
 m=verify(expected);p=D/'wire_client_r206b.py'
 if sha(p)!=m['files_SHA256']['wire_client_r206b.py']:raise ValueError('Pinned HTTP transport mismatch')
 spec=importlib.util.spec_from_file_location('deep281_real_wire',p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
