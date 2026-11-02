"""Exact Common/runtime path and hash checks; no library is opened by this module."""
import pathlib,hashlib
PREFIX='/refactor/'
EXT={'GLM':'16910b4d649f81ba3b63eea9e5bc7c71be5e698c06e00cc669b4ad4d0ec88b1a','Deep':'6e1a7e7643f84a52ab147cfc3c55eee4816affc6898109dbef31d559feaa86d1'}
RUNTIME={'GLM':'dc8542f14f70b85c67c227532f61756d432bbfee02308d373ff7ae5b9a540c36','Deep':'c04e19df5fa07bb52c0acf7be9768d258f5e2aa706647db32761e23426b153ec'}
def expected(model):
 assert model in EXT
 return {PREFIX+'candidates/latest-family-native-r269/'+model+'/python/_infinilm.cpython-310-x86_64-linux-gnu.so':EXT[model],PREFIX+'candidates/rebased-5f91ad7-r3668/Common/'+model+'/InfiniLM/python/infinilm/_lib/libinfinilm_runtime.so':RUNTIME[model]}
def check_metadata(model,paths,hashes):
 pins=expected(model);assert set(paths)==set(pins),('exact Common/runtime set',set(paths),set(pins))
 assert hashes==pins,('Common/runtime SHA mismatch',hashes,pins)
 return {'passed':True,'paths':sorted(pins),'SHA256':pins,'Model_qualified':False,'scope':'Actual mapping paths and bytes; not proof of all internal relocations or Model correctness'}
def validate_loaded(model,bindings=None):
 pins=expected(model)
 if bindings is not None:
  from_receipt={str(pathlib.Path(p).resolve()):v['SHA256']for p,v in bindings.items()if '_infinilm.cpython-'in p or 'libinfinilm_runtime.so'in p}
  assert from_receipt==pins,'Verified prebuild bindings disagree with pinned CPU600 identities'
 paths={str(pathlib.Path(v.split()[-1]).resolve())for v in pathlib.Path('/proc/self/maps').read_text().splitlines()if '_infinilm.cpython-'in v or 'libinfinilm_runtime.so'in v}
 assert paths==set(pins),('reject missing, foreign, or dual Common mappings',paths,set(pins))
 hashes={p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()for p in paths}
 return check_metadata(model,paths,hashes)
def validate_deep(bindings):return validate_loaded('Deep',bindings)
