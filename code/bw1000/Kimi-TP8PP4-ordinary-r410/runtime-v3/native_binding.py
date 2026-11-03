"""Typed ordinary ABI only; no old H6/batch/capture/commit/replay entry binding."""
import ctypes as C
import json,hashlib
from pathlib import Path
from lifecycle import GuardedLibrary
P=C.c_void_p;I=C.c_int;S=C.c_size_t;B=C.c_char_p
SIGNATURES={
 'modern_kimi_engine_create_ordinary_tp8_pp4_r410':([B,I,B,I],P),
 'modern_kimi_engine_forward_ordinary_r410':([P,I,S,S,I]+[P]*9,I),
 'modern_kimi_ordinary_capabilities_r410':([P],B),
 'modern_kimi_engine_error_r2475':([],B),
 'modern_kimi_engine_load_r2475':([P,B],I),
 'modern_kimi_engine_partition_r2475':([I,I,C.POINTER(S)],I),
 'modern_kimi_shard_head_info_r9183':([],B),
 'modern_kimi_shard_head_error_r9183':([],B),
 'modern_kimi_shard_head_expected_r9183':([B],I),
 'modern_kimi_vocab_head_get_r8341':([],I),
 'modern_kimi_vocab_head_set_r8341':([I],I),
 'modern_cpu_copy_stats_r2515':([C.POINTER(C.c_uint64),S],I),
 'modern_owner_affinity_report_r2392':([],B)}
for name in ['finish','reset','destroy']:SIGNATURES['modern_kimi_engine_'+name+'_r2475']=([P],I)
for name in ['keys','layout','cache_geometry','output_metrics']:SIGNATURES['modern_kimi_engine_'+name+'_r2475']=([P],B)
def declare(raw):
 for name,(args,result)in SIGNATURES.items():
  f=getattr(raw,name);f.argtypes=args;f.restype=result
 return raw
def bind(library,gate):
 library=Path(library);expected=json.loads((Path(__file__).resolve().parent/'CAPI_PROVIDERS.json').read_bytes())['symbols'];assert set(expected)==set(SIGNATURES)
 files={Path(spec['path']).resolve():spec['SHA256']for spec in expected.values()};assert len(files)==2
 actual_hashes={}
 for path,digest in files.items():actual_hashes[path]=hashlib.sha256(path.read_bytes()).hexdigest();assert actual_hashes[path]==digest,str(path)
 raw=declare(C.CDLL(str(library)))
 class Dl(C.Structure):_fields_=[('fname',B),('base',P),('sname',B),('addr',P)]
 dl=C.CDLL(None).dladdr;dl.argtypes=[P,C.POINTER(Dl)];dl.restype=I;providers={}
 for name in SIGNATURES:
  info=Dl();assert dl(C.cast(getattr(raw,name),P),C.byref(info)),name
  actual=Path(info.fname.decode()).resolve();spec=expected[name];actual_SHA256=actual_hashes.get(actual)
  if actual_SHA256 is None:actual_SHA256=hashlib.sha256(actual.read_bytes()).hexdigest()
  assert actual==Path(spec['path']).resolve()and actual_SHA256==spec['SHA256'],dict(symbol=name,actual=str(actual),actual_SHA256=actual_SHA256,expected=spec)
  providers[name]=str(actual)
 lib=GuardedLibrary(raw,gate)
 assert lib.modern_kimi_vocab_head_get_r8341()==-1 and lib.modern_kimi_vocab_head_set_r8341(0)!=0
 return lib,providers
