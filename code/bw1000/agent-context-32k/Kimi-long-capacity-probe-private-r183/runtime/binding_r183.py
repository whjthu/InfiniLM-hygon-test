"""Exact new long factory binding, never the legacy create_batch entry."""
import ctypes as C,json,pathlib,hashlib
P=C.c_void_p;I=C.c_int;S=C.c_size_t
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def bind(pins):
 path=pathlib.Path(pins['Native']);assert sha(path)==pins['Native_SHA256'];lib=C.CDLL(str(path))
 class Dl(C.Structure):_fields_=[('fname',C.c_char_p),('base',P),('sname',C.c_char_p),('addr',P)]
 dl=C.CDLL(None).dladdr;dl.argtypes=[P,C.POINTER(Dl)];dl.restype=I
 names=['modern_kimi_engine_create_long_agent_r179','modern_kimi_long_agent_capabilities_r179','modern_kimi_engine_error_r2475','modern_kimi_engine_destroy_r2475','modern_kimi_engine_layout_r2475','modern_kimi_engine_cache_geometry_r2475','modern_kimi_engine_keys_r2475','modern_kimi_engine_output_metrics_r2475','modern_kimi_engine_partition_r2475','modern_kimi_batch_reset_r9122','modern_kimi_batch_mode_r9122','modern_kimi_batch_capabilities_r9122','modern_kimi_shard_head_info_r9183']
 providers={}
 for name in names:
  info=Dl();assert dl(C.cast(getattr(lib,name),P),C.byref(info));actual=pathlib.Path(info.fname.decode()).resolve();assert actual==path.resolve();providers[name]={'path':str(actual),'SHA256':sha(actual)}
 lib.modern_kimi_engine_create_long_agent_r179.argtypes=[C.c_char_p,I,I,I,C.c_char_p,I];lib.modern_kimi_engine_create_long_agent_r179.restype=P
 for name in ['modern_kimi_engine_error_r2475','modern_kimi_shard_head_info_r9183']:getattr(lib,name).restype=C.c_char_p
 lib.modern_kimi_shard_head_info_r9183.argtypes=[]
 for name in ['modern_kimi_long_agent_capabilities_r179','modern_kimi_engine_layout_r2475','modern_kimi_engine_cache_geometry_r2475','modern_kimi_engine_keys_r2475','modern_kimi_engine_output_metrics_r2475','modern_kimi_batch_capabilities_r9122']:getattr(lib,name).argtypes=[P];getattr(lib,name).restype=C.c_char_p
 for name in ['modern_kimi_batch_reset_r9122','modern_kimi_batch_mode_r9122']:getattr(lib,name).argtypes=[P,I];getattr(lib,name).restype=I
 lib.modern_kimi_engine_destroy_r2475.argtypes=[P];lib.modern_kimi_engine_destroy_r2475.restype=I
 lib.modern_kimi_engine_partition_r2475.argtypes=[I,I,C.POINTER(S)];lib.modern_kimi_engine_partition_r2475.restype=I
 return lib,providers

def maps(pins):
 actual={};lines=pathlib.Path('/proc/self/maps').read_text().splitlines()
 for name,h in pins['loaded_library_SHA256'].items():
  paths={line.split()[-1]for line in lines if pathlib.Path(line.split()[-1]).name==name};assert len(paths)==1;path=next(iter(paths));assert sha(path)==h;actual[name]={'path':path,'SHA256':h}
 return actual
