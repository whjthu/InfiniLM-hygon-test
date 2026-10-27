import ctypes as C,json,pathlib
from entry_contract_r9143 import sha
P=C.c_void_p;I=C.c_int;S=C.c_size_t

def bind(m,b):
 library=pathlib.Path(b['native']);lib=C.CDLL(str(library));lib.modern_kimi_engine_error_r2475.restype=C.c_char_p
 class Dl(C.Structure):_fields_=[('fname',C.c_char_p),('base',P),('sname',C.c_char_p),('addr',P)]
 dl=C.CDLL(None).dladdr;dl.argtypes=[P,C.POINTER(Dl)];dl.restype=I;providers={}
 names=['modern_kimi_model_events_configure_r8669zq','modern_kimi_model_events_read_r8669zq','modern_kimi_model_events_disable_r8669zq','modern_kimi_shard_head_info_r9183','modern_kimi_shard_head_expected_r9183','modern_kimi_shard_head_error_r9183','modern_kimi_vocab_head_get_r8341','modern_kimi_vocab_head_set_r8341','modern_kimi_engine_error_r2475','modern_kimi_engine_output_metrics_r2475','modern_kimi_engine_create_batch_r9122c','modern_kimi_engine_keys_r2475','modern_kimi_engine_load_r2475','modern_kimi_engine_finish_r2475','modern_kimi_engine_destroy_r2475','modern_kimi_engine_layout_r2475','modern_kimi_engine_cache_geometry_r2475','modern_kimi_engine_partition_r2475','modern_kimi_I192_T_counts_r9129']+['modern_kimi_batch_'+n+'_r9122'for n in ['reset','mode','prefill','forward','commit','replay','rows','stats','capabilities','taps_begin','taps_count','taps_info','taps_copy','taps_capabilities']]
 for name in names:
  f=getattr(lib,name);v=Dl();assert dl(C.cast(f,P),C.byref(v))and pathlib.Path(v.fname.decode()).resolve()==library.resolve();providers[name]=str(library)
 lib.modern_kimi_engine_create_batch_r9122c.argtypes=[C.c_char_p,I,I,I,C.c_char_p,I,S];lib.modern_kimi_engine_create_batch_r9122c.restype=P
 for name in ['keys','layout','cache_geometry','output_metrics']:
  f=getattr(lib,'modern_kimi_engine_'+name+'_r2475');f.argtypes=[P];f.restype=C.c_char_p
 for name in ['finish','reset','destroy']:getattr(lib,'modern_kimi_engine_'+name+'_r2475').argtypes=[P]
 lib.modern_kimi_engine_load_r2475.argtypes=[P,C.c_char_p];lib.modern_kimi_engine_partition_r2475.argtypes=[I,I,C.POINTER(S)];lib.modern_kimi_batch_capabilities_r9122.argtypes=[P];lib.modern_kimi_batch_capabilities_r9122.restype=C.c_char_p
 lib.modern_kimi_batch_taps_capabilities_r9122.argtypes=[];lib.modern_kimi_batch_taps_capabilities_r9122.restype=C.c_uint64;assert lib.modern_kimi_batch_taps_capabilities_r9122()==0x01200500
 lib.modern_kimi_shard_head_info_r9183.argtypes=[];lib.modern_kimi_shard_head_info_r9183.restype=C.c_char_p;lib.modern_kimi_shard_head_expected_r9183.argtypes=[C.c_char_p];lib.modern_kimi_shard_head_expected_r9183.restype=I;lib.modern_kimi_shard_head_error_r9183.argtypes=[];lib.modern_kimi_shard_head_error_r9183.restype=C.c_char_p;lib.modern_kimi_vocab_head_get_r8341.argtypes=[];lib.modern_kimi_vocab_head_get_r8341.restype=I;lib.modern_kimi_vocab_head_set_r8341.argtypes=[I];lib.modern_kimi_vocab_head_set_r8341.restype=I
 assert lib.modern_kimi_vocab_head_get_r8341()==-1 and lib.modern_kimi_vocab_head_set_r8341(0)!=0,'legacy full-head switch must be unsupported'
 return library,lib,providers

def verify_maps(m):
 maps=pathlib.Path('/proc/self/maps').read_text().splitlines();actual={}
 for name,h in m['loaded_library_SHA256'].items():
  paths={line.split()[-1]for line in maps if pathlib.Path(line.split()[-1]).name==name};assert len(paths)==1;path=next(iter(paths));assert sha(path)==h;actual[name]={'path':path,'SHA256':h}
 return actual
