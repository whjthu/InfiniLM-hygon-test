"""Bind the smoke853-qualified shared-actual-caller Model; no fallback to a differently bound native."""
import pathlib,json,hashlib,ctypes,os
def load(r):
 sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
 m=json.loads((r/'incoming/Kimi-TP16PP2-I192-manifest-r8669zh.json').read_text());bpath=r/'results/Kimi-I192-smoke-static-closure-r8669zg.json';assert sha(bpath)==m['CPU_receipt_SHA256'];b=json.loads(bpath.read_text());assert b['completed']and b['CPU_only']and not b['GPU']and not b['dlopen'];library=pathlib.Path(b['native']);assert sha(library)==b['native_SHA256']==m['native_DSO_SHA256']
 for p,h in b['objects'].items():assert sha(p)==h,p
 for p,h in b['unchanged_objects'].items():assert sha(p)==h,p
 assert os.environ.get('INFINILM_KIMI_I192_GROUPED_R8669Y')=='0'
 assert os.environ.get('INFINILM_KIMI_I192_COMPACT_R8669ZC')=='1'
 for name,value in json.loads((r/'incoming/Kimi-head-Model-manifest-r8235.json').read_text())['ordinary_initial_flags'].items():assert os.environ.get(name)==value,(name,os.environ.get(name),value)
 assert all(os.environ.get(k)=='0'for k in ['INFINILM_KIMI_CCL512_R8603','INFINILM_KIMI_PREFILL512_R8562','INFINILM_KIMI_PREFILL512_ELIGIBLE_R8562'])
 lib=ctypes.CDLL(str(library));providers={}
 class Dl(ctypes.Structure):_fields_=[('fname',ctypes.c_char_p),('base',ctypes.c_void_p),('sname',ctypes.c_char_p),('addr',ctypes.c_void_p)]
 dl=ctypes.CDLL(None).dladdr;dl.argtypes=[ctypes.c_void_p,ctypes.POINTER(Dl)];dl.restype=ctypes.c_int
 for name in ['modern_kimi_engine_create_r2475','modern_kimi_engine_forward_r2475','modern_kimi_engine_destroy_r2475','modern_kimi_I192_counts_r8669y','modern_kimi_I192_workspace_r8669zc','modern_kimi_grouped_I192_r8669y','modern_kimi_vocab_head_set_r8341']:
  z=Dl();assert dl(ctypes.cast(getattr(lib,name),ctypes.c_void_p),ctypes.byref(z));p=pathlib.Path(z.fname.decode()).resolve();assert p==library.resolve();providers[name]={'path':str(p),'SHA256':sha(p)}
 c=lib.modern_kimi_I192_counts_r8669y;c.argtypes=[ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];c.restype=ctypes.c_int
 for d in range(8):
  values=(ctypes.c_uint64*2)();assert c(d,values,2)==0 and list(values)==[0,0]
 lib.modern_kimi_vocab_head_set_r8341.argtypes=[ctypes.c_int];assert lib.modern_kimi_vocab_head_set_r8341(0)==0;assert lib.modern_kimi_vocab_head_get_r8341()==0
 return library,lib,lib.modern_kimi_n4_tile_counts_r8167,{'CPU_receipt_SHA256':sha(bpath),'actualproviders':providers,'Model_SHA256':sha(library),'currentI192component':'841 fullMoE+21.47%,all16OFFONbyteexact; notModelqualification','defaultOFF':True,'provider_scope':'T16 TP16 only;flags3/7wideTP;allotherNfallback'}
