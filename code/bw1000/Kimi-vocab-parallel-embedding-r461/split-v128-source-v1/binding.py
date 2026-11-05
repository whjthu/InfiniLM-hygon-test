import ctypes as C,json
FACTORY='modern_kimi_engine_create_k192_v128_private'
CAPS='modern_kimi_k192_v128_capabilities_private'
def declare(lib):
 f=getattr(lib,FACTORY);f.argtypes=[C.c_char_p,C.c_int,C.c_char_p,C.c_int];f.restype=C.c_void_p
 f=getattr(lib,CAPS);f.argtypes=[C.c_void_p];f.restype=C.c_char_p
 return lib
def create(lib,config,node,address,port):
 if not config.get('infinilm_kimi_k192_v128_private')or not(0<=node<4)or not(0<port<65536):raise ValueError('explicit private TP32/C1 profile')
 ctx=getattr(declare(lib),FACTORY)(json.dumps(config,separators=(',',':')).encode(),node,address.encode(),port)
 if not ctx:raise RuntimeError('new private constructor returned null')
 return ctx
