"""Only typed r438 CABI declaration; loading a library belongs to device task."""
import ctypes as C
V=C.c_void_p;Z=C.c_size_t;I=C.c_int;U=C.c_uint64;D=C.c_double
SPECS={
 'error':(C.c_char_p,[]),'id_size':(Z,[]),'id':(I,[V]),
 'create':(V,[I,C.c_char_p,V]),'keys':(C.c_char_p,[V]),
 'load':(I,[V,C.c_char_p,V,Z,C.POINTER(Z),I]),'finish':(I,[V]),
 'forward':(I,[V,Z,V,V,I,I]),'compare_gu':(I,[V]),
 'proof_info':(I,[V,I,C.POINTER(Z),C.POINTER(Z),C.POINTER(I)]),
 'proof_read':(I,[V,I,V,Z]),'stats':(I,[V,C.POINTER(D),C.POINTER(U)]),
 'dtype':(I,[I]),'destroy':(I,[V]),'shutdown':(I,[])}
def bind(lib):
 for name,(result,args)in SPECS.items():
  function=getattr(lib,'deep_actquant_'+name+'_r438');function.argtypes=args;function.restype=result
 return lib
def check(lib,code):
 if code:raise RuntimeError(lib.deep_actquant_error_r438().decode())
