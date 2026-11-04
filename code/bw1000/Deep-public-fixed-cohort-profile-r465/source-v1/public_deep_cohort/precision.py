"""Select the original existing mode0. No new math, provider or SDK code."""
import ctypes as C
import hashlib
from pathlib import Path
NATIVE_SHA='a314ae6891977316189f875b793ed3f2e2eb0898110ad0cc382d2569eaf31dc1'

def verify_provider(fn,path):
 class Info(C.Structure):_fields_=[('fname',C.c_char_p),('base',C.c_void_p),('sname',C.c_char_p),('saddr',C.c_void_p)]
 dl=C.CDLL(None).dladdr;dl.argtypes=[C.c_void_p,C.POINTER(Info)];dl.restype=C.c_int
 q=Info();assert dl(C.cast(fn,C.c_void_p),C.byref(q))
 actual=Path(q.fname.decode()).resolve();assert actual==Path(path).resolve()
 assert hashlib.sha256(actual.read_bytes()).hexdigest()==NATIVE_SHA
 return dict(path=str(actual),SHA256=NATIVE_SHA)

def select_original(lib,path):
 fn=lib.modern_deep_precision_policy_r3443;fn.argtypes=[C.c_int];fn.restype=C.c_int
 provider=verify_provider(fn,path);status=fn(0)
 assert status==0,'Original precision policy0 rejected; no alternate mode'
 return dict(policy=0,status=status,provider=provider,setting_before_context=True,
   FP32_expected_delta_per_owner=0,F64_allowed=False,
   readback_scope='Native setter acknowledges0 at quiescence; every original forward additionally proves all8 FP32 counters delta0')
