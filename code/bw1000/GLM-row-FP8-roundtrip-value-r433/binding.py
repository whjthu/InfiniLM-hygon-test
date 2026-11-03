"""New policy only. Existing constructor/query/loader remain original APIs."""
import ctypes
def bind_policy(lib):
 fn=lib.glm_attention_value_roundtrip_r433
 fn.argtypes=[ctypes.c_void_p,ctypes.c_int]
 fn.restype=ctypes.c_int
 return fn
