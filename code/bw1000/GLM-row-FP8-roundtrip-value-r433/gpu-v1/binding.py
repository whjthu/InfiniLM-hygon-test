"""New policy only. Existing constructor/query/loader remain original APIs."""
import ctypes
def bind_policy(lib):
 fn=lib.glm_attention_value_roundtrip_r433
 fn.argtypes=[ctypes.c_void_p,ctypes.c_int]
 fn.restype=ctypes.c_int
 return fn
def bind_diagnostic(lib):
 lib.glm_attention_diag_arm_r433.argtypes=[ctypes.c_void_p,ctypes.c_int];lib.glm_attention_diag_arm_r433.restype=ctypes.c_int
 lib.glm_attention_diag_desc_r433.argtypes=[ctypes.c_void_p];lib.glm_attention_diag_desc_r433.restype=ctypes.c_char_p
 lib.glm_attention_diag_read_r433.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t];lib.glm_attention_diag_read_r433.restype=ctypes.c_int
 lib.glm_attention_diag_clear_r433.argtypes=[ctypes.c_void_p];lib.glm_attention_diag_clear_r433.restype=ctypes.c_int
 return lib
